"""
Job Executor Service
Consumes jobs from queue, selects optimal instance, and executes training on Lambda Labs

Features:
- Smart job selection based on max_spend constraints
- Discord approval workflow before launching instances
- Handles approval timeouts and rejections
- SSH-based training execution on Lambda Labs instances
- CONCURRENT job execution - multiple jobs run in parallel on separate VMs
"""
import logging
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Set
import tempfile
import os
import requests
from concurrent.futures import ThreadPoolExecutor, Future

from ..config import Config, LambdaConfig, TrainingConfig, GCSConfig
from ..models.job import TrainingJob, JobStatus, PriceData, BuyOption, ApprovalStatus
from .cache import CacheService
from .price_monitor import PriceMonitorService
from .discord import DiscordService
from .lambda_client import LambdaLabsClient, LambdaAPIError

logger = logging.getLogger(__name__)

# Maximum concurrent jobs
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "10"))


class JobExecutorService:
    """
    Service that:
    1. Consumes jobs from the queue using smart selection
    2. Selects the optimal instance type based on cached prices and max_spend
    3. Requests Discord approval before launching
    4. Launches Lambda Labs instance and runs training via SSH
    5. Terminates instances after completion
    
    Smart Selection:
    - Considers all jobs in queue, not just the first one
    - Matches jobs to available prices based on max_spend constraint
    - If job A can't afford any price but job B can, job B gets dispatched first
    
    Concurrency:
    - Multiple jobs can run simultaneously on different VMs
    - Each job execution runs in its own thread
    - Non-blocking: main loop continues processing new jobs
    """
    
    def __init__(
        self,
        config: Config,
        cache: CacheService,
        price_monitor: PriceMonitorService,
        discord_service: Optional[DiscordService] = None
    ):
        self.config = config
        self.cache = cache
        self.price_monitor = price_monitor
        self.discord = discord_service
        self.lambda_client = LambdaLabsClient(config.lambda_labs.api_key)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Concurrent execution tracking
        self._executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)
        self._running_jobs: Dict[str, Future] = {}  # job_id -> Future
        self._running_jobs_lock = threading.Lock()
    
    @property
    def running_job_count(self) -> int:
        """Number of currently running jobs"""
        with self._running_jobs_lock:
            # Clean up completed futures
            completed = [jid for jid, future in self._running_jobs.items() if future.done()]
            for jid in completed:
                del self._running_jobs[jid]
            return len(self._running_jobs)
    
    @property
    def running_job_ids(self) -> List[str]:
        """List of currently running job IDs"""
        with self._running_jobs_lock:
            return list(self._running_jobs.keys())
    
    def _send_callback(self, job: TrainingJob) -> None:
        """Send callback notification when job completes/fails"""
        if not job.callback_url:
            return
            
        try:
            payload = {
                "job_id": job.job_id,
                "status": job.status.value,
                "data_gcs_path": job.data_gcs_path,
                "output_gcs_path": job.output_gcs_path,
                "instance_type": job.instance_type,
                "region": job.region,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "error_message": job.error_message,
                "buy_option": job.buy_option,
            }
            
            response = requests.post(
                job.callback_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.ok:
                logger.info(f"Callback sent for job {job.job_id} to {job.callback_url}")
            else:
                logger.warning(f"Callback failed for job {job.job_id}: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send callback for job {job.job_id}: {e}")
    
    def start(self) -> None:
        """Start the job executor loop in a background thread"""
        if self._running:
            logger.warning("Job executor already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._executor_loop, daemon=True)
        self._thread.start()
        logger.info("Job executor started (Lambda Labs)")
    
    def stop(self) -> None:
        """Stop the job executor loop"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        
        # Shutdown thread pool (don't wait for running jobs)
        self._executor.shutdown(wait=False)
        logger.info(f"Job executor stopped ({self.running_job_count} jobs still running)")
    
    def _executor_loop(self) -> None:
        """
        Main executor loop - smart job selection and CONCURRENT execution.
        
        Instead of simple FIFO, we:
        1. Look at all queued jobs
        2. Look at all available prices
        3. Find the best match (earliest job that can afford a price)
        4. Request approval before launching
        5. Execute in a separate thread (non-blocking)
        
        Concurrency:
        - Jobs are submitted to a thread pool
        - Loop continues immediately to process more jobs
        - Multiple jobs can run simultaneously on different VMs
        """
        while self._running:
            try:
                # Check if we can accept more jobs
                if self.running_job_count >= MAX_CONCURRENT_JOBS:
                    logger.debug(f"At max concurrent jobs ({MAX_CONCURRENT_JOBS}), waiting...")
                    time.sleep(10)
                    continue
                
                # Use smart selection to find a job that can be executed
                result = self._smart_select_job()
                
                if result:
                    job, buy_option = result
                    
                    # Submit job to thread pool (non-blocking)
                    future = self._executor.submit(
                        self._execute_job_with_approval, 
                        job, 
                        buy_option
                    )
                    
                    with self._running_jobs_lock:
                        self._running_jobs[job.job_id] = future
                    
                    logger.info(
                        f"Job {job.job_id} submitted for execution "
                        f"(running: {self.running_job_count}/{MAX_CONCURRENT_JOBS})"
                    )
                    
                    # Brief pause before checking for more jobs
                    time.sleep(2)
                else:
                    # No viable job/price combination, wait and retry
                    time.sleep(10)
                    
            except Exception as e:
                logger.error(f"Error in executor loop: {e}")
                time.sleep(10)
    
    def _smart_select_job(self) -> Optional[Tuple[TrainingJob, BuyOption]]:
        """
        Smart job selection:
        - Get all jobs in queue
        - Get all available prices
        - For each job (in queue order), check if any price is within max_spend
        - Return the first job that has an affordable price, along with the cheapest option
        """
        # Get all queued jobs
        queued_jobs = self.cache.get_all_queued_jobs()
        
        if not queued_jobs:
            return None
        
        # Get all available prices sorted by price_per_gpu_hour
        all_prices = self.price_monitor.get_current_prices()
        available_prices = [p for p in all_prices if p.available]
        available_prices.sort(key=lambda p: p.price_per_gpu_hour)
        
        if not available_prices:
            # No prices available, trigger a poll
            logger.debug("No available prices, polling Lambda Labs...")
            self.price_monitor.poll_once()
            available_prices = [p for p in self.price_monitor.get_current_prices() if p.available]
            available_prices.sort(key=lambda p: p.price_per_gpu_hour)
            
            if not available_prices:
                return None
        
        logger.debug(f"Smart selection: {len(queued_jobs)} jobs, {len(available_prices)} price options")
        
        # For each job in queue order, find the best affordable option
        for job in queued_jobs:
            best_option = self._find_best_option_for_job(job, available_prices)
            
            if best_option:
                # Found a match! Remove job from queue and return
                removed = self.cache.remove_job_from_queue(job.job_id)
                if removed:
                    logger.info(
                        f"Smart selection: Job {job.job_id} matched with "
                        f"{best_option.instance_type} @ ${best_option.price_per_gpu_hour:.2f}/GPU/hr "
                        f"(max_spend: ${job.max_spend or 'unlimited'})"
                    )
                    return (removed, best_option)
        
        # No job could afford any available price
        logger.debug("Smart selection: No jobs can afford current prices")
        return None
    
    def _find_best_option_for_job(
        self, 
        job: TrainingJob, 
        available_prices: List[PriceData]
    ) -> Optional[BuyOption]:
        """
        Find the best price option that a job can afford.
        
        Preference order:
        1. More GPUs are preferred (8 > 4 > 1) for faster training
        2. Within same GPU count, prefer cheaper options
        
        Considers:
        - max_spend: max price per GPU per hour
        - max_total_cost: total budget cap (estimated based on max_duration_hours)
        - min_gpus/max_gpus: GPU count requirements
        """
        # Sort by GPU count (descending), then by price_per_gpu_hour (ascending)
        # This prefers more GPUs when affordable
        sorted_prices = sorted(
            available_prices,
            key=lambda p: (-p.gpu_count, p.price_per_gpu_hour)
        )
        
        for price in sorted_prices:
            # Check GPU count constraint
            if not (job.min_gpus <= price.gpu_count <= job.max_gpus):
                continue
            
            # Check max_spend constraint (per GPU per hour)
            if job.max_spend is not None and price.price_per_gpu_hour > job.max_spend:
                continue
            
            # Check total cost constraint (estimated based on max_duration)
            estimated_total = price.price_per_gpu_hour * price.gpu_count * job.max_duration_hours
            if job.max_total_cost is not None and estimated_total > job.max_total_cost:
                continue
            
            # Get best region for this instance type
            region = self.price_monitor.get_best_region_for_instance_type(price.gpu_type)
            if not region:
                continue
            
            # Get hourly price from metadata
            price_per_hour = price.price_per_gpu_hour * price.gpu_count
            if price.metadata:
                price_per_hour = price.metadata.get("price_per_hour", price_per_hour)
            
            # This option works!
            return BuyOption(
                gpu_count=price.gpu_count,
                duration_hours=job.max_duration_hours,
                price_per_gpu_hour=price.price_per_gpu_hour,
                total_price=estimated_total,
                available=True,
                instance_type=price.gpu_type,
                region=region,
                price_per_hour=price_per_hour
            )
        
        return None
    
    def _execute_job_with_approval(self, job: TrainingJob, buy_option: BuyOption) -> None:
        """
        Execute a job with Discord approval workflow.
        
        Flow:
        1. Request approval via Discord
        2. Wait for approval response
        3. If approved, launch instance
        4. If rejected, move job to back of queue
        5. If approval times out, treat as rejection
        6. If launch fails after approval, keep job at front of queue
        """
        logger.info(f"Executing job {job.job_id} with approval workflow")
        
        try:
            job.status = JobStatus.SELECTING
            job.started_at = datetime.utcnow().isoformat()
            job.buy_option = buy_option.to_dict()
            job.instance_type = buy_option.instance_type
            job.region = buy_option.region
            self.cache.update_job(job)
            
            # Check if Discord approval is required
            if self.discord and self.discord.is_enabled():
                # Request approval
                approval_result = self._request_and_wait_for_approval(job, buy_option)
                
                if approval_result == ApprovalStatus.REJECTED:
                    # Move job to back of queue
                    logger.info(f"Job {job.job_id} rejected, moving to back of queue")
                    job.status = JobStatus.PENDING
                    job.approval_request_id = None
                    self.cache.update_job(job)
                    self.cache.requeue_job_to_back(job)
                    if self.discord:
                        self.discord.send_rejection_notification(job, self.cache.get_approval_request_by_job(job.job_id))
                    return
                
                elif approval_result == ApprovalStatus.EXPIRED:
                    # Treat expired as soft rejection - move to back
                    logger.info(f"Job {job.job_id} approval expired, moving to back of queue")
                    job.status = JobStatus.PENDING
                    job.approval_request_id = None
                    self.cache.update_job(job)
                    self.cache.requeue_job_to_back(job)
                    return
            
                # approval_result == ApprovalStatus.APPROVED - continue with launch
            
            # Execute the job (launch instance and run training)
            self._execute_job(job, buy_option)
            
        except Exception as e:
            logger.error(f"Error in approval workflow for job {job.job_id}: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow().isoformat()
            self.cache.update_job(job)
            self._send_callback(job)
        finally:
            # Remove from running jobs tracking
            with self._running_jobs_lock:
                self._running_jobs.pop(job.job_id, None)
    
    def _request_and_wait_for_approval(
        self, 
        job: TrainingJob, 
        buy_option: BuyOption
    ) -> ApprovalStatus:
        """Send approval request and wait for response."""
        job.status = JobStatus.AWAITING_APPROVAL
        self.cache.update_job(job)
        
        # Send approval request
        request = self.discord.send_approval_request(job, buy_option)
        
        if not request:
            logger.warning(f"Failed to send approval request for job {job.job_id}, auto-approving")
            return ApprovalStatus.APPROVED
        
        job.approval_request_id = request.request_id
        self.cache.update_job(job)
        
        logger.info(f"Waiting for approval of request {request.request_id}")
        
        # Wait for response
        status = self.discord.wait_for_approval(request)
        
        return status
    
    def _execute_job(self, job: TrainingJob, buy_option: BuyOption) -> None:
        """Execute a single training job (after approval)"""
        logger.info(f"Executing job {job.job_id}: {job.data_gcs_path}")
        
        try:
            logger.info(
                f"Selected option: {buy_option.instance_type} in {buy_option.region} "
                f"({buy_option.gpu_count} GPUs @ ${buy_option.price_per_hour:.2f}/hr)"
            )
            
            # Step 1: Launch instance
            job.status = JobStatus.BUYING  # Using BUYING status for launching
            self.cache.update_job(job)
            
            instance_id = self._launch_instance(job, buy_option)
            
            if not instance_id:
                # Launch failed - notify Discord and keep at front of queue
                job.status = JobStatus.PENDING
                job.error_message = "Failed to launch instance - capacity may have changed"
                self.cache.update_job(job)
                
                if self.discord:
                    self.discord.send_buy_result(job, success=False, message="Launch failed, job staying at front of queue")
                
                # Requeue to front for immediate retry
                self.cache.requeue_job_to_front(job)
                logger.warning(f"Job {job.job_id} launch failed, requeued to front")
                return
            
            job.instance_id = instance_id
            job.vm_id = instance_id  # Backwards compatibility
            
            # Notify Discord of successful launch
            if self.discord:
                self.discord.send_buy_result(job, success=True, message=f"Launched instance {instance_id}")
            
            # Step 2: Wait for instance to be ready
            job.status = JobStatus.PROVISIONING
            self.cache.update_job(job)
            
            instance = self._wait_for_instance_ready(job)
            if not instance:
                job.status = JobStatus.FAILED
                job.error_message = "Instance provisioning timed out"
                job.completed_at = datetime.utcnow().isoformat()
                self.cache.update_job(job)
                self._send_callback(job)
                self._terminate_instance(job)
                return
            
            job.instance_ip = instance.ip
            self.cache.update_job(job)
            
            # Step 3: Run training via SSH
            job.status = JobStatus.RUNNING
            self.cache.update_job(job)
            
            success = self._run_training(job, buy_option)
            
            # Step 4: Cleanup
            if success:
                job.status = JobStatus.COMPLETED
            else:
                job.status = JobStatus.FAILED
                if not job.error_message:
                    job.error_message = "Training failed"
            
            job.completed_at = datetime.utcnow().isoformat()
            self.cache.update_job(job)
            
            # Send callback notification
            self._send_callback(job)
            
            # Terminate instance
            self._terminate_instance(job)
            
            logger.info(f"Job {job.job_id} completed with status: {job.status.value}")
            
        except Exception as e:
            logger.error(f"Error executing job {job.job_id}: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow().isoformat()
            self.cache.update_job(job)
            self._send_callback(job)
            
            # Always try to terminate on error
            self._terminate_instance(job)
    
    def _launch_instance(self, job: TrainingJob, buy_option: BuyOption) -> Optional[str]:
        """Launch a Lambda Labs instance"""
        if self.config.dry_run:
            logger.info(
                f"[DRY RUN] Would launch: {buy_option.instance_type} "
                f"in {buy_option.region}"
            )
            return "dry-run-instance-id"
        
        try:
            # Use the configured SSH key
            ssh_key_name = self.config.lambda_labs.ssh_key_name
            
            # Generate user_data script for cloud-init
            user_data = self._generate_user_data(job, buy_option)
            
            logger.info(
                f"Launching {buy_option.instance_type} in {buy_option.region} "
                f"with SSH key '{ssh_key_name}'"
            )
            
            instance_ids = self.lambda_client.launch_instance(
                instance_type_name=buy_option.instance_type,
                region_name=buy_option.region,
                ssh_key_names=[ssh_key_name],
                name=f"training-{job.job_id}",
                user_data=user_data
            )
            
            if instance_ids:
                instance_id = instance_ids[0]
                logger.info(f"Launched instance: {instance_id}")
                return instance_id
            else:
                logger.error("No instance IDs returned")
                return None
            
        except LambdaAPIError as e:
            logger.error(f"Lambda API error launching instance: {e}")
            job.error_message = str(e)
            return None
        except Exception as e:
            logger.error(f"Error launching instance: {e}")
            job.error_message = str(e)
            return None
    
    def _wait_for_instance_ready(self, job: TrainingJob) -> Optional[Any]:
        """Wait for instance to be ready and return it"""
        if self.config.dry_run:
            logger.info("[DRY RUN] Would wait for instance to be ready")
            from dataclasses import dataclass
            @dataclass
            class MockInstance:
                ip: str = "1.2.3.4"
                is_ready: bool = True
            return MockInstance()
        
        if not job.instance_id:
            return None
        
        return self.lambda_client.wait_for_instance_ready(
            job.instance_id,
            timeout_seconds=self.config.lambda_labs.instance_ready_timeout
        )
    
    def _generate_user_data(self, job: TrainingJob, buy_option: BuyOption) -> str:
        """
        Generate cloud-init user_data script.
        
        This script runs automatically when the instance boots.
        It sets up the environment and starts training.
        """
        # Get training params with defaults
        batch_size = job.batch_size or self.config.training.batch_size
        max_steps = job.max_steps or self.config.training.max_steps
        chunk_size = job.chunk_size or self.config.training.chunk_size
        learning_rate = job.learning_rate or self.config.training.learning_rate
        num_workers = job.num_workers or self.config.training.num_workers
        output_gcs_path = job.output_gcs_path or self.config.gcs.default_output_bucket
        
        return f'''#!/bin/bash
# =============================================================================
# Auto-generated startup script for job: {job.job_id}
# Generated: {datetime.utcnow().isoformat()}
# Instance: {buy_option.instance_type} in {buy_option.region}
# =============================================================================

exec > >(tee -a /var/log/training-startup.log) 2>&1
echo "========================================"
echo "🚀 Lambda Labs Training Job: {job.job_id}"
echo "========================================"

# Configuration
DATA_GCS_PATH="{job.data_gcs_path}"
OUTPUT_GCS_PATH="{output_gcs_path}"
LOCAL_DATA_DIR="/data/dataset"
CHECKPOINT_DIR="/data/checkpoints"

# Training parameters
BATCH_SIZE={batch_size}
MAX_STEPS={max_steps}
CHUNK_SIZE={chunk_size}
LEARNING_RATE="{learning_rate}"
NUM_WORKERS={num_workers}
GPU_COUNT={buy_option.gpu_count}

# =============================================================================
# Step 1: Install Dependencies
# =============================================================================
echo ""
echo "📦 [1/5] Installing system dependencies..."

apt-get update -qq
apt-get install -y -qq \\
    python3-pip \\
    python3-venv \\
    git \\
    htop \\
    tmux \\
    nvtop \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    apt-transport-https \\
    ca-certificates \\
    gnupg \\
    curl

echo "✅ System dependencies installed"

# =============================================================================
# Step 2: Setup GCS Authentication
# =============================================================================
echo ""
echo "☁️ [2/5] Setting up GCS authentication..."

mkdir -p /root/.config/gcloud

# Install gcloud CLI
if ! command -v gcloud &> /dev/null; then
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg 2>/dev/null || true
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list > /dev/null
    apt-get update -qq && apt-get install -y -qq google-cloud-cli
fi

echo "✅ gcloud CLI installed"

# =============================================================================
# Step 3: Setup Python Environment
# =============================================================================
echo ""
echo "🐍 [3/5] Setting up Python environment..."

GITHUB_TOKEN="{os.getenv('GITHUB_TOKEN', '')}"
BRANCH="lambda_refactor"

if [ -n "${{GITHUB_TOKEN}}" ]; then
    REPO_URL="https://${{GITHUB_TOKEN}}@github.com/innate-inc/ACT-test.git"
    echo "   Using authenticated GitHub URL"
else
    REPO_URL="https://github.com/innate-inc/ACT-test.git"
    echo "   ⚠️ Warning: No GITHUB_TOKEN - private repo clone may fail"
fi

cd /root
if [ -d "/root/ACT-test" ]; then
    cd /root/ACT-test && git fetch origin && git checkout ${{BRANCH}} && git pull origin ${{BRANCH}} || true
else
    echo "   Cloning repo..."
    git clone -b "${{BRANCH}}" "${{REPO_URL}}" /root/ACT-test
    CLONE_EXIT=$?
    if [ $CLONE_EXIT -ne 0 ]; then
        echo "❌ Git clone failed (exit code: $CLONE_EXIT)"
        echo "   If repo is private, ensure GITHUB_TOKEN is set in .env"
        exit 1
    fi
    cd /root/ACT-test
fi

python3 -m venv /root/venv
source /root/venv/bin/activate

pip install --upgrade pip

# Detect GPU architecture and install appropriate PyTorch version
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "   Detected GPU: ${{GPU_NAME}}"

if echo "${{GPU_NAME}}" | grep -qi "B200\|B100\|blackwell"; then
    echo "   ⚠️ Blackwell GPU detected (sm_100) - installing PyTorch nightly with CUDA 12.8"
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
else
    echo "   Installing PyTorch with CUDA 12.1"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

pip install -r /root/ACT-test/requirements.txt
pip install -e /root/ACT-test

echo "✅ Python environment ready"

# =============================================================================
# Step 4: GCS Authentication & Download Training Data
# =============================================================================
echo ""
echo "📥 [4/5] Setting up GCS and downloading training data..."

# Setup GCS authentication
GCS_KEY_B64="{os.getenv('GCS_SERVICE_ACCOUNT_KEY_B64', '')}"
if [ -n "${{GCS_KEY_B64}}" ]; then
    echo "   Setting up GCS service account authentication..."
    mkdir -p /root/.config/gcloud
    echo "${{GCS_KEY_B64}}" | base64 -d > /root/.config/gcloud/service-account.json
    export GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/service-account.json
    
    # Activate service account
    gcloud auth activate-service-account --key-file=/root/.config/gcloud/service-account.json
    PROJECT_ID=$(cat /root/.config/gcloud/service-account.json | python3 -c "import json,sys; print(json.load(sys.stdin).get('project_id',''))")
    gcloud config set project "${{PROJECT_ID}}"
    echo "   ✅ GCS authentication configured (project: ${{PROJECT_ID}})"
else
    echo "   ⚠️ No GCS_SERVICE_ACCOUNT_KEY_B64 set"
    echo "   GCS download will likely fail without authentication"
fi

mkdir -p "${{LOCAL_DATA_DIR}}"
mkdir -p "${{CHECKPOINT_DIR}}"
chmod -R 777 /data

# Strip trailing slashes from path to avoid // issues
DATA_GCS_PATH_CLEAN="${{DATA_GCS_PATH%/}}"

echo "   Source: ${{DATA_GCS_PATH_CLEAN}}"
echo "   Destination: ${{LOCAL_DATA_DIR}}"

START_TIME=$(date +%s)
gcloud storage cp -r "${{DATA_GCS_PATH_CLEAN}}/*" "${{LOCAL_DATA_DIR}}/"
DOWNLOAD_EXIT=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $DOWNLOAD_EXIT -ne 0 ]; then
    echo "❌ Failed to download data from GCS (exit code: $DOWNLOAD_EXIT)"
    echo "   Ensure GCS_SERVICE_ACCOUNT_KEY_B64 is set in .env"
    echo "$DOWNLOAD_EXIT" > /tmp/training_complete
    exit 1
fi

echo "✅ Data downloaded in ${{DURATION}}s"
echo "   Size: $(du -sh ${{LOCAL_DATA_DIR}} | cut -f1)"

# =============================================================================
# Step 5: Run Training
# =============================================================================
echo ""
echo "🏋️ [5/5] Starting distributed training..."
echo "   Job ID: {job.job_id}"
echo "   GPUs: ${{GPU_COUNT}}"
echo "   Batch size: ${{BATCH_SIZE}}"
echo "   Max steps: ${{MAX_STEPS}}"

source /root/venv/bin/activate
export PYTHONPATH="/root/ACT-test/act_test:${{PYTHONPATH}}"
cd /root/ACT-test

# Disable torch.compile temporarily (SIGSEGV issues on some instances)
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

python3 -m act_test.train_dist \\
    --data_dir "${{LOCAL_DATA_DIR}}" \\
    --chunk_size ${{CHUNK_SIZE}} \\
    --max_steps ${{MAX_STEPS}} \\
    --batch_size ${{BATCH_SIZE}} \\
    --world_size ${{GPU_COUNT}} \\
    --num_workers ${{NUM_WORKERS}}

TRAIN_EXIT=$?

# =============================================================================
# Upload Results
# =============================================================================
echo ""
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    echo "📤 Uploading checkpoints to GCS..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    gcloud storage cp -r "${{CHECKPOINT_DIR}}/*" "${{OUTPUT_GCS_PATH}}/job_{job.job_id}_${{TIMESTAMP}}/" 2>/dev/null || true
    gcloud storage cp -r "${{LOCAL_DATA_DIR}}/checkpoints/*" "${{OUTPUT_GCS_PATH}}/job_{job.job_id}_${{TIMESTAMP}}/" 2>/dev/null || true
    echo "✅ Checkpoints uploaded"
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT"
fi

echo ""
echo "========================================"
echo "🏁 Job {job.job_id} complete - $(date)"
echo "========================================"

# Signal completion (write marker file)
echo "$TRAIN_EXIT" > /tmp/training_complete
'''
    
    def _run_training(self, job: TrainingJob, buy_option: BuyOption) -> bool:
        """
        Run training on the instance.
        
        For Lambda Labs, we use cloud-init user_data for setup,
        so training should start automatically. We just monitor progress.
        """
        if self.config.dry_run:
            logger.info("[DRY RUN] Would monitor training progress")
            return True
        
        # Monitor training progress by checking instance status
        return self._monitor_training(job)
    
    def _monitor_training(self, job: TrainingJob, timeout_hours: int = 48) -> bool:
        """Monitor training progress by checking for completion marker via SSH"""
        if self.config.dry_run:
            logger.info("[DRY RUN] Would monitor training progress")
            return True
        
        timeout_seconds = timeout_hours * 3600
        start_time = time.time()
        ssh_check_interval = 60  # Check every minute
        last_ssh_check = 0
        
        # Import SSHExecutor here to avoid circular imports
        from .ssh_executor import SSHExecutor
        
        while time.time() - start_time < timeout_seconds:
            try:
                # Check instance status first
                instance = self.lambda_client.get_instance(job.instance_id)
                
                if instance is None:
                    logger.info("Instance terminated - training likely complete")
                    return True
                
                if instance.status == "terminated":
                    logger.info("Instance status: terminated - training complete")
                    return True
                
                if instance.status == "unhealthy":
                    logger.warning("Instance became unhealthy")
                    job.error_message = "Instance became unhealthy during training"
                    return False
                
                # Check for completion marker via SSH (every minute)
                if job.instance_ip and time.time() - last_ssh_check >= ssh_check_interval:
                    last_ssh_check = time.time()
                    try:
                        ssh = SSHExecutor(cache=self.cache)
                        if ssh.is_configured:
                            # Check if training completion marker exists
                            exit_code_ssh, stdout, stderr = ssh.execute_command(
                                job.instance_ip, 
                                "cat /tmp/training_complete 2>/dev/null || echo 'RUNNING'"
                            )
                            result = stdout.strip() if stdout else ""
                            if result and result != "RUNNING":
                                train_exit_code = result
                                if train_exit_code == "0":
                                    logger.info(f"Training completed successfully (marker found)")
                                    return True
                                else:
                                    logger.warning(f"Training failed with exit code: {train_exit_code}")
                                    job.error_message = f"Training script exited with code {train_exit_code}"
                                    return False
                            else:
                                logger.debug(f"Training still running on {job.instance_ip}")
                    except Exception as ssh_err:
                        logger.debug(f"SSH check failed (may still be booting): {ssh_err}")
                
                logger.debug(f"Instance {job.instance_id} status: {instance.status}")
                time.sleep(30)  # Short sleep between checks
                
            except Exception as e:
                logger.error(f"Error monitoring training: {e}")
                time.sleep(60)
        
        job.error_message = "Training timed out"
        return False
    
    def _terminate_instance(self, job: TrainingJob) -> None:
        """Terminate the instance after training"""
        if self.config.dry_run:
            logger.info("[DRY RUN] Would terminate instance")
            return
        
        if not job.instance_id:
            return
        
        try:
            # Check if instance still exists
            instance = self.lambda_client.get_instance(job.instance_id)
            
            if instance and instance.status not in ["terminated", "terminating"]:
                logger.info(f"Terminating instance {job.instance_id}")
                self.lambda_client.terminate_instances([job.instance_id])
                logger.info(f"Instance {job.instance_id} terminated")
            else:
                logger.info(f"Instance {job.instance_id} already terminated")
                
        except Exception as e:
            logger.warning(f"Error terminating instance: {e}")
