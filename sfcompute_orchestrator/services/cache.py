"""
Redis Cache Service
Provides caching for price data and message queue for jobs
"""
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import redis

from ..config import RedisConfig
from ..models.job import TrainingJob, PriceData, JobStatus, ApprovalRequest, ApprovalStatus

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-based cache for prices and job queue"""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self._client: Optional[redis.Redis] = None
    
    @property
    def client(self) -> redis.Redis:
        """Lazy connection to Redis"""
        if self._client is None:
            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password if self.config.password else None,
                decode_responses=True,
            )
        return self._client
    
    def ping(self) -> bool:
        """Check Redis connectivity"""
        try:
            return self.client.ping()
        except redis.ConnectionError:
            return False
    
    # =========================================================================
    # Price Data Cache
    # =========================================================================
    
    def store_price(self, price_data: PriceData) -> None:
        """Store price data with TTL"""
        key = f"{self.config.price_prefix}{price_data.cache_key}"
        self.client.setex(
            key,
            self.config.cache_ttl_seconds,
            price_data.to_json()
        )
        logger.debug(f"Stored price: {price_data.cache_key} = ${price_data.price_per_gpu_hour}/gpu/hr")
    
    def get_price(self, gpu_type: str, gpu_count: int, duration_hours: int) -> Optional[PriceData]:
        """Get cached price data"""
        key = f"{self.config.price_prefix}{gpu_type}:{gpu_count}:{duration_hours}"
        data = self.client.get(key)
        if data:
            return PriceData.from_json(data)
        return None
    
    def get_all_prices(self) -> List[PriceData]:
        """Get all cached price data"""
        pattern = f"{self.config.price_prefix}*"
        prices = []
        for key in self.client.scan_iter(match=pattern):
            # Skip history keys (they are lists, not strings)
            if ":history:" in key:
                continue
            try:
                data = self.client.get(key)
                if data:
                    prices.append(PriceData.from_json(data))
            except Exception:
                continue  # Skip keys with wrong type
        return prices
    
    def store_availability(self, gpu_type: str, gpu_count: int, available: bool) -> None:
        """Store availability status"""
        key = f"{self.config.availability_prefix}{gpu_type}:{gpu_count}"
        self.client.setex(
            key,
            self.config.cache_ttl_seconds,
            json.dumps({"available": available, "timestamp": datetime.utcnow().isoformat()})
        )
    
    def get_availability(self, gpu_type: str, gpu_count: int) -> Optional[bool]:
        """Get availability status"""
        key = f"{self.config.availability_prefix}{gpu_type}:{gpu_count}"
        data = self.client.get(key)
        if data:
            return json.loads(data).get("available")
        return None
    
    # =========================================================================
    # Job Queue (using Redis List as FIFO queue)
    # =========================================================================
    
    def enqueue_job(self, job: TrainingJob) -> None:
        """Add job to the queue"""
        self.client.rpush(self.config.job_queue, job.to_json())
        self._update_job_status(job)
        logger.info(f"Enqueued job {job.job_id}: {job.data_gcs_path}")
    
    def dequeue_job(self, timeout: int = 0) -> Optional[TrainingJob]:
        """
        Pop job from queue (blocking).
        timeout=0 means block indefinitely.
        """
        result = self.client.blpop(self.config.job_queue, timeout=timeout)
        if result:
            _, job_json = result
            job = TrainingJob.from_json(job_json)
            logger.info(f"Dequeued job {job.job_id}")
            return job
        return None
    
    def peek_queue(self, count: int = 10) -> List[TrainingJob]:
        """View jobs in queue without removing them"""
        jobs_json = self.client.lrange(self.config.job_queue, 0, count - 1)
        return [TrainingJob.from_json(j) for j in jobs_json]
    
    def queue_length(self) -> int:
        """Get number of jobs in queue"""
        return self.client.llen(self.config.job_queue)
    
    def get_all_queued_jobs(self) -> List[TrainingJob]:
        """Get all jobs in queue without removing them"""
        jobs_json = self.client.lrange(self.config.job_queue, 0, -1)
        return [TrainingJob.from_json(j) for j in jobs_json]
    
    def remove_job_from_queue(self, job_id: str) -> Optional[TrainingJob]:
        """
        Remove a specific job from the queue by job_id.
        Returns the job if found and removed, None otherwise.
        
        This is used for smart selection where we may skip jobs
        that can't afford current prices.
        """
        # Get all jobs
        all_jobs_json = self.client.lrange(self.config.job_queue, 0, -1)
        
        for i, job_json in enumerate(all_jobs_json):
            job = TrainingJob.from_json(job_json)
            if job.job_id == job_id:
                # Remove this job from queue
                # We use LREM to remove the first occurrence
                self.client.lrem(self.config.job_queue, 1, job_json)
                logger.info(f"Removed job {job_id} from queue at position {i}")
                return job
        
        return None
    
    def requeue_job_to_back(self, job: TrainingJob) -> None:
        """
        Add a job to the back of the queue.
        Used when a job is rejected or skipped.
        """
        self.client.rpush(self.config.job_queue, job.to_json())
        logger.info(f"Requeued job {job.job_id} to back of queue")
    
    def requeue_job_to_front(self, job: TrainingJob) -> None:
        """
        Add a job to the front of the queue.
        Used when approval fails but job should retry immediately.
        """
        self.client.lpush(self.config.job_queue, job.to_json())
        logger.info(f"Requeued job {job.job_id} to front of queue")
    
    # =========================================================================
    # Job Status Tracking
    # =========================================================================
    
    def _update_job_status(self, job: TrainingJob) -> None:
        """Update job status in cache"""
        key = f"{self.config.job_status_prefix}{job.job_id}"
        # Keep job status for 24 hours
        self.client.setex(key, 86400, job.to_json())
    
    def update_job(self, job: TrainingJob) -> None:
        """Update job in cache"""
        self._update_job_status(job)
        logger.debug(f"Updated job {job.job_id}: status={job.status.value}")
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID"""
        key = f"{self.config.job_status_prefix}{job_id}"
        data = self.client.get(key)
        if data:
            return TrainingJob.from_json(data)
        return None
    
    def get_all_jobs(self) -> List[TrainingJob]:
        """Get all tracked jobs"""
        pattern = f"{self.config.job_status_prefix}*"
        jobs = []
        for key in self.client.scan_iter(match=pattern):
            data = self.client.get(key)
            if data:
                jobs.append(TrainingJob.from_json(data))
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    # =========================================================================
    # Price History (for trend analysis)
    # =========================================================================
    
    def append_price_history(self, price_data: PriceData, max_entries: int = 100) -> None:
        """Append to price history list"""
        key = f"{self.config.price_prefix}history:{price_data.cache_key}"
        self.client.lpush(key, price_data.to_json())
        self.client.ltrim(key, 0, max_entries - 1)
    
    def get_price_history(self, gpu_type: str, gpu_count: int, duration_hours: int, limit: int = 50) -> List[PriceData]:
        """Get recent price history"""
        key = f"{self.config.price_prefix}history:{gpu_type}:{gpu_count}:{duration_hours}"
        entries = self.client.lrange(key, 0, limit - 1)
        return [PriceData.from_json(e) for e in entries]
    
    # =========================================================================
    # Approval Request Tracking
    # =========================================================================
    
    def store_approval_request(self, request: ApprovalRequest, ttl_seconds: int = 3600) -> None:
        """Store an approval request with TTL"""
        key = f"sfcompute:approvals:{request.request_id}"
        self.client.setex(key, ttl_seconds, request.to_json())
        
        # Also index by job_id for quick lookup
        job_key = f"sfcompute:approvals:by_job:{request.job_id}"
        self.client.setex(job_key, ttl_seconds, request.request_id)
        
        logger.debug(f"Stored approval request {request.request_id} for job {request.job_id}")
    
    def get_approval_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get approval request by ID"""
        key = f"sfcompute:approvals:{request_id}"
        data = self.client.get(key)
        if data:
            return ApprovalRequest.from_json(data)
        return None
    
    def get_approval_request_by_job(self, job_id: str) -> Optional[ApprovalRequest]:
        """Get the pending approval request for a job"""
        job_key = f"sfcompute:approvals:by_job:{job_id}"
        request_id = self.client.get(job_key)
        if request_id:
            return self.get_approval_request(request_id)
        return None
    
    def update_approval_request(self, request: ApprovalRequest) -> None:
        """Update an existing approval request"""
        key = f"sfcompute:approvals:{request.request_id}"
        # Get remaining TTL
        ttl = self.client.ttl(key)
        if ttl > 0:
            self.client.setex(key, ttl, request.to_json())
        else:
            # Default to 1 hour if TTL expired
            self.client.setex(key, 3600, request.to_json())
        logger.debug(f"Updated approval request {request.request_id}: status={request.status.value}")
    
    def delete_approval_request(self, request_id: str) -> None:
        """Delete an approval request"""
        request = self.get_approval_request(request_id)
        if request:
            self.client.delete(f"sfcompute:approvals:{request_id}")
            self.client.delete(f"sfcompute:approvals:by_job:{request.job_id}")
            logger.debug(f"Deleted approval request {request_id}")
    
    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """Get all pending approval requests"""
        pattern = "sfcompute:approvals:*"
        requests = []
        for key in self.client.scan_iter(match=pattern):
            if ":by_job:" in key:
                continue  # Skip index keys
            data = self.client.get(key)
            if data:
                try:
                    req = ApprovalRequest.from_json(data)
                    if req.status == ApprovalStatus.PENDING:
                        requests.append(req)
                except Exception:
                    continue
        return requests
    
    def get_all_approval_requests(self) -> List[ApprovalRequest]:
        """Get all approval requests (any status)"""
        pattern = "sfcompute:approvals:*"
        requests = []
        for key in self.client.scan_iter(match=pattern):
            if ":by_job:" in key:
                continue  # Skip index keys
            data = self.client.get(key)
            if data:
                try:
                    req = ApprovalRequest.from_json(data)
                    requests.append(req)
                except Exception:
                    continue
        return requests
    
    # =========================================================================
    # SSH Key Storage
    # =========================================================================
    
    def store_ssh_private_key(self, private_key: str) -> None:
        """Store SSH private key for Lambda Labs instance access"""
        self.client.set("lambda:ssh_private_key", private_key)
        logger.info("SSH private key stored in Redis")
    
    def get_ssh_private_key(self) -> Optional[str]:
        """Get stored SSH private key"""
        return self.client.get("lambda:ssh_private_key")
    
    def delete_ssh_private_key(self) -> None:
        """Delete stored SSH private key"""
        self.client.delete("lambda:ssh_private_key")
        logger.info("SSH private key deleted from Redis")
    
    # =========================================================================
    # Admin Operations
    # =========================================================================
    
    def clear_job_queue(self) -> int:
        """Clear all jobs from the queue. Returns count of jobs cleared."""
        count = self.client.llen(self.config.job_queue)
        self.client.delete(self.config.job_queue)
        logger.info(f"Cleared {count} jobs from queue")
        return count
    
    def clear_all(self) -> Dict[str, int]:
        """Clear all cached data. Returns counts of items cleared."""
        result = {
            "prices": 0,
            "availability": 0,
            "jobs": 0,
            "approvals": 0,
        }
        
        # Clear prices
        for key in self.client.scan_iter(match=f"{self.config.price_prefix}*"):
            self.client.delete(key)
            result["prices"] += 1
        
        # Clear availability
        for key in self.client.scan_iter(match=f"{self.config.availability_prefix}*"):
            self.client.delete(key)
            result["availability"] += 1
        
        # Clear job statuses (not the queue itself)
        for key in self.client.scan_iter(match=f"{self.config.job_status_prefix}*"):
            self.client.delete(key)
            result["jobs"] += 1
        
        # Clear approvals
        for key in self.client.scan_iter(match="sfcompute:approvals:*"):
            self.client.delete(key)
            result["approvals"] += 1
        
        logger.info(f"Cleared all cache: {result}")
        return result

