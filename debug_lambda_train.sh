#!/bin/bash
# =============================================================================
# Debug Script 2: Launch Lambda VM and run training with custom GCS bucket
# =============================================================================
# This script launches a Lambda Labs VM and runs training with data from
# a specified GCS bucket.
#
# Usage:
#   ./debug_lambda_train.sh <gcs_data_path> [instance_type] [region]
#
# Examples:
#   ./debug_lambda_train.sh gs://my-bucket/data/episode_folder
#   ./debug_lambda_train.sh gs://innate-manipulation-training-data/test-data gpu_8x_a100 us-west-1
#
# Training parameters can be set via environment variables:
#   BATCH_SIZE=8 MAX_STEPS=1000 ./debug_lambda_train.sh gs://my-bucket/data
# =============================================================================

set -e

# Configuration - Load from .env if exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/sfcompute_orchestrator/.env"

if [[ -f "$ENV_FILE" ]]; then
    echo "📂 Loading environment from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
fi

# Required arguments
GCS_DATA_PATH="${1:-}"
if [[ -z "$GCS_DATA_PATH" ]]; then
    echo "Usage: $0 <gcs_data_path> [instance_type] [region]"
    echo ""
    echo "Examples:"
    echo "  $0 gs://innate-manipulation-training-data/test/data"
    echo "  $0 gs://my-bucket/data gpu_8x_a100 us-west-1"
    echo ""
    echo "Training params (set via env vars):"
    echo "  BATCH_SIZE=8 MAX_STEPS=1000 $0 gs://my-bucket/data"
    exit 1
fi

# Required environment variables
LAMBDA_API_KEY="${LAMBDA_API_KEY:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
SSH_KEY_FILE="${SCRIPT_DIR}/sfcompute_orchestrator/manipulation_training_new.pem"
GCS_KEY_FILE="${SCRIPT_DIR}/sfcompute_orchestrator/gcs-key-auth.json"

# Instance configuration
INSTANCE_TYPE="${2:-gpu_8x_a100}"
REGION="${3:-us-west-1}"
SSH_KEY_NAME="manipulation_training_new"
BRANCH="lambda_refactor"

# Training parameters (can be overridden via env vars)
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-30}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

# Validate requirements
echo "========================================"
echo "🏋️ Lambda Labs Debug Script - Training"
echo "========================================"
echo ""

if [[ -z "$LAMBDA_API_KEY" ]]; then
    print_error "LAMBDA_API_KEY not set. Add it to $ENV_FILE or export it."
    exit 1
fi

if [[ ! -f "$SSH_KEY_FILE" ]]; then
    print_error "SSH key not found: $SSH_KEY_FILE"
    exit 1
fi
chmod 600 "$SSH_KEY_FILE"

if [[ ! -f "$GCS_KEY_FILE" ]]; then
    print_error "GCS key not found: $GCS_KEY_FILE - Required for data download"
    exit 1
fi
GCS_KEY_B64=$(base64 -w0 "$GCS_KEY_FILE")

print_info "Data source: $GCS_DATA_PATH"
print_info "Instance type: $INSTANCE_TYPE"
print_info "Region: $REGION"
print_info "Branch: $BRANCH"
echo ""
print_info "Training params:"
echo "  Batch size: $BATCH_SIZE"
echo "  Max steps: $MAX_STEPS"
echo "  Chunk size: $CHUNK_SIZE"
echo "  Num workers: $NUM_WORKERS"
echo "  Learning rate: $LEARNING_RATE"
echo ""

# =============================================================================
# Step 1: Launch Instance
# =============================================================================
print_info "Launching Lambda Labs instance..."

LAUNCH_RESPONSE=$(curl -s -X POST "https://cloud.lambdalabs.com/api/v1/instance-operations/launch" \
    -H "Authorization: Bearer $LAMBDA_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
        \"instance_type_name\": \"$INSTANCE_TYPE\",
        \"region_name\": \"$REGION\",
        \"ssh_key_names\": [\"$SSH_KEY_NAME\"],
        \"name\": \"debug-train-$(date +%s)\"
    }")

# Check for errors (using Python instead of jq)
parse_json() {
    python3 -c "import json,sys; data=json.load(sys.stdin); $1"
}

ERROR_MSG=$(echo "$LAUNCH_RESPONSE" | parse_json "print(data.get('error',{}).get('message',''))")
if [[ -n "$ERROR_MSG" ]]; then
    print_error "Failed to launch instance: $ERROR_MSG"
    echo "$LAUNCH_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$LAUNCH_RESPONSE"
    exit 1
fi

INSTANCE_ID=$(echo "$LAUNCH_RESPONSE" | parse_json "ids=data.get('data',{}).get('instance_ids',[]); print(ids[0] if ids else '')")
if [[ -z "$INSTANCE_ID" ]]; then
    print_error "Failed to get instance ID"
    echo "$LAUNCH_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$LAUNCH_RESPONSE"
    exit 1
fi

print_success "Instance launched: $INSTANCE_ID"

# Function to terminate instance
terminate_instance() {
    print_info "Terminating instance $INSTANCE_ID..."
    curl -s -X POST "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate" \
        -H "Authorization: Bearer $LAMBDA_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"instance_ids\": [\"$INSTANCE_ID\"]}" > /dev/null
    print_success "Instance terminated"
}

# Trap to ensure cleanup on script exit
trap 'echo ""; print_warning "Script interrupted. Instance $INSTANCE_ID may still be running!"' EXIT

# =============================================================================
# Step 2: Wait for Instance to be Ready
# =============================================================================
print_info "Waiting for instance to be ready..."

INSTANCE_IP=""
MAX_WAIT=300  # 5 minutes
WAITED=0

while [[ -z "$INSTANCE_IP" && $WAITED -lt $MAX_WAIT ]]; do
    sleep 10
    WAITED=$((WAITED + 10))
    
    STATUS_RESPONSE=$(curl -s "https://cloud.lambdalabs.com/api/v1/instances/$INSTANCE_ID" \
        -H "Authorization: Bearer $LAMBDA_API_KEY")
    
    STATUS=$(echo "$STATUS_RESPONSE" | parse_json "print(data.get('data',{}).get('status',''))")
    INSTANCE_IP=$(echo "$STATUS_RESPONSE" | parse_json "print(data.get('data',{}).get('ip',''))")
    
    echo -n "."
    
    if [[ "$STATUS" == "active" && -n "$INSTANCE_IP" ]]; then
        break
    fi
done
echo ""

if [[ -z "$INSTANCE_IP" ]]; then
    print_error "Instance did not become ready within ${MAX_WAIT}s"
    terminate_instance
    exit 1
fi

print_success "Instance ready: $INSTANCE_IP"
echo ""
echo "📋 To SSH manually (in another terminal):"
echo "  ssh -i $SSH_KEY_FILE ubuntu@$INSTANCE_IP"
echo ""

# =============================================================================
# Step 3: Wait for SSH to be available
# =============================================================================
print_info "Waiting for SSH to be available (this can take 1-3 minutes)..."

SSH_CMD="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -i $SSH_KEY_FILE ubuntu@$INSTANCE_IP"

SSH_READY=false
SSH_WAIT=0
SSH_MAX_WAIT=180  # 3 minutes

while [[ "$SSH_READY" == "false" && $SSH_WAIT -lt $SSH_MAX_WAIT ]]; do
    if $SSH_CMD "echo 'SSH ready'" 2>/dev/null | grep -q "SSH ready"; then
        SSH_READY=true
    else
        echo -n "."
        sleep 10
        SSH_WAIT=$((SSH_WAIT + 10))
    fi
done
echo ""

if [[ "$SSH_READY" == "false" ]]; then
    print_error "SSH not available after ${SSH_MAX_WAIT}s"
    print_info "Instance may still be booting. Try SSH manually:"
    echo "  ssh -i $SSH_KEY_FILE ubuntu@$INSTANCE_IP"
    exit 1
fi

print_success "SSH connection established"

# =============================================================================
# Step 4: Get GPU count from instance
# =============================================================================

GPU_COUNT=$($SSH_CMD "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l" || echo "1")
print_info "Detected $GPU_COUNT GPUs"

# =============================================================================
# Step 4: Run Full Setup and Training
# =============================================================================
print_info "Starting setup and training on instance..."

# Create the full training script
TRAIN_SCRIPT=$(cat << EOF
#!/bin/bash
set -e

exec > >(tee -a /var/log/debug-training.log) 2>&1
echo "========================================"
echo "🏋️ Debug Training Script"
echo "========================================"
echo "Started at: \$(date)"

# Configuration
DATA_GCS_PATH="$GCS_DATA_PATH"
LOCAL_DATA_DIR="/data/dataset"
BATCH_SIZE=$BATCH_SIZE
MAX_STEPS=$MAX_STEPS
CHUNK_SIZE=$CHUNK_SIZE
NUM_WORKERS=$NUM_WORKERS
LEARNING_RATE="$LEARNING_RATE"
GPU_COUNT=$GPU_COUNT

echo ""
echo "📋 Configuration:"
echo "   Data source: \$DATA_GCS_PATH"
echo "   GPUs: \$GPU_COUNT"
echo "   Batch size: \$BATCH_SIZE"
echo "   Max steps: \$MAX_STEPS"
echo "   Chunk size: \$CHUNK_SIZE"
echo "   Num workers: \$NUM_WORKERS"
echo ""

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

# Install gcloud CLI
if ! command -v gcloud &> /dev/null; then
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg 2>/dev/null || true
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list > /dev/null
    apt-get update -qq && apt-get install -y -qq google-cloud-cli
fi

echo "✅ System dependencies installed"

# =============================================================================
# Step 2: Setup GCS Auth
# =============================================================================
echo ""
echo "☁️ [2/5] Setting up GCS authentication..."

mkdir -p /root/.config/gcloud
echo "$GCS_KEY_B64" | base64 -d > /root/.config/gcloud/service-account.json
export GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/service-account.json
gcloud auth activate-service-account --key-file=/root/.config/gcloud/service-account.json
PROJECT_ID=\$(cat /root/.config/gcloud/service-account.json | python3 -c "import json,sys; print(json.load(sys.stdin).get('project_id',''))")
gcloud config set project "\$PROJECT_ID"
echo "✅ GCS authentication configured (project: \$PROJECT_ID)"

# =============================================================================
# Step 3: Setup Python Environment
# =============================================================================
echo ""
echo "🐍 [3/5] Setting up Python environment..."

GITHUB_TOKEN="$GITHUB_TOKEN"
BRANCH="$BRANCH"

cd /root
if [ -n "\$GITHUB_TOKEN" ]; then
    REPO_URL="https://\${GITHUB_TOKEN}@github.com/innate-inc/ACT-test.git"
    echo "   Using authenticated GitHub URL"
else
    REPO_URL="https://github.com/innate-inc/ACT-test.git"
    echo "   ⚠️ No GITHUB_TOKEN - using public URL"
fi

if [ -d "/root/ACT-test" ]; then
    cd /root/ACT-test && git fetch origin && git checkout \$BRANCH && git pull origin \$BRANCH
else
    git clone -b "\$BRANCH" "\$REPO_URL" /root/ACT-test
fi

python3 -m venv /root/venv
source /root/venv/bin/activate

pip install --upgrade pip

# Detect GPU and install PyTorch
GPU_NAME=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "   Detected GPU: \$GPU_NAME"

if echo "\$GPU_NAME" | grep -qi "B200\|B100\|blackwell"; then
    echo "   Installing PyTorch nightly for Blackwell..."
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
else
    echo "   Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

pip install -r /root/ACT-test/requirements.txt
pip install -e /root/ACT-test

echo "✅ Python environment ready"

# =============================================================================
# Step 4: Download Data
# =============================================================================
echo ""
echo "📥 [4/5] Downloading training data..."

mkdir -p "\$LOCAL_DATA_DIR"
chmod -R 777 /data

# Strip trailing slashes
DATA_GCS_PATH_CLEAN="\${DATA_GCS_PATH%/}"

echo "   Source: \$DATA_GCS_PATH_CLEAN"
echo "   Destination: \$LOCAL_DATA_DIR"

START_TIME=\$(date +%s)
gcloud storage cp -r "\${DATA_GCS_PATH_CLEAN}/*" "\$LOCAL_DATA_DIR/"
DOWNLOAD_EXIT=\$?
END_TIME=\$(date +%s)
DURATION=\$((END_TIME - START_TIME))

if [ \$DOWNLOAD_EXIT -ne 0 ]; then
    echo "❌ Failed to download data from GCS (exit code: \$DOWNLOAD_EXIT)"
    exit 1
fi

echo "✅ Data downloaded in \${DURATION}s"
echo "   Size: \$(du -sh \$LOCAL_DATA_DIR | cut -f1)"
echo "   Contents:"
ls -la "\$LOCAL_DATA_DIR"

# =============================================================================
# Step 5: Run Training
# =============================================================================
echo ""
echo "🏋️ [5/5] Starting distributed training..."
echo ""

source /root/venv/bin/activate
export PYTHONPATH="/root/ACT-test:$PYTHONPATH"
cd /root/ACT-test

# Disable torch.compile for debugging
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

echo "Running training with:"
echo "  python3 -m act_test.train_dist \\"
echo "    --data_dir \$LOCAL_DATA_DIR \\"
echo "    --chunk_size \$CHUNK_SIZE \\"
echo "    --max_steps \$MAX_STEPS \\"
echo "    --batch_size \$BATCH_SIZE \\"
echo "    --world_size \$GPU_COUNT \\"
echo "    --num_workers \$NUM_WORKERS"
echo ""

python3 -m act_test.train_dist \\
    --data_dir "\$LOCAL_DATA_DIR" \\
    --chunk_size \$CHUNK_SIZE \\
    --max_steps \$MAX_STEPS \\
    --batch_size \$BATCH_SIZE \\
    --world_size \$GPU_COUNT \\
    --num_workers \$NUM_WORKERS

TRAIN_EXIT=\$?

echo ""
if [ \$TRAIN_EXIT -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code: \$TRAIN_EXIT"
fi

echo ""
echo "========================================"
echo "🏁 Debug training complete - \$(date)"
echo "========================================"

exit \$TRAIN_EXIT
EOF
)

# Run the training script
echo "$TRAIN_SCRIPT" | $SSH_CMD "sudo bash"
TRAIN_EXIT=$?

echo ""
echo "========================================"
if [[ $TRAIN_EXIT -eq 0 ]]; then
    print_success "Training completed successfully!"
else
    print_error "Training failed with exit code: $TRAIN_EXIT"
fi
echo "========================================"

# =============================================================================
# View Logs
# =============================================================================
echo ""
echo "📋 To view full logs:"
echo "  ssh -i $SSH_KEY_FILE ubuntu@$INSTANCE_IP 'sudo cat /var/log/debug-training.log'"
echo ""
echo "To SSH manually:"
echo "  ssh -i $SSH_KEY_FILE ubuntu@$INSTANCE_IP"
echo ""

# =============================================================================
# Cleanup
# =============================================================================
read -p "Terminate instance now? (y/N): " TERMINATE
if [[ "$TERMINATE" =~ ^[Yy]$ ]]; then
    terminate_instance
else
    print_info "Instance left running at: $INSTANCE_IP"
    print_warning "Remember to terminate it manually when done!"
    echo ""
    echo "To terminate later:"
    echo "  curl -X POST 'https://cloud.lambdalabs.com/api/v1/instance-operations/terminate' \\"
    echo "    -H 'Authorization: Bearer \$LAMBDA_API_KEY' \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"instance_ids\": [\"$INSTANCE_ID\"]}'"
fi

# Clear trap
trap - EXIT

