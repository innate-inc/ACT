#!/bin/bash
# =============================================================================
# Debug Script 1: Launch Lambda VM and run test.py
# =============================================================================
# This script launches a Lambda Labs VM, sets up the environment,
# and runs test.py for debugging instead of the full training script.
#
# Usage:
#   ./debug_lambda_test.sh [instance_type] [region]
#
# Examples:
#   ./debug_lambda_test.sh                           # Use defaults (gpu_1x_a100, us-west-1)
#   ./debug_lambda_test.sh gpu_8x_a100               # 8x A100 in default region
#   ./debug_lambda_test.sh gpu_8x_h100_sxm5 us-west-3  # 8x H100 in us-west-3
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

# Required environment variables
LAMBDA_API_KEY="${LAMBDA_API_KEY:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
SSH_KEY_FILE="${SCRIPT_DIR}/sfcompute_orchestrator/manipulation_training_new.pem"
GCS_KEY_FILE="${SCRIPT_DIR}/sfcompute_orchestrator/gcs-key-auth.json"

# Instance configuration
INSTANCE_TYPE="${1:-gpu_1x_a100}"
REGION="${2:-us-west-1}"
SSH_KEY_NAME="manipulation_training_new"
BRANCH="lambda_refactor"

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
echo "🔧 Lambda Labs Debug Script - test.py"
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
    print_warning "GCS key not found: $GCS_KEY_FILE - GCS operations will fail"
    GCS_KEY_B64=""
else
    GCS_KEY_B64=$(base64 -w0 "$GCS_KEY_FILE")
fi

print_info "Instance type: $INSTANCE_TYPE"
print_info "Region: $REGION"
print_info "SSH key: $SSH_KEY_NAME"
print_info "Branch: $BRANCH"
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
        \"name\": \"debug-test-$(date +%s)\"
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
    print_info "Terminating instance $INSTANCE_ID..."
    curl -s -X POST "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate" \
        -H "Authorization: Bearer $LAMBDA_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"instance_ids\": [\"$INSTANCE_ID\"]}"
    exit 1
fi

print_success "Instance ready: $INSTANCE_IP"

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
# Step 4: Setup Instance
# =============================================================================
print_info "Setting up instance..."

# Create setup script
SETUP_SCRIPT=$(cat << 'SETUP_EOF'
#!/bin/bash
set -e

echo "========================================"
echo "🚀 Debug Setup Script"
echo "========================================"

# Install dependencies
echo "📦 Installing dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv git htop nvtop libgl1-mesa-glx libglib2.0-0 apt-transport-https ca-certificates gnupg curl

# Install gcloud CLI
if ! command -v gcloud &> /dev/null; then
    echo "📦 Installing gcloud CLI..."
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg 2>/dev/null || true
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list > /dev/null
    sudo apt-get update -qq && sudo apt-get install -y -qq google-cloud-cli
fi

echo "✅ Dependencies installed"
SETUP_EOF
)

echo "$SETUP_SCRIPT" | $SSH_CMD "cat > /tmp/setup.sh && chmod +x /tmp/setup.sh && sudo bash /tmp/setup.sh"

# =============================================================================
# Step 4: Clone Repo and Setup Python
# =============================================================================
print_info "Cloning repo and setting up Python..."

PYTHON_SETUP=$(cat << EOF
#!/bin/bash
set -e

cd /root

# Clone repo
GITHUB_TOKEN="$GITHUB_TOKEN"
BRANCH="$BRANCH"

if [ -n "\$GITHUB_TOKEN" ]; then
    REPO_URL="https://\${GITHUB_TOKEN}@github.com/innate-inc/ACT-test.git"
else
    REPO_URL="https://github.com/innate-inc/ACT-test.git"
fi

if [ -d "/root/ACT-test" ]; then
    cd /root/ACT-test && git fetch origin && git checkout \$BRANCH && git pull origin \$BRANCH
else
    git clone -b "\$BRANCH" "\$REPO_URL" /root/ACT-test
fi

# Setup venv
python3 -m venv /root/venv
source /root/venv/bin/activate

pip install --upgrade pip

# Detect GPU and install PyTorch
GPU_NAME=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "Detected GPU: \$GPU_NAME"

if echo "\$GPU_NAME" | grep -qi "B200\|B100\|blackwell"; then
    echo "Installing PyTorch nightly for Blackwell..."
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
else
    echo "Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

pip install -r /root/ACT-test/requirements.txt
pip install -e /root/ACT-test

echo "✅ Python environment ready"
EOF
)

echo "$PYTHON_SETUP" | $SSH_CMD "sudo bash"

# =============================================================================
# Step 5: Setup GCS Auth
# =============================================================================
if [[ -n "$GCS_KEY_B64" ]]; then
    print_info "Setting up GCS authentication..."
    
    GCS_SETUP=$(cat << EOF
#!/bin/bash
mkdir -p /root/.config/gcloud
echo "$GCS_KEY_B64" | base64 -d > /root/.config/gcloud/service-account.json
export GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/service-account.json
gcloud auth activate-service-account --key-file=/root/.config/gcloud/service-account.json
PROJECT_ID=\$(cat /root/.config/gcloud/service-account.json | python3 -c "import json,sys; print(json.load(sys.stdin).get('project_id',''))")
gcloud config set project "\$PROJECT_ID"
echo "✅ GCS authentication configured (project: \$PROJECT_ID)"
EOF
)
    echo "$GCS_SETUP" | $SSH_CMD "sudo bash"
fi

# =============================================================================
# Step 6: Run test.py
# =============================================================================
print_info "Running test.py..."
echo ""
echo "========================================"
echo "📋 test.py output:"
echo "========================================"

$SSH_CMD "sudo bash -c 'source /root/venv/bin/activate && cd /root/ACT-test && python3 -m act_test.test'"

TEST_EXIT=$?

echo ""
echo "========================================"
if [[ $TEST_EXIT -eq 0 ]]; then
    print_success "test.py completed successfully!"
else
    print_error "test.py failed with exit code: $TEST_EXIT"
fi
echo "========================================"

# =============================================================================
# Cleanup Options
# =============================================================================
echo ""
echo "Instance is still running: $INSTANCE_IP"
echo ""
echo "To SSH manually:"
echo "  ssh -i $SSH_KEY_FILE ubuntu@$INSTANCE_IP"
echo ""
echo "To terminate:"
echo "  curl -X POST 'https://cloud.lambdalabs.com/api/v1/instance-operations/terminate' \\"
echo "    -H 'Authorization: Bearer \$LAMBDA_API_KEY' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"instance_ids\": [\"$INSTANCE_ID\"]}'"
echo ""

read -p "Terminate instance now? (y/N): " TERMINATE
if [[ "$TERMINATE" =~ ^[Yy]$ ]]; then
    print_info "Terminating instance..."
    curl -s -X POST "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate" \
        -H "Authorization: Bearer $LAMBDA_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"instance_ids\": [\"$INSTANCE_ID\"]}"
    print_success "Instance terminated"
else
    print_info "Instance left running. Remember to terminate it manually!"
fi

