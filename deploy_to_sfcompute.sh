#!/bin/bash

# =============================================================================
# Deploy ACT Training to SFCompute
# =============================================================================
# This script creates an SFCompute node and prepares it for training.
# Data is downloaded from GCS bucket to the node's NVMe storage at /data.
#
# Usage:
#   ./deploy_to_sfcompute.sh <DATA_GCS_PATH> <OUTPUT_GCS_PATH> [OPTIONS]
#
# Examples:
#   ./deploy_to_sfcompute.sh gs://my-bucket/data gs://my-bucket/outputs
#   ./deploy_to_sfcompute.sh gs://my-bucket/data gs://my-bucket/outputs --node-name my-job
#   ./deploy_to_sfcompute.sh gs://my-bucket/data gs://my-bucket/outputs --auto --duration 24h
# =============================================================================

set -e

# Default configuration
NODE_NAME="act-training-$(date +%Y%m%d-%H%M%S)"
ZONE="landsend"
MAX_PRICE="25.00"
DURATION="12h"
NODE_TYPE="reserved"  # "reserved" or "auto"
CLOUD_INIT_FILE="cloud-init.yaml"

# Training defaults
MAX_STEPS=120000
LEARNING_RATE="5e-5"
CHUNK_SIZE=30

# Parse required arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <DATA_GCS_PATH> <OUTPUT_GCS_PATH> [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  DATA_GCS_PATH     GCS path to training data (e.g., gs://my-bucket/data)"
    echo "  OUTPUT_GCS_PATH   GCS path for outputs (e.g., gs://my-bucket/outputs)"
    echo ""
    echo "Options:"
    echo "  --node-name NAME    Node name (default: auto-generated)"
    echo "  --zone ZONE         SFCompute zone (default: landsend)"
    echo "  --max-price PRICE   Max price per hour (default: 25.00)"
    echo "  --duration TIME     Duration for reserved nodes (default: 12h)"
    echo "  --auto              Use auto-reserved nodes instead of reserved"
    echo "  --max-steps NUM     Maximum training steps (default: 120000)"
    echo "  --learning-rate LR  Learning rate (default: 5e-5)"
    echo "  --chunk-size SIZE   Action chunk size (default: 30)"
    echo "  --dry-run           Only generate configs, don't create node"
    echo ""
    echo "Examples:"
    echo "  $0 gs://my-data-bucket gs://my-output-bucket"
    echo "  $0 gs://my-data gs://my-outputs --node-name my-training --duration 24h"
    echo "  $0 gs://my-data gs://my-outputs --auto --max-price 20.00"
    exit 1
fi

DATA_GCS_PATH="$1"
OUTPUT_GCS_PATH="$2"
shift 2

# Parse optional arguments
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --node-name)
            NODE_NAME="$2"
            shift 2
            ;;
        --zone)
            ZONE="$2"
            shift 2
            ;;
        --max-price)
            MAX_PRICE="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --auto)
            NODE_TYPE="auto"
            shift
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🚀 Deploying ACT Training to SFCompute"
echo "======================================="
echo ""
echo "📊 Configuration:"
echo "   Data source:     ${DATA_GCS_PATH}"
echo "   Output dest:     ${OUTPUT_GCS_PATH}"
echo "   Node name:       ${NODE_NAME}"
echo "   Zone:            ${ZONE}"
echo "   Max price:       \$${MAX_PRICE}/hour"
echo "   Node type:       ${NODE_TYPE}"
if [ "$NODE_TYPE" = "reserved" ]; then
    echo "   Duration:        ${DURATION}"
fi
echo ""
echo "🏋️ Training Config:"
echo "   Max steps:       ${MAX_STEPS}"
echo "   Learning rate:   ${LEARNING_RATE}"
echo "   Chunk size:      ${CHUNK_SIZE}"
echo ""

# Check if sf CLI is installed
if ! command -v sf &> /dev/null; then
    echo "❌ Error: sf CLI is not installed."
    echo ""
    echo "Install it with:"
    echo "   curl -fsSL https://sfcompute.com/cli/install | bash"
    echo "   source ~/.bashrc"
    echo ""
    exit 1
fi

# Check if logged in
echo "🔐 Checking SFCompute authentication..."
if ! sf nodes list &> /dev/null; then
    echo "⚠️  Not logged in to SFCompute. Running: sf login"
    sf login
fi

# Get SSH public keys for cloud-init
echo ""
echo "🔑 Gathering SSH public keys..."
SSH_KEYS=""
for keyfile in ~/.ssh/id_*.pub; do
    if [ -f "$keyfile" ]; then
        key=$(cat "$keyfile")
        SSH_KEYS="${SSH_KEYS}      - ${key}\n"
        echo "   Found: $(basename $keyfile)"
    fi
done

if [ -z "$SSH_KEYS" ]; then
    echo "⚠️  No SSH public keys found in ~/.ssh/"
    echo "   Generate one with: ssh-keygen -t ed25519"
    exit 1
fi

# Generate cloud-init file with embedded configuration
GENERATED_CLOUD_INIT="cloud-init-${NODE_NAME}.yaml"
echo ""
echo "📝 Generating cloud-init file: ${GENERATED_CLOUD_INIT}"

cat > "${GENERATED_CLOUD_INIT}" << EOF
#cloud-config
# Auto-generated cloud-init for ACT training job: ${NODE_NAME}
# Generated: $(date -Iseconds)

disable_root: false
ssh_pwauth: false

users:
  - name: root
    ssh_authorized_keys:
$(echo -e "$SSH_KEYS")

package_update: true
package_upgrade: true

packages:
  - git
  - wget
  - curl
  - htop
  - nvtop
  - tmux
  - unzip
  - build-essential
  - python3-pip
  - python3-dev
  - apt-transport-https
  - ca-certificates
  - gnupg

write_files:
  - path: /data/job_config.sh
    permissions: '0644'
    content: |
      # ACT Training Job Configuration
      export DATA_GCS_PATH="${DATA_GCS_PATH}"
      export OUTPUT_GCS_PATH="${OUTPUT_GCS_PATH}"
      export LOCAL_DATA_DIR="/data/dataset"
      export CHECKPOINT_DIR="/data/checkpoints"
      export MAX_STEPS=${MAX_STEPS}
      export LEARNING_RATE=${LEARNING_RATE}
      export CHUNK_SIZE=${CHUNK_SIZE}
      export WANDB_MODE="online"
      export NCCL_DEBUG="INFO"

  - path: /data/download_and_train.sh
    permissions: '0755'
    content: |
      #!/bin/bash
      set -e
      
      echo "🚀 ACT Training Job Starting"
      echo "============================"
      
      # Load configuration
      source /data/job_config.sh
      
      # Download data from GCS
      echo ""
      echo "📥 Downloading data from GCS..."
      echo "   Source: \${DATA_GCS_PATH}"
      echo "   Destination: \${LOCAL_DATA_DIR}"
      mkdir -p \${LOCAL_DATA_DIR}
      
      # Use gcloud storage for faster parallel downloads
      gcloud storage cp -r "\${DATA_GCS_PATH}/*" "\${LOCAL_DATA_DIR}/"
      
      echo "✅ Data download complete!"
      echo "   Size: \$(du -sh \${LOCAL_DATA_DIR} | cut -f1)"
      ls -la \${LOCAL_DATA_DIR}/
      
      # Run training
      echo ""
      echo "🏋️ Starting distributed training..."
      cd /data/ACT-test
      
      python3 -m act_test.train_dist \\
        --data_dir "\${LOCAL_DATA_DIR}" \\
        --chunk_size \${CHUNK_SIZE} \\
        --max_steps \${MAX_STEPS} \\
        --learning_rate \${LEARNING_RATE} \\
        --learning_rate_backbone \${LEARNING_RATE}
      
      TRAIN_EXIT=\$?
      
      # Upload checkpoints
      echo ""
      echo "📤 Uploading checkpoints to GCS..."
      CKPT_DIR=\$(find \${LOCAL_DATA_DIR}/checkpoints -maxdepth 1 -type d -name "*_ddp" | head -1)
      
      if [ -n "\${CKPT_DIR}" ] && [ -d "\${CKPT_DIR}" ]; then
        gcloud storage cp -r "\${CKPT_DIR}" "\${OUTPUT_GCS_PATH}/"
        echo "✅ Checkpoints uploaded to: \${OUTPUT_GCS_PATH}/\$(basename \${CKPT_DIR})"
      else
        echo "⚠️  No checkpoint directory found"
      fi
      
      if [ \$TRAIN_EXIT -eq 0 ]; then
        echo ""
        echo "🎉 Training completed successfully!"
      else
        echo ""
        echo "❌ Training failed with exit code: \$TRAIN_EXIT"
      fi
      
      exit \$TRAIN_EXIT

  - path: /etc/profile.d/act_env.sh
    permissions: '0644'
    content: |
      export PYTHONUNBUFFERED=1
      export PATH="/root/.local/bin:\$PATH"

runcmd:
  # Create directories on NVMe storage
  - mkdir -p /data/dataset /data/checkpoints /data/outputs
  - chmod -R 777 /data
  
  # Install Google Cloud SDK for gsutil/gcloud storage
  - curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  - echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
  - apt-get update && apt-get install -y google-cloud-cli
  
  # Install NVIDIA Container Toolkit
  - curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  - curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  - apt-get update && apt-get install -y nvidia-container-toolkit docker.io
  - nvidia-ctk runtime configure --runtime=docker || true
  - systemctl restart docker || true
  
  # Install Python ML dependencies
  - pip3 install --upgrade pip
  - pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  - pip3 install numpy einops opencv-python-headless pillow huggingface-hub timm safetensors h5py webdataset PyYAML matplotlib tqdm requests click pydantic wandb
  
  # Clone ACT repository
  - cd /data && git clone https://github.com/yourusername/ACT-test.git || echo "Clone skipped"
  - cd /data/ACT-test && pip3 install -e . || echo "Package install skipped"
  
  # Print setup completion
  - echo "========================================"
  - echo "SFCompute node ready!"
  - echo "NVMe storage: /data (2TB+)"
  - nvidia-smi --query-gpu=name,memory.total --format=csv
  - echo "========================================"
  - echo "Next steps:"
  - echo "1. gcloud auth login"
  - echo "2. /data/download_and_train.sh"

final_message: |
  ========================================
  SFCompute ACT Training Node Ready!
  
  To start training:
  1. SSH: sf nodes ssh root@${NODE_NAME}
  2. Auth: gcloud auth login
  3. Train: /data/download_and_train.sh
  ========================================
EOF

echo "✅ Cloud-init file generated"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "🔍 DRY RUN - Commands that would be executed:"
    echo ""
    if [ "$NODE_TYPE" = "auto" ]; then
        echo "   sf nodes create ${NODE_NAME} --auto --zone ${ZONE} --max-price ${MAX_PRICE} --user-data-file ./${GENERATED_CLOUD_INIT}"
    else
        echo "   sf nodes create ${NODE_NAME} --zone ${ZONE} --duration ${DURATION} --max-price ${MAX_PRICE} --user-data-file ./${GENERATED_CLOUD_INIT}"
    fi
    echo ""
    echo "Run without --dry-run to create the node."
    exit 0
fi

# Create the node
echo ""
echo "🔄 Creating SFCompute node..."
if [ "$NODE_TYPE" = "auto" ]; then
    sf nodes create ${NODE_NAME} \
        --auto \
        --zone ${ZONE} \
        --max-price ${MAX_PRICE} \
        --user-data-file ./${GENERATED_CLOUD_INIT}
else
    sf nodes create ${NODE_NAME} \
        --zone ${ZONE} \
        --duration ${DURATION} \
        --max-price ${MAX_PRICE} \
        --user-data-file ./${GENERATED_CLOUD_INIT}
fi

echo ""
echo "✅ Node creation initiated!"
echo ""
echo "📋 Next Steps:"
echo "======================================="
echo ""
echo "1. Check node status:"
echo "   sf nodes list"
echo ""
echo "2. Wait for node to be 'Running' (takes ~5 minutes)"
echo ""
echo "3. SSH into the node:"
echo "   sf nodes ssh root@${NODE_NAME}"
echo ""
echo "4. Once connected, authenticate with GCS:"
echo "   gcloud auth login"
echo ""
echo "5. Start training:"
echo "   /data/download_and_train.sh"
echo ""
echo "📊 Monitoring:"
echo "   View logs: sf nodes logs ${NODE_NAME}"
echo ""
if [ "$NODE_TYPE" = "reserved" ]; then
    echo "⏰ Extend time: sf nodes extend ${NODE_NAME} --duration 2h --max-price ${MAX_PRICE}"
else
    echo "🛑 Release node: sf nodes release ${NODE_NAME}"
fi
echo "🗑️  Delete node: sf nodes delete ${NODE_NAME}"
echo ""
echo "======================================="

