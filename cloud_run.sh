#!/bin/bash
# =============================================================================
# cloud_run.sh
# =============================================================================
# Training script for GPU instances
# Runs distributed training using local data
#
# Environment Variables:
#   DATA_DIR       Path to training data (default: /training/data)
#   OUTPUT_DIR     Path for checkpoints and outputs (default: /training/out)
#   WORLD_SIZE     Number of GPUs (default: 4)
#   CHUNK_SIZE     Action sequence length (default: 30)
#   MAX_STEPS      Maximum training steps (default: 120000)
#   LEARNING_RATE  Main learning rate (default: 5e-5)
#   LEARNING_RATE_BACKBONE  Backbone learning rate (default: 5e-5)
#   BATCH_SIZE     Batch size per GPU (default: 96)
#   NUM_WORKERS    DataLoader workers per GPU (default: 4)
# =============================================================================

set -e

# =============================================================================
# Configuration (all via environment variables)
# =============================================================================
DATA_DIR="${DATA_DIR:-/training/data/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/training/out}"
WORLD_SIZE="${WORLD_SIZE:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-30}"
MAX_STEPS="${MAX_STEPS:-120000}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
LEARNING_RATE_BACKBONE="${LEARNING_RATE_BACKBONE:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-96}"
NUM_WORKERS="${NUM_WORKERS:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# =============================================================================
# Print Configuration
# =============================================================================
echo "🚀 ACT Training Pipeline"
echo "====================================="
echo ""
echo "📊 Configuration:"
echo "   World size (GPUs):     ${WORLD_SIZE}"
echo "   Data directory:        ${DATA_DIR}"
echo "   Output directory:      ${OUTPUT_DIR}"
echo "   Chunk size:            ${CHUNK_SIZE}"
echo "   Max steps:             ${MAX_STEPS}"
echo "   Learning rate:         ${LEARNING_RATE}"
echo "   LR backbone:           ${LEARNING_RATE_BACKBONE}"
echo "   Batch size per GPU:    ${BATCH_SIZE}"
echo "   Workers per GPU:       ${NUM_WORKERS}"
echo ""

# =============================================================================
# Step 1: System Checks
# =============================================================================
echo "🔍 System Checks"
echo "================"

# Check GPU availability
echo ""
echo "🎮 GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo ""
    echo "   Found ${GPU_COUNT} GPUs"
    
    if [ "$GPU_COUNT" -lt "$WORLD_SIZE" ]; then
        echo "⚠️  Warning: Requested ${WORLD_SIZE} GPUs but only ${GPU_COUNT} available"
        WORLD_SIZE=$GPU_COUNT
        echo "   Adjusted world_size to ${WORLD_SIZE}"
    fi
else
    echo "❌ nvidia-smi not found! GPU training will fail."
    exit 1
fi

# =============================================================================
# Step 2: Setup Directories
# =============================================================================
echo ""
echo "📁 Setting Up Directories"
echo "========================="

# Verify data exists
if [ ! -d "${DATA_DIR}" ] || [ -z "$(ls -A ${DATA_DIR} 2>/dev/null)" ]; then
    echo "❌ No data found at ${DATA_DIR}!"
    echo "   Set DATA_DIR to point to your training data."
    exit 1
fi
echo "   Data:   ${DATA_DIR} ($(du -sh ${DATA_DIR} | cut -f1))"

mkdir -p "${OUTPUT_DIR}"
echo "   Output: ${OUTPUT_DIR}"

# =============================================================================
# Step 3: Install System Dependencies
# =============================================================================
echo ""
echo "📦 Installing system dependencies..."
echo "====================================="

apt-get update -qq
apt-get install -y -qq \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0
echo "✅ System dependencies installed"

# =============================================================================
# Step 4: Setup Python Environment
# =============================================================================
echo ""
echo "🐍 Setting up Python environment..."
echo "====================================="

# Create and activate venv
if [ ! -d "${SCRIPT_DIR}/.venv" ]; then
    python3 -m venv "${SCRIPT_DIR}/.venv"
fi
source "${SCRIPT_DIR}/.venv/bin/activate"

pip install --upgrade pip --quiet

# Detect GPU architecture and install appropriate PyTorch build
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "   Detected GPU: ${GPU_NAME}"

if echo "${GPU_NAME}" | grep -qi "B200\|B100\|blackwell"; then
    echo "   ⚡ Blackwell GPU detected - installing PyTorch nightly with CUDA 12.8"
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --quiet
else
    echo "   Installing PyTorch with CUDA 12.1"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
fi

# Install project and all dependencies
pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet
pip install -e "${SCRIPT_DIR}" --quiet
echo "✅ Python environment ready"

# Disable torch.compile (SIGSEGV issues on some instances)
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Add act_test/ to PYTHONPATH so bare imports (from ACT, from data_utils, etc.) resolve
export PYTHONPATH="${SCRIPT_DIR}/act_test:${PYTHONPATH}"

# Verify key dependencies
python3 -c "
import torch
import numpy
import cv2
print(f'   PyTorch:  {torch.__version__}')
print(f'   CUDA:     {torch.cuda.is_available()} (devices: {torch.cuda.device_count()})')
print(f'   NumPy:    {numpy.__version__}')
print(f'   OpenCV:   {cv2.__version__}')
"

# =============================================================================
# Step 5: Run Distributed Training
# =============================================================================
echo ""
echo "🏋️ Starting Distributed Training"
echo "================================="
echo ""
echo "Training command:"
echo "   python3 -m act_test.train_dist \\"
echo "     --data_dir ${DATA_DIR} \\"
echo "     --chunk_size ${CHUNK_SIZE} \\"
echo "     --max_steps ${MAX_STEPS} \\"
echo "     --learning_rate ${LEARNING_RATE} \\"
echo "     --learning_rate_backbone ${LEARNING_RATE_BACKBONE} \\"
echo "     --batch_size ${BATCH_SIZE} \\"
echo "     --num_workers ${NUM_WORKERS} \\"
echo "     --world_size ${WORLD_SIZE}"
echo ""

# Run training
TRAIN_START=$(date +%s)

python3 -m act_test.train_dist \
    --data_dir "${DATA_DIR}" \

    --chunk_size ${CHUNK_SIZE} \
    --max_steps ${MAX_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --learning_rate_backbone ${LEARNING_RATE_BACKBONE} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --world_size ${WORLD_SIZE}

TRAIN_EXIT_CODE=$?
TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))

echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "   Duration: $((TRAIN_DURATION / 3600))h $((TRAIN_DURATION % 3600 / 60))m $((TRAIN_DURATION % 60))s"
else
    echo "❌ Training failed with exit code: ${TRAIN_EXIT_CODE}"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "======================================="
echo "🎉 Training Pipeline Complete!"
echo "======================================="
echo ""
echo "Summary:"
echo "   Exit code:   ${TRAIN_EXIT_CODE}"
echo "   Duration:    $((TRAIN_DURATION / 3600))h $((TRAIN_DURATION % 3600 / 60))m"
echo "   Data:        ${DATA_DIR}"
echo "   Output:      ${OUTPUT_DIR}"
echo ""

exit $TRAIN_EXIT_CODE

