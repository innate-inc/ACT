#!/bin/bash
# =============================================================================
# lambda_train.sh
# =============================================================================
# Training script for Lambda Labs GPU instances
# Downloads data from GCS to local storage and runs distributed training
#
# Usage:
#   ./lambda_train.sh [OPTIONS]
#
# This script is designed to run on Lambda Labs GPU instances with:
#   - Local storage (typically /data or /home)
#   - Multiple GPUs (A100, H100, etc.)
#   - GCS access for data download
# =============================================================================

set -e

# =============================================================================
# Default Configuration
# =============================================================================
WORLD_SIZE=4
DATA_GCS_PATH="${DATA_GCS_PATH:-}"  # GCS path (gs://bucket/path)
OUTPUT_GCS_PATH="${OUTPUT_GCS_PATH:-}"  # GCS output path
LOCAL_DATA_DIR="/data/dataset"
CHUNK_SIZE=30
MAX_STEPS=120000
LEARNING_RATE=5e-5
LEARNING_RATE_BACKBONE=5e-5
BATCH_SIZE=96
NUM_WORKERS=4

# =============================================================================
# Parse Command Line Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --world_size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --data_gcs_path)
            DATA_GCS_PATH="$2"
            shift 2
            ;;
        --output_gcs_path)
            OUTPUT_GCS_PATH="$2"
            shift 2
            ;;
        --local_data_dir)
            LOCAL_DATA_DIR="$2"
            shift 2
            ;;
        --chunk_size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --learning_rate_backbone)
            LEARNING_RATE_BACKBONE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Lambda Labs Training Script - Downloads data from GCS and runs distributed training"
            echo ""
            echo "Options:"
            echo "  --world_size INT              Number of GPUs (default: 4)"
            echo "  --data_gcs_path STR           GCS path to dataset (gs://bucket/path)"
            echo "  --output_gcs_path STR         GCS path for outputs (gs://bucket/path)"
            echo "  --local_data_dir STR          Local data directory (default: /data/dataset)"
            echo "  --chunk_size INT              Action sequence length (default: 30)"
            echo "  --max_steps INT               Maximum training steps (default: 120000)"
            echo "  --learning_rate FLOAT         Main learning rate (default: 5e-5)"
            echo "  --learning_rate_backbone FLOAT Backbone learning rate (default: 5e-5)"
            echo "  --batch_size INT              Batch size per GPU (default: 96)"
            echo "  --num_workers INT             DataLoader workers per GPU (default: 4)"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  DATA_GCS_PATH     Alternative to --data_gcs_path"
            echo "  OUTPUT_GCS_PATH   Alternative to --output_gcs_path"
            echo ""
            echo "Examples:"
            echo "  $0 --data_gcs_path gs://my-bucket/data --output_gcs_path gs://my-bucket/outputs"
            echo "  $0 --max_steps 200000 --learning_rate 1e-4"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Print Configuration
# =============================================================================
echo "🚀 Lambda Labs ACT Training Pipeline"
echo "====================================="
echo ""
echo "📊 Configuration:"
echo "   World size (GPUs):     ${WORLD_SIZE}"
echo "   Data GCS path:         ${DATA_GCS_PATH:-'(local data)'}"
echo "   Output GCS path:       ${OUTPUT_GCS_PATH:-'(no upload)'}"
echo "   Local data directory:  ${LOCAL_DATA_DIR}"
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

# Check storage
echo ""
echo "💾 Storage Information:"
if [ -d "/data" ]; then
    df -h /data
else
    echo "   /data not available - using ${LOCAL_DATA_DIR}"
fi

# =============================================================================
# Step 2: Setup Data Directory
# =============================================================================
echo ""
echo "📁 Setting Up Data Directory"
echo "============================"

mkdir -p "${LOCAL_DATA_DIR}"
mkdir -p "${LOCAL_DATA_DIR}/checkpoints"
mkdir -p "${LOCAL_DATA_DIR}/outputs"

echo "✅ Directories created:"
echo "   Data:        ${LOCAL_DATA_DIR}"
echo "   Checkpoints: ${LOCAL_DATA_DIR}/checkpoints"
echo "   Outputs:     ${LOCAL_DATA_DIR}/outputs"

# =============================================================================
# Step 3: Download Data from GCS (if specified)
# =============================================================================
if [ -n "${DATA_GCS_PATH}" ]; then
    echo ""
    echo "📥 Downloading Data from GCS"
    echo "============================"
    echo "   Source:      ${DATA_GCS_PATH}"
    echo "   Destination: ${LOCAL_DATA_DIR}"
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        echo "❌ gcloud CLI not installed!"
        echo "   Install with: apt-get install -y google-cloud-cli"
        exit 1
    fi
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1 | grep -q "@"; then
        echo "⚠️  Not authenticated with GCS."
        echo "   Please run: gcloud auth login"
        echo "   Or provide service account: gcloud auth activate-service-account --key-file=key.json"
        exit 1
    fi
    
    # Check if data already exists
    if [ -d "${LOCAL_DATA_DIR}" ] && [ "$(ls -A ${LOCAL_DATA_DIR} 2>/dev/null)" ]; then
        echo ""
        echo "📂 Data already exists locally:"
        ls -la "${LOCAL_DATA_DIR}/" | head -10
        echo ""
        read -p "Re-download data? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "⏭️  Skipping download, using existing data"
        else
            echo "🔄 Downloading fresh data..."
            rm -rf "${LOCAL_DATA_DIR}"/*
            gcloud storage cp -r "${DATA_GCS_PATH}/*" "${LOCAL_DATA_DIR}/"
        fi
    else
        # Download data
        echo "🔄 Starting download..."
        START_TIME=$(date +%s)
        
        gcloud storage cp -r "${DATA_GCS_PATH}/*" "${LOCAL_DATA_DIR}/"
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo ""
        echo "✅ Download complete! (${DURATION}s)"
    fi
    
    # Show data stats
    echo ""
    echo "📊 Data Statistics:"
    echo "   Size: $(du -sh ${LOCAL_DATA_DIR} | cut -f1)"
    echo "   Files: $(find ${LOCAL_DATA_DIR} -type f | wc -l)"
    ls -la "${LOCAL_DATA_DIR}/" | head -10
else
    echo ""
    echo "📁 Using Local Data"
    echo "==================="
    echo "   No GCS path specified, expecting data at: ${LOCAL_DATA_DIR}"
    
    if [ ! -d "${LOCAL_DATA_DIR}" ] || [ -z "$(ls -A ${LOCAL_DATA_DIR} 2>/dev/null)" ]; then
        echo "❌ No data found at ${LOCAL_DATA_DIR}!"
        echo "   Either:"
        echo "   1. Specify --data_gcs_path gs://your-bucket/data"
        echo "   2. Manually copy data to ${LOCAL_DATA_DIR}"
        exit 1
    fi
    
    echo "   Size: $(du -sh ${LOCAL_DATA_DIR} | cut -f1)"
fi

# =============================================================================
# Step 4: Install Dependencies (if needed)
# =============================================================================
echo ""
echo "🐍 Checking Python Environment"
echo "=============================="

# Check if act_test is installed
if ! python3 -c "import act_test" 2>/dev/null; then
    echo "📦 Installing act_test package..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${SCRIPT_DIR}/.."
    pip3 install -e . --quiet
    echo "✅ Package installed"
else
    echo "✅ act_test package already installed"
fi

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
echo "     --data_dir ${LOCAL_DATA_DIR} \\"
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
    --data_dir "${LOCAL_DATA_DIR}" \
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
# Step 6: Upload Checkpoints to GCS
# =============================================================================
if [ -n "${OUTPUT_GCS_PATH}" ]; then
    echo ""
    echo "📤 Uploading Checkpoints to GCS"
    echo "================================"
    
    # Find checkpoint directory
    CHECKPOINT_BASE="${LOCAL_DATA_DIR}/checkpoints"
    
    if [ -d "${CHECKPOINT_BASE}" ]; then
        # Get the most recent checkpoint directory
        LATEST_CKPT=$(find "${CHECKPOINT_BASE}" -maxdepth 1 -type d -name "*_ddp" | sort | tail -1)
        
        if [ -n "${LATEST_CKPT}" ] && [ -d "${LATEST_CKPT}" ]; then
            CKPT_NAME=$(basename "${LATEST_CKPT}")
            GCS_DEST="${OUTPUT_GCS_PATH}/${CKPT_NAME}"
            
            echo "   Source:      ${LATEST_CKPT}"
            echo "   Destination: ${GCS_DEST}"
            echo ""
            echo "   Contents:"
            ls -la "${LATEST_CKPT}/" | head -10
            echo ""
            
            echo "🔄 Uploading..."
            gcloud storage cp -r "${LATEST_CKPT}" "${OUTPUT_GCS_PATH}/"
            
            echo ""
            echo "✅ Checkpoints uploaded to: ${GCS_DEST}"
        else
            echo "⚠️  No checkpoint directory found matching pattern *_ddp"
        fi
    else
        echo "⚠️  Checkpoint base directory not found: ${CHECKPOINT_BASE}"
    fi
else
    echo ""
    echo "📁 Checkpoints saved locally"
    echo "   No OUTPUT_GCS_PATH specified, skipping upload"
    echo "   Checkpoints at: ${LOCAL_DATA_DIR}/checkpoints/"
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
echo "   Data:        ${LOCAL_DATA_DIR}"
echo "   Checkpoints: ${LOCAL_DATA_DIR}/checkpoints/"
if [ -n "${OUTPUT_GCS_PATH}" ]; then
    echo "   GCS Output:  ${OUTPUT_GCS_PATH}"
fi
echo ""

exit $TRAIN_EXIT_CODE

