#!/bin/bash

set -e  # Exit on any error

echo "🚀 Starting ACT Training Job"
echo "================================"


# --- Environment Sanity Check ---
echo "🐍 Checking Python package versions..."
python -c "import numpy; import cv2; print(f'✅ NumPy version: {numpy.__version__}'); print(f'✅ OpenCV version: {cv2.__version__}')"
echo "--------------------------------"

# Parse environment variables
DATA_BUCKET=${DATA_BUCKET:-"gs://mauricearm-act-data"}
OUTPUT_BUCKET=${OUTPUT_BUCKET:-"gs://mauricearm-act-outputs"}

# Use /cache for all data storage (4x local SSDs RAIDed)
CACHE_DIR="/cache/data"
mkdir -p "${CACHE_DIR}"

# Extract the last folder name from the bucket URL
# gs://bucket_name/dir1/dir2/dir3 -> dir3
LAST_FOLDER=$(basename "${DATA_BUCKET}")
ACTUAL_DATA_DIR="${CACHE_DIR}/${LAST_FOLDER}"

echo "📥 Downloading data from ${DATA_BUCKET} to ${CACHE_DIR}"
echo "📁 Expected data directory: ${ACTUAL_DATA_DIR}"
echo "📤 Checkpoints will be uploaded to ${OUTPUT_BUCKET} after training"
echo "💾 Using /cache (4x local SSDs RAIDed) for maximum speed"

# Check available storage
echo "💽 Storage info for /cache:"
df -h /cache

# Download all data to local SSD RAID
echo "⏬ Starting data download..."
gcloud storage cp --recursive "${DATA_BUCKET}" "${CACHE_DIR}/"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "✅ Data download completed successfully!"
    echo "📊 Cache directory contents:"
    ls -la "${CACHE_DIR}/"
    echo "📊 Data directory contents:"
    ls -la "${ACTUAL_DATA_DIR}/"
    echo "📏 Data directory size:"
    du -sh "${ACTUAL_DATA_DIR}/"
else
    echo "❌ Data download failed!"
    exit 1
fi

# Start training with the actual data directory
echo "🏋️ Starting training with data directory: ${ACTUAL_DATA_DIR}"
python -m act_test.train_dist --data_dir "${ACTUAL_DATA_DIR}" "$@"

# Training finished, get the exit code
TRAIN_EXIT_CODE=$?

# Upload checkpoints after training completes
echo "🔄 Training finished, uploading checkpoints..."
if [ -d "${ACTUAL_DATA_DIR}/checkpoints" ]; then
    # Find the specific checkpoint directory created by train_dist.py
    CHECKPOINT_RUN_DIR=$(find "${ACTUAL_DATA_DIR}/checkpoints" -maxdepth 1 -type d -name "*_ddp" | head -1)
    
    if [ -n "$CHECKPOINT_RUN_DIR" ] && [ -d "$CHECKPOINT_RUN_DIR" ]; then
        RUN_DIR_NAME=$(basename "$CHECKPOINT_RUN_DIR")
        echo "📁 Uploading checkpoint directory: $RUN_DIR_NAME"
        
        # Upload the specific run directory
        gcloud storage cp --recursive "$CHECKPOINT_RUN_DIR" "${OUTPUT_BUCKET}/"
        
        echo "✅ Checkpoints uploaded to: ${OUTPUT_BUCKET}/${RUN_DIR_NAME}/"
    else
        echo "⚠️  No checkpoint directory found matching pattern *_ddp"
    fi
else
    echo "⚠️  No checkpoints directory found"
fi

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code $TRAIN_EXIT_CODE"
fi

exit $TRAIN_EXIT_CODE 