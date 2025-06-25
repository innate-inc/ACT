#!/bin/bash

set -e  # Exit on any error

echo "🚀 Starting ACT Training Job"
echo "================================"

# Parse environment variables
DATA_BUCKET=${DATA_BUCKET:-"gs://mauricearm-act-data"}
OUTPUT_BUCKET=${OUTPUT_BUCKET:-"gs://mauricearm-act-outputs"}
LOCAL_DATA_DIR="/app/data"

echo "📥 Downloading data from ${DATA_BUCKET} to ${LOCAL_DATA_DIR}"
echo "📤 Checkpoints will be uploaded to ${OUTPUT_BUCKET} after training"

# Download all data to local disk
echo "⏬ Starting data download..."
gsutil -m cp -r "${DATA_BUCKET}/*" "${LOCAL_DATA_DIR}/"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "✅ Data download completed successfully!"
    echo "📊 Data directory contents:"
    ls -la "${LOCAL_DATA_DIR}/"
    echo "📏 Data directory size:"
    du -sh "${LOCAL_DATA_DIR}/"
else
    echo "❌ Data download failed!"
    exit 1
fi

# Start training with local data path
echo "🏋️ Starting training..."
python -m act_test.train_dist --data_dir "${LOCAL_DATA_DIR}" "$@"

# Training finished, get the exit code
TRAIN_EXIT_CODE=$?

# Upload checkpoints after training completes
echo "🔄 Training finished, uploading checkpoints..."
if [ -d "${LOCAL_DATA_DIR}/checkpoints" ]; then
    # Find the specific checkpoint directory created by train_dist.py
    CHECKPOINT_RUN_DIR=$(find "${LOCAL_DATA_DIR}/checkpoints" -maxdepth 1 -type d -name "*_ddp" | head -1)
    
    if [ -n "$CHECKPOINT_RUN_DIR" ] && [ -d "$CHECKPOINT_RUN_DIR" ]; then
        RUN_DIR_NAME=$(basename "$CHECKPOINT_RUN_DIR")
        echo "📁 Uploading checkpoint directory: $RUN_DIR_NAME"
        
        # Upload the specific run directory
        gsutil -m cp -r "$CHECKPOINT_RUN_DIR" "${OUTPUT_BUCKET}/"
        
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