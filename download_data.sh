#!/bin/bash

set -e  # Exit on any error

# Production messaging (comment out for testing):
# echo "🚀 Starting ACT Training Job"
# echo "================================"

# Test messaging (comment out for production):
echo "🧪 ACT RAID TEST MODE - Data Download & Storage Validation Only"
echo "==============================================================="
echo "⚠️  This is a TEST version - training is DISABLED"
echo "🎯 Purpose: Validate RAID setup and data download performance"
echo ""


# --- Environment Sanity Check ---
echo "🐍 Checking Python package versions..."
python -c "import numpy; import cv2; print(f'✅ NumPy version: {numpy.__version__}'); print(f'✅ OpenCV version: {cv2.__version__}')"
echo "--------------------------------"

# Parse environment variables
DATA_BUCKET=${DATA_BUCKET:-"gs://mauricearm-act-data"}
OUTPUT_BUCKET=${OUTPUT_BUCKET:-"gs://mauricearm-act-outputs"}

# --- Setup High-Performance Storage ---
echo "🚀 Setting up high-performance storage for training..."
if [ -f "/app/setup_vertex_raid.sh" ]; then
    echo "📦 Found Local SSD RAID setup script, configuring high-speed storage..."
    bash /app/setup_vertex_raid.sh
    if [ $? -eq 0 ]; then
        echo "✅ High-performance storage configured successfully!"
    else
        echo "⚠️  RAID setup failed, falling back to standard storage..."
        mkdir -p /cache
    fi
else
    echo "⚠️  RAID setup script not found, using standard storage..."
    mkdir -p /cache
fi

# Use /cache for all data storage (now optimized with Local SSDs if available)
CACHE_DIR="/cache/data"
mkdir -p "${CACHE_DIR}"

# Extract the last folder name from the bucket URL
# gs://bucket_name/dir1/dir2/dir3 -> dir3
LAST_FOLDER=$(basename "${DATA_BUCKET}")
ACTUAL_DATA_DIR="${CACHE_DIR}/${LAST_FOLDER}"

echo "📥 Downloading data from ${DATA_BUCKET} to ${CACHE_DIR}"
echo "📁 Expected data directory: ${ACTUAL_DATA_DIR}"
echo "📤 Checkpoints will be uploaded to ${OUTPUT_BUCKET} after training"

# Check available storage
echo "💽 Storage info for /cache:"
df -h /cache

# Download all data to high-performance storage
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

# =============================================================================
# 🧪 RAID TEST MODE SECTION - COMMENT OUT THIS ENTIRE SECTION FOR PRODUCTION
# =============================================================================
echo "🧪 RAID TEST MODE: Validating RAID setup and data download only"
echo "⚠️  Training is DISABLED in this test version"
echo ""

# Validate RAID setup worked
echo "🔍 Validating RAID setup..."
if mountpoint -q /cache; then
    echo "✅ /cache is properly mounted"
    echo "📊 Storage details:"
    df -h /cache
    
    # Check if it's actually using RAID (look for md0 device)
    if mount | grep -q "/dev/md0 on /cache"; then
        echo "✅ RAID array is active and mounted!"
        echo "📦 RAID details:"
        sudo mdadm --detail /dev/md0 2>/dev/null || echo "   (RAID details not available)"
    else
        echo "⚠️  /cache is mounted but not using RAID array"
        echo "   This might be fallback to boot disk storage"
    fi
else
    echo "❌ /cache is not mounted - RAID setup may have failed"
fi

# Test file I/O performance on the mounted storage
echo ""
echo "🚀 Testing storage performance..."
TEST_FILE="${CACHE_DIR}/raid_test_file"
echo "📝 Writing test file to ${TEST_FILE}..."

# Create a 100MB test file and measure write speed
dd if=/dev/zero of="${TEST_FILE}" bs=1M count=100 2>&1 | grep -E "(copied|MB/s)" || echo "Write test completed"

if [ -f "${TEST_FILE}" ]; then
    echo "✅ Write test successful"
    
    # Test read speed
    echo "📖 Testing read speed..."
    dd if="${TEST_FILE}" of=/dev/null bs=1M 2>&1 | grep -E "(copied|MB/s)" || echo "Read test completed"
    
    # Clean up test file
    rm -f "${TEST_FILE}"
    echo "🧹 Cleaned up test file"
else
    echo "❌ Write test failed"
fi

echo ""
echo "🎯 RAID Test Summary:"
echo "================================"
echo "✅ Data download: $([ -d "${ACTUAL_DATA_DIR}" ] && echo "SUCCESS" || echo "FAILED")"
echo "✅ Storage mount: $(mountpoint -q /cache && echo "SUCCESS" || echo "FAILED")"
echo "✅ Storage type: $(mount | grep -q "/dev/md0 on /cache" && echo "RAID ARRAY" || echo "FALLBACK STORAGE")"
echo "📊 Available space: $(df -h /cache | awk 'NR==2 {print $4}')"
echo ""
echo "🧪 RAID test completed - ready for production deployment!"

# Exit successfully for RAID test
exit 0
# =============================================================================
# END OF RAID TEST MODE SECTION
# =============================================================================

# =============================================================================
# 🚀 PRODUCTION MODE SECTION - UNCOMMENT THIS ENTIRE SECTION FOR PRODUCTION
# =============================================================================
# Start training with the actual data directory
# echo "🏋️ Starting training with data directory: ${ACTUAL_DATA_DIR}"
# python -m act_test.train_dist --data_dir "${ACTUAL_DATA_DIR}" "$@"
# 
# # Training finished, get the exit code
# TRAIN_EXIT_CODE=$?
# 
# # Upload checkpoints after training completes
# echo "🔄 Training finished, uploading checkpoints..."
# if [ -d "${ACTUAL_DATA_DIR}/checkpoints" ]; then
#     # Find the specific checkpoint directory created by train_dist.py
#     CHECKPOINT_RUN_DIR=$(find "${ACTUAL_DATA_DIR}/checkpoints" -maxdepth 1 -type d -name "*_ddp" | head -1)
#     
#     if [ -n "$CHECKPOINT_RUN_DIR" ] && [ -d "$CHECKPOINT_RUN_DIR" ]; then
#         RUN_DIR_NAME=$(basename "$CHECKPOINT_RUN_DIR")
#         echo "📁 Uploading checkpoint directory: $RUN_DIR_NAME"
#         
#         # Upload the specific run directory
#         gcloud storage cp --recursive "$CHECKPOINT_RUN_DIR" "${OUTPUT_BUCKET}/"
#         
#         echo "✅ Checkpoints uploaded to: ${OUTPUT_BUCKET}/${RUN_DIR_NAME}/"
#     else
#         echo "⚠️  No checkpoint directory found matching pattern *_ddp"
#     fi
# else
#     echo "⚠️  No checkpoints directory found"
# fi
# 
# if [ $TRAIN_EXIT_CODE -eq 0 ]; then
#     echo "✅ Training completed successfully!"
# else
#     echo "❌ Training failed with exit code $TRAIN_EXIT_CODE"
# fi
# 
# exit $TRAIN_EXIT_CODE
# =============================================================================
# END OF PRODUCTION MODE SECTION
# ============================================================================= 