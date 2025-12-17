#!/bin/bash
# vertex_train.sh
# This script downloads data from GCS and runs distributed training

set -e

# Add /root/.local/bin to PATH
export PATH="/root/.local/bin:$PATH"

# Default values (matching train_dist.py defaults)
WORLD_SIZE=4
DATA_DIR="/data/PaperMulti_1_2_Filtered"
CHUNK_SIZE=30
MAX_STEPS=15000
LEARNING_RATE=5e-5
LEARNING_RATE_BACKBONE=5e-5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --world_size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --world_size INT              Number of GPUs to use (default: 4)"
            echo "  --data_dir STR               Path/GCS URI to dataset (default: /data/PaperMulti_1_2_Filtered)"
            echo "  --chunk_size INT             Action sequence length (default: 30)"
            echo "  --max_steps INT              Maximum training steps (default: 15000)"
            echo "  --learning_rate FLOAT        Main learning rate (default: 5e-5)"
            echo "  --learning_rate_backbone FLOAT  Backbone learning rate (default: 5e-5)"
            echo "  -h, --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Starting Vertex AI Training Pipeline"
echo "========================================"
echo "Configuration:"
echo "  World size: $WORLD_SIZE"
echo "  Data dir: $DATA_DIR"
echo "  Chunk size: $CHUNK_SIZE"
echo "  Max steps: $MAX_STEPS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Learning rate backbone: $LEARNING_RATE_BACKBONE"
echo ""

# Ensure script is run as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run this script as root (e.g., sudo $0 ...)"
    exit 1
fi

# Step 1: Create data directory
echo "📁 Setting up data directory..."
echo "==============================="

# Create the data directory
mkdir -p /data
chmod 777 /data
echo "✅ Data directory created at /data"
echo ""

# Step 2: Download data from GCS if data_dir is a GCS URI
echo "📥 Handling data directory..."
echo "============================="

# Determine the proper user for running commands
if [ -n "$SUDO_USER" ]; then
    ACTUAL_USER="$SUDO_USER"
    USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
else
    ACTUAL_USER=$(whoami)
    USER_HOME="$HOME"
fi

LOCAL_DATA_DIR="/data/$(basename "$DATA_DIR")"

# Check if DATA_DIR is a GCS URI (starts with gs://)
if [[ "$DATA_DIR" == gs://* ]]; then
    echo "🌐 Detected GCS URI: $DATA_DIR"
    echo "📂 Local destination: $LOCAL_DATA_DIR"
    
    # Check if data already exists locally
    if [ -d "$LOCAL_DATA_DIR" ] && [ "$(ls -A "$LOCAL_DATA_DIR" 2>/dev/null)" ]; then
        echo "✅ Data already exists locally at $LOCAL_DATA_DIR. Skipping download."
    else
        echo "🔄 Downloading data from GCS..."
        # Run gsutil as the actual user (not root)
        sudo -u "$ACTUAL_USER" gsutil -m cp -r "$DATA_DIR" "/data/"
        
        # Ensure proper permissions
        chown -R "$ACTUAL_USER:$ACTUAL_USER" "$LOCAL_DATA_DIR"
        echo "✅ Data downloaded successfully to $LOCAL_DATA_DIR"
    fi
else
    echo "📁 Using local/existing data directory: $DATA_DIR"
    LOCAL_DATA_DIR="$DATA_DIR"
fi

# Step 3: Install Python dependencies
echo ""
echo "🐍 Installing Python dependencies..."
echo "===================================="

# Install the act_test package and its dependencies
echo "🔄 Installing act_test package..."
cd "$(dirname "$0")/.."
sudo -u "$ACTUAL_USER" pip3 install -e .

echo "✅ Python dependencies installed successfully"
echo ""

# Step 4: Run distributed training
echo ""
echo "🎯 Starting distributed training..."
echo "==================================="

# Build the python command with arguments
PYTHON_CMD="python3 act_test/train_dist.py --data_dir \"$LOCAL_DATA_DIR\" --chunk_size $CHUNK_SIZE --max_steps $MAX_STEPS --learning_rate $LEARNING_RATE --learning_rate_backbone $LEARNING_RATE_BACKBONE --world_size $WORLD_SIZE"

echo "🚀 Executing: $PYTHON_CMD"
echo ""

# Change to the actual user and run the training
sudo -u "$ACTUAL_USER" bash -c "$PYTHON_CMD"

# Step 5: Upload checkpoints to GCS
echo ""
echo "📤 Uploading checkpoints to GCS..."
echo "=================================="

# Find the checkpoint directory (it's created inside the local data directory)
CHECKPOINT_BASE_DIR="$LOCAL_DATA_DIR/checkpoints"

if [ -d "$CHECKPOINT_BASE_DIR" ]; then
    echo "🔍 Found checkpoint directory: $CHECKPOINT_BASE_DIR"
    
    # Get the checkpoint directory
    # If RUN_NAME is set, look for that specific directory
    # Otherwise, find the most recent *_ddp directory
    if [ -n "$RUN_NAME" ]; then
        LATEST_CHECKPOINT_DIR="$CHECKPOINT_BASE_DIR/$RUN_NAME"
    else
        LATEST_CHECKPOINT_DIR=$(find "$CHECKPOINT_BASE_DIR" -maxdepth 1 -type d -name "*_ddp" | sort | tail -1)
    fi
    
    if [ -n "$LATEST_CHECKPOINT_DIR" ] && [ -d "$LATEST_CHECKPOINT_DIR" ]; then
        # Use RUN_NAME from environment variable if available, otherwise fall back to directory name
        if [ -n "$RUN_NAME" ]; then
            GCS_DESTINATION="${OUTPUT_BUCKET}/${RUN_NAME}"
        else
            CHECKPOINT_DIR_NAME=$(basename "$LATEST_CHECKPOINT_DIR")
            GCS_DESTINATION="gs://maurice-prod-data/ckpts/${CHECKPOINT_DIR_NAME}"
        fi
        
        echo "🔄 Uploading checkpoint contents"
        echo "📂 Source: $LATEST_CHECKPOINT_DIR"
        echo "☁️  Destination: $GCS_DESTINATION"
        
        # Upload the checkpoint directory contents to GCS
        sudo -u "$ACTUAL_USER" gsutil -m cp -r "$LATEST_CHECKPOINT_DIR/"* "$GCS_DESTINATION/"
        
        echo "✅ Checkpoints uploaded successfully to $GCS_DESTINATION"
    else
        echo "⚠️  No checkpoint directory found in $CHECKPOINT_BASE_DIR"
    fi
else
    echo "⚠️  Checkpoint base directory not found: $CHECKPOINT_BASE_DIR"
fi

echo ""
echo "🎉 Training completed!"
