#!/bin/bash
# vertex_train.sh
# This script sets up RAID, downloads data from GCS, and runs distributed training

set -e

# Default values (matching train_dist.py defaults)
WORLD_SIZE=4
DATA_DIR="/home/vignesh/raid/PaperMulti_1_2_Filtered"
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
            echo "  --data_dir STR               Path/GCS URI to dataset (default: /home/vignesh/raid/PaperMulti_1_2_Filtered)"
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

# Ensure script is run as root for RAID setup
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run this script as root (e.g., sudo $0 ...)"
    exit 1
fi

# Step 0: Install system dependencies
echo "📦 Installing system dependencies..."
echo "===================================="

# Update package list
echo "🔄 Updating package list..."
apt update -y

# Install required packages
echo "🔄 Installing required packages..."
apt install -y \
    mdadm \
    google-cloud-cli \
    python3-pip \
    build-essential \
    curl \
    wget

echo "✅ System dependencies installed successfully"
echo ""

# Step 1: Set up RAID
echo "🔧 Setting up RAID 0 array..."
echo "================================"

# Define the NVMe devices to be used
DEVICES="/dev/nvme0n1 /dev/nvme0n2 /dev/nvme0n3 /dev/nvme0n4"

# Check if RAID array already exists
if [ -e /dev/md0 ]; then
    echo "⚠️  RAID array /dev/md0 already exists. Checking if mounted..."
    if mount | grep -q "/dev/md0"; then
        echo "✅ RAID array already mounted. Skipping RAID setup."
    else
        echo "🔄 RAID array exists but not mounted. Mounting..."
        # Determine the proper home directory
        if [ -n "$SUDO_USER" ]; then
            USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
        else
            USER_HOME="$HOME"
        fi
        MOUNTPOINT="$USER_HOME/raid"
        mkdir -p "$MOUNTPOINT"
        mount /dev/md0 "$MOUNTPOINT"
        chmod 777 "$MOUNTPOINT"
        echo "✅ RAID array mounted to $MOUNTPOINT"
    fi
else
    echo "🔄 Creating RAID 0 array /dev/md0 using devices: $DEVICES"
    mdadm --create --verbose /dev/md0 --level=0 --raid-devices=4 $DEVICES

    echo "⏳ Waiting 10 seconds for /dev/md0 to initialize..."
    sleep 10

    echo "🔄 Creating ext4 filesystem on /dev/md0..."
    mkfs.ext4 -F /dev/md0

    # Determine the proper home directory
    if [ -n "$SUDO_USER" ]; then
        USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    else
        USER_HOME="$HOME"
    fi

    MOUNTPOINT="$USER_HOME/raid"
    echo "🔄 Creating mount point at $MOUNTPOINT..."
    mkdir -p "$MOUNTPOINT"

    echo "🔄 Mounting /dev/md0 to $MOUNTPOINT..."
    mount /dev/md0 "$MOUNTPOINT"

    echo "🔍 Verifying mount..."
    df -h "$MOUNTPOINT"

    chmod 777 "$MOUNTPOINT"
    echo "✅ RAID 0 array setup complete. Mounted at $MOUNTPOINT"
fi

# Step 2: Download data from GCS if data_dir is a GCS URI
echo ""
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

LOCAL_DATA_DIR="$USER_HOME/raid/$(basename "$DATA_DIR")"

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
        sudo -u "$ACTUAL_USER" gsutil -m cp -r "$DATA_DIR" "$USER_HOME/raid/"
        
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
cd "$(dirname "$0")"
sudo -u "$ACTUAL_USER" pip3 install -e .

echo "✅ Python dependencies installed successfully"
echo ""

# Step 4: Run distributed training
echo ""
echo "🎯 Starting distributed training..."
echo "==================================="

# Build the python command with arguments
PYTHON_CMD="python act_test/train_dist.py --data_dir \"$LOCAL_DATA_DIR\" --chunk_size $CHUNK_SIZE --max_steps $MAX_STEPS --learning_rate $LEARNING_RATE --learning_rate_backbone $LEARNING_RATE_BACKBONE --world_size $WORLD_SIZE"

echo "🚀 Executing: $PYTHON_CMD"
echo ""

# Change to the actual user and run the training
sudo -u "$ACTUAL_USER" bash -c "$PYTHON_CMD"

echo ""
echo "🎉 Training completed!"
