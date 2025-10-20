#!/bin/bash

# Setup RAID 0 array from Local SSDs for Vertex AI training
# This version is optimized for use inside Docker containers on Vertex AI
# WARNING: This will erase all data on the specified NVMe devices.

set -e

echo "🔧 Setting up Local SSD RAID for maximum training performance..."

# Check if we have the necessary permissions
if ! command -v mdadm &> /dev/null || ! command -v mkfs.ext4 &> /dev/null; then
    echo "❌ Required tools (mdadm, mkfs) not available"
    exit 1
fi

# Check if we can run as root or have sudo access
if [ "$EUID" -ne 0 ] && ! sudo -n true 2>/dev/null; then
    echo "❌ This script requires root privileges to create RAID arrays"
    echo "   Either run as root or configure sudo access"
    exit 1
fi

# Use sudo if not root
SUDO=""
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
    echo "🔑 Using sudo for privileged operations..."
fi

# Auto-detect available NVMe Local SSD devices
LOCAL_SSDS=($(lsblk -d -o NAME,TYPE | grep disk | grep nvme | awk '{print "/dev/"$1}'))

if [ ${#LOCAL_SSDS[@]} -eq 0 ]; then
    echo "⚠️  No Local SSDs found. Using boot disk storage."
    echo "   This is normal for local development or smaller instance types."
    exit 0
fi

echo "📦 Found ${#LOCAL_SSDS[@]} Local SSD(s): ${LOCAL_SSDS[*]}"

# Create RAID 0 array from all available Local SSDs
echo "🛠️  Creating RAID 0 array /dev/md0..."
$SUDO mdadm --create --verbose /dev/md0 --level=0 --raid-devices=${#LOCAL_SSDS[@]} "${LOCAL_SSDS[@]}"

# Wait for array initialization
echo "⏳ Waiting for RAID array to initialize..."
sleep 10

# Create filesystem
echo "💾 Creating ext4 filesystem..."
$SUDO mkfs.ext4 -F /dev/md0

# Create and mount to /cache (for Vertex AI)
echo "📁 Creating mount point /cache..."
$SUDO mkdir -p /cache

echo "🔗 Mounting RAID array to /cache..."
$SUDO mount /dev/md0 /cache

# Set permissions for container user (get the actual username)
CONTAINER_USER=${SUDO_USER:-$(whoami)}
echo "👤 Setting permissions for user: $CONTAINER_USER"
$SUDO chown -R $CONTAINER_USER:$CONTAINER_USER /cache
$SUDO chmod 755 /cache

# Display results
echo "✅ RAID setup complete!"
echo "📊 Storage info:"
df -h /cache

echo "🚀 /cache is now ready for high-performance data storage!"
echo "   Total capacity: $(df -h /cache | awk 'NR==2 {print $2}')"
echo "   Performance: ~3-4x faster than persistent disk"
