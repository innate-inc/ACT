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
echo "🔍 Checking permissions - EUID: $EUID, USER: $(whoami)"

if [ "$EUID" -ne 0 ]; then
    echo "🔑 Not running as root, testing sudo access..."
    if sudo -n true 2>/dev/null; then
        echo "✅ Sudo access confirmed"
        SUDO="sudo"
    else
        echo "❌ This script requires root privileges to create RAID arrays"
        echo "   Either run as root or configure sudo access"
        echo "   Current user: $(whoami), EUID: $EUID"
        echo "   Testing sudo with: sudo -n whoami"
        sudo -n whoami 2>&1 || echo "   Sudo test failed"
        exit 1
    fi
else
    echo "✅ Running as root"
    SUDO=""
fi

echo "🔧 Will use SUDO command: '${SUDO}'"

# Auto-detect available NVMe Local SSD devices
echo "🔍 Detecting Local SSD devices..."
echo "📋 Full lsblk output:"
lsblk -d -o NAME,TYPE,SIZE

echo ""
echo "🔎 Filtering for NVMe disks:"
lsblk -d -o NAME,TYPE | grep disk | grep nvme || echo "   (no nvme disks found)"

echo ""
echo "🎯 Auto-detecting Local SSDs..."
LOCAL_SSDS=($(lsblk -d -o NAME,TYPE | grep disk | grep nvme | awk '{print "/dev/"$1}'))

if [ ${#LOCAL_SSDS[@]} -eq 0 ]; then
    echo "⚠️  No Local SSDs found. Using boot disk storage."
    echo "   This is normal for local development or smaller instance types."
    echo "   Available devices:"
    lsblk -d -o NAME,TYPE,SIZE
    exit 0
fi

echo "📦 Found ${#LOCAL_SSDS[@]} Local SSD(s): ${LOCAL_SSDS[*]}"

# Verify each device actually exists
echo "✅ Verifying detected devices exist:"
for device in "${LOCAL_SSDS[@]}"; do
    if [ -b "$device" ]; then
        echo "   ✅ $device exists"
    else
        echo "   ❌ $device does NOT exist!"
    fi
done

echo ""
echo "🔍 Investigating /dev/ directory for NVMe devices:"
echo "📂 Contents of /dev/ matching nvme*:"
ls -la /dev/nvme* 2>/dev/null || echo "   (no nvme* files found in /dev/)"

echo ""
echo "📂 Contents of /dev/ matching *nvme*:"
ls -la /dev/ | grep nvme || echo "   (no nvme entries found in /dev/)"

echo ""
echo "🔍 Checking /dev/disk/by-id/ for NVMe devices:"
ls -la /dev/disk/by-id/ | grep nvme 2>/dev/null || echo "   (no nvme entries in /dev/disk/by-id/)"

echo ""
echo "⏰ Waiting 5 seconds for device files to appear..."
sleep 5

echo "🔄 Re-checking device existence after wait:"
for device in "${LOCAL_SSDS[@]}"; do
    if [ -b "$device" ]; then
        echo "   ✅ $device now exists!"
    else
        echo "   ❌ $device still does not exist"
    fi
done

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
