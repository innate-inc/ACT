#!/bin/bash

# Create RAID 0 array from NVMe Local SSDs for maximum performance
# WARNING: This will erase all data on the specified NVMe devices.

set -e

# Ensure the script is run as root.
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script as root (e.g., sudo ./create_raid.sh)"
  exit 1
fi

# Define the NVMe devices to be used.
DEVICES="/dev/nvme0n1 /dev/nvme0n2 /dev/nvme0n3 /dev/nvme0n4"
echo "Creating RAID 0 array /dev/md0 using devices: $DEVICES"
mdadm --create --verbose /dev/md0 --level=0 --raid-devices=4 $DEVICES

# Allow time for the array to finish initializing.
echo "Waiting 10 seconds for /dev/md0 to initialize..."
sleep 10

echo "Creating ext4 filesystem on /dev/md0..."
mkfs.ext4 -F /dev/md0

# Determine the proper home directory:
# If running via sudo, retrieve the SUDO_USER home directory.
if [ -n "$SUDO_USER" ]; then
    USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
else
    USER_HOME="$HOME"
fi

# Create the mount point ~/raid in the user's home directory.
MOUNTPOINT="$USER_HOME/raid"
echo "Creating mount point at $MOUNTPOINT..."
mkdir -p "$MOUNTPOINT"

echo "Mounting /dev/md0 to $MOUNTPOINT..."
mount /dev/md0 "$MOUNTPOINT"

echo "Verifying mount..."
df -h "$MOUNTPOINT"

chmod 777 "$MOUNTPOINT"
echo "RAID 0 array setup complete. It is mounted at $MOUNTPOINT."
