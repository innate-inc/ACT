#!/bin/bash
# =============================================================================
# download_data_sfcompute.sh
# =============================================================================
# Download data from GCS bucket to SFCompute node's NVMe storage
#
# Usage:
#   ./download_data_sfcompute.sh <GCS_PATH> [LOCAL_PATH]
#
# Examples:
#   ./download_data_sfcompute.sh gs://my-bucket/training-data
#   ./download_data_sfcompute.sh gs://my-bucket/training-data /data/my-dataset
# =============================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

# =============================================================================
# Parse Arguments
# =============================================================================
if [ $# -lt 1 ]; then
    echo "Usage: $0 <GCS_PATH> [LOCAL_PATH]"
    echo ""
    echo "Arguments:"
    echo "  GCS_PATH     GCS bucket path (e.g., gs://my-bucket/data)"
    echo "  LOCAL_PATH   Local destination (default: /data/dataset)"
    echo ""
    echo "Examples:"
    echo "  $0 gs://maurice-prod-data/PaperMulti_1_2_Filtered"
    echo "  $0 gs://my-bucket/training-data /data/my-custom-dir"
    echo ""
    echo "Environment Variables:"
    echo "  GOOGLE_APPLICATION_CREDENTIALS  Path to service account JSON key"
    exit 1
fi

GCS_PATH="$1"
LOCAL_PATH="${2:-/data/dataset}"

# Extract bucket name and path for validation
if [[ ! "$GCS_PATH" =~ ^gs:// ]]; then
    print_error "Invalid GCS path. Must start with gs://"
    exit 1
fi

# =============================================================================
# Header
# =============================================================================
echo ""
echo "========================================"
echo "📥 SFCompute Data Download"
echo "========================================"
echo ""
print_info "Source:      ${GCS_PATH}"
print_info "Destination: ${LOCAL_PATH}"
echo ""

# =============================================================================
# Step 1: Check Prerequisites
# =============================================================================
echo "🔍 Checking Prerequisites..."
echo "----------------------------"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not found!"
    echo ""
    echo "Install with:"
    echo "  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg"
    echo "  echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' | tee /etc/apt/sources.list.d/google-cloud-sdk.list"
    echo "  apt-get update && apt-get install -y google-cloud-cli"
    exit 1
fi
print_success "gcloud CLI found"

# Check authentication
ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1)
if [ -z "$ACTIVE_ACCOUNT" ]; then
    print_warning "Not authenticated with GCS"
    echo ""
    echo "Please authenticate using one of these methods:"
    echo ""
    echo "  Method 1 - Interactive login:"
    echo "    gcloud auth login"
    echo ""
    echo "  Method 2 - Service account:"
    echo "    gcloud auth activate-service-account --key-file=/path/to/key.json"
    echo ""
    echo "  Method 3 - Environment variable:"
    echo "    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"
    echo ""
    exit 1
fi
print_success "Authenticated as: ${ACTIVE_ACCOUNT}"

# Check if bucket is accessible
echo ""
echo "🔗 Validating GCS access..."
if ! gcloud storage ls "${GCS_PATH}" &> /dev/null; then
    print_error "Cannot access ${GCS_PATH}"
    echo ""
    echo "Please check:"
    echo "  1. The GCS path is correct"
    echo "  2. You have permission to access this bucket"
    echo "  3. The bucket/path exists"
    exit 1
fi
print_success "GCS path accessible"

# =============================================================================
# Step 2: Check Storage
# =============================================================================
echo ""
echo "💾 Checking Storage..."
echo "----------------------"

# Check available storage
if df -h /data &> /dev/null; then
    AVAILABLE=$(df -h /data | awk 'NR==2 {print $4}')
    MOUNT_POINT=$(df -h /data | awk 'NR==2 {print $6}')
    print_info "Available storage: ${AVAILABLE}"
    print_info "Mount point: ${MOUNT_POINT}"
else
    print_warning "/data not mounted as separate filesystem"
    print_info "Using local storage"
fi

# Get size of data to download
echo ""
echo "📊 Calculating download size..."
DATA_SIZE=$(gcloud storage du -s "${GCS_PATH}" 2>/dev/null | awk '{print $1}')
if [ -n "$DATA_SIZE" ]; then
    # Convert to human readable
    DATA_SIZE_HR=$(numfmt --to=iec-i --suffix=B ${DATA_SIZE} 2>/dev/null || echo "${DATA_SIZE} bytes")
    print_info "Data size: ${DATA_SIZE_HR}"
fi

# =============================================================================
# Step 3: Prepare Local Directory
# =============================================================================
echo ""
echo "📁 Preparing Local Directory..."
echo "--------------------------------"

# Check if directory exists and has content
if [ -d "${LOCAL_PATH}" ] && [ "$(ls -A ${LOCAL_PATH} 2>/dev/null)" ]; then
    EXISTING_SIZE=$(du -sh "${LOCAL_PATH}" 2>/dev/null | cut -f1)
    print_warning "Directory already exists with data (${EXISTING_SIZE})"
    echo ""
    echo "Options:"
    echo "  1) Skip download (use existing data)"
    echo "  2) Delete and re-download"
    echo "  3) Download to different directory"
    echo ""
    read -p "Choose option [1/2/3]: " -n 1 -r
    echo
    
    case $REPLY in
        1)
            print_info "Using existing data"
            echo ""
            echo "📂 Contents:"
            ls -la "${LOCAL_PATH}/"
            exit 0
            ;;
        2)
            print_info "Removing existing data..."
            rm -rf "${LOCAL_PATH}"/*
            ;;
        3)
            read -p "Enter new directory path: " NEW_PATH
            LOCAL_PATH="${NEW_PATH}"
            ;;
        *)
            print_info "Using existing data (default)"
            exit 0
            ;;
    esac
fi

# Create directory
mkdir -p "${LOCAL_PATH}"
chmod 777 "${LOCAL_PATH}"
print_success "Directory ready: ${LOCAL_PATH}"

# =============================================================================
# Step 4: Download Data
# =============================================================================
echo ""
echo "⬇️  Starting Download..."
echo "------------------------"
echo ""

START_TIME=$(date +%s)

# Use gcloud storage for parallel downloads (faster than gsutil)
# The -r flag handles recursive copy
# The -m flag would enable parallel but gcloud storage already does this
gcloud storage cp -r "${GCS_PATH}/*" "${LOCAL_PATH}/"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Calculate speed
DOWNLOADED_SIZE=$(du -sb "${LOCAL_PATH}" 2>/dev/null | cut -f1)
if [ -n "$DOWNLOADED_SIZE" ] && [ "$DURATION" -gt 0 ]; then
    SPEED=$((DOWNLOADED_SIZE / DURATION))
    SPEED_HR=$(numfmt --to=iec-i --suffix=B/s ${SPEED} 2>/dev/null || echo "${SPEED} B/s")
fi

# =============================================================================
# Step 5: Verify Download
# =============================================================================
echo ""
echo "✔️  Verifying Download..."
echo "-------------------------"

# Count files
FILE_COUNT=$(find "${LOCAL_PATH}" -type f | wc -l)
DIR_COUNT=$(find "${LOCAL_PATH}" -type d | wc -l)
FINAL_SIZE=$(du -sh "${LOCAL_PATH}" | cut -f1)

print_success "Download completed!"
echo ""
echo "📊 Statistics:"
echo "   Duration:     $((DURATION / 60))m $((DURATION % 60))s"
echo "   Total size:   ${FINAL_SIZE}"
echo "   Files:        ${FILE_COUNT}"
echo "   Directories:  ${DIR_COUNT}"
if [ -n "$SPEED_HR" ]; then
    echo "   Avg speed:    ${SPEED_HR}"
fi

echo ""
echo "📂 Contents:"
ls -la "${LOCAL_PATH}/"

# Check for common data formats
echo ""
echo "🔍 Data format detection:"
if ls "${LOCAL_PATH}"/*.hdf5 &> /dev/null 2>&1 || ls "${LOCAL_PATH}"/*.h5 &> /dev/null 2>&1; then
    HDF5_COUNT=$(find "${LOCAL_PATH}" -name "*.hdf5" -o -name "*.h5" | wc -l)
    print_info "Found ${HDF5_COUNT} HDF5 files"
fi
if ls "${LOCAL_PATH}"/*.tar &> /dev/null 2>&1; then
    TAR_COUNT=$(find "${LOCAL_PATH}" -name "*.tar" | wc -l)
    print_info "Found ${TAR_COUNT} WebDataset tar files"
fi
if [ -d "${LOCAL_PATH}/webdataset" ]; then
    print_info "WebDataset directory found"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
print_success "Data ready for training!"
echo "========================================"
echo ""
echo "Data location: ${LOCAL_PATH}"
echo ""
echo "Next steps:"
echo "  1. Run training:"
echo "     python3 -m act_test.train_dist --data_dir ${LOCAL_PATH}"
echo ""
echo "  2. Or use the training script:"
echo "     ./sfcompute_train.sh --local_data_dir ${LOCAL_PATH}"
echo ""

