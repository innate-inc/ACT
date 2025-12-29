#!/bin/bash
# =============================================================================
# download_data_lambda.sh
# =============================================================================
# Download data from GCS bucket to Lambda Labs instance storage
#
# Usage:
#   ./download_data_lambda.sh <GCS_PATH> [LOCAL_PATH]
#
# Examples:
#   ./download_data_lambda.sh gs://my-bucket/training-data
#   ./download_data_lambda.sh gs://my-bucket/training-data /data/my-dataset
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
    echo "  $0 gs://my-bucket/training-data"
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
echo "📥 Lambda Labs Data Download"
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
    exit 1
fi
print_success "Authenticated as: ${ACTIVE_ACCOUNT}"

# =============================================================================
# Step 2: Check Storage
# =============================================================================
echo ""
echo "💾 Storage Information"
echo "----------------------"
df -h "${LOCAL_PATH}" 2>/dev/null || df -h "$(dirname ${LOCAL_PATH})" 2>/dev/null || df -h /
echo ""

# =============================================================================
# Step 3: Create Destination Directory
# =============================================================================
echo "📁 Creating destination directory..."
mkdir -p "${LOCAL_PATH}"
print_success "Directory ready: ${LOCAL_PATH}"

# =============================================================================
# Step 4: Check if Data Already Exists
# =============================================================================
if [ -d "${LOCAL_PATH}" ] && [ "$(ls -A ${LOCAL_PATH} 2>/dev/null)" ]; then
    echo ""
    print_warning "Data already exists at ${LOCAL_PATH}"
    echo ""
    echo "Contents:"
    ls -lah "${LOCAL_PATH}" | head -10
    echo ""
    read -p "Re-download? This will overwrite existing data. (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping download, using existing data"
        exit 0
    fi
    print_info "Removing existing data..."
    rm -rf "${LOCAL_PATH}"/*
fi

# =============================================================================
# Step 5: Download Data
# =============================================================================
echo ""
echo "⏬ Starting Download"
echo "-------------------"
print_info "Source: ${GCS_PATH}"
print_info "Destination: ${LOCAL_PATH}"
echo ""

START_TIME=$(date +%s)

# Use gcloud storage cp for faster parallel transfers
print_info "Downloading with gcloud storage cp..."
gcloud storage cp -r "${GCS_PATH}/*" "${LOCAL_PATH}/"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# =============================================================================
# Step 6: Verify Download
# =============================================================================
echo ""
echo "✅ Download Complete!"
echo "-------------------"
echo "   Duration: ${DURATION}s ($(($DURATION / 60))m $(($DURATION % 60))s)"
echo ""

if [ -d "${LOCAL_PATH}" ] && [ "$(ls -A ${LOCAL_PATH} 2>/dev/null)" ]; then
    print_success "Data verified at ${LOCAL_PATH}"
    echo ""
    echo "📊 Download Statistics:"
    echo "   Size:  $(du -sh ${LOCAL_PATH} | cut -f1)"
    echo "   Files: $(find ${LOCAL_PATH} -type f | wc -l)"
    echo "   Dirs:  $(find ${LOCAL_PATH} -type d | wc -l)"
    echo ""
    echo "📁 Contents:"
    ls -lah "${LOCAL_PATH}" | head -20
    echo ""
    print_success "Ready for training!"
    echo ""
    echo "Next steps:"
    echo "   ./lambda_train.sh --local_data_dir ${LOCAL_PATH}"
    echo ""
    echo "   Or directly:"
    echo "   python -m act_test.train_dist --data_dir ${LOCAL_PATH}"
else
    print_error "Download verification failed - directory is empty!"
    exit 1
fi

