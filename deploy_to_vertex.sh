#!/bin/bash

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <DATA_GCS_PATH> <OUTPUT_GCS_PATH> [JOB_NAME]"
    echo "Example: $0 gs://my-existing-data-bucket gs://my-existing-output-bucket my-training-job"
    exit 1
fi

# Configuration
PROJECT_ID="mauricearm"
REGION="us-east5"
IMAGE_NAME="act-training"
TAG="latest"

# Get arguments
DATA_GCS_PATH="$1"
OUTPUT_GCS_PATH="$2"
JOB_NAME="${3:-act-training-$(date +%Y%m%d-%H%M%S)}"

echo "🚀 Deploying ACT Training Job"
echo "================================"
echo "📊 Data source: ${DATA_GCS_PATH}"
echo "📤 Output destination: ${OUTPUT_GCS_PATH}"
echo "🏷️  Job name: ${JOB_NAME}"
echo ""

# Validate that buckets exist
echo "🔍 Validating GCS paths..."
if ! gsutil ls "${DATA_GCS_PATH}" > /dev/null 2>&1; then
    echo "❌ Error: Data bucket ${DATA_GCS_PATH} does not exist or is not accessible"
    exit 1
fi

if ! gsutil ls "${OUTPUT_GCS_PATH}" > /dev/null 2>&1; then
    echo "❌ Error: Output bucket ${OUTPUT_GCS_PATH} does not exist or is not accessible"
    exit 1
fi

echo "✅ GCS paths validated successfully"

# Build and push container
echo "🐳 Building and pushing container..."
./build_container.sh

# Submit training job
echo "🚀 Submitting training job to Vertex AI..."
python vertex_job_config.py \
    --project_id ${PROJECT_ID} \
    --region ${REGION} \
    --image_uri "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}" \
    --data_path ${DATA_GCS_PATH} \
    --output_path ${OUTPUT_GCS_PATH} \
    --job_name ${JOB_NAME}

echo "✅ Job submitted successfully!"
echo "📝 Monitor your job at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}" 