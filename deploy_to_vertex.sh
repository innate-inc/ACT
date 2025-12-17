#!/bin/bash

# Check if required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <DATA_GCS_PATH> <OUTPUT_GCS_PATH> <RUN_NAME> [JOB_NAME]"
    echo "Example: $0 gs://my-existing-data-bucket gs://my-existing-output-bucket experiment-001 my-training-job"
    exit 1
fi

# Configuration
PROJECT_ID="mauricearm"
REGION="us-central1"
IMAGE_NAME="act-training"
TAG="h100-latest"

# Get arguments
DATA_GCS_PATH="$1"
OUTPUT_GCS_PATH="$2"
RUN_NAME="$3"
JOB_NAME="${4:-act-training-$(date +%Y%m%d-%H%M%S)}"

# Construct the full output path with run name
FULL_OUTPUT_PATH="${OUTPUT_GCS_PATH%/}/${RUN_NAME}"

echo "🚀 Deploying ACT Training Job"
echo "================================"
echo "📊 Data source: ${DATA_GCS_PATH}"
echo "📤 Output destination: ${FULL_OUTPUT_PATH}/"
echo "🏷️  Run name: ${RUN_NAME}"
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

# Check if the run folder already exists
echo "🔍 Checking if run folder already exists..."
if gsutil ls "${FULL_OUTPUT_PATH}/" > /dev/null 2>&1; then
    echo "❌ Error: Output folder ${FULL_OUTPUT_PATH}/ already exists"
    echo "   Please choose a different run name or delete the existing folder"
    exit 1
fi

echo "✅ GCS paths validated successfully"
echo "✅ Run folder ${FULL_OUTPUT_PATH}/ will be created"

# Build and push container
echo "🐳 Building and pushing container..."
# ./build_container.sh  # Comment out or remove this line

# Submit training job
echo "🚀 Submitting training job to Vertex AI..."
python3 vertex_job_config.py \
    --project_id ${PROJECT_ID} \
    --region ${REGION} \
    --image_uri "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}" \
    --data_path ${DATA_GCS_PATH} \
    --output_path ${OUTPUT_GCS_PATH} \
    --run_name ${RUN_NAME} \
    --job_name ${JOB_NAME}

echo "✅ Job submitted successfully!"
echo "📝 Monitor your job at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}" 