#!/bin/bash

# Check if required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <DATA_GCS_PATH> <OUTPUT_GCS_PATH> <RUN_NAME> [JOB_NAME]"
    echo "Example: $0 gs://my-bucket/data gs://my-bucket/outputs my-run-001 my-job"
    echo ""
    echo "Arguments:"
    echo "  DATA_GCS_PATH    - GCS path to your training data"
    echo "  OUTPUT_GCS_PATH  - GCS path where checkpoints will be saved"
    echo "  RUN_NAME         - Name for this training run (used as checkpoint folder name)"
    echo "  JOB_NAME         - (Optional) Name for the Vertex AI job"
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

echo "🚀 Deploying ACT Training Job"
echo "================================"
echo "📊 Data source: ${DATA_GCS_PATH}"
echo "📤 Output destination: ${OUTPUT_GCS_PATH}"
echo "📁 Checkpoint folder: ${OUTPUT_GCS_PATH}/${RUN_NAME}"
echo "🏷️  Job name: ${JOB_NAME}"
echo ""

# Validate that buckets exist
echo "🔍 Validating GCS paths..."
if ! gsutil ls "${DATA_GCS_PATH}" > /dev/null 2>&1; then
    echo "❌ Error: Data path ${DATA_GCS_PATH} does not exist or is not accessible"
    exit 1
fi

# For output path, only check that the bucket exists (GCS will create folders automatically)
OUTPUT_BUCKET=$(echo "${OUTPUT_GCS_PATH}" | sed 's|gs://\([^/]*\).*|gs://\1|')
if ! gsutil ls "${OUTPUT_BUCKET}" > /dev/null 2>&1; then
    echo "❌ Error: Output bucket ${OUTPUT_BUCKET} does not exist or is not accessible"
    exit 1
fi

# Check if checkpoint folder already exists
echo "🔍 Checking if checkpoint folder already exists..."
if gsutil ls "${OUTPUT_GCS_PATH}/${RUN_NAME}/" > /dev/null 2>&1; then
    echo "❌ Error: Checkpoint folder ${OUTPUT_GCS_PATH}/${RUN_NAME}/ already exists!"
    echo "   Please choose a different run name or delete the existing folder with:"
    echo "   gsutil -m rm -r ${OUTPUT_GCS_PATH}/${RUN_NAME}/"
    exit 1
fi

echo "✅ GCS paths validated successfully"

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