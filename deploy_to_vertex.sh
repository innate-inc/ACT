#!/bin/bash

# Parse command line arguments
usage() {
    echo "Usage: $0 <DATA_GCS_PATH> <OUTPUT_GCS_PATH> [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  DATA_GCS_PATH     GCS path to training data"
    echo "  OUTPUT_GCS_PATH   GCS path for outputs"
    echo ""
    echo "Options:"
    echo "  --job-name NAME           Job name (default: auto-generated)"
    echo "  --max-steps STEPS         Maximum training steps (default: 15000)"
    echo "  --learning-rate LR        Main learning rate (default: 5e-5)"
    echo "  --learning-rate-backbone LR  Backbone learning rate (default: 1e-5)"
    echo "  --help                    Show this help message"
    echo ""
    echo "Example: $0 gs://my-data gs://my-output --job-name my-job --max-steps 20000 --learning-rate 1e-4"
}

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    usage
    exit 1
fi

# Configuration
PROJECT_ID="mauricearm"
REGION="us-east5"
IMAGE_NAME="act-training-test"
TAG="latest"

# Get required arguments
DATA_GCS_PATH="$1"
OUTPUT_GCS_PATH="$2"

# Initialize optional arguments with defaults
JOB_NAME="act-training-$(date +%Y%m%d-%H%M%S)"
MAX_STEPS="15000"
LEARNING_RATE="5e-5"
LEARNING_RATE_BACKBONE="1e-5"

# Parse optional arguments
shift 2  # Remove the first two required arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --learning-rate-backbone)
            LEARNING_RATE_BACKBONE="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

echo "🚀 Deploying ACT Training Job"
echo "================================"
echo "📊 Data source: ${DATA_GCS_PATH}"
echo "📤 Output destination: ${OUTPUT_GCS_PATH}"
echo "🏷️  Job name: ${JOB_NAME}"
echo "🔢 Max steps: ${MAX_STEPS}"
echo "📈 Learning rate: ${LEARNING_RATE}"
echo "🧠 Backbone learning rate: ${LEARNING_RATE_BACKBONE}"
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
# ./build_container.sh  # Comment out or remove this line

# Submit training job
echo "🚀 Submitting training job to Vertex AI..."
python3 vertex_job_config.py \
    --project_id ${PROJECT_ID} \
    --region ${REGION} \
    --image_uri "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}" \
    --data_path ${DATA_GCS_PATH} \
    --output_path ${OUTPUT_GCS_PATH} \
    --job_name ${JOB_NAME} \
    --max_steps ${MAX_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --learning_rate_backbone ${LEARNING_RATE_BACKBONE}

echo "✅ Job submitted successfully!"
echo "📝 Monitor your job at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}" 