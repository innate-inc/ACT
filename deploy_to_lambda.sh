#!/bin/bash
set -e

# =============================================================================
# Deploy ACT Training to Lambda Labs
# =============================================================================
# This script helps deploy the ACT training pipeline to Lambda Labs instances
#
# Usage:
#   ./deploy_to_lambda.sh [OPTIONS]
#
# Examples:
#   ./deploy_to_lambda.sh --build                    # Build & push image first
#   ./deploy_to_lambda.sh --data-path gs://bucket    # Custom data path
# =============================================================================

# Default configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-heemyk}"
IMAGE_NAME="act-training"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DATA_GCS_PATH="${DATA_GCS_PATH:-}"
OUTPUT_GCS_PATH="${OUTPUT_GCS_PATH:-}"

# Parse command line arguments
BUILD_IMAGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_IMAGE=true
            shift
            ;;
        --data-path)
            DATA_GCS_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_GCS_PATH="$2"
            shift 2
            ;;
        --docker-username)
            DOCKER_USERNAME="$2"
            shift 2
            ;;
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Deploy ACT training to Lambda Labs GPU instances"
            echo ""
            echo "Options:"
            echo "  --build              Build and push Docker image first"
            echo "  --data-path PATH     GCS path to training data"
            echo "  --output-path PATH   GCS path for outputs/checkpoints"
            echo "  --docker-username    Docker Hub username (default: heemyk)"
            echo "  --image-tag          Docker image tag (default: latest)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  DATA_GCS_PATH        Alternative to --data-path"
            echo "  OUTPUT_GCS_PATH      Alternative to --output-path"
            echo "  DOCKER_USERNAME       Alternative to --docker-username"
            echo ""
            echo "Examples:"
            echo "  $0 --build"
            echo "  $0 --data-path gs://my-bucket/data --output-path gs://my-bucket/outputs"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

IMAGE_URI="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "🚀 Lambda Labs Deployment"
echo "========================="
echo ""

# Build image if requested
if [ "$BUILD_IMAGE" = true ]; then
    echo "🔨 Building Docker image..."
    ./build_container_lambda.sh "${DOCKER_USERNAME}" "${IMAGE_TAG}"
    echo ""
fi

# Print deployment instructions
echo "📋 Deployment Instructions"
echo "========================="
echo ""
echo "1. SSH into your Lambda Labs instance:"
echo "   ssh ubuntu@<your-lambda-instance-ip>"
echo ""
echo "2. Pull the Docker image:"
echo "   docker pull ${IMAGE_URI}"
echo ""
echo "3. Set up GCS authentication (if using GCS):"
echo "   gcloud auth login"
echo "   # OR"
echo "   gcloud auth activate-service-account --key-file=/path/to/key.json"
echo ""
echo "4. Run the training container:"
if [ -n "${DATA_GCS_PATH}" ] && [ -n "${OUTPUT_GCS_PATH}" ]; then
    echo "   docker run --gpus all -v /data:/data \\"
    echo "     -v ~/.config/gcloud:/root/.config/gcloud \\"
    echo "     ${IMAGE_URI} \\"
    echo "     ./lambda_train.sh --data_gcs_path ${DATA_GCS_PATH} --output_gcs_path ${OUTPUT_GCS_PATH}"
elif [ -n "${DATA_GCS_PATH}" ]; then
    echo "   docker run --gpus all -v /data:/data \\"
    echo "     -v ~/.config/gcloud:/root/.config/gcloud \\"
    echo "     ${IMAGE_URI} \\"
    echo "     ./lambda_train.sh --data_gcs_path ${DATA_GCS_PATH}"
else
    echo "   docker run --gpus all -v /data:/data ${IMAGE_URI} \\"
    echo "     ./lambda_train.sh --local_data_dir /data/dataset"
fi
echo ""
echo "5. Or run interactively:"
echo "   docker run -it --gpus all -v /data:/data ${IMAGE_URI}"
echo ""
echo "✅ Deployment instructions complete!"

