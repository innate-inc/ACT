#!/bin/bash
set -e

# =============================================================================
# Build and Push Container for SFCompute (Docker Hub)
# =============================================================================
# This script builds the Docker container and pushes to Docker Hub
#
# Usage:
#   ./build_container_sfcompute.sh [DOCKER_USERNAME] [IMAGE_TAG]
#
# Examples:
#   ./build_container_sfcompute.sh                    # Uses defaults
#   ./build_container_sfcompute.sh myusername         # Custom username
#   ./build_container_sfcompute.sh myusername v1.0    # Custom username and tag
# =============================================================================

# Configuration
DOCKER_USERNAME="${1:-heemyk}"
IMAGE_NAME="act-training"
TAG="${2:-latest}"

# Full image URI
IMAGE_URI="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "🐳 Building Docker Image for SFCompute"
echo "======================================="
echo "📦 Image: ${IMAGE_URI}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if logged in to Docker Hub
echo "🔐 Checking Docker Hub authentication..."
if ! docker info 2>/dev/null | grep -q "Username"; then
    echo "⚠️  Not logged in to Docker Hub. Please login:"
    docker login
fi

# Build the image using the sfcompute Dockerfile
echo ""
echo "🔨 Building image (no cache - fresh build)..."
docker build --no-cache -t ${IMAGE_URI} -f Dockerfile.sfcompute .

# Tag with additional tags
docker tag ${IMAGE_URI} ${DOCKER_USERNAME}/${IMAGE_NAME}:h100
docker tag ${IMAGE_URI} ${DOCKER_USERNAME}/${IMAGE_NAME}:sfcompute

# Push to Docker Hub
echo ""
echo "🚀 Pushing image to Docker Hub..."
docker push ${IMAGE_URI}
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:h100
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:sfcompute

echo ""
echo "✅ Image built and pushed successfully!"
echo "======================================="
echo "Image URIs:"
echo "  - ${IMAGE_URI}"
echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:h100"
echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:sfcompute"
echo ""
echo "📝 To use this image on SFCompute K8s:"
echo "   kubectl apply -f k8s-act-training.yaml"
echo ""
echo "📝 To use this image on SFCompute (direct SSH):"
echo "   1. SSH into your node: sf nodes ssh root@<node-name>"
echo "   2. Pull the image: docker pull ${IMAGE_URI}"
echo "   3. Run with GPU: docker run --gpus all -v /data:/data ${IMAGE_URI}"
