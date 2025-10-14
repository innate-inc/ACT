#!/bin/bash
set -e

# Set your project details
PROJECT_ID="mauricearm"
IMAGE_NAME="act-training"
TAG="latest"
REGION="us-east5"

# Use Google Container Registry (simpler - no repository creation needed)
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

echo "🐳 Building Docker image: ${IMAGE_URI}"
echo "📦 Building for linux/amd64 platform (required for Vertex AI)"

# Build the image for AMD64 architecture (required for Vertex AI)
docker build --platform linux/amd64 -t ${IMAGE_URI} .

# Configure Docker for GCR
gcloud auth configure-docker

# Push to GCR
echo "🚀 Pushing image to Google Container Registry..."
docker push ${IMAGE_URI}

echo "✅ Image built and pushed successfully!"
echo "Image URI: ${IMAGE_URI}" 