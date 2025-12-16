#!/bin/bash
set -e

# Set your project details
PROJECT_ID="mauricearm"
IMAGE_NAME="act-training"
TAG="h100-latest"
REGION="us-central1"

# Use Google Container Registry (simpler - no repository creation needed)
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

echo "🐳 Building Docker image: ${IMAGE_URI}"
echo "⚙️  Building for AMD64 (x86_64) platform for Vertex AI compatibility"

# Build the image for AMD64/x86_64 platform (required for Vertex AI)
docker build --platform linux/amd64 -t ${IMAGE_URI} .

# Configure Docker for GCR
gcloud auth configure-docker

# Push to GCR
echo "🚀 Pushing image to Google Container Registry..."
docker push ${IMAGE_URI}

echo "✅ Image built and pushed successfully!"
echo "Image URI: ${IMAGE_URI}" 