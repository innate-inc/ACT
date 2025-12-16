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

# Build the image
docker build -t ${IMAGE_URI} .

# Configure Docker for GCR
gcloud auth configure-docker

# Push to GCR
echo "🚀 Pushing image to Google Container Registry..."
docker push ${IMAGE_URI}

echo "✅ Image built and pushed successfully!"
echo "Image URI: ${IMAGE_URI}" 