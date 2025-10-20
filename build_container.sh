#!/bin/bash
set -e

# Set your project details
PROJECT_ID="mauricearm"
IMAGE_NAME="act-training"
# Production version:
# TAG="latest"
# Test version (UNCOMMENT for RAID testing, COMMENT OUT for production):
TAG="raid-test"
REGION="us-east5"

# Use Google Container Registry (simpler - no repository creation needed)
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

# Production messaging (comment out for testing):
# echo "🐳 Building Docker image: ${IMAGE_URI}"

# Test messaging (comment out for production):
echo "🧪 Building RAID TEST Docker image: ${IMAGE_URI}"
echo "⚠️  This is a TEST version for validating RAID setup - not for production training!"

# Build the image
docker build -t ${IMAGE_URI} .

# Configure Docker for GCR
gcloud auth configure-docker

# Push to GCR
echo "🚀 Pushing image to Google Container Registry..."
docker push ${IMAGE_URI}

echo "✅ Image built and pushed successfully!"
echo "Image URI: ${IMAGE_URI}" 