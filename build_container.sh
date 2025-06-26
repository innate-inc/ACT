#!/bin/bash

# Set your project details
PROJECT_ID="mauricearm"
IMAGE_NAME="act-training"
TAG="latest"
REGION="us-east5"

# Use Google Container Registry (simpler - no repository creation needed)
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${IMAGE_URI}"

# Build the image
docker build -t ${IMAGE_URI} .

# Configure Docker for GCR
gcloud auth configure-docker

# Push to GCR
echo "Pushing to Google Container Registry..."
docker push ${IMAGE_URI}

echo "Container built and pushed successfully!"
echo "Image URI: ${IMAGE_URI}" 