#!/bin/bash

# Set your project details
PROJECT_ID="mauricearm"
IMAGE_NAME="act-training"
TAG="latest"
REGION="us-east5"

# Full image URI
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${IMAGE_URI}"

# Build the image
docker build -t ${IMAGE_URI} .

# Push to Google Container Registry
echo "Pushing to GCR..."
docker push ${IMAGE_URI}

echo "Container built and pushed successfully!"
echo "Image URI: ${IMAGE_URI}" 