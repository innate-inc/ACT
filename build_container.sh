#!/bin/bash
set -euo pipefail

PROJECT_ID="mauricearm"
IMAGE_NAME="act-training"
TAG="h100-latest"
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

echo "🐳 Building (linux/amd64) and pushing: ${IMAGE_URI}"

# Ensure buildx is available and initialized
docker buildx create --use --name multiarch-builder >/dev/null 2>&1 || true
docker buildx inspect --bootstrap >/dev/null

# Push directly to registry as amd64
docker buildx build \
  --platform=linux/amd64 \
  -t "${IMAGE_URI}" \
  --push \
  .

echo "✅ Done: ${IMAGE_URI}"