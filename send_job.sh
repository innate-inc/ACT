#!/bin/bash
# send_job.sh
# Script to send ACT training job to Vertex AI

set -e

echo "🚀 Vertex AI Job Submission Script"
echo "=================================="
echo ""

# Prompt for GS bucket name
read -p "Enter GS bucket name (e.g., maurice-prod-data): " BUCKET_NAME

# Validate bucket name
if [[ -z "$BUCKET_NAME" ]]; then
    echo "❌ Error: Bucket name cannot be empty"
    exit 1
fi

# Prompt for number of training steps
read -p "Enter number of training steps (default: 15000): " MAX_STEPS

# Use default if empty
if [[ -z "$MAX_STEPS" ]]; then
    MAX_STEPS=15000
fi

# Validate steps is a number
if ! [[ "$MAX_STEPS" =~ ^[0-9]+$ ]]; then
    echo "❌ Error: Number of steps must be a positive integer"
    exit 1
fi

# Set service account (you may want to make this configurable too)
SA="vertex-training@maurice-production.iam.gserviceaccount.com"

# Generate a unique job name with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="act-dist-training-${TIMESTAMP}"

echo ""
echo "📋 Job Configuration:"
echo "  Job Name: $JOB_NAME"
echo "  Bucket: gs://$BUCKET_NAME"
echo "  Max Steps: $MAX_STEPS"
echo "  Service Account: $SA"
echo ""

# Confirm before submitting
read -p "Submit job to Vertex AI? (y/N): " CONFIRM

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "❌ Job submission cancelled"
    exit 0
fi

echo ""
echo "🔄 Submitting job to Vertex AI..."
echo "================================="

# Submit the job
gcloud ai custom-jobs create \
  --region=us-east5 \
  --display-name="$JOB_NAME" \
  --service-account="$SA" \
  --python-package-uris="gs://$BUCKET_NAME/trainer/act_test-0.1.0.tar.gz" \
  --worker-pool-spec="machine-type=a2-ultragpu-4g,replica-count=1,executor-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest,python-module=act_test.run_vertex,accelerator-type=NVIDIA_A100_80GB,accelerator-count=4" \
  --args="--data_dir=gs://$BUCKET_NAME/data/PaperMulti_1_2_Filtered --max_steps=$MAX_STEPS"

echo ""
echo "✅ Job submitted successfully!"
echo "🔍 You can monitor the job in the Google Cloud Console:"
echo "   https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$(gcloud config get-value project)"
