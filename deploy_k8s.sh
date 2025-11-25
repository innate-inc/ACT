#!/bin/bash
# =============================================================================
# deploy_k8s.sh
# =============================================================================
# Deploy ACT training to SFCompute Kubernetes cluster
#
# Usage:
#   ./deploy_k8s.sh [OPTIONS]
#
# Options:
#   --gcs-key FILE     Path to GCS service account JSON key
#   --dry-run          Show what would be applied without applying
# =============================================================================

set -e

GCS_KEY_FILE=""
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gcs-key)
            GCS_KEY_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gcs-key FILE   Path to GCS service account JSON key"
            echo "  --dry-run        Show what would be applied"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🚀 Deploying ACT Training to Kubernetes"
echo "========================================"
echo ""

# Check kubectl connection
echo "🔍 Checking Kubernetes connection..."
if ! kubectl get nodes &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster"
    echo "   Run: sf clusters list"
    echo "   Then: sf clusters users add --cluster <name> --user $USER"
    exit 1
fi

echo "✅ Connected to Kubernetes"
kubectl get nodes
echo ""

# Check for GPUs
echo "🎮 Checking GPU resources..."
GPU_COUNT=$(kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' 2>/dev/null | tr ' ' '\n' | awk '{s+=$1} END {print s}')
echo "   Available GPUs: ${GPU_COUNT:-0}"
echo ""

# Create GCS secret if key file provided
if [ -n "$GCS_KEY_FILE" ]; then
    if [ -f "$GCS_KEY_FILE" ]; then
        echo "🔐 Creating GCS credentials secret..."
        kubectl delete secret gcs-key --ignore-not-found
        kubectl create secret generic gcs-key --from-file=gcs-key.json="$GCS_KEY_FILE"
        echo "✅ GCS secret created"
        
        # Update the job manifest to use the secret
        echo "📝 Updating job manifest to use GCS secret..."
    else
        echo "❌ GCS key file not found: $GCS_KEY_FILE"
        exit 1
    fi
fi

# Check if job already exists
if kubectl get job act-training &> /dev/null; then
    echo "⚠️  Job 'act-training' already exists"
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete job act-training
        echo "✅ Old job deleted"
    else
        echo "Exiting without changes"
        exit 0
    fi
fi

# Apply the job
echo ""
if [ "$DRY_RUN" = true ]; then
    echo "🔍 Dry run - would apply:"
    kubectl apply -f k8s-act-training.yaml --dry-run=client
else
    echo "📦 Applying Kubernetes job..."
    kubectl apply -f k8s-act-training.yaml
    
    echo ""
    echo "✅ Job submitted!"
    echo ""
    echo "📋 Monitor your training:"
    echo "   Watch pods:  kubectl get pods -w"
    echo "   View logs:   kubectl logs -f act-training-0"
    echo "   Describe:    kubectl describe job act-training"
    echo ""
    echo "🗑️  To delete the job:"
    echo "   kubectl delete job act-training"
fi

