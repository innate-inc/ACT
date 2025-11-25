#!/bin/bash
# =============================================================================
# Deploy ACT Training to SFCompute Kubernetes
# =============================================================================
# This script builds, pushes the Docker image, and deploys to SFCompute K8s
#
# Usage:
#   ./deploy_to_sfcompute_k8s.sh [OPTIONS]
#
# Examples:
#   ./deploy_to_sfcompute_k8s.sh                           # Deploy training job
#   ./deploy_to_sfcompute_k8s.sh --ssh                     # Deploy SSH pod
#   ./deploy_to_sfcompute_k8s.sh --build                   # Build & push image first
#   ./deploy_to_sfcompute_k8s.sh --data-path gs://bucket   # Custom data path
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
DOCKER_USERNAME="${DOCKER_USERNAME:-yourusername}"
IMAGE_NAME="act-training"
IMAGE_TAG="latest"
IMAGE_URI="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# Data paths (GCS)
DATA_GCS_PATH="${DATA_GCS_PATH:-gs://maurice-prod-data/data/socks1wed_socks2wed_filt_merged}"
OUTPUT_GCS_PATH="${OUTPUT_GCS_PATH:-gs://maurice-prod-data/ckpts}"

# Training parameters
MAX_STEPS=120000
LEARNING_RATE="5e-5"
CHUNK_SIZE=30
BATCH_SIZE=96
WORLD_SIZE=8

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="${SCRIPT_DIR}/k8s"

# =============================================================================
# Parse Arguments
# =============================================================================
BUILD_IMAGE=false
DEPLOY_SSH=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build|-b)
            BUILD_IMAGE=true
            shift
            ;;
        --ssh)
            DEPLOY_SSH=true
            shift
            ;;
        --docker-user)
            DOCKER_USERNAME="$2"
            IMAGE_URI="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
            shift 2
            ;;
        --data-path)
            DATA_GCS_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_GCS_PATH="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Deploy ACT training to SFCompute Kubernetes"
            echo ""
            echo "Options:"
            echo "  --build, -b         Build and push Docker image first"
            echo "  --ssh               Deploy SSH pod instead of training job"
            echo "  --docker-user USER  Docker Hub username (default: yourusername)"
            echo "  --data-path PATH    GCS data path (default: maurice-prod-data/...)"
            echo "  --output-path PATH  GCS output path for checkpoints"
            echo "  --max-steps NUM     Maximum training steps (default: 120000)"
            echo "  --dry-run           Generate manifests without applying"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --build                    # Build image and deploy"
            echo "  $0 --ssh                      # Deploy SSH pod for development"
            echo "  $0 --docker-user myuser       # Use custom Docker Hub user"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Header
# =============================================================================
echo ""
echo "========================================"
echo "🚀 SFCompute Kubernetes Deployment"
echo "========================================"
echo ""
echo "📦 Image:       ${IMAGE_URI}"
echo "📊 Data:        ${DATA_GCS_PATH}"
echo "📤 Output:      ${OUTPUT_GCS_PATH}"
echo "🏋️ Max steps:   ${MAX_STEPS}"
echo "🎮 GPUs:        ${WORLD_SIZE}"
echo ""

# =============================================================================
# Prerequisites Check
# =============================================================================
echo "🔍 Checking prerequisites..."

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found!"
    echo "   Install: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi
echo "   ✅ kubectl found"

# Check sf CLI
if ! command -v sf &> /dev/null; then
    echo "❌ sf CLI not found!"
    echo "   Install: curl -fsSL https://sfcompute.com/cli/install | bash"
    exit 1
fi
echo "   ✅ sf CLI found"

# Check cluster connection
echo ""
echo "🔗 Checking Kubernetes cluster..."
if ! kubectl get nodes &> /dev/null 2>&1; then
    echo "❌ Cannot connect to Kubernetes cluster!"
    echo ""
    echo "Setup steps:"
    echo "  1. Buy K8s nodes: sf buy -d '12h' -t h100i"
    echo "  2. List clusters: sf clusters list"
    echo "  3. Add user:      sf clusters users add --cluster <name> --user $USER"
    echo "  4. Test:          kubectl get nodes"
    exit 1
fi

echo "   ✅ Connected to cluster"
kubectl get nodes

# =============================================================================
# Build Docker Image (if requested)
# =============================================================================
if [ "$BUILD_IMAGE" = true ]; then
    echo ""
    echo "🐳 Building Docker Image..."
    echo "============================"
    
    cd "${SCRIPT_DIR}"
    
    # Build with buildx for amd64 (in case running on ARM Mac)
    if [[ "$(uname -m)" == "arm64" ]]; then
        echo "   Building for amd64 (cross-compile from ARM)..."
        docker buildx build --platform linux/amd64 -t ${IMAGE_URI} -f Dockerfile.sfcompute --push .
    else
        docker build -t ${IMAGE_URI} -f Dockerfile.sfcompute .
        docker push ${IMAGE_URI}
    fi
    
    echo "   ✅ Image pushed: ${IMAGE_URI}"
fi

# =============================================================================
# Generate Kubernetes Manifest
# =============================================================================
echo ""
echo "📝 Generating Kubernetes manifest..."

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

if [ "$DEPLOY_SSH" = true ]; then
    # SSH Pod manifest
    MANIFEST_FILE="${K8S_DIR}/act-ssh-${TIMESTAMP}.yaml"
    JOB_NAME="act-ssh-${TIMESTAMP}"
    
    cat > "${MANIFEST_FILE}" << EOF
# Auto-generated SSH Pod - ${TIMESTAMP}
apiVersion: v1
kind: Pod
metadata:
  name: ${JOB_NAME}
  labels:
    app: act-training
spec:
  containers:
  - name: cuda
    image: ${IMAGE_URI}
    imagePullPolicy: Always
    command:
    - /bin/bash
    - -c
    - |
      apt-get update && apt-get install -y openssh-server
      passwd -d root
      echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
      echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
      echo 'PermitEmptyPasswords yes' >> /etc/ssh/sshd_config
      mkdir -p /var/run/sshd
      echo "SSH Pod Ready! Port-forward and connect:"
      echo "  kubectl port-forward pod/${JOB_NAME} 2222:22"
      echo "  ssh -p 2222 root@localhost"
      nvidia-smi
      /usr/sbin/sshd -D
    ports:
    - containerPort: 22
    env:
    - name: DATA_GCS_PATH
      value: "${DATA_GCS_PATH}"
    - name: OUTPUT_GCS_PATH
      value: "${OUTPUT_GCS_PATH}"
    resources:
      requests:
        nvidia.com/gpu: 8
        nvidia.com/hostdev: 8
        memory: "512Gi"
        cpu: "32"
      limits:
        nvidia.com/gpu: 8
        nvidia.com/hostdev: 8
        memory: "512Gi"
        cpu: "32"
    volumeMounts:
    - name: data-volume
      mountPath: /data
    - name: shm
      mountPath: /dev/shm
  volumes:
  - name: data-volume
    emptyDir: {}
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: "64Gi"
  restartPolicy: Never
EOF

else
    # Training Job manifest
    MANIFEST_FILE="${K8S_DIR}/act-training-${TIMESTAMP}.yaml"
    JOB_NAME="act-training-${TIMESTAMP}"
    
    cat > "${MANIFEST_FILE}" << EOF
# Auto-generated Training Job - ${TIMESTAMP}
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  labels:
    app: act-training
spec:
  completions: 1
  parallelism: 1
  completionMode: Indexed
  backoffLimit: 2
  template:
    metadata:
      labels:
        job-name: ${JOB_NAME}
    spec:
      containers:
      - name: trainer
        image: ${IMAGE_URI}
        imagePullPolicy: Always
        command:
        - /bin/bash
        - -c
        - |
          set -e
          echo "🚀 ACT Training Job: ${JOB_NAME}"
          echo "========================================"
          
          # GPU check
          nvidia-smi --query-gpu=name,memory.total --format=csv
          
          # Download data
          echo "📥 Downloading data from GCS..."
          mkdir -p /data/dataset
          gcloud storage cp -r "${DATA_GCS_PATH}/*" /data/dataset/
          echo "✅ Downloaded: \$(du -sh /data/dataset/)"
          
          # Training
          echo "🏋️ Starting training..."
          cd /app
          torchrun --standalone --nproc_per_node=${WORLD_SIZE} \\
            -m act_test.train_dist \\
            --data_dir /data/dataset \\
            --chunk_size ${CHUNK_SIZE} \\
            --max_steps ${MAX_STEPS} \\
            --learning_rate ${LEARNING_RATE} \\
            --learning_rate_backbone ${LEARNING_RATE} \\
            --batch_size ${BATCH_SIZE} \\
            --world_size ${WORLD_SIZE}
          
          # Upload checkpoints
          echo "📤 Uploading checkpoints..."
          CKPT=\$(find /data/dataset/checkpoints -maxdepth 1 -type d -name "*_ddp" | head -1)
          if [ -n "\$CKPT" ]; then
            gcloud storage cp -r "\$CKPT" "${OUTPUT_GCS_PATH}/"
            echo "✅ Uploaded to: ${OUTPUT_GCS_PATH}/\$(basename \$CKPT)"
          fi
          
          echo "🎉 Training complete!"
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: WANDB_MODE
          value: "online"
        - name: NCCL_DEBUG
          value: "INFO"
        resources:
          requests:
            nvidia.com/gpu: 8
            nvidia.com/hostdev: 8
            memory: "512Gi"
            cpu: "32"
          limits:
            nvidia.com/gpu: 8
            nvidia.com/hostdev: 8
            memory: "512Gi"
            cpu: "32"
        volumeMounts:
        - name: data-volume
          mountPath: /data
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: data-volume
        emptyDir: {}
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: "64Gi"
      restartPolicy: Never
EOF
fi

echo "   ✅ Generated: ${MANIFEST_FILE}"

# =============================================================================
# Deploy
# =============================================================================
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "🔍 DRY RUN - Manifest content:"
    echo "========================================"
    cat "${MANIFEST_FILE}"
    echo ""
    echo "========================================"
    echo "Run without --dry-run to apply"
    exit 0
fi

echo ""
echo "🚀 Deploying to Kubernetes..."
kubectl apply -f "${MANIFEST_FILE}"

# =============================================================================
# Post-deployment Info
# =============================================================================
echo ""
echo "========================================"
echo "✅ Deployment Complete!"
echo "========================================"
echo ""

if [ "$DEPLOY_SSH" = true ]; then
    echo "📋 SSH Pod: ${JOB_NAME}"
    echo ""
    echo "Connect with:"
    echo "  1. Wait for pod:    kubectl get pods -w"
    echo "  2. Port forward:    kubectl port-forward pod/${JOB_NAME} 2222:22"
    echo "  3. SSH:             ssh -p 2222 root@localhost"
    echo ""
    echo "Once connected:"
    echo "  gcloud auth login"
    echo "  gcloud storage cp -r '${DATA_GCS_PATH}/*' /data/dataset/"
    echo "  cd /app && python -m act_test.train_dist --data_dir /data/dataset"
else
    echo "📋 Training Job: ${JOB_NAME}"
    echo ""
    echo "Monitor with:"
    echo "  kubectl get pods -w"
    echo "  kubectl logs -f ${JOB_NAME}-0"
    echo ""
    echo "Cancel with:"
    echo "  kubectl delete job ${JOB_NAME}"
fi

echo ""
echo "All pods:"
kubectl get pods

