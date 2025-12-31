#!/bin/bash
# =============================================================================
# Deploy Lambda Labs Orchestrator to GCP
# =============================================================================
# Deploys the orchestrator service to Google Cloud Run with Redis (Memorystore)
#
# Setup:
#   1. Copy env.template to .env: cp env.template .env
#   2. Fill in your secrets in .env (especially LAMBDA_API_KEY)
#   3. Run: ./deploy_to_gcp.sh
#
# Usage:
#   ./deploy_to_gcp.sh                    # Deploy/update with .env settings
#   ./deploy_to_gcp.sh --project my-proj  # Override project
#   ./deploy_to_gcp.sh --dry-run          # Show what would be done
#   ./deploy_to_gcp.sh --status           # Check current deployment status
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

# =============================================================================
# Load .env file
# =============================================================================
load_env() {
    if [[ -f "${ENV_FILE}" ]]; then
        print_info "Loading configuration from .env"
        # Export all variables from .env (ignore comments and empty lines)
        set -a
        source "${ENV_FILE}"
        set +a
    else
        print_warning "No .env file found at ${ENV_FILE}"
        print_info "Create one from template: cp env.template .env"
        echo ""
    fi
}

# =============================================================================
# Configuration (with defaults)
# =============================================================================
load_env

PROJECT_ID="${GCP_PROJECT:-innate-agent}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="lambda-orchestrator"
REDIS_INSTANCE="lambda-redis"
DOCKERHUB_USER="${DOCKERHUB_USER:-heemyk}"
# Use timestamped tag to avoid Cloud Run caching issues
IMAGE_TAG=$(date +%Y%m%d%H%M%S)
IMAGE_NAME="docker.io/${DOCKERHUB_USER}/${SERVICE_NAME}:${IMAGE_TAG}"
DRY_RUN=false
ACTION="deploy"

# =============================================================================
# Parse arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Deploy Lambda Labs Orchestrator to GCP Cloud Run"
            echo ""
            echo "Options:"
            echo "  --project ID     GCP project ID (default: ${PROJECT_ID})"
            echo "  --region REGION  GCP region (default: ${REGION})"
            echo "  --status         Show current deployment status"
            echo "  --dry-run        Show what would be done without executing"
            echo "  -h, --help       Show this help"
            echo ""
            echo "Configuration:"
            echo "  1. Copy env.template to .env: cp env.template .env"
            echo "  2. Edit .env with your Lambda Labs API key"
            echo "  3. Run: ./deploy_to_gcp.sh"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Status check
# =============================================================================
if [[ "${ACTION}" == "status" ]]; then
    echo ""
    echo "========================================"
    echo "📊 Lambda Labs Orchestrator Status"
    echo "========================================"
    echo ""
    
    gcloud config set project "${PROJECT_ID}" 2>/dev/null
    
    # Check Cloud Run service
    if gcloud run services describe "${SERVICE_NAME}" --platform=managed --region="${REGION}" &>/dev/null; then
        print_success "Cloud Run service: DEPLOYED"
        SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
            --platform=managed --region="${REGION}" --format='value(status.url)')
        print_info "URL: ${SERVICE_URL}"
        
        # Get revision info
        REVISION=$(gcloud run services describe "${SERVICE_NAME}" \
            --platform=managed --region="${REGION}" --format='value(status.latestReadyRevisionName)')
        print_info "Latest revision: ${REVISION}"
    else
        print_warning "Cloud Run service: NOT DEPLOYED"
    fi
    
    # Check Redis
    if gcloud redis instances describe "${REDIS_INSTANCE}" --region="${REGION}" &>/dev/null; then
        print_success "Redis: RUNNING"
        REDIS_HOST=$(gcloud redis instances describe "${REDIS_INSTANCE}" \
            --region="${REGION}" --format='value(host)')
        print_info "Redis host: ${REDIS_HOST}"
    else
        print_warning "Redis: NOT CREATED"
    fi
    
    # Check VPC connector
    if gcloud compute networks vpc-access connectors describe "sfcompute-connector" --region="${REGION}" &>/dev/null; then
        print_success "VPC Connector: READY (sfcompute-connector)"
    elif gcloud compute networks vpc-access connectors describe "lambda-vpc-connector" --region="${REGION}" &>/dev/null; then
        print_success "VPC Connector: READY (lambda-vpc-connector)"
    else
        print_warning "VPC Connector: NOT CREATED"
    fi
    
    echo ""
    exit 0
fi

# =============================================================================
# Validate required secrets
# =============================================================================
validate_secrets() {
    local missing=0
    
    if [[ -z "${LAMBDA_API_KEY}" ]]; then
        print_error "LAMBDA_API_KEY not set!"
        print_info "Get your API key from: https://cloud.lambda.ai/api-keys"
        missing=1
    else
        print_success "Lambda API key configured"
    fi
    
    if [[ $missing -eq 1 ]]; then
        echo ""
        print_error "Missing required Lambda Labs credentials!"
        echo ""
        echo "Add to .env file:"
        echo "  LAMBDA_API_KEY=your-api-key-here"
        echo ""
        exit 1
    fi
}

# =============================================================================
# Main deployment
# =============================================================================
echo ""
echo "========================================"
echo "🚀 Lambda Labs Orchestrator Deployment"
echo "========================================"
echo ""
print_info "Project: ${PROJECT_ID}"
print_info "Region: ${REGION}"
print_info "Service: ${SERVICE_NAME}"
echo ""

# Check prerequisites
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not installed!"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    print_error "docker not installed!"
    exit 1
fi

# Validate secrets
validate_secrets

# Set project
gcloud config set project "${PROJECT_ID}" 2>/dev/null

# =============================================================================
# Step 1: Create Redis (Memorystore) instance
# =============================================================================
echo ""
echo "📦 Step 1: Setting up Redis (Memorystore)..."

if $DRY_RUN; then
    print_info "[DRY RUN] Would create Redis instance: ${REDIS_INSTANCE}"
    REDIS_HOST="10.0.0.1"
else
    if gcloud redis instances describe "${REDIS_INSTANCE}" --region="${REGION}" &>/dev/null; then
        print_success "Redis instance already exists"
    else
        print_info "Creating Redis instance (this may take a few minutes)..."
        gcloud redis instances create "${REDIS_INSTANCE}" \
            --size=1 \
            --region="${REGION}" \
            --redis-version=redis_7_0 \
            --tier=basic
        print_success "Redis instance created"
    fi
    
    REDIS_HOST=$(gcloud redis instances describe "${REDIS_INSTANCE}" \
        --region="${REGION}" --format='value(host)')
    print_info "Redis host: ${REDIS_HOST}"
fi

# =============================================================================
# Step 2: Build and push Docker image (using Cloud Build)
# =============================================================================
echo ""
echo "🐳 Step 2: Building Docker image with Cloud Build..."

cd "${SCRIPT_DIR}"

# Use Google Artifact Registry
AR_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/lambda-orchestrator/app:${IMAGE_TAG}"

if $DRY_RUN; then
    print_info "[DRY RUN] Would build: ${AR_IMAGE}"
else
    # Enable required APIs
    gcloud services enable run.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com 2>/dev/null || true
    
    # Build using Cloud Build with layer caching
    print_info "Submitting build to Cloud Build (with layer caching)..."
    print_info "This builds in GCP - no slow local upload required"
    
    # Submit build asynchronously using cloudbuild.yaml for caching
    BUILD_OUTPUT=$(gcloud builds submit --config=cloudbuild.yaml --async . 2>&1)
    BUILD_ID=$(echo "${BUILD_OUTPUT}" | grep -oP 'builds/\K[a-f0-9-]+' | head -1)
    
    if [[ -z "${BUILD_ID}" ]]; then
        print_error "Failed to submit build"
        echo "${BUILD_OUTPUT}"
        exit 1
    fi
    
    print_info "Build submitted: ${BUILD_ID}"
    print_info "Waiting for build to complete..."
    
    # Poll for build status (avoids log streaming permission issues)
    while true; do
        BUILD_STATUS=$(gcloud builds describe "${BUILD_ID}" --format='value(status)' 2>/dev/null)
        case "${BUILD_STATUS}" in
            SUCCESS)
                # Use the BUILD_ID tagged image for deployment
                AR_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/lambda-orchestrator/app:${BUILD_ID}"
                print_success "Image built: ${AR_IMAGE}"
                IMAGE_NAME="${AR_IMAGE}"
                break
                ;;
            FAILURE|TIMEOUT|CANCELLED)
                print_error "Cloud Build failed with status: ${BUILD_STATUS}"
                print_info "View logs: https://console.cloud.google.com/cloud-build/builds/${BUILD_ID}?project=${PROJECT_ID}"
                exit 1
                ;;
            *)
                printf "."
                sleep 5
                ;;
        esac
    done
    echo ""
fi

# =============================================================================
# Step 3: Create VPC Connector (for Redis access)
# =============================================================================
echo ""
echo "🔌 Step 3: Setting up VPC Connector..."

VPC_CONNECTOR="sfcompute-connector"

if $DRY_RUN; then
    print_info "[DRY RUN] Would use VPC connector: ${VPC_CONNECTOR}"
else
    if gcloud compute networks vpc-access connectors describe "${VPC_CONNECTOR}" --region="${REGION}" &>/dev/null; then
        print_success "VPC connector already exists: ${VPC_CONNECTOR}"
    else
        # Try alternative connector name with different IP range
        VPC_CONNECTOR="lambda-vpc-connector"
        if gcloud compute networks vpc-access connectors describe "${VPC_CONNECTOR}" --region="${REGION}" &>/dev/null; then
            print_success "VPC connector already exists: ${VPC_CONNECTOR}"
        else
            print_info "Creating VPC connector..."
            gcloud compute networks vpc-access connectors create "${VPC_CONNECTOR}" \
                --region="${REGION}" \
                --range="10.9.0.0/28"
            print_success "VPC connector created"
        fi
    fi
fi

# =============================================================================
# Step 4: Deploy to Cloud Run
# =============================================================================
echo ""
echo "☁️ Step 4: Deploying to Cloud Run..."

# Build environment variables string
ENV_VARS="REDIS_HOST=${REDIS_HOST}"
ENV_VARS="${ENV_VARS},REDIS_PORT=6379"
ENV_VARS="${ENV_VARS},DRY_RUN=${DRY_RUN_MODE:-false}"

# Lambda Labs API key
ENV_VARS="${ENV_VARS},LAMBDA_API_KEY=${LAMBDA_API_KEY}"
if [[ -n "${LAMBDA_SSH_KEY_NAME}" ]]; then
    ENV_VARS="${ENV_VARS},LAMBDA_SSH_KEY_NAME=${LAMBDA_SSH_KEY_NAME}"
fi

# GitHub token for private repo access on Lambda instances
if [[ -n "${GITHUB_TOKEN}" ]]; then
    print_info "GitHub token configured for private repo access"
    ENV_VARS="${ENV_VARS},GITHUB_TOKEN=${GITHUB_TOKEN}"
else
    print_warning "No GITHUB_TOKEN set - private repo clone will fail on Lambda instances"
fi

# GCS service account key for data access on Lambda instances
# First check if gcs-key-auth.json exists, then fall back to env var
GCS_KEY_FILE="${SCRIPT_DIR}/gcs-key-auth.json"
if [[ -f "${GCS_KEY_FILE}" ]]; then
    print_info "Found GCS service account key at ${GCS_KEY_FILE}"
    GCS_SERVICE_ACCOUNT_KEY_B64=$(cat "${GCS_KEY_FILE}" | base64 -w0)
    ENV_VARS="${ENV_VARS},GCS_SERVICE_ACCOUNT_KEY_B64=${GCS_SERVICE_ACCOUNT_KEY_B64}"
    print_success "GCS key loaded and base64 encoded"
elif [[ -n "${GCS_SERVICE_ACCOUNT_KEY_B64}" ]]; then
    print_info "GCS service account key configured from .env"
    ENV_VARS="${ENV_VARS},GCS_SERVICE_ACCOUNT_KEY_B64=${GCS_SERVICE_ACCOUNT_KEY_B64}"
else
    print_warning "No GCS key found - Lambda instances won't be able to download training data from GCS"
    print_info "Create gcs-key-auth.json with your service account key, or set GCS_SERVICE_ACCOUNT_KEY_B64 in .env"
fi

# Discord configuration
if [[ -n "${DISCORD_WEBHOOK_URL}" ]]; then
    print_info "Discord webhook configured"
    ENV_VARS="${ENV_VARS},DISCORD_WEBHOOK_URL=${DISCORD_WEBHOOK_URL}"
    ENV_VARS="${ENV_VARS},DISCORD_APPROVAL_TIMEOUT=${DISCORD_APPROVAL_TIMEOUT:-300}"
    ENV_VARS="${ENV_VARS},DISCORD_REQUIRE_APPROVAL=${DISCORD_REQUIRE_APPROVAL:-true}"
    
    # Generate callback secret for Discord links if not set
    # This allows Discord approve/reject links to bypass GCP auth securely
    if [[ -z "${DISCORD_CALLBACK_SECRET}" ]]; then
        DISCORD_CALLBACK_SECRET=$(openssl rand -hex 32)
        print_info "Generated new Discord callback secret"
    fi
    ENV_VARS="${ENV_VARS},DISCORD_CALLBACK_SECRET=${DISCORD_CALLBACK_SECRET}"
else
    print_warning "Discord webhook not configured (set DISCORD_WEBHOOK_URL in .env to enable)"
    ENV_VARS="${ENV_VARS},DISCORD_REQUIRE_APPROVAL=false"
fi

# Generate API secret key for application-level authentication
# This is used when Cloud Run allows unauthenticated access (for Discord callbacks)
if [[ -z "${API_SECRET_KEY}" ]]; then
    API_SECRET_KEY=$(openssl rand -hex 32)
    print_info "Generated new API secret key"
fi
ENV_VARS="${ENV_VARS},API_SECRET_KEY=${API_SECRET_KEY}"

if $DRY_RUN; then
    print_info "[DRY RUN] Would deploy to Cloud Run with env vars"
    echo ""
    echo "Environment variables that would be set:"
    echo "  REDIS_HOST=${REDIS_HOST}"
    echo "  LAMBDA_API_KEY=<set>"
    [[ -n "${LAMBDA_SSH_KEY_NAME}" ]] && echo "  LAMBDA_SSH_KEY_NAME=${LAMBDA_SSH_KEY_NAME}"
    [[ -n "${DISCORD_WEBHOOK_URL}" ]] && echo "  DISCORD_WEBHOOK_URL=<set>"
else
    # Check if service exists (update vs create)
    if gcloud run services describe "${SERVICE_NAME}" --platform=managed --region="${REGION}" &>/dev/null; then
        print_info "Updating existing service..."
        
        # Update existing service with --allow-unauthenticated for Discord callbacks
        # Application-level auth protects sensitive endpoints
        ORCHESTRATOR_SA="training-orchestrator@${PROJECT_ID}.iam.gserviceaccount.com"
        gcloud run services update "${SERVICE_NAME}" \
            --platform=managed \
            --region="${REGION}" \
            --image="${IMAGE_NAME}" \
            --service-account="${ORCHESTRATOR_SA}" \
            --update-env-vars="${ENV_VARS}"
        
        # Ensure unauthenticated access is allowed (for Discord callbacks)
        gcloud run services add-iam-policy-binding "${SERVICE_NAME}" \
            --platform=managed \
            --region="${REGION}" \
            --member="allUsers" \
            --role="roles/run.invoker" 2>/dev/null || true
        
        print_success "Service updated!"
    else
        print_info "Creating new service..."
        
        # Create new service with --allow-unauthenticated for Discord callbacks
        # Application-level auth protects sensitive endpoints
        ORCHESTRATOR_SA="training-orchestrator@${PROJECT_ID}.iam.gserviceaccount.com"
        gcloud run deploy "${SERVICE_NAME}" \
            --image="${IMAGE_NAME}" \
            --platform=managed \
            --region="${REGION}" \
            --vpc-connector="${VPC_CONNECTOR}" \
            --set-env-vars="${ENV_VARS}" \
            --service-account="${ORCHESTRATOR_SA}" \
            --memory=1Gi \
            --cpu=1 \
            --min-instances=1 \
            --max-instances=3 \
            --timeout=3600 \
            --allow-unauthenticated
        
        print_success "Service created!"
    fi
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
        --platform=managed --region="${REGION}" --format='value(status.url)')
    
    # Update Discord callback URL if needed
    if [[ -n "${DISCORD_WEBHOOK_URL}" ]]; then
        print_info "Setting Discord callback URL..."
        gcloud run services update "${SERVICE_NAME}" \
            --platform=managed --region="${REGION}" \
            --update-env-vars="DISCORD_CALLBACK_URL=${SERVICE_URL}"
    fi
    
    # Wait for service to be ready
    sleep 5
    
    # =============================================================================
    # Step 5: Post-deployment setup
    # =============================================================================
    echo ""
    echo "🔧 Step 5: Post-deployment setup..."
    
    # Get auth token for API calls
    TOKEN=$(gcloud auth print-identity-token)
    
    # Clear the job queue and cache on fresh deployment
    print_info "Clearing job queue and cache..."
    CLEAR_RESULT=$(curl -s -X POST "${SERVICE_URL}/admin/clear-all" \
        -H "Authorization: Bearer ${TOKEN}" \
        -H "Content-Type: application/json" 2>/dev/null || echo '{"error": "failed"}')
    
    if echo "${CLEAR_RESULT}" | grep -q '"status":"success"'; then
        print_success "Queue and cache cleared"
    else
        print_warning "Could not clear queue (may need manual cleanup)"
    fi
    
    # Ensure SSH key exists for Lambda Labs
    print_info "Checking SSH key '${LAMBDA_SSH_KEY_NAME}'..."
    SSH_RESULT=$(curl -s -X POST "${SERVICE_URL}/debug/ssh-keys/generate" \
        -H "Authorization: Bearer ${TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"${LAMBDA_SSH_KEY_NAME}\"}" 2>/dev/null || echo '{"error": "failed"}')
    
    # Check SSH key status
    if echo "${SSH_RESULT}" | grep -q '"status":"exists"'; then
        print_success "SSH key '${LAMBDA_SSH_KEY_NAME}' already exists"
    elif echo "${SSH_RESULT}" | grep -q '"status":"created"'; then
        print_success "SSH key '${LAMBDA_SSH_KEY_NAME}' created"
        
        # Extract and store the private key
        PRIVATE_KEY=$(echo "${SSH_RESULT}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('key',{}).get('private_key',''))" 2>/dev/null || echo "")
        
        if [[ -n "${PRIVATE_KEY}" ]]; then
            # Update the service with the private key (base64 encode for safety)
            PRIVATE_KEY_B64=$(echo "${PRIVATE_KEY}" | base64 -w0)
            
            gcloud run services update "${SERVICE_NAME}" \
                --platform=managed --region="${REGION}" \
                --update-env-vars="LAMBDA_SSH_PRIVATE_KEY=${PRIVATE_KEY_B64}" --quiet
            
            print_success "SSH private key stored in Cloud Run"
            
            # Also save locally for reference
            echo "${PRIVATE_KEY}" > "${SCRIPT_DIR}/.ssh_private_key.pem"
            chmod 600 "${SCRIPT_DIR}/.ssh_private_key.pem"
            print_warning "Private key saved to .ssh_private_key.pem - keep this safe!"
        else
            print_warning "Could not extract private key from response"
            echo "${SSH_RESULT}" | python3 -m json.tool 2>/dev/null || echo "${SSH_RESULT}"
        fi
    else
        print_warning "Could not verify SSH key - check Lambda Labs dashboard"
        echo "${SSH_RESULT}"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
if ! $DRY_RUN; then
    SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
        --platform=managed --region="${REGION}" --format='value(status.url)' 2>/dev/null || echo "")
    
    echo "========================================"
    print_success "Deployment Complete!"
    echo "========================================"
    echo ""
    print_info "Service URL: ${SERVICE_URL}"
    echo ""
    echo "Authentication:"
    echo "  - Discord callbacks: Protected by secret token (auto-generated)"
    echo "  - API endpoints: Protected by GCP identity token OR API key"
    echo ""
    echo "Test endpoints:"
    echo "  # Using GCP identity token:"
    echo "  TOKEN=\$(gcloud auth print-identity-token)"
    echo "  curl -H \"Authorization: Bearer \$TOKEN\" ${SERVICE_URL}/health"
    echo "  curl -H \"Authorization: Bearer \$TOKEN\" ${SERVICE_URL}/prices"
    echo ""
    echo "  # Health check (unauthenticated):"
    echo "  curl ${SERVICE_URL}/health"
    echo ""
    
    if [[ -n "${DISCORD_WEBHOOK_URL}" ]]; then
        print_success "Discord integration: ENABLED"
    else
        print_warning "Discord integration: DISABLED (add DISCORD_WEBHOOK_URL to .env)"
    fi
    echo ""
fi

