# =============================================================================
# Lambda Labs Orchestrator - Docker Image
# =============================================================================
# Runs the orchestrator services for managing Lambda Labs training jobs
#
# Build: docker build -t lambda-orchestrator .
# Run:   docker run -p 8080:8080 -e REDIS_HOST=redis -e LAMBDA_API_KEY=xxx lambda-orchestrator
# =============================================================================

FROM python:3.11-slim

# Labels
LABEL org.opencontainers.image.title="Lambda Labs Orchestrator"
LABEL org.opencontainers.image.description="Event-driven architecture for Lambda Labs GPU training jobs"
LABEL org.opencontainers.image.version="0.2.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    ca-certificates \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK (for gcloud commands in scripts)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the module directory structure
RUN mkdir -p /app/sfcompute_orchestrator

# Copy application code into the module
COPY __init__.py /app/sfcompute_orchestrator/
COPY __main__.py /app/sfcompute_orchestrator/
COPY config.py /app/sfcompute_orchestrator/
COPY models /app/sfcompute_orchestrator/models/
COPY services /app/sfcompute_orchestrator/services/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8080
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the orchestrator
CMD ["python", "-m", "sfcompute_orchestrator"]
