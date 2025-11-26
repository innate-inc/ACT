# Use NVIDIA's PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH="/app/act_test:/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Install system dependencies including gsutil
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    gnupg \
    apt-transport-https \
    ca-certificates \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# Use docker-specific requirements (excludes packages already in base image)
COPY requirements.docker.txt ./
COPY setup.py ./
COPY act_test/ ./act_test/

# Install all dependencies in a single layer to minimize image size
RUN pip install --no-cache-dir -r requirements.docker.txt && \
    pip install --no-cache-dir -e . && \
    rm -rf ~/.cache/pip

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/checkpoints

# Create a data download script
COPY download_data.sh /app/
RUN chmod +x /app/download_data.sh

# Create a non-root user with a home directory for security
RUN groupadd -r appuser && useradd --no-log-init -r -m -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Default entrypoint - download data first, then train
ENTRYPOINT ["/app/download_data.sh"] 