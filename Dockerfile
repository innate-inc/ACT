# Use NVIDIA's PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Install system dependencies including gsutil
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK for gsutil
RUN curl https://sdk.cloud.google.com | bash
ENV PATH=$PATH:/root/google-cloud-sdk/bin

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package
COPY act_test/ ./act_test/
COPY setup.py ./

# Install the package in development mode
RUN pip install -e .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/checkpoints

# Create a data download script
COPY download_data.sh /app/
RUN chmod +x /app/download_data.sh

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Default entrypoint - download data first, then train
ENTRYPOINT ["/app/download_data.sh"] 