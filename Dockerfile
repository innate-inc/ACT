# Use NVIDIA's PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH="/app/act_test:/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Install system dependencies including gsutil and RAID tools
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    gnupg \
    apt-transport-https \
    ca-certificates \
    sudo \
    mdadm \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

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

# Copy scripts
COPY download_data.sh /app/
COPY setup_vertex_raid.sh /app/
RUN chmod +x /app/download_data.sh /app/setup_vertex_raid.sh

# Create a non-root user with a home directory for security
RUN groupadd -r appuser && useradd --no-log-init -r -m -g appuser appuser

# Configure sudo for appuser to run RAID commands without password
RUN echo "appuser ALL=(ALL) NOPASSWD: /sbin/mdadm, /sbin/mkfs.ext4, /bin/mount, /bin/mkdir, /bin/chown, /bin/chmod" >> /etc/sudoers.d/appuser \
    && chmod 0440 /etc/sudoers.d/appuser

RUN chown -R appuser:appuser /app
USER appuser

# Default entrypoint - download data first, then train
ENTRYPOINT ["/app/download_data.sh"] 