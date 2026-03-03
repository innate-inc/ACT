# Minimal base — let cloud_run.sh handle Python env setup at runtime
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

WORKDIR /app

ENV PYTHONPATH="/app/act_test:/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy repo
COPY act_test/ ./act_test/
COPY requirements.txt setup.py manifest.in ./
COPY cloud_run.sh ./
RUN chmod +x cloud_run.sh

# cloud_run.sh sets up venv + installs PyTorch (nightly cu128 for Blackwell) at runtime
ENTRYPOINT ["/app/cloud_run.sh"]
