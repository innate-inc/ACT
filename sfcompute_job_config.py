#!/usr/bin/env python3
"""
SFCompute Job Configuration and Management

This module provides configuration and utilities for managing ACT training jobs
on SFCompute infrastructure. Unlike Vertex AI which uses Python SDK for job submission,
SFCompute uses CLI commands, so this module generates configs and helper scripts.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List


@dataclass
class SFComputeJobConfig:
    """Configuration for an SFCompute training job."""
    
    # Node configuration
    node_name: str = "act-training"
    zone: str = "landsend"
    max_price: float = 25.00  # Maximum price per node hour
    duration: str = "12h"  # Duration for reserved nodes
    
    # Node type: "reserved" or "auto"
    node_type: str = "reserved"
    
    # Number of nodes (for multi-node training)
    num_nodes: int = 1
    
    # Data configuration
    data_gcs_path: str = ""  # GCS path to training data (gs://bucket/path)
    output_gcs_path: str = ""  # GCS path for outputs (gs://bucket/path)
    
    # Training configuration
    max_steps: int = 120000
    learning_rate: float = 5e-5
    learning_rate_backbone: float = 5e-5
    chunk_size: int = 30
    batch_size: int = 96
    world_size: int = 4  # Number of GPUs per node
    
    # Docker configuration
    docker_image: str = ""  # Docker Hub image (e.g., username/act-training:latest)
    use_docker: bool = True  # Whether to use containerized training
    
    # Cloud-init configuration
    cloud_init_file: str = "cloud-init.yaml"
    
    # GCS service account key (optional - for automated auth)
    gcs_key_file: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "node_name": self.node_name,
            "zone": self.zone,
            "max_price": self.max_price,
            "duration": self.duration,
            "node_type": self.node_type,
            "num_nodes": self.num_nodes,
            "data_gcs_path": self.data_gcs_path,
            "output_gcs_path": self.output_gcs_path,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "learning_rate_backbone": self.learning_rate_backbone,
            "chunk_size": self.chunk_size,
            "batch_size": self.batch_size,
            "world_size": self.world_size,
            "docker_image": self.docker_image,
            "use_docker": self.use_docker,
            "cloud_init_file": self.cloud_init_file,
        }
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Configuration saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SFComputeJobConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def generate_training_cloud_init(config: SFComputeJobConfig, output_path: str = "cloud-init-training.yaml"):
    """
    Generate a cloud-init file specifically configured for this training job.
    This embeds the training configuration directly into the cloud-init.
    """
    
    cloud_init_content = f'''#cloud-config
# Auto-generated cloud-init for ACT training job
# Generated: {datetime.now().isoformat()}

disable_root: false
ssh_pwauth: false

users:
  - name: root
    ssh_authorized_keys:
      # SSH keys will be added from your local machine by deploy script

package_update: true
package_upgrade: true

packages:
  - git
  - wget
  - curl
  - htop
  - nvtop
  - tmux
  - unzip
  - build-essential
  - python3-pip
  - python3-dev
  - apt-transport-https
  - ca-certificates
  - gnupg

write_files:
  - path: /data/job_config.json
    permissions: '0644'
    content: |
      {json.dumps(config.to_dict(), indent=6).replace(chr(10), chr(10) + "      ")}
  
  - path: /data/run_training.sh
    permissions: '0755'
    content: |
      #!/bin/bash
      set -e
      
      echo "🚀 Starting ACT Training Job"
      echo "============================"
      
      # Load environment
      source /etc/profile.d/act_training.sh
      
      # Configuration
      DATA_GCS_PATH="{config.data_gcs_path}"
      OUTPUT_GCS_PATH="{config.output_gcs_path}"
      LOCAL_DATA_DIR="/data/dataset"
      
      # Download data from GCS
      echo "📥 Downloading data from $DATA_GCS_PATH..."
      mkdir -p $LOCAL_DATA_DIR
      gcloud storage cp -r "$DATA_GCS_PATH/*" "$LOCAL_DATA_DIR/"
      
      echo "✅ Data downloaded successfully"
      ls -la $LOCAL_DATA_DIR
      
      # Run training
      cd /data/ACT-test
      python3 -m act_test.train_dist \\
        --data_dir "$LOCAL_DATA_DIR" \\
        --chunk_size {config.chunk_size} \\
        --max_steps {config.max_steps} \\
        --learning_rate {config.learning_rate} \\
        --learning_rate_backbone {config.learning_rate_backbone} \\
        --batch_size {config.batch_size} \\
        --world_size {config.world_size}
      
      # Upload checkpoints
      echo "📤 Uploading checkpoints to $OUTPUT_GCS_PATH..."
      CHECKPOINT_DIR=$(find $LOCAL_DATA_DIR/checkpoints -maxdepth 1 -type d -name "*_ddp" | head -1)
      if [ -n "$CHECKPOINT_DIR" ]; then
        gcloud storage cp -r "$CHECKPOINT_DIR" "$OUTPUT_GCS_PATH/"
        echo "✅ Checkpoints uploaded successfully"
      fi
      
      echo "🎉 Training complete!"
  
  - path: /etc/profile.d/act_training.sh
    permissions: '0644'
    content: |
      export DATA_DIR="/data/dataset"
      export OUTPUT_DIR="/data/outputs"
      export CHECKPOINT_DIR="/data/checkpoints"
      export PYTHONUNBUFFERED=1
      export WANDB_MODE="online"
      export NCCL_DEBUG="INFO"

runcmd:
  # Create directories
  - mkdir -p /data/dataset /data/outputs /data/checkpoints
  - chmod -R 777 /data
  
  # Install Google Cloud SDK
  - curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  - echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
  - apt-get update && apt-get install -y google-cloud-cli
  
  # Install NVIDIA Container Toolkit
  - curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  - curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  - apt-get update && apt-get install -y nvidia-container-toolkit docker.io
  - nvidia-ctk runtime configure --runtime=docker
  - systemctl restart docker
  
  # Install Python dependencies
  - pip3 install --upgrade pip
  - pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  - pip3 install numpy einops opencv-python pillow huggingface-hub timm safetensors h5py webdataset PyYAML matplotlib tqdm requests click pydantic wandb
  
  # Clone ACT repository
  - cd /data && git clone https://github.com/yourusername/ACT-test.git || echo "Clone failed or repo exists"
  
  # Print setup complete
  - echo "Node setup complete! GPU info:"
  - nvidia-smi

final_message: |
  SFCompute node ready for training!
  Run: /data/run_training.sh
'''
    
    with open(output_path, 'w') as f:
        f.write(cloud_init_content)
    
    print(f"✅ Generated cloud-init file: {output_path}")
    return output_path


def generate_sf_commands(config: SFComputeJobConfig) -> List[str]:
    """Generate sf CLI commands for node creation and management."""
    
    commands = []
    
    # Generate node creation command
    if config.node_type == "reserved":
        create_cmd = (
            f"sf nodes create {config.node_name} "
            f"--zone {config.zone} "
            f"--duration {config.duration} "
            f"--max-price {config.max_price:.2f} "
            f"--user-data-file ./cloud-init-training.yaml"
        )
    else:  # auto reserved
        create_cmd = (
            f"sf nodes create {config.node_name} "
            f"--auto "
            f"--zone {config.zone} "
            f"--max-price {config.max_price:.2f} "
            f"--user-data-file ./cloud-init-training.yaml"
        )
    
    if config.num_nodes > 1:
        create_cmd = create_cmd.replace(
            f"sf nodes create {config.node_name}",
            f"sf nodes create -n {config.num_nodes}"
        )
    
    commands.append(("Create node(s)", create_cmd))
    commands.append(("Check node status", "sf nodes list --verbose"))
    commands.append(("SSH into node", f"sf nodes ssh root@{config.node_name}"))
    commands.append(("View logs", f"sf nodes logs {config.node_name}"))
    
    if config.node_type == "reserved":
        commands.append(("Extend node time", f"sf nodes extend {config.node_name} --duration 2h --max-price {config.max_price:.2f}"))
    else:
        commands.append(("Release node", f"sf nodes release {config.node_name}"))
    
    commands.append(("Delete node", f"sf nodes delete {config.node_name}"))
    
    return commands


def print_deployment_instructions(config: SFComputeJobConfig, cloud_init_path: str):
    """Print step-by-step deployment instructions."""
    
    commands = generate_sf_commands(config)
    
    print("\n" + "=" * 70)
    print("📋 SFCompute Deployment Instructions")
    print("=" * 70)
    
    print("\n📝 Pre-requisites:")
    print("   1. Install sf CLI: curl -fsSL https://sfcompute.com/cli/install | bash")
    print("   2. Login: sf login")
    print("   3. Ensure your SSH public key is in the cloud-init file")
    
    print("\n📁 Generated Files:")
    print(f"   - Cloud-init: {cloud_init_path}")
    print(f"   - Job config: job_config.json")
    
    print("\n🚀 Deployment Commands:")
    for i, (desc, cmd) in enumerate(commands, 1):
        print(f"\n   {i}. {desc}:")
        print(f"      $ {cmd}")
    
    print("\n📊 Job Configuration:")
    print(f"   - Node type: {config.node_type}")
    print(f"   - Zone: {config.zone}")
    print(f"   - Max price: ${config.max_price:.2f}/hour")
    print(f"   - Duration: {config.duration}")
    print(f"   - GPUs per node: {config.world_size}")
    print(f"   - Data source: {config.data_gcs_path}")
    print(f"   - Output dest: {config.output_gcs_path}")
    
    print("\n💡 After SSH into node:")
    print("   1. Authenticate GCS: gcloud auth login")
    print("   2. Run training: /data/run_training.sh")
    print("   3. Or run manually:")
    print(f"      $ gcloud storage cp -r '{config.data_gcs_path}/*' /data/dataset/")
    print(f"      $ cd /data/ACT-test && python3 -m act_test.train_dist --data_dir /data/dataset")
    
    print("\n" + "=" * 70)


def create_training_job(
    data_gcs_path: str,
    output_gcs_path: str,
    node_name: Optional[str] = None,
    zone: str = "landsend",
    max_price: float = 25.00,
    duration: str = "12h",
    node_type: str = "reserved",
    max_steps: int = 120000,
    learning_rate: float = 5e-5,
    chunk_size: int = 30,
    dry_run: bool = True,
) -> SFComputeJobConfig:
    """
    Create and configure an SFCompute training job.
    
    Args:
        data_gcs_path: GCS path to training data (gs://bucket/path)
        output_gcs_path: GCS path for outputs
        node_name: Name for the node (auto-generated if not provided)
        zone: SFCompute zone (default: landsend)
        max_price: Maximum price per node hour
        duration: Duration for reserved nodes (e.g., "12h", "24h")
        node_type: "reserved" or "auto"
        max_steps: Maximum training steps
        learning_rate: Learning rate
        chunk_size: Action sequence length
        dry_run: If True, only generate configs without creating node
    
    Returns:
        SFComputeJobConfig with the job configuration
    """
    
    # Auto-generate node name if not provided
    if node_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        node_name = f"act-train-{timestamp}"
    
    # Create configuration
    config = SFComputeJobConfig(
        node_name=node_name,
        zone=zone,
        max_price=max_price,
        duration=duration,
        node_type=node_type,
        data_gcs_path=data_gcs_path,
        output_gcs_path=output_gcs_path,
        max_steps=max_steps,
        learning_rate=learning_rate,
        learning_rate_backbone=learning_rate,
        chunk_size=chunk_size,
    )
    
    # Generate cloud-init file
    cloud_init_path = generate_training_cloud_init(config)
    
    # Save job configuration
    config.save("job_config.json")
    
    # Print deployment instructions
    print_deployment_instructions(config, cloud_init_path)
    
    if not dry_run:
        print("\n🚀 Creating SFCompute node...")
        commands = generate_sf_commands(config)
        create_cmd = commands[0][1]  # First command is create
        
        try:
            result = subprocess.run(
                create_cmd.split(),
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Node created successfully!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create node: {e.stderr}")
            raise
        except FileNotFoundError:
            print("❌ sf CLI not found. Please install it first:")
            print("   curl -fsSL https://sfcompute.com/cli/install | bash")
            raise
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Configure and deploy ACT training jobs on SFCompute',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate configuration only (dry run)
  python sfcompute_job_config.py \\
    --data_path gs://my-bucket/training-data \\
    --output_path gs://my-bucket/outputs
  
  # Create node and deploy
  python sfcompute_job_config.py \\
    --data_path gs://my-bucket/training-data \\
    --output_path gs://my-bucket/outputs \\
    --no-dry-run
  
  # Custom configuration
  python sfcompute_job_config.py \\
    --data_path gs://my-bucket/data \\
    --output_path gs://my-bucket/outputs \\
    --node_name my-training-job \\
    --zone landsend \\
    --duration 24h \\
    --max_price 30.00 \\
    --max_steps 200000
        '''
    )
    
    # Required arguments
    parser.add_argument("--data_path", required=True,
                        help="GCS path to training data (e.g., gs://bucket/data)")
    parser.add_argument("--output_path", required=True,
                        help="GCS path for outputs (e.g., gs://bucket/outputs)")
    
    # Node configuration
    parser.add_argument("--node_name", default=None,
                        help="Name for the node (auto-generated if not provided)")
    parser.add_argument("--zone", default="landsend",
                        help="SFCompute zone (default: landsend)")
    parser.add_argument("--max_price", type=float, default=25.00,
                        help="Maximum price per node hour (default: 25.00)")
    parser.add_argument("--duration", default="12h",
                        help="Duration for reserved nodes (default: 12h)")
    parser.add_argument("--node_type", choices=["reserved", "auto"], default="reserved",
                        help="Node type: reserved or auto (default: reserved)")
    
    # Training configuration
    parser.add_argument("--max_steps", type=int, default=120000,
                        help="Maximum training steps (default: 120000)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--chunk_size", type=int, default=30,
                        help="Action sequence length (default: 30)")
    
    # Deployment options
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                        help="Only generate configs, don't create node (default)")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                        help="Actually create the node")
    
    args = parser.parse_args()
    
    try:
        config = create_training_job(
            data_gcs_path=args.data_path,
            output_gcs_path=args.output_path,
            node_name=args.node_name,
            zone=args.zone,
            max_price=args.max_price,
            duration=args.duration,
            node_type=args.node_type,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            chunk_size=args.chunk_size,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

