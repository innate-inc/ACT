#!/usr/bin/env python3
"""
SFCompute Kubernetes Configuration Generator

Generates Kubernetes Job manifests for ACT training on SFCompute.

Usage:
    python sfcompute_k8s_config.py --data-path gs://bucket/data --output-path gs://bucket/ckpts
    python sfcompute_k8s_config.py --ssh  # Generate SSH pod for development
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml


def generate_training_job(
    job_name: str,
    image_uri: str,
    data_gcs_path: str,
    output_gcs_path: str,
    max_steps: int = 120000,
    learning_rate: float = 5e-5,
    chunk_size: int = 30,
    batch_size: int = 96,
    num_gpus: int = 8,
    num_nodes: int = 1,
    wandb_api_key: Optional[str] = None,
) -> dict:
    """Generate a Kubernetes Job manifest for ACT training."""
    
    # Training command
    if num_nodes == 1:
        # Single-node: use standalone mode
        train_command = f"""set -e
echo "🚀 ACT Training Job: {job_name}"
echo "========================================"

# GPU check
nvidia-smi --query-gpu=name,memory.total --format=csv

# Download data from GCS
echo "📥 Downloading data from GCS.n.."
mkdir -p /data/dataset
gcloud storage cp -r "{data_gcs_path}/*" /data/dataset/
echo "✅ Downloaded: $(du -sh /data/dataset/)"
ls -la /data/dataset/

# Training
echo "🏋️ Starting distributed training..."
cd /app
torchrun --standalone --nproc_per_node={num_gpus} \\
    -m act_test.train_dist \\
    --data_dir /data/dataset \\
    --chunk_size {chunk_size} \\
    --max_steps {max_steps} \\
    --learning_rate {learning_rate} \\
    --learning_rate_backbone {learning_rate} \\
    --batch_size {batch_size} \\
    --world_size {num_gpus}

TRAIN_EXIT=$?

# Upload checkpoints
echo "📤 Uploading checkpoints..."
CKPT=$(find /data/dataset/checkpoints -maxdepth 1 -type d -name "*_ddp" | head -1)
if [ -n "$CKPT" ]; then
    gcloud storage cp -r "$CKPT" "{output_gcs_path}/"
    echo "✅ Uploaded to: {output_gcs_path}/$(basename $CKPT)"
fi

echo "🎉 Training complete! Exit code: $TRAIN_EXIT"
exit $TRAIN_EXIT
"""
    else:
        # Multi-node: use rendezvous
        train_command = f"""set -e
echo "🚀 ACT Multi-Node Training: {job_name}"
echo "========================================"

# GPU check
nvidia-smi --query-gpu=name,memory.total --format=csv

# Download data from GCS
echo "📥 Downloading data from GCS..."
mkdir -p /data/dataset
gcloud storage cp -r "{data_gcs_path}/*" /data/dataset/
echo "✅ Downloaded: $(du -sh /data/dataset/)"

# Training with multi-node rendezvous
echo "🏋️ Starting multi-node distributed training..."
cd /app
torchrun --nnodes={num_nodes} --nproc_per_node={num_gpus} \\
    --rdzv-backend=c10d --rdzv-endpoint={job_name}-0.{job_name}-svc:29500 \\
    -m act_test.train_dist \\
    --data_dir /data/dataset \\
    --chunk_size {chunk_size} \\
    --max_steps {max_steps} \\
    --learning_rate {learning_rate} \\
    --learning_rate_backbone {learning_rate} \\
    --batch_size {batch_size} \\
    --world_size {num_gpus * num_nodes}

TRAIN_EXIT=$?

# Upload checkpoints (only from rank 0)
if [ "$JOB_COMPLETION_INDEX" = "0" ]; then
    echo "📤 Uploading checkpoints..."
    CKPT=$(find /data/dataset/checkpoints -maxdepth 1 -type d -name "*_ddp" | head -1)
    if [ -n "$CKPT" ]; then
        gcloud storage cp -r "$CKPT" "{output_gcs_path}/"
        echo "✅ Uploaded to: {output_gcs_path}/$(basename $CKPT)"
    fi
fi

echo "🎉 Training complete! Exit code: $TRAIN_EXIT"
exit $TRAIN_EXIT
"""
    
    # Environment variables
    env_vars = [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "WANDB_MODE", "value": "online"},
        {"name": "NCCL_DEBUG", "value": "INFO"},
        {"name": "DATA_GCS_PATH", "value": data_gcs_path},
        {"name": "OUTPUT_GCS_PATH", "value": output_gcs_path},
    ]
    
    if wandb_api_key:
        env_vars.append({"name": "WANDB_API_KEY", "value": wandb_api_key})
    
    # Job spec
    job_spec = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "labels": {"app": "act-training"},
        },
        "spec": {
            "completions": num_nodes,
            "parallelism": num_nodes,
            "completionMode": "Indexed",
            "backoffLimit": 2,
            "template": {
                "metadata": {
                    "labels": {"job-name": job_name, "app": "act-training"},
                },
                "spec": {
                    "containers": [{
                        "name": "trainer",
                        "image": image_uri,
                        "imagePullPolicy": "Always",
                        "command": ["/bin/bash", "-c", train_command],
                        "ports": [{"containerPort": 29500, "name": "dist-port"}],
                        "env": env_vars,
                        "resources": {
                            "requests": {
                                "nvidia.com/gpu": num_gpus,
                                "nvidia.com/hostdev": num_gpus,
                                "memory": "512Gi",
                                "cpu": "32",
                            },
                            "limits": {
                                "nvidia.com/gpu": num_gpus,
                                "nvidia.com/hostdev": num_gpus,
                                "memory": "512Gi",
                                "cpu": "32",
                            },
                        },
                        "volumeMounts": [
                            {"name": "data-volume", "mountPath": "/data"},
                            {"name": "shm", "mountPath": "/dev/shm"},
                        ],
                    }],
                    "volumes": [
                        {"name": "data-volume", "emptyDir": {}},
                        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"}},
                    ],
                    "restartPolicy": "Never",
                },
            },
        },
    }
    
    # Add subdomain for multi-node
    if num_nodes > 1:
        job_spec["spec"]["template"]["spec"]["subdomain"] = f"{job_name}-svc"
    
    return job_spec


def generate_service(job_name: str) -> dict:
    """Generate a headless Service for multi-node training."""
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": f"{job_name}-svc"},
        "spec": {
            "clusterIP": "None",
            "selector": {"job-name": job_name},
            "ports": [{"port": 29500, "name": "dist-port"}],
        },
    }


def generate_ssh_pod(
    pod_name: str,
    image_uri: str,
    data_gcs_path: str,
    num_gpus: int = 8,
) -> dict:
    """Generate an SSH-enabled Pod for development."""
    
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "labels": {"app": "act-training"},
        },
        "spec": {
            "containers": [{
                "name": "cuda",
                "image": image_uri,
                "imagePullPolicy": "Always",
                "command": ["/bin/bash", "-c", """
apt-get update && apt-get install -y openssh-server
passwd -d root
echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
echo 'PermitEmptyPasswords yes' >> /etc/ssh/sshd_config
mkdir -p /var/run/sshd
echo "========================================"
echo "SSH Pod Ready!"
echo "  kubectl port-forward pod/""" + pod_name + """ 2222:22"
echo "  ssh -p 2222 root@localhost"
echo "========================================"
nvidia-smi
/usr/sbin/sshd -D
"""],
                "ports": [
                    {"containerPort": 22, "name": "ssh"},
                    {"containerPort": 29500, "name": "dist-port"},
                ],
                "env": [
                    {"name": "DATA_GCS_PATH", "value": data_gcs_path},
                ],
                "resources": {
                    "requests": {
                        "nvidia.com/gpu": num_gpus,
                        "nvidia.com/hostdev": num_gpus,
                        "memory": "512Gi",
                        "cpu": "32",
                    },
                    "limits": {
                        "nvidia.com/gpu": num_gpus,
                        "nvidia.com/hostdev": num_gpus,
                        "memory": "512Gi",
                        "cpu": "32",
                    },
                },
                "volumeMounts": [
                    {"name": "data-volume", "mountPath": "/data"},
                    {"name": "shm", "mountPath": "/dev/shm"},
                ],
            }],
            "volumes": [
                {"name": "data-volume", "emptyDir": {}},
                {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"}},
            ],
            "restartPolicy": "Never",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate Kubernetes manifests for ACT training on SFCompute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training job manifest
  python sfcompute_k8s_config.py \\
    --data-path gs://maurice-prod-data/data/socks1wed_socks2wed_filt_merged \\
    --output-path gs://maurice-prod-data/ckpts
  
  # Generate SSH pod for development
  python sfcompute_k8s_config.py --ssh
  
  # Multi-node training (2 nodes, 16 GPUs total)
  python sfcompute_k8s_config.py --num-nodes 2 \\
    --data-path gs://bucket/data \\
    --output-path gs://bucket/ckpts
        """
    )
    
    # Required for training job
    parser.add_argument("--data-path", 
                        default="gs://maurice-prod-data/data/socks1wed_socks2wed_filt_merged",
                        help="GCS path to training data")
    parser.add_argument("--output-path", 
                        default="gs://maurice-prod-data/ckpts",
                        help="GCS path for checkpoints")
    
    # Docker image
    parser.add_argument("--image", default="yourusername/act-training:latest",
                        help="Docker image URI")
    
    # Training config
    parser.add_argument("--max-steps", type=int, default=120000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-gpus", type=int, default=8, help="GPUs per node")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    
    # W&B
    parser.add_argument("--wandb-key", help="Weights & Biases API key")
    
    # Output
    parser.add_argument("--output-dir", default="k8s", help="Output directory for manifests")
    parser.add_argument("--job-name", help="Job name (auto-generated if not provided)")
    
    # Mode
    parser.add_argument("--ssh", action="store_true", help="Generate SSH pod instead of training job")
    parser.add_argument("--apply", action="store_true", help="Apply manifest with kubectl")
    
    args = parser.parse_args()
    
    # Generate job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.ssh:
        job_name = args.job_name or f"act-ssh-{timestamp}"
    else:
        job_name = args.job_name or f"act-training-{timestamp}"
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate manifest
    if args.ssh:
        manifest = generate_ssh_pod(
            pod_name=job_name,
            image_uri=args.image,
            data_gcs_path=args.data_path,
            num_gpus=args.num_gpus,
        )
        output_file = output_dir / f"{job_name}.yaml"
    else:
        manifests = []
        
        # Add service for multi-node
        if args.num_nodes > 1:
            manifests.append(generate_service(job_name))
        
        # Add job
        manifests.append(generate_training_job(
            job_name=job_name,
            image_uri=args.image,
            data_gcs_path=args.data_path,
            output_gcs_path=args.output_path,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus,
            num_nodes=args.num_nodes,
            wandb_api_key=args.wandb_key,
        ))
        
        manifest = manifests
        output_file = output_dir / f"{job_name}.yaml"
    
    # Write manifest
    with open(output_file, 'w') as f:
        if isinstance(manifest, list):
            yaml.dump_all(manifest, f, default_flow_style=False)
        else:
            yaml.dump(manifest, f, default_flow_style=False)
    
    print(f"✅ Generated: {output_file}")
    
    # Print summary
    print("\n" + "=" * 50)
    if args.ssh:
        print(f"📋 SSH Pod: {job_name}")
        print("\nTo deploy:")
        print(f"  kubectl apply -f {output_file}")
        print("\nTo connect:")
        print(f"  kubectl port-forward pod/{job_name} 2222:22")
        print("  ssh -p 2222 root@localhost")
    else:
        print(f"📋 Training Job: {job_name}")
        print(f"   Nodes: {args.num_nodes}")
        print(f"   GPUs:  {args.num_gpus * args.num_nodes}")
        print(f"   Data:  {args.data_path}")
        print("\nTo deploy:")
        print(f"  kubectl apply -f {output_file}")
        print("\nTo monitor:")
        print(f"  kubectl get pods -w")
        print(f"  kubectl logs -f {job_name}-0")
    print("=" * 50)
    
    # Apply if requested
    if args.apply:
        import subprocess
        print("\n🚀 Applying manifest...")
        result = subprocess.run(["kubectl", "apply", "-f", str(output_file)], 
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ Error: {result.stderr}")
            sys.exit(1)


if __name__ == "__main__":
    main()

