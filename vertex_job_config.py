#!/usr/bin/env python3

import argparse
from google.cloud import aiplatform
from typing import Optional

def create_training_job(
    project_id: str,
    region: str,
    image_uri: str,
    data_path: str,
    output_path: str,
    job_name: Optional[str] = None
):
    """Create and submit a Vertex AI training job."""
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Define the custom training job
    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name or "act-distributed-training",
        container_uri=image_uri,
        # The container will handle data download and training
        # Pass training arguments after the download script
        args=[
            "--max_steps", "15000",
            "--learning_rate", "5e-5",
            "--chunk_size", "30"
        ],
        # Machine configuration - larger disk for local data storage
        machine_type="n1-standard-8",  # More CPU for data download
        accelerator_type="NVIDIA_TESLA_V100",
        accelerator_count=2,
        # Larger disk to store all data locally
        boot_disk_type="pd-ssd",
        boot_disk_size_gb=500,  # Increased for local data storage
    )
    
    # Run the training job
    job.run(
        # This is where outputs (checkpoints, logs) will be saved
        base_output_dir=output_path,
        # Environment variables for the container
        environment_variables={
            "WANDB_MODE": "online",
            "NCCL_DEBUG": "INFO",
            "DATA_BUCKET": data_path,
            "OUTPUT_BUCKET": output_path,
            # Ensure we have enough disk space
            "TMPDIR": "/app/tmp"
        },
        # Restart policy
        restart_job_on_worker_restart=True,
        # Longer timeout for data download + training
        timeout=3600 * 48,  # 48 hours
        # Enable early stopping
        enable_web_access=True,
    )
    
    print(f"Training job started: {job.resource_name}")
    print(f"Data will be downloaded from: {data_path}")
    print(f"Outputs will be synced to: {output_path}")
    return job

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--region", default="us-east5")
    parser.add_argument("--image_uri", required=True)
    parser.add_argument("--data_path", required=True, help="GCS path to data (e.g., gs://your-bucket/data)")
    parser.add_argument("--output_path", required=True, help="GCS path for outputs (e.g., gs://your-bucket/outputs)")
    parser.add_argument("--job_name", help="Job name")
    
    args = parser.parse_args()
    
    job = create_training_job(
        project_id=args.project_id,
        region=args.region,
        image_uri=args.image_uri,
        data_path=args.data_path,
        output_path=args.output_path,
        job_name=args.job_name
    ) 