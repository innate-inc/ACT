"""
Job API Service
HTTP API for submitting and monitoring training jobs
"""
import functools
import hmac
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

import requests
from flask import Flask, request, jsonify, redirect

from ..config import APIConfig, DiscordConfig
from ..models.job import TrainingJob, JobStatus, PriceData, ApprovalStatus
from .cache import CacheService
from .lambda_client import LambdaLabsClient, LambdaAPIError
from .ssh_executor import SSHExecutor

logger = logging.getLogger(__name__)

# Regex to validate request IDs (UUID-like format, alphanumeric with hyphens, max 64 chars)
REQUEST_ID_PATTERN = re.compile(r'^[a-zA-Z0-9\-]{1,64}$')

# API key for application-level authentication (when Cloud Run allows unauthenticated)
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")


def create_app(
    cache: CacheService, 
    api_config: APIConfig,
    discord_config: Optional[DiscordConfig] = None
) -> Flask:
    """Create Flask application for Job API"""
    
    app = Flask(__name__)
    app.config["cache"] = cache
    app.config["api_config"] = api_config
    app.config["discord_config"] = discord_config
    
    def require_auth(f):
        """
        Decorator to require authentication for protected endpoints.
        Accepts either:
        1. GCP Identity Token in Authorization: Bearer header (validated via Google)
        2. API secret key in X-API-Key header (for programmatic access)
        """
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Check for API key first (simpler, for programmatic access)
            api_key = request.headers.get("X-API-Key", "")
            if API_SECRET_KEY and api_key:
                if hmac.compare_digest(api_key, API_SECRET_KEY):
                    logger.debug("Authenticated via API key")
                    return f(*args, **kwargs)
            
            # Check for GCP Identity Token
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                logger.debug(f"Validating Bearer token (length={len(token)})")
                # Validate token with Google
                if _validate_gcp_token(token):
                    logger.debug("Authenticated via GCP identity token")
                    return f(*args, **kwargs)
                else:
                    logger.warning("GCP token validation failed")
            else:
                logger.debug(f"No valid auth header found (got: {auth_header[:20]}...)")
            
            # No valid auth found
            return jsonify({
                "error": "Authentication required",
                "hint": "Use 'Authorization: Bearer <gcloud-identity-token>' or 'X-API-Key: <api-key>'"
            }), 401
        
        return decorated_function
    
    def _validate_gcp_token(token: str) -> bool:
        """Validate GCP identity token by calling Google's tokeninfo endpoint"""
        if not token or len(token) < 20:
            return False
        
        try:
            # Validate using Google's tokeninfo endpoint
            response = requests.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={token}",
                timeout=10
            )
            
            if response.status_code == 200:
                token_info = response.json()
                # Check if token is not expired and has valid issuer
                if token_info.get("iss") in ["https://accounts.google.com", "accounts.google.com"]:
                    logger.debug(f"Token validated for: {token_info.get('email', 'unknown')}")
                    return True
                logger.warning(f"Token has invalid issuer: {token_info.get('iss')}")
                return False
            
            # Log the error for debugging
            logger.warning(f"Token validation failed with status {response.status_code}: {response.text[:200]}")
            return False
            
        except requests.exceptions.Timeout:
            logger.warning("Token validation timed out - allowing request")
            # On timeout, allow the request (fail-open for availability)
            return True
        except Exception as e:
            logger.warning(f"Token validation error: {e}")
            return False
    
    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint"""
        redis_ok = cache.ping()
        lambda_key = bool(os.environ.get("LAMBDA_API_KEY"))
        return jsonify({
            "status": "healthy" if redis_ok else "degraded",
            "redis": "connected" if redis_ok else "disconnected",
            "lambda_api_key": "configured" if lambda_key else "not set",
            "provider": "lambda_labs",
            "timestamp": datetime.utcnow().isoformat()
        }), 200 if redis_ok else 503
    
    @app.route("/jobs/running", methods=["GET"])
    @require_auth
    def get_running_jobs():
        """
        Get all currently running jobs.
        
        These are jobs that have been assigned to a VM and are actively training.
        Multiple jobs can run concurrently on different VMs.
        """
        try:
            all_jobs = cache.get_all_jobs()
            running_jobs = [
                j for j in all_jobs 
                if j.status in [
                    JobStatus.RUNNING, 
                    JobStatus.PROVISIONING, 
                    JobStatus.BUYING,
                    JobStatus.AWAITING_APPROVAL
                ]
            ]
            
            return jsonify({
                "count": len(running_jobs),
                "jobs": [
                    {
                        "job_id": j.job_id,
                        "status": j.status.value,
                        "data_gcs_path": j.data_gcs_path,
                        "instance_id": j.instance_id,
                        "instance_ip": j.instance_ip,
                        "instance_type": j.instance_type,
                        "region": j.region,
                        "started_at": j.started_at
                    }
                    for j in running_jobs
                ]
            })
        except Exception as e:
            logger.error(f"Error getting running jobs: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/jobs", methods=["POST"])
    @require_auth
    def submit_job():
        """
        Submit a new training job.
        
        Request body:
        {
            "job_id": "custom-id-123",                      # Optional - custom ID for tracking
            "data_gcs_path": "gs://bucket/path/to/data",   # Required
            "output_gcs_path": "gs://bucket/outputs",       # Optional
            "callback_url": "https://...",                  # Optional - POST on completion/failure
            "batch_size": 96,                               # Optional
            "max_steps": 120000,                            # Optional
            "chunk_size": 30,                               # Optional
            "learning_rate": "5e-5",                        # Optional
            "num_workers": 4,                               # Optional
            "min_gpus": 1,                                  # Optional (Lambda Labs)
            "max_gpus": 8,                                  # Optional (Lambda Labs)
            "max_duration_hours": 24,                       # Optional - for cost estimation
            "max_total_cost": 500.0,                        # Optional - total budget cap
            "max_spend": 5.00                               # Optional - max $/GPU/hr willing to pay
        }
        """
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({"error": "Request body required"}), 400
            
            if "data_gcs_path" not in data:
                return jsonify({"error": "data_gcs_path is required"}), 400
            
            # Create job from request
            # Allow custom job_id for cross-service tracking, or auto-generate
            custom_job_id = data.get("job_id")
            job = TrainingJob(
                job_id=custom_job_id if custom_job_id else str(uuid.uuid4())[:8],
                data_gcs_path=data["data_gcs_path"],
                output_gcs_path=data.get("output_gcs_path"),
                batch_size=data.get("batch_size"),
                max_steps=data.get("max_steps"),
                chunk_size=data.get("chunk_size"),
                learning_rate=data.get("learning_rate"),
                num_workers=data.get("num_workers"),
                min_gpus=data.get("min_gpus", 1),
                max_gpus=data.get("max_gpus", 8),
                max_duration_hours=data.get("max_duration_hours", 24),
                max_total_cost=data.get("max_total_cost"),
                max_spend=data.get("max_spend"),
                callback_url=data.get("callback_url"),
            )
            
            # Enqueue the job
            cache.enqueue_job(job)
            
            logger.info(f"Job {job.job_id} submitted: {job.data_gcs_path} (max_spend: ${job.max_spend}/GPU/hr)")
            
            return jsonify({
                "job_id": job.job_id,
                "status": job.status.value,
                "max_spend": job.max_spend,
                "message": "Job submitted successfully",
                "queue_position": cache.queue_length()
            }), 201
            
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/jobs", methods=["GET"])
    @require_auth
    def list_jobs():
        """List all jobs"""
        try:
            jobs = cache.get_all_jobs()
            return jsonify({
                "jobs": [j.to_dict() for j in jobs],
                "count": len(jobs)
            })
        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/jobs/<job_id>", methods=["GET"])
    @require_auth
    def get_job(job_id: str):
        """Get job by ID"""
        # Validate job ID format (same as request_id - alphanumeric with hyphens)
        if not REQUEST_ID_PATTERN.match(job_id):
            return jsonify({"error": "Invalid job ID format"}), 400
        
        try:
            job = cache.get_job(job_id)
            if not job:
                return jsonify({"error": "Job not found"}), 404
            return jsonify(job.to_dict())
        except Exception as e:
            logger.error(f"Error getting job {job_id}: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/jobs/<job_id>/cancel", methods=["POST"])
    @require_auth
    def cancel_job(job_id: str):
        """Cancel a job"""
        # Validate job ID format
        if not REQUEST_ID_PATTERN.match(job_id):
            return jsonify({"error": "Invalid job ID format"}), 400
        
        try:
            job = cache.get_job(job_id)
            if not job:
                return jsonify({"error": "Job not found"}), 404
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return jsonify({"error": f"Cannot cancel job in {job.status.value} state"}), 400
            
            job.status = JobStatus.CANCELLED
            cache.update_job(job)
            
            return jsonify({
                "job_id": job_id,
                "status": job.status.value,
                "message": "Job cancelled"
            })
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/queue", methods=["GET"])
    @require_auth
    def get_queue():
        """Get queue status"""
        try:
            queued_jobs = cache.peek_queue(count=50)
            return jsonify({
                "queue_length": cache.queue_length(),
                "jobs": [j.to_dict() for j in queued_jobs]
            })
        except Exception as e:
            logger.error(f"Error getting queue: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/prices", methods=["GET"])
    @require_auth
    def get_prices():
        """Get current cached prices (Lambda Labs instance types)"""
        try:
            prices = cache.get_all_prices()
            
            # Separate available and unavailable
            available = [p for p in prices if p.available]
            unavailable = [p for p in prices if not p.available]
            
            return jsonify({
                "provider": "lambda_labs",
                "prices": [p.to_dict() for p in prices],
                "available_count": len(available),
                "unavailable_count": len(unavailable),
                "total_count": len(prices)
            })
        except Exception as e:
            logger.error(f"Error getting prices: {e}")
            return jsonify({"error": str(e)}), 500
    
    # =========================================================================
    # Debug Endpoints (Lambda Labs)
    # =========================================================================
    
    @app.route("/debug/lambda-test", methods=["GET"])
    @require_auth
    def debug_lambda_test():
        """Test Lambda Labs API connectivity"""
        try:
            result = {
                "api_key_set": bool(os.environ.get("LAMBDA_API_KEY")),
                "ssh_key_name": os.environ.get("LAMBDA_SSH_KEY_NAME", "not set"),
            }
            
            # Try to list instance types
            try:
                client = LambdaLabsClient()
                instance_types = client.list_instance_types()
                
                result["api_connection"] = "success"
                result["instance_types_count"] = len(instance_types)
                
                # Get available ones
                available = [it for it in instance_types if it.is_available]
                result["available_count"] = len(available)
                
                # Sample of available types
                result["available_sample"] = [
                    {
                        "name": it.name,
                        "gpus": it.gpus,
                        "price_per_hour": it.price_per_hour,
                        "regions": it.regions_available[:3]
                    }
                    for it in available[:5]
                ]
                
            except LambdaAPIError as e:
                result["api_connection"] = "failed"
                result["api_error"] = str(e)
                
            return jsonify(result)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/debug/ssh-keys", methods=["GET"])
    @require_auth
    def debug_ssh_keys():
        """List SSH keys in Lambda Labs account"""
        try:
            client = LambdaLabsClient()
            keys = client.list_ssh_keys()
            
            return jsonify({
                "ssh_keys": [
                    {"id": k.id, "name": k.name}
                    for k in keys
                ],
                "count": len(keys),
                "configured_key": os.environ.get("LAMBDA_SSH_KEY_NAME", "not set")
            })
            
        except LambdaAPIError as e:
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/debug/ssh-keys/generate", methods=["POST"])
    @require_auth
    def generate_ssh_key():
        """Generate a new SSH key in Lambda Labs account and store in Redis"""
        try:
            data = request.get_json() or {}
            key_name = data.get("name") or os.environ.get("LAMBDA_SSH_KEY_NAME", "orchestrator-key")
            
            client = LambdaLabsClient()
            
            # Check if key already exists
            existing_keys = client.list_ssh_keys()
            for k in existing_keys:
                if k.name == key_name:
                    return jsonify({
                        "status": "exists",
                        "message": f"SSH key '{key_name}' already exists",
                        "key": {"id": k.id, "name": k.name}
                    })
            
            # Generate new key (Lambda will generate the keypair)
            result = client.add_ssh_key(name=key_name)
            
            # Store private key in Redis for persistent access
            private_key = result.get("private_key")
            if private_key:
                cache.set("lambda:ssh_private_key", private_key)
                logger.info(f"SSH private key stored in Redis for key '{key_name}'")
            
            return jsonify({
                "status": "created",
                "message": f"SSH key '{key_name}' generated and stored in Redis",
                "key": result,
                "stored_in_redis": bool(private_key),
                "warning": "Private key also returned - save as backup!"
            })
            
        except LambdaAPIError as e:
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/debug/ssh-keys/add", methods=["POST"])
    @require_auth
    def add_ssh_key():
        """Add an existing SSH public key to Lambda Labs account"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body required"}), 400
            
            key_name = data.get("name") or os.environ.get("LAMBDA_SSH_KEY_NAME", "orchestrator-key")
            public_key = data.get("public_key")
            
            if not public_key:
                return jsonify({"error": "public_key is required"}), 400
            
            client = LambdaLabsClient()
            
            # Check if key already exists
            existing_keys = client.list_ssh_keys()
            for k in existing_keys:
                if k.name == key_name:
                    return jsonify({
                        "status": "exists",
                        "message": f"SSH key '{key_name}' already exists",
                        "key": {"id": k.id, "name": k.name}
                    })
            
            # Add the public key
            result = client.add_ssh_key(name=key_name, public_key=public_key)
            
            return jsonify({
                "status": "created",
                "message": f"SSH key '{key_name}' added",
                "key": result
            })
            
        except LambdaAPIError as e:
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/debug/ssh-keys/store", methods=["POST"])
    @require_auth
    def store_ssh_private_key():
        """Store an SSH private key in Redis (for connecting to Lambda instances)"""
        try:
            data = request.get_json()
            if not data or "private_key" not in data:
                return jsonify({"error": "private_key required in request body"}), 400
            
            private_key = data["private_key"]
            
            # Validate it looks like a PEM key
            if not private_key.strip().startswith("-----BEGIN"):
                return jsonify({"error": "Invalid private key format - must be PEM encoded"}), 400
            
            cache.store_ssh_private_key(private_key)
            logger.info("SSH private key stored in Redis")
            
            return jsonify({
                "status": "success",
                "message": "SSH private key stored in Redis",
                "key_length": len(private_key)
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/debug/ssh-keys/check", methods=["GET"])
    @require_auth
    def check_ssh_key():
        """Check if SSH private key is available"""
        try:
            # Check env var
            env_key = os.getenv("LAMBDA_SSH_PRIVATE_KEY", "")
            
            # Check Redis
            redis_key = cache.get("lambda:ssh_private_key") or ""
            
            return jsonify({
                "env_var_set": bool(env_key),
                "env_var_length": len(env_key) if env_key else 0,
                "redis_key_set": bool(redis_key),
                "redis_key_length": len(redis_key) if redis_key else 0,
                "ssh_available": bool(env_key or redis_key)
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/admin/clear-queue", methods=["POST"])
    @require_auth
    def clear_queue():
        """Clear all jobs from the queue"""
        try:
            count = cache.clear_job_queue()
            logger.info(f"Cleared {count} jobs from queue")
            return jsonify({
                "status": "success",
                "message": f"Cleared {count} jobs from queue"
            })
        except Exception as e:
            logger.error(f"Error clearing queue: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/admin/clear-cache", methods=["POST"])
    @require_auth
    def clear_cache():
        """Clear all cached data (prices, jobs, approvals)"""
        try:
            result = cache.clear_all()
            logger.info(f"Cache cleared: {result}")
            return jsonify({
                "status": "success",
                "message": "All cached data cleared",
                "details": result
            })
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/admin/clear-all", methods=["POST"])
    @require_auth
    def clear_all():
        """Clear everything - queue, jobs, prices, approvals"""
        try:
            queue_count = cache.clear_job_queue()
            cache_result = cache.clear_all()
            logger.info(f"Cleared queue ({queue_count} jobs) and cache")
            return jsonify({
                "status": "success",
                "message": "All data cleared",
                "queue_cleared": queue_count,
                "cache_cleared": cache_result
            })
        except Exception as e:
            logger.error(f"Error clearing all: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/jobs/<job_id>/ssh-info", methods=["GET"])
    @require_auth
    def get_ssh_info(job_id: str):
        """Get SSH connection info for a running job's instance"""
        if not REQUEST_ID_PATTERN.match(job_id):
            return jsonify({"error": "Invalid job ID format"}), 400
        
        try:
            job = cache.get_job(job_id)
            if not job:
                return jsonify({"error": "Job not found"}), 404
            
            if not job.instance_ip:
                return jsonify({
                    "error": "Instance not ready or no IP assigned",
                    "status": job.status.value,
                    "instance_id": job.instance_id
                }), 400
            
            ssh_key_name = os.environ.get("LAMBDA_SSH_KEY_NAME", "manipulation-training")
            
            return jsonify({
                "job_id": job_id,
                "instance_id": job.instance_id,
                "instance_ip": job.instance_ip,
                "instance_type": job.instance_type,
                "region": job.region,
                "status": job.status.value,
                "ssh_command": f"ssh -i ~/.ssh/{ssh_key_name} ubuntu@{job.instance_ip}",
                "log_commands": {
                    "startup_log": f"ssh -i ~/.ssh/{ssh_key_name} ubuntu@{job.instance_ip} 'tail -100 /var/log/training-startup.log'",
                    "cloud_init_log": f"ssh -i ~/.ssh/{ssh_key_name} ubuntu@{job.instance_ip} 'tail -100 /var/log/cloud-init-output.log'",
                    "training_status": f"ssh -i ~/.ssh/{ssh_key_name} ubuntu@{job.instance_ip} 'cat /tmp/training_complete 2>/dev/null || echo running'",
                    "gpu_status": f"ssh -i ~/.ssh/{ssh_key_name} ubuntu@{job.instance_ip} 'nvidia-smi'"
                }
            })
        except Exception as e:
            logger.error(f"Error getting SSH info for job {job_id}: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/jobs/<job_id>/logs", methods=["GET"])
    @require_auth
    def get_job_logs(job_id: str):
        """
        Get training logs from a running job's instance via SSH.
        
        Query params:
            lines: Number of log lines to fetch (default: 100)
            log_type: Type of log to fetch - 'training', 'cloud_init', 'all' (default: 'all')
        """
        if not REQUEST_ID_PATTERN.match(job_id):
            return jsonify({"error": "Invalid job ID format"}), 400
        
        try:
            job = cache.get_job(job_id)
            if not job:
                return jsonify({"error": "Job not found"}), 404
            
            if not job.instance_ip:
                return jsonify({
                    "error": "Instance not ready or no IP assigned",
                    "status": job.status.value,
                    "instance_id": job.instance_id,
                    "hint": "Wait for the instance to be provisioned and receive an IP address"
                }), 400
            
            # Get query params
            lines = request.args.get("lines", default=100, type=int)
            log_type = request.args.get("log_type", default="all", type=str)
            
            # Initialize SSH executor with cache to check Redis for key
            ssh = SSHExecutor(cache=cache)
            
            if not ssh.is_configured:
                return jsonify({
                    "error": "SSH not configured",
                    "hint": "Generate SSH key via /debug/ssh-keys/generate or store via /debug/ssh-keys/store",
                    "job_id": job_id,
                    "instance_ip": job.instance_ip,
                    "status": job.status.value
                }), 503
            
            result = {
                "job_id": job_id,
                "instance_id": job.instance_id,
                "instance_ip": job.instance_ip,
                "status": job.status.value,
                "log_type": log_type,
                "lines_requested": lines
            }
            
            if log_type in ("all", "training"):
                success, logs = ssh.get_training_logs(job.instance_ip, lines)
                result["training_logs"] = logs
                result["training_logs_success"] = success
            
            if log_type in ("all", "cloud_init"):
                success, logs = ssh.get_cloud_init_logs(job.instance_ip, lines)
                result["cloud_init_logs"] = logs
                result["cloud_init_logs_success"] = success
            
            if log_type == "all":
                success, status = ssh.get_training_status(job.instance_ip)
                result["training_status"] = status
                result["training_status_success"] = success
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error getting logs for job {job_id}: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/jobs/<job_id>/status", methods=["GET"])
    @require_auth
    def get_job_training_status(job_id: str):
        """
        Get training status and GPU status from a running job's instance via SSH.
        """
        if not REQUEST_ID_PATTERN.match(job_id):
            return jsonify({"error": "Invalid job ID format"}), 400
        
        try:
            job = cache.get_job(job_id)
            if not job:
                return jsonify({"error": "Job not found"}), 404
            
            if not job.instance_ip:
                return jsonify({
                    "job_id": job_id,
                    "status": job.status.value,
                    "instance_id": job.instance_id,
                    "message": "Instance not ready or no IP assigned"
                })
            
            # Initialize SSH executor with cache to check Redis for key
            ssh = SSHExecutor(cache=cache)
            
            result = {
                "job_id": job_id,
                "instance_id": job.instance_id,
                "instance_ip": job.instance_ip,
                "job_status": job.status.value,
                "ssh_configured": ssh.is_configured
            }
            
            if ssh.is_configured:
                # Get training status
                success, status = ssh.get_training_status(job.instance_ip)
                result["training_status"] = status if success else f"Error: {status}"
                
                # Get GPU status
                success, gpu = ssh.get_gpu_status(job.instance_ip)
                result["gpu_status"] = gpu if success else f"Error: {gpu}"
            else:
                result["error"] = "SSH not configured - set LAMBDA_SSH_PRIVATE_KEY"
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error getting training status for job {job_id}: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/debug/instances/<instance_id>/details", methods=["GET"])
    @require_auth
    def get_instance_details(instance_id: str):
        """Get detailed info about a Lambda Labs instance"""
        if not REQUEST_ID_PATTERN.match(instance_id):
            return jsonify({"error": "Invalid instance ID format"}), 400
        
        try:
            client = LambdaLabsClient()
            instance = client.get_instance(instance_id)
            
            if not instance:
                return jsonify({"error": "Instance not found"}), 404
            
            ssh_key_name = os.environ.get("LAMBDA_SSH_KEY_NAME", "manipulation-training")
            
            result = {
                "id": instance.id,
                "name": instance.name,
                "status": instance.status,
                "instance_type": instance.instance_type,
                "region": instance.region,
                "ip": instance.ip,
                "hostname": instance.hostname,
                "is_ready": instance.is_ready,
            }
            
            if instance.ip:
                result["ssh_command"] = f"ssh -i ~/.ssh/{ssh_key_name} ubuntu@{instance.ip}"
            
            return jsonify(result)
            
        except LambdaAPIError as e:
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/debug/instances", methods=["GET"])
    @require_auth
    def debug_instances():
        """List running Lambda Labs instances"""
        try:
            client = LambdaLabsClient()
            instances = client.list_instances()
            
            return jsonify({
                "instances": [
                    {
                        "id": inst.id,
                        "name": inst.name,
                        "status": inst.status,
                        "instance_type": inst.instance_type,
                        "region": inst.region,
                        "ip": inst.ip,
                    }
                    for inst in instances
                ],
                "count": len(instances)
            })
            
        except LambdaAPIError as e:
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/debug/force-poll", methods=["POST"])
    @require_auth
    def debug_force_poll():
        """Force a price poll from Lambda Labs API"""
        try:
            client = LambdaLabsClient()
            instance_types = client.list_instance_types()
            
            available = [it for it in instance_types if it.is_available and it.gpus > 0]
            
            return jsonify({
                "status": "polled",
                "available_gpu_instances": [
                    {
                        "name": it.name,
                        "gpus": it.gpus,
                        "price_per_hour": it.price_per_hour,
                        "price_per_gpu_hour": it.price_per_gpu_hour,
                        "regions": it.regions_available
                    }
                    for it in sorted(available, key=lambda x: x.price_per_gpu_hour)
                ],
                "total_available": len(available)
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # =========================================================================
    # Discord Approval Endpoints
    # =========================================================================
    
    def _validate_discord_token() -> bool:
        """Validate the Discord callback secret token using constant-time comparison"""
        if not discord_config or not discord_config.callback_secret:
            # No secret configured, allow (relies on GCP auth)
            return True
        
        token = request.args.get("token", "")
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(token, discord_config.callback_secret)
    
    def _validate_request_id(request_id: str) -> bool:
        """
        Validate request ID format to prevent path traversal and injection attacks.
        Only allows alphanumeric characters and hyphens, max 64 chars.
        """
        if not request_id:
            return False
        # Check against whitelist pattern
        if not REQUEST_ID_PATTERN.match(request_id):
            logger.warning(f"Invalid request_id format rejected: {request_id[:20]}...")
            return False
        # Extra check: no path traversal sequences
        if '..' in request_id or '/' in request_id or '\\' in request_id:
            logger.warning(f"Path traversal attempt detected in request_id")
            return False
        return True
    
    @app.route("/approvals", methods=["GET"])
    @require_auth
    def list_approvals():
        """List all pending approval requests"""
        try:
            requests_list = cache.get_all_approval_requests()
            pending = [r for r in requests_list if r.status == ApprovalStatus.PENDING]
            return jsonify({
                "pending_approvals": [r.to_dict() for r in pending],
                "all_approvals": [r.to_dict() for r in requests_list],
                "pending_count": len(pending),
                "total_count": len(requests_list)
            })
        except Exception as e:
            logger.error(f"Error listing approvals: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/discord/approve/<request_id>", methods=["GET"])
    def approve_request(request_id: str):
        """Approve a launch request (called from Discord button)"""
        # Validate request ID format (prevent path traversal/injection)
        if not _validate_request_id(request_id):
            return jsonify({"error": "Invalid request ID format"}), 400
        
        # Validate secret token (allows bypassing GCP auth for Discord links)
        if not _validate_discord_token():
            return jsonify({"error": "Invalid or missing token"}), 403
        
        try:
            approval = cache.get_approval_request(request_id)
            
            if not approval:
                return jsonify({"error": "Approval request not found"}), 404
            
            if approval.status != ApprovalStatus.PENDING:
                return jsonify({
                    "error": f"Request already {approval.status.value}",
                    "status": approval.status.value
                }), 400
            
            # Update approval status
            approval.status = ApprovalStatus.APPROVED
            approval.responded_at = datetime.utcnow().isoformat()
            cache.update_approval_request(approval)
            
            logger.info(f"Approval request {request_id} APPROVED")
            
            # Return a nice HTML response for Discord users
            return f"""
            <!DOCTYPE html>
            <html>
            <head><title>Launch Approved</title></head>
            <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: green;">✅ Launch Approved</h1>
                <p>Job {approval.job_id} has been approved.</p>
                <p>The instance will be launched shortly.</p>
            </body>
            </html>
            """
            
        except Exception as e:
            logger.error(f"Error approving request {request_id}: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/discord/reject/<request_id>", methods=["GET"])
    def reject_request(request_id: str):
        """Reject a launch request (called from Discord button)"""
        # Validate request ID format (prevent path traversal/injection)
        if not _validate_request_id(request_id):
            return jsonify({"error": "Invalid request ID format"}), 400
        
        # Validate secret token (allows bypassing GCP auth for Discord links)
        if not _validate_discord_token():
            return jsonify({"error": "Invalid or missing token"}), 403
        
        try:
            approval = cache.get_approval_request(request_id)
            
            if not approval:
                return jsonify({"error": "Approval request not found"}), 404
            
            if approval.status != ApprovalStatus.PENDING:
                return jsonify({
                    "error": f"Request already {approval.status.value}",
                    "status": approval.status.value
                }), 400
            
            # Update approval status
            approval.status = ApprovalStatus.REJECTED
            approval.responded_at = datetime.utcnow().isoformat()
            cache.update_approval_request(approval)
            
            logger.info(f"Approval request {request_id} REJECTED")
            
            # Return a nice HTML response for Discord users
            return f"""
            <!DOCTYPE html>
            <html>
            <head><title>Launch Rejected</title></head>
            <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: red;">❌ Launch Rejected</h1>
                <p>Job {approval.job_id} has been rejected.</p>
                <p>The job will be moved to the back of the queue.</p>
            </body>
            </html>
            """
            
        except Exception as e:
            logger.error(f"Error rejecting request {request_id}: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/discord/status/<request_id>", methods=["GET"])
    def approval_status(request_id: str):
        """Check approval status (called from Discord button)"""
        # Validate request ID format (prevent path traversal/injection)
        if not _validate_request_id(request_id):
            return jsonify({"error": "Invalid request ID format"}), 400
        
        try:
            approval = cache.get_approval_request(request_id)
            
            if not approval:
                return jsonify({"error": "Approval request not found"}), 404
            
            return jsonify(approval.to_dict())
            
        except Exception as e:
            logger.error(f"Error checking approval status {request_id}: {e}")
            return jsonify({"error": str(e)}), 500
    
    return app


class JobAPIService:
    """Wrapper service for the Job API"""
    
    def __init__(
        self, 
        cache: CacheService, 
        api_config: APIConfig,
        discord_config: Optional[DiscordConfig] = None
    ):
        self.cache = cache
        self.api_config = api_config
        self.discord_config = discord_config
        self.app = create_app(cache, api_config, discord_config)
    
    def run(self, threaded: bool = False) -> None:
        """Run the Flask API server"""
        self.app.run(
            host=self.api_config.host,
            port=self.api_config.port,
            threaded=threaded
        )
