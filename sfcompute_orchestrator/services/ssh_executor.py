"""
SSH Executor Service
Executes commands on Lambda Labs instances via SSH
"""
import io
import logging
import os
from typing import Optional, Tuple

from .cache import CacheService

logger = logging.getLogger(__name__)

# GCP Project for Secret Manager
GCP_PROJECT = os.getenv("GCP_PROJECT", "innate-agent")
SSH_KEY_SECRET_NAME = "lambda-ssh-private-key"


def _get_secret_from_secret_manager(secret_id: str, project_id: str = None) -> Optional[str]:
    """
    Fetch a secret from Google Cloud Secret Manager.
    
    Returns None if not available or on error.
    """
    try:
        from google.cloud import secretmanager
        
        project = project_id or GCP_PROJECT
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project}/secrets/{secret_id}/versions/latest"
        
        response = client.access_secret_version(request={"name": name})
        secret_value = response.payload.data.decode("UTF-8")
        logger.debug(f"Retrieved secret '{secret_id}' from Secret Manager")
        return secret_value
        
    except ImportError:
        logger.debug("google-cloud-secret-manager not installed")
        return None
    except Exception as e:
        logger.debug(f"Could not fetch secret '{secret_id}' from Secret Manager: {e}")
        return None


class SSHExecutor:
    """
    Execute commands on remote instances via SSH.
    
    Uses paramiko for SSH connections with the private key
    from environment variable or Redis cache.
    """
    
    def __init__(
        self, 
        private_key_pem: Optional[str] = None,
        cache: Optional[CacheService] = None
    ):
        """
        Initialize SSH executor.
        
        Args:
            private_key_pem: PEM-encoded private key string.
                           If not provided, checks LAMBDA_SSH_PRIVATE_KEY env var,
                           then Redis cache.
                           Can be base64-encoded.
            cache: Optional CacheService instance to check for key in Redis.
        """
        self._cache = cache
        raw_key = private_key_pem or self._get_private_key()
        
        # Check if key is base64 encoded (doesn't start with '-----')
        if raw_key and not raw_key.strip().startswith("-----"):
            try:
                import base64
                raw_key = base64.b64decode(raw_key).decode('utf-8')
                logger.debug("Decoded base64-encoded private key")
            except Exception as e:
                logger.warning(f"Failed to base64-decode private key: {e}")
        
        self.private_key_pem = raw_key
        self._pkey = None
        
        if self.private_key_pem:
            self._load_private_key()
    
    def _get_private_key(self) -> str:
        """
        Get private key from various sources (in order of priority):
        1. LAMBDA_SSH_PRIVATE_KEY environment variable
        2. Google Cloud Secret Manager
        3. Redis cache (fallback)
        """
        # 1. First check environment variable
        env_key = os.getenv("LAMBDA_SSH_PRIVATE_KEY", "")
        if env_key:
            logger.debug("Using SSH key from LAMBDA_SSH_PRIVATE_KEY env var")
            return env_key
        
        # 2. Check Secret Manager (preferred for Cloud Run)
        secret_key = _get_secret_from_secret_manager(SSH_KEY_SECRET_NAME)
        if secret_key:
            logger.info("Using SSH key from Secret Manager")
            return secret_key
        
        # 3. Fallback to Redis cache
        if self._cache:
            try:
                redis_key = self._cache.get_ssh_private_key()
                if redis_key:
                    logger.debug("Using SSH key from Redis cache")
                    return redis_key
            except Exception as e:
                logger.warning(f"Error reading SSH key from Redis: {e}")
        
        logger.debug("No SSH private key found in env, Secret Manager, or Redis")
        return ""
    
    def _load_private_key(self):
        """Load the private key from PEM string"""
        try:
            import paramiko
            
            # Try RSA key first
            key_file = io.StringIO(self.private_key_pem)
            try:
                self._pkey = paramiko.RSAKey.from_private_key(key_file)
                logger.debug("Loaded RSA private key")
                return
            except paramiko.SSHException:
                pass
            
            # Try Ed25519
            key_file = io.StringIO(self.private_key_pem)
            try:
                self._pkey = paramiko.Ed25519Key.from_private_key(key_file)
                logger.debug("Loaded Ed25519 private key")
                return
            except paramiko.SSHException:
                pass
            
            # Try ECDSA
            key_file = io.StringIO(self.private_key_pem)
            try:
                self._pkey = paramiko.ECDSAKey.from_private_key(key_file)
                logger.debug("Loaded ECDSA private key")
                return
            except paramiko.SSHException:
                pass
            
            logger.error("Could not load private key - unsupported key type")
            
        except Exception as e:
            logger.error(f"Failed to load private key: {e}")
    
    @property
    def is_configured(self) -> bool:
        """Check if SSH executor is properly configured with a valid key"""
        return self._pkey is not None
    
    def execute_command(
        self, 
        host: str, 
        command: str, 
        username: str = "ubuntu",
        timeout: int = 30,
        port: int = 22
    ) -> Tuple[int, str, str]:
        """
        Execute a command on a remote host via SSH.
        
        Args:
            host: IP address or hostname
            command: Command to execute
            username: SSH username (default: ubuntu for Lambda instances)
            timeout: Command timeout in seconds
            port: SSH port (default: 22)
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        if not self.is_configured:
            return (-1, "", "SSH executor not configured - no private key available")
        
        try:
            import paramiko
            
            # Create SSH client
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            logger.debug(f"Connecting to {username}@{host}:{port}")
            
            client.connect(
                hostname=host,
                port=port,
                username=username,
                pkey=self._pkey,
                timeout=timeout,
                allow_agent=False,
                look_for_keys=False
            )
            
            # Execute command
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            
            # Get output
            exit_code = stdout.channel.recv_exit_status()
            stdout_text = stdout.read().decode('utf-8', errors='replace')
            stderr_text = stderr.read().decode('utf-8', errors='replace')
            
            client.close()
            
            return (exit_code, stdout_text, stderr_text)
            
        except Exception as e:
            error_msg = f"SSH execution failed: {str(e)}"
            logger.error(error_msg)
            return (-1, "", error_msg)
    
    def get_training_logs(
        self, 
        host: str, 
        lines: int = 100,
        username: str = "ubuntu"
    ) -> Tuple[bool, str]:
        """
        Get training startup logs from an instance.
        
        Args:
            host: Instance IP address
            lines: Number of lines to fetch
            username: SSH username
            
        Returns:
            Tuple of (success, log_content)
        """
        command = f"tail -{lines} /var/log/training-startup.log 2>/dev/null || echo 'Log file not found'"
        exit_code, stdout, stderr = self.execute_command(host, command, username)
        
        if exit_code == 0:
            return (True, stdout)
        else:
            return (False, f"Failed to get logs: {stderr or stdout}")
    
    def get_cloud_init_logs(
        self, 
        host: str, 
        lines: int = 100,
        username: str = "ubuntu"
    ) -> Tuple[bool, str]:
        """
        Get cloud-init output logs from an instance.
        
        Args:
            host: Instance IP address
            lines: Number of lines to fetch
            username: SSH username
            
        Returns:
            Tuple of (success, log_content)
        """
        command = f"tail -{lines} /var/log/cloud-init-output.log 2>/dev/null || echo 'Log file not found'"
        exit_code, stdout, stderr = self.execute_command(host, command, username)
        
        if exit_code == 0:
            return (True, stdout)
        else:
            return (False, f"Failed to get cloud-init logs: {stderr or stdout}")
    
    def get_training_status(
        self, 
        host: str, 
        username: str = "ubuntu"
    ) -> Tuple[bool, str]:
        """
        Check if training is complete and get the exit code.
        
        Args:
            host: Instance IP address
            username: SSH username
            
        Returns:
            Tuple of (success, status_message)
        """
        command = "cat /tmp/training_complete 2>/dev/null || echo 'running'"
        exit_code, stdout, stderr = self.execute_command(host, command, username, timeout=10)
        
        if exit_code == 0:
            status = stdout.strip()
            if status == "running":
                return (True, "Training is still running")
            elif status == "0":
                return (True, "Training completed successfully")
            else:
                return (True, f"Training failed with exit code: {status}")
        else:
            return (False, f"Failed to check status: {stderr or stdout}")
    
    def get_gpu_status(
        self, 
        host: str, 
        username: str = "ubuntu"
    ) -> Tuple[bool, str]:
        """
        Get GPU status from nvidia-smi.
        
        Args:
            host: Instance IP address
            username: SSH username
            
        Returns:
            Tuple of (success, nvidia_smi_output)
        """
        exit_code, stdout, stderr = self.execute_command(host, "nvidia-smi", username, timeout=15)
        
        if exit_code == 0:
            return (True, stdout)
        else:
            return (False, f"Failed to get GPU status: {stderr or stdout}")
    
    def get_combined_status(
        self, 
        host: str, 
        log_lines: int = 50,
        username: str = "ubuntu"
    ) -> dict:
        """
        Get combined status information from an instance.
        
        Args:
            host: Instance IP address
            log_lines: Number of log lines to fetch
            username: SSH username
            
        Returns:
            Dict with training_status, training_logs, gpu_status
        """
        result = {
            "host": host,
            "ssh_configured": self.is_configured
        }
        
        if not self.is_configured:
            result["error"] = "SSH not configured - private key not available"
            return result
        
        # Get training status
        success, status = self.get_training_status(host, username)
        result["training_status"] = status if success else f"Error: {status}"
        
        # Get training logs
        success, logs = self.get_training_logs(host, log_lines, username)
        result["training_logs"] = logs if success else f"Error: {logs}"
        
        # Get GPU status (optional, may fail if training is using GPUs heavily)
        success, gpu = self.get_gpu_status(host, username)
        result["gpu_status"] = gpu if success else f"Error: {gpu}"
        
        return result

