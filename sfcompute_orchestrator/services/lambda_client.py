"""
Lambda Labs Cloud API Client

Provides a Python interface to the Lambda Labs Cloud API for:
- Listing instance types and availability
- Launching instances
- Terminating instances
- Managing SSH keys
"""
import logging
import os
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Lambda Labs API base URL
LAMBDA_API_BASE = "https://cloud.lambda.ai"


@dataclass
class LambdaInstanceType:
    """Represents a Lambda Labs instance type"""
    name: str
    description: str
    gpu_description: str
    price_cents_per_hour: int
    gpus: int
    vcpus: int
    memory_gib: int
    storage_gib: int
    regions_available: List[str] = field(default_factory=list)
    
    @property
    def price_per_hour(self) -> float:
        """Price in dollars per hour"""
        return self.price_cents_per_hour / 100.0
    
    @property
    def price_per_gpu_hour(self) -> float:
        """Price per GPU per hour"""
        if self.gpus == 0:
            return 0.0
        return self.price_per_hour / self.gpus
    
    @property
    def is_available(self) -> bool:
        """Check if any region has capacity"""
        return len(self.regions_available) > 0


@dataclass
class LambdaInstance:
    """Represents a running Lambda Labs instance"""
    id: str
    name: Optional[str]
    ip: Optional[str]
    private_ip: Optional[str]
    status: str  # booting, active, unhealthy, terminated, terminating
    instance_type: str
    region: str
    ssh_key_names: List[str]
    hostname: Optional[str] = None
    jupyter_url: Optional[str] = None
    
    @property
    def is_ready(self) -> bool:
        """Check if instance is ready for SSH"""
        return self.status == "active" and self.ip is not None


@dataclass
class LambdaSSHKey:
    """Represents an SSH key in Lambda Labs"""
    id: str
    name: str
    public_key: str


class LambdaLabsClient:
    """
    Client for Lambda Labs Cloud API.
    
    Uses Bearer token authentication with the Lambda API key.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("LAMBDA_API_KEY", "")
        self.base_url = LAMBDA_API_BASE
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Make an API request"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                timeout=timeout
            )
            
            # Parse response
            result = response.json()
            
            # Check for errors
            if "error" in result:
                error = result["error"]
                raise LambdaAPIError(
                    code=error.get("code", "unknown"),
                    message=error.get("message", "Unknown error"),
                    suggestion=error.get("suggestion")
                )
            
            return result.get("data", result)
            
        except requests.exceptions.Timeout:
            raise LambdaAPIError("timeout", f"Request to {endpoint} timed out")
        except requests.exceptions.RequestException as e:
            raise LambdaAPIError("request_failed", str(e))
    
    # =========================================================================
    # Instance Types
    # =========================================================================
    
    def list_instance_types(self) -> List[LambdaInstanceType]:
        """
        List all available instance types with pricing and regional availability.
        
        Returns:
            List of LambdaInstanceType objects
        """
        data = self._request("GET", "/api/v1/instance-types")
        
        instance_types = []
        for name, info in data.items():
            it = info.get("instance_type", {})
            specs = it.get("specs", {})
            
            regions = [
                r.get("name") 
                for r in info.get("regions_with_capacity_available", [])
            ]
            
            instance_types.append(LambdaInstanceType(
                name=name,
                description=it.get("description", ""),
                gpu_description=it.get("gpu_description", ""),
                price_cents_per_hour=it.get("price_cents_per_hour", 0),
                gpus=specs.get("gpus", 0),
                vcpus=specs.get("vcpus", 0),
                memory_gib=specs.get("memory_gib", 0),
                storage_gib=specs.get("storage_gib", 0),
                regions_available=regions
            ))
        
        return instance_types
    
    def get_available_instance_types(self) -> List[LambdaInstanceType]:
        """Get only instance types that have capacity available"""
        return [it for it in self.list_instance_types() if it.is_available]
    
    def get_gpu_instance_types(self, min_gpus: int = 1) -> List[LambdaInstanceType]:
        """Get GPU instance types with at least min_gpus GPUs"""
        return [
            it for it in self.list_instance_types() 
            if it.gpus >= min_gpus
        ]
    
    # =========================================================================
    # Instances
    # =========================================================================
    
    def list_instances(self) -> List[LambdaInstance]:
        """List all running instances"""
        data = self._request("GET", "/api/v1/instances")
        
        instances = []
        for inst in data:
            region_info = inst.get("region", {})
            it_info = inst.get("instance_type", {})
            
            instances.append(LambdaInstance(
                id=inst.get("id", ""),
                name=inst.get("name"),
                ip=inst.get("ip"),
                private_ip=inst.get("private_ip"),
                status=inst.get("status", "unknown"),
                instance_type=it_info.get("name", ""),
                region=region_info.get("name", ""),
                ssh_key_names=inst.get("ssh_key_names", []),
                hostname=inst.get("hostname"),
                jupyter_url=inst.get("jupyter_url")
            ))
        
        return instances
    
    def get_instance(self, instance_id: str) -> Optional[LambdaInstance]:
        """Get details for a specific instance"""
        try:
            data = self._request("GET", f"/api/v1/instances/{instance_id}")
            
            region_info = data.get("region", {})
            it_info = data.get("instance_type", {})
            
            return LambdaInstance(
                id=data.get("id", ""),
                name=data.get("name"),
                ip=data.get("ip"),
                private_ip=data.get("private_ip"),
                status=data.get("status", "unknown"),
                instance_type=it_info.get("name", ""),
                region=region_info.get("name", ""),
                ssh_key_names=data.get("ssh_key_names", []),
                hostname=data.get("hostname"),
                jupyter_url=data.get("jupyter_url")
            )
        except LambdaAPIError as e:
            if e.code == "global/object-does-not-exist":
                return None
            raise
    
    def launch_instance(
        self,
        instance_type_name: str,
        region_name: str,
        ssh_key_names: List[str],
        name: Optional[str] = None,
        file_system_names: Optional[List[str]] = None,
        user_data: Optional[str] = None
    ) -> List[str]:
        """
        Launch a new instance.
        
        Args:
            instance_type_name: The instance type (e.g., "gpu_8x_h100_sxm5")
            region_name: Region code (e.g., "us-west-1")
            ssh_key_names: List of SSH key names for access
            name: Optional instance name
            file_system_names: Optional filesystem names to mount
            user_data: Optional cloud-init user data script
        
        Returns:
            List of instance IDs that were launched
        """
        payload = {
            "instance_type_name": instance_type_name,
            "region_name": region_name,
            "ssh_key_names": ssh_key_names,
        }
        
        if name:
            payload["name"] = name
        if file_system_names:
            payload["file_system_names"] = file_system_names
        if user_data:
            payload["user_data"] = user_data
        
        data = self._request(
            "POST", 
            "/api/v1/instance-operations/launch",
            data=payload,
            timeout=60
        )
        
        return data.get("instance_ids", [])
    
    def terminate_instances(self, instance_ids: List[str]) -> List[Dict]:
        """
        Terminate one or more instances.
        
        Args:
            instance_ids: List of instance IDs to terminate
        
        Returns:
            List of terminated instance info
        """
        data = self._request(
            "POST",
            "/api/v1/instance-operations/terminate",
            data={"instance_ids": instance_ids},
            timeout=60
        )
        
        return data.get("terminated_instances", [])
    
    def restart_instances(self, instance_ids: List[str]) -> List[Dict]:
        """Restart one or more instances"""
        data = self._request(
            "POST",
            "/api/v1/instance-operations/restart",
            data={"instance_ids": instance_ids},
            timeout=60
        )
        
        return data.get("restarted_instances", [])
    
    # =========================================================================
    # SSH Keys
    # =========================================================================
    
    def list_ssh_keys(self) -> List[LambdaSSHKey]:
        """List all SSH keys"""
        data = self._request("GET", "/api/v1/ssh-keys")
        
        return [
            LambdaSSHKey(
                id=key.get("id", ""),
                name=key.get("name", ""),
                public_key=key.get("public_key", "")
            )
            for key in data
        ]
    
    def get_ssh_key_by_name(self, name: str) -> Optional[LambdaSSHKey]:
        """Get an SSH key by name"""
        for key in self.list_ssh_keys():
            if key.name == name:
                return key
        return None
    
    def add_ssh_key(
        self, 
        name: str, 
        public_key: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Add an SSH key.
        
        Args:
            name: Name for the SSH key
            public_key: Optional public key. If not provided, a new key pair 
                       will be generated and the private key returned.
        
        Returns:
            Dict with id, name, public_key, and optionally private_key
        """
        payload = {"name": name}
        if public_key:
            payload["public_key"] = public_key
        
        data = self._request("POST", "/api/v1/ssh-keys", data=payload)
        
        return {
            "id": data.get("id", ""),
            "name": data.get("name", ""),
            "public_key": data.get("public_key", ""),
            "private_key": data.get("private_key")  # Only if generated
        }
    
    def delete_ssh_key(self, key_id: str) -> bool:
        """Delete an SSH key by ID"""
        try:
            self._request("DELETE", f"/api/v1/ssh-keys/{key_id}")
            return True
        except LambdaAPIError:
            return False
    
    # =========================================================================
    # Filesystems
    # =========================================================================
    
    def list_filesystems(self) -> List[Dict]:
        """List all filesystems"""
        return self._request("GET", "/api/v1/file-systems")
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def wait_for_instance_ready(
        self, 
        instance_id: str, 
        timeout_seconds: int = 600,
        poll_interval: int = 10
    ) -> Optional[LambdaInstance]:
        """
        Wait for an instance to become ready (active with IP).
        
        Returns:
            The instance if ready, None if timeout
        """
        import time
        start = time.time()
        
        while time.time() - start < timeout_seconds:
            instance = self.get_instance(instance_id)
            
            if instance is None:
                logger.warning(f"Instance {instance_id} not found")
                return None
            
            if instance.is_ready:
                logger.info(f"Instance {instance_id} is ready at {instance.ip}")
                return instance
            
            logger.debug(f"Instance {instance_id} status: {instance.status}")
            time.sleep(poll_interval)
        
        logger.warning(f"Timeout waiting for instance {instance_id}")
        return None
    
    def ensure_ssh_key_exists(self, key_name: str, public_key: str) -> LambdaSSHKey:
        """
        Ensure an SSH key exists, creating it if necessary.
        
        Args:
            key_name: Name for the SSH key
            public_key: The public key content
        
        Returns:
            The SSH key (existing or newly created)
        """
        existing = self.get_ssh_key_by_name(key_name)
        if existing:
            logger.info(f"SSH key '{key_name}' already exists")
            return existing
        
        logger.info(f"Creating SSH key '{key_name}'")
        result = self.add_ssh_key(key_name, public_key)
        
        return LambdaSSHKey(
            id=result["id"],
            name=result["name"],
            public_key=result["public_key"]
        )


class LambdaAPIError(Exception):
    """Exception for Lambda Labs API errors"""
    
    def __init__(self, code: str, message: str, suggestion: Optional[str] = None):
        self.code = code
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"[{code}] {message}")
    
    def __str__(self):
        msg = f"[{self.code}] {self.message}"
        if self.suggestion:
            msg += f" (Suggestion: {self.suggestion})"
        return msg

