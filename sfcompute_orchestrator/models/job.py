"""
Job and pricing data models
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import json
import uuid


class JobStatus(str, Enum):
    """Training job status"""
    PENDING = "pending"                       # Waiting in queue
    SELECTING = "selecting"                   # Finding optimal buy option
    AWAITING_APPROVAL = "awaiting_approval"   # Waiting for Discord approval
    BUYING = "buying"                         # Purchasing compute
    PROVISIONING = "provisioning"             # VM spinning up
    RUNNING = "running"                       # Training in progress
    UPLOADING = "uploading"                   # Uploading checkpoints
    COMPLETED = "completed"                   # Successfully finished
    FAILED = "failed"                         # Error occurred
    CANCELLED = "cancelled"                   # Manually cancelled


class ApprovalStatus(str, Enum):
    """Approval request status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class TrainingJob:
    """A training job to be executed on Lambda Labs"""
    # Unique job identifier
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Required: Path to training data
    data_gcs_path: str = ""
    
    # Optional: Output path (defaults to config)
    output_gcs_path: Optional[str] = None
    
    # Training hyperparameters (optional overrides)
    batch_size: Optional[int] = None
    max_steps: Optional[int] = None
    chunk_size: Optional[int] = None
    learning_rate: Optional[str] = None
    num_workers: Optional[int] = None
    
    # Compute preferences
    min_gpus: int = 1              # Minimum GPUs acceptable
    max_gpus: int = 8              # Maximum GPUs to consider
    max_duration_hours: int = 24   # Maximum duration (for cost estimation)
    max_total_cost: Optional[float] = None  # Budget cap (total estimated)
    max_spend: Optional[float] = None  # Max price per GPU per hour willing to pay
    
    # Status tracking
    status: JobStatus = JobStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Execution details (Lambda Labs instance)
    instance_id: Optional[str] = None  # Lambda Labs instance ID
    instance_ip: Optional[str] = None  # Instance public IP
    instance_type: Optional[str] = None  # Lambda Labs instance type
    region: Optional[str] = None  # Lambda Labs region
    vm_id: Optional[str] = None  # Alias for instance_id (backwards compat)
    buy_option: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Approval tracking
    approval_request_id: Optional[str] = None
    
    # Callback URL for status updates (optional)
    # When job completes/fails, POST to this URL with job details
    callback_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "job_id": self.job_id,
            "data_gcs_path": self.data_gcs_path,
            "output_gcs_path": self.output_gcs_path,
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "chunk_size": self.chunk_size,
            "learning_rate": self.learning_rate,
            "num_workers": self.num_workers,
            "min_gpus": self.min_gpus,
            "max_gpus": self.max_gpus,
            "max_duration_hours": self.max_duration_hours,
            "max_total_cost": self.max_total_cost,
            "max_spend": self.max_spend,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "instance_id": self.instance_id,
            "instance_ip": self.instance_ip,
            "instance_type": self.instance_type,
            "region": self.region,
            "vm_id": self.vm_id,
            "buy_option": self.buy_option,
            "error_message": self.error_message,
            "approval_request_id": self.approval_request_id,
            "callback_url": self.callback_url,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingJob":
        """Create from dictionary"""
        # Handle status enum
        if "status" in data and isinstance(data["status"], str):
            data["status"] = JobStatus(data["status"])
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TrainingJob":
        """Deserialize from JSON"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class BuyOption:
    """A potential buy/launch configuration"""
    gpu_count: int
    duration_hours: int  # Estimated duration for cost calculation
    price_per_gpu_hour: float
    total_price: float  # Estimated total based on duration
    available: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    # Lambda Labs specific
    instance_type: Optional[str] = None  # Lambda instance type name
    region: Optional[str] = None  # Lambda region
    price_per_hour: Optional[float] = None  # Total hourly cost
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "gpu_count": self.gpu_count,
            "duration_hours": self.duration_hours,
            "price_per_gpu_hour": self.price_per_gpu_hour,
            "total_price": self.total_price,
            "available": self.available,
            "timestamp": self.timestamp,
        }
        if self.instance_type:
            result["instance_type"] = self.instance_type
        if self.region:
            result["region"] = self.region
        if self.price_per_hour:
            result["price_per_hour"] = self.price_per_hour
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuyOption":
        return cls(**data)


@dataclass
class PriceData:
    """Price and availability snapshot"""
    gpu_type: str
    gpu_count: int
    duration_hours: int
    price_per_gpu_hour: float
    available: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    # Extra metadata (e.g., Lambda Labs instance info)
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def total_price(self) -> float:
        return self.gpu_count * self.duration_hours * self.price_per_gpu_hour
    
    @property
    def cache_key(self) -> str:
        return f"{self.gpu_type}:{self.gpu_count}:{self.duration_hours}"
    
    @property
    def instance_type(self) -> Optional[str]:
        """Get Lambda Labs instance type from metadata"""
        if self.metadata:
            return self.metadata.get("instance_type")
        return self.gpu_type
    
    @property
    def regions_available(self) -> list:
        """Get available regions from metadata"""
        if self.metadata:
            return self.metadata.get("regions_available", [])
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "duration_hours": self.duration_hours,
            "price_per_gpu_hour": self.price_per_gpu_hour,
            "total_price": self.total_price,
            "available": self.available,
            "timestamp": self.timestamp,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriceData":
        # Remove computed field if present
        data.pop("total_price", None)
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "PriceData":
        return cls.from_dict(json.loads(json_str))


@dataclass
class ApprovalRequest:
    """A pending approval request for a buy operation"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    job_id: str = ""
    buy_option: Optional[Dict[str, Any]] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    responded_at: Optional[str] = None
    discord_message_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "job_id": self.job_id,
            "buy_option": self.buy_option,
            "status": self.status.value,
            "created_at": self.created_at,
            "responded_at": self.responded_at,
            "discord_message_id": self.discord_message_id,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalRequest":
        if "status" in data and isinstance(data["status"], str):
            data["status"] = ApprovalStatus(data["status"])
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ApprovalRequest":
        return cls.from_dict(json.loads(json_str))

