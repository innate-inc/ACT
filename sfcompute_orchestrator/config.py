"""
Configuration for GPU Orchestrator (Lambda Labs)
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class LambdaConfig:
    """Lambda Labs specific configuration"""
    # API key for Lambda Labs
    api_key: str = field(default_factory=lambda: os.getenv("LAMBDA_API_KEY", ""))
    
    # SSH key name to use for instances
    ssh_key_name: str = field(default_factory=lambda: os.getenv("LAMBDA_SSH_KEY_NAME", "orchestrator-key"))
    
    # SSH private key (PEM format, stored in env for Cloud Run)
    # This is the private key generated when creating the SSH key in Lambda Labs
    ssh_private_key: str = field(default_factory=lambda: os.getenv("LAMBDA_SSH_PRIVATE_KEY", ""))
    
    # Preferred instance types (in order of preference)
    # Priority: SXM (NVLink) >> PCIe, H100 > A100, more GPUs > fewer GPUs
    preferred_instance_types: List[str] = field(default_factory=lambda: [
        # === 8 GPU SXM (NVLink - BEST for multi-GPU training) ===
        "gpu_8x_h100_sxm5",      # 8x H100 80GB SXM5 - fastest
        "gpu_8x_h100_sxm5gdr",   # 8x H100 SXM5 with GDR
        "gpu_8x_a100_80gb_sxm4", # 8x A100 80GB SXM4 - great value
        # === 4 GPU SXM ===
        "gpu_4x_h100_sxm5",      # 4x H100 SXM5
        "gpu_4x_a100_sxm4",      # 4x A100 SXM4
        # === 2 GPU SXM ===
        "gpu_2x_h100_sxm5",      # 2x H100 SXM5
        "gpu_2x_a100_sxm4",      # 2x A100 SXM4
        # === 1 GPU SXM ===
        "gpu_1x_h100_sxm5",      # 1x H100 SXM5
        "gpu_1x_a100_sxm4",      # 1x A100 SXM4
        # === PCIe fallback (slower interconnect) ===
        "gpu_8x_h100_pcie",      # 8x H100 PCIe
        "gpu_8x_a100",           # 8x A100 40GB PCIe
        "gpu_4x_h100_pcie",      # 4x H100 PCIe
        "gpu_4x_a100",           # 4x A100 40GB
        "gpu_2x_h100_pcie",      # 2x H100 PCIe
        "gpu_2x_a100",           # 2x A100 40GB
        "gpu_1x_h100_pcie",      # 1x H100 PCIe
        "gpu_1x_a100",           # 1x A100 40GB
    ])
    
    # Preferred regions (in order of preference)
    preferred_regions: List[str] = field(default_factory=lambda: [
        "us-west-1",
        "us-west-2", 
        "us-east-1",
        "us-south-1",
        "europe-central-1",
        "asia-south-1",
    ])
    
    # Minimum GPUs required for training
    min_gpus: int = 1
    
    # Maximum GPUs to consider
    max_gpus: int = 8
    
    # Price polling interval in seconds
    poll_interval_seconds: int = 60
    
    # Maximum acceptable price per GPU per hour (dollars)
    max_price_per_gpu_hour: float = 5.00
    
    # Instance launch timeout (seconds)
    instance_launch_timeout: int = 600
    
    # Instance ready timeout (seconds)  
    instance_ready_timeout: int = 600


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    password: str = field(default_factory=lambda: os.getenv("REDIS_PASSWORD", ""))
    
    # Key prefixes
    price_prefix: str = "lambda:prices:"
    availability_prefix: str = "lambda:availability:"
    job_queue: str = "lambda:jobs:queue"
    job_status_prefix: str = "lambda:jobs:status:"
    
    # TTL for cached data (5 minutes for Lambda, API is fast)
    cache_ttl_seconds: int = 300


@dataclass
class TrainingConfig:
    """Default training configuration"""
    batch_size: int = 96
    max_steps: int = 120000
    chunk_size: int = 30
    learning_rate: str = "5e-5"
    num_workers: int = 4


@dataclass
class GCSConfig:
    """Google Cloud Storage configuration"""
    default_output_bucket: str = field(
        default_factory=lambda: os.getenv("OUTPUT_GCS_PATH", "gs://mauricearm-training-outputs")
    )
    credentials_path: str = field(
        default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    )


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8080")))


@dataclass
class DiscordConfig:
    """Discord integration configuration for approval workflow"""
    # Webhook URL to send approval requests to Discord
    webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))
    
    # Bot token for sending messages via API (optional, can use webhook instead)
    bot_token: str = field(default_factory=lambda: os.getenv("DISCORD_BOT_TOKEN", ""))
    
    # Channel ID for approval messages
    channel_id: str = field(default_factory=lambda: os.getenv("DISCORD_CHANNEL_ID", ""))
    
    # The external URL where Discord callbacks will be sent
    # (This service's public URL for the bot to call back)
    callback_base_url: str = field(default_factory=lambda: os.getenv("DISCORD_CALLBACK_URL", "http://localhost:8080"))
    
    # Secret token for Discord callback URLs (bypasses GCP auth)
    # This allows Discord links to work without requiring the user to authenticate
    callback_secret: str = field(default_factory=lambda: os.getenv("DISCORD_CALLBACK_SECRET", ""))
    
    # Approval timeout in seconds (how long to wait for a response)
    approval_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("DISCORD_APPROVAL_TIMEOUT", "300")))
    
    # Whether Discord approval is required (can be disabled for testing)
    require_approval: bool = field(default_factory=lambda: os.getenv("DISCORD_REQUIRE_APPROVAL", "true").lower() == "true")
    
    @property
    def is_configured(self) -> bool:
        """Check if Discord is properly configured"""
        return bool(self.webhook_url or (self.bot_token and self.channel_id))


@dataclass
class Config:
    """Main configuration container"""
    lambda_labs: LambdaConfig = field(default_factory=LambdaConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gcs: GCSConfig = field(default_factory=GCSConfig)
    api: APIConfig = field(default_factory=APIConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    
    # Service toggles
    enable_price_monitor: bool = True
    enable_job_executor: bool = True
    enable_api: bool = True
    
    # Dry run mode (don't actually launch/terminate instances)
    dry_run: bool = field(default_factory=lambda: os.getenv("DRY_RUN", "false").lower() == "true")
    
    # Legacy alias for backwards compatibility
    @property
    def sfcompute(self) -> LambdaConfig:
        """Alias for lambda_labs config (backwards compatibility)"""
        return self.lambda_labs


def load_config() -> Config:
    """Load configuration from environment"""
    return Config()
