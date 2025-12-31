"""
Lambda Labs Orchestrator - Main Entry Point

Runs all services:
1. Price Monitor - Polls Lambda Labs for pricing/availability
2. Job API - HTTP API for job submission
3. Job Executor - Processes jobs from queue

Usage:
    python -m sfcompute_orchestrator
    python -m sfcompute_orchestrator --dry-run
    python -m sfcompute_orchestrator --api-only
"""
import argparse
import logging
import os
import signal
import sys
import threading
import time

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Log startup immediately
logger.info("=" * 60)
logger.info("🚀 Lambda Labs Orchestrator Starting...")
logger.info("=" * 60)
logger.info(f"Python: {sys.version}")
logger.info(f"REDIS_HOST: {os.environ.get('REDIS_HOST', 'not set')}")
logger.info(f"API_PORT: {os.environ.get('API_PORT', '8080')}")
logger.info(f"LAMBDA_API_KEY: {'set' if os.environ.get('LAMBDA_API_KEY') else 'not set'}")

# Now import modules (these may take time)
try:
    from .config import load_config
    from .services.cache import CacheService
    from .services.price_monitor import PriceMonitorService
    from .services.job_api import JobAPIService
    from .services.job_executor import JobExecutorService
    from .services.discord import DiscordService
    from .services.lambda_client import LambdaLabsClient
    logger.info("✅ All modules imported successfully")
except Exception as e:
    logger.error(f"❌ Failed to import modules: {e}")
    raise


class Orchestrator:
    """Main orchestrator that manages all services"""
    
    def __init__(self, config):
        self.config = config
        self.cache = None
        self.price_monitor = None
        self.discord_service = None
        self.job_executor = None
        self.job_api = None
        self._shutdown = threading.Event()
        self._services_initialized = False
    
    def _init_services(self) -> bool:
        """Initialize all services. Returns True on success."""
        if self._services_initialized:
            return True
            
        try:
            self.cache = CacheService(self.config.redis)
            
            # Check Redis connectivity (quick check, don't block startup)
            if self.cache.ping():
                logger.info("✅ Redis connected")
            else:
                logger.warning(f"Redis not available at {self.config.redis.host}:{self.config.redis.port}")
                logger.warning("Will retry in background...")
            
            # Initialize price monitor with Lambda Labs config
            self.price_monitor = PriceMonitorService(self.config.lambda_labs, self.cache)
            
            # Initialize Discord service if configured
            if self.config.discord.is_configured:
                self.discord_service = DiscordService(self.config.discord, self.cache)
            
            # Initialize executor with Discord service
            self.job_executor = JobExecutorService(
                self.config, 
                self.cache, 
                self.price_monitor,
                self.discord_service
            )
            
            self._services_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize services: {e}")
            return False
    
    def start(self) -> None:
        """Start all services"""
        # Initialize services first
        if not self._init_services():
            logger.error("Service initialization failed, but API will still start")
            # Create minimal cache for API even if Redis is down
            self.cache = CacheService(self.config.redis)
        
        # Initialize API with Discord config (do this even if other services failed)
        self.job_api = JobAPIService(self.cache, self.config.api, self.config.discord)
        
        # Start background services (only if properly initialized)
        if self._services_initialized:
            if self.config.enable_price_monitor and self.price_monitor:
                self.price_monitor.start()
                logger.info("✅ Price Monitor started (Lambda Labs)")
                logger.info(f"   Polling every {self.config.lambda_labs.poll_interval_seconds}s")
                logger.info(f"   Min GPUs: {self.config.lambda_labs.min_gpus}")
                logger.info(f"   Max GPUs: {self.config.lambda_labs.max_gpus}")
            
            if self.config.enable_job_executor and self.job_executor:
                self.job_executor.start()
                logger.info("✅ Job Executor started")
            
            # Discord status
            if self.discord_service and self.discord_service.is_enabled():
                logger.info("✅ Discord approval workflow enabled")
                logger.info(f"   Callback URL: {self.config.discord.callback_base_url}")
                logger.info(f"   Approval timeout: {self.config.discord.approval_timeout_seconds}s")
            else:
                logger.warning("⚠️  Discord approval disabled or not configured")
        else:
            logger.warning("⚠️  Background services NOT started (initialization failed)")
        
        if self.config.dry_run:
            logger.warning("⚠️  DRY RUN MODE - No actual launch/terminate commands")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"📡 API Server: http://{self.config.api.host}:{self.config.api.port}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Endpoints:")
        logger.info("  POST /jobs              - Submit a training job")
        logger.info("  GET  /jobs              - List all jobs")
        logger.info("  GET  /jobs/<id>         - Get job status")
        logger.info("  POST /jobs/<id>/cancel  - Cancel a job")
        logger.info("  GET  /queue             - View job queue")
        logger.info("  GET  /prices            - View cached prices")
        logger.info("  GET  /health            - Health check")
        logger.info("  GET  /approvals         - List pending approvals")
        logger.info("")
        logger.info("Discord Callback Endpoints:")
        logger.info("  GET /discord/approve/<request_id>  - Approve a launch")
        logger.info("  GET /discord/reject/<request_id>   - Reject a launch")
        logger.info("  GET /discord/status/<request_id>   - Check approval status")
        logger.info("")
        
        # Start API server (blocking)
        if self.config.enable_api:
            self.job_api.run(threaded=True)
    
    def stop(self) -> None:
        """Stop all services"""
        logger.info("Shutting down...")
        if self.price_monitor:
            self.price_monitor.stop()
        if self.job_executor:
            self.job_executor.stop()
        logger.info("Shutdown complete")


def main():
    logger.info("main() called - starting orchestrator...")
    
    parser = argparse.ArgumentParser(
        description="Lambda Labs Orchestrator - Event-driven training job management"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't execute actual launch/terminate commands"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Only run the API server"
    )
    parser.add_argument(
        "--price-poll",
        action="store_true",
        help="Run a single price poll and exit"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    logger.info("Arguments parsed")
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config
    logger.info("Loading config...")
    config = load_config()
    logger.info(f"Config loaded - Redis: {config.redis.host}:{config.redis.port}")
    logger.info(f"Lambda API Key: {'configured' if config.lambda_labs.api_key else 'NOT SET'}")
    
    if args.dry_run:
        config.dry_run = True
    
    if args.api_only:
        config.enable_price_monitor = False
        config.enable_job_executor = False
    
    # Single price poll mode
    if args.price_poll:
        cache = CacheService(config.redis)
        price_monitor = PriceMonitorService(config.lambda_labs, cache)
        prices = price_monitor.poll_once()
        
        print("\n📊 Lambda Labs Instance Availability")
        print("=" * 70)
        available = [p for p in prices if p.available]
        unavailable = [p for p in prices if not p.available]
        
        print("\n✅ Available:")
        for p in sorted(available, key=lambda x: x.price_per_gpu_hour):
            regions = ", ".join(p.regions_available[:2]) if p.regions_available else "unknown"
            print(f"   {p.gpu_type}: {p.gpu_count} GPUs @ ${p.price_per_gpu_hour:.2f}/GPU/hr ({regions})")
        
        if unavailable:
            print("\n❌ No Capacity:")
            for p in sorted(unavailable, key=lambda x: x.gpu_count):
                print(f"   {p.gpu_type}: {p.gpu_count} GPUs @ ${p.price_per_gpu_hour:.2f}/GPU/hr")
        print("=" * 70)
        return
    
    # Create and run orchestrator
    logger.info("Creating orchestrator...")
    orchestrator = Orchestrator(config)
    logger.info("Orchestrator created")
    
    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        orchestrator.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    orchestrator.start()


if __name__ == "__main__":
    main()
