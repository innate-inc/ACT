"""
Price Monitor Service
Continuously polls Lambda Labs for pricing and availability information
"""
import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from ..config import LambdaConfig, RedisConfig
from ..models.job import PriceData
from .cache import CacheService
from .lambda_client import LambdaLabsClient, LambdaInstanceType, LambdaAPIError

logger = logging.getLogger(__name__)


class PriceMonitorService:
    """
    Service that continuously polls Lambda Labs for pricing data.
    Stores results in Redis cache for the selector to consume.
    
    Unlike SF Compute, Lambda Labs has:
    - Fixed pricing (per instance type)
    - Dynamic availability (varies by region)
    - Instance types rather than GPU counts + durations
    """
    
    def __init__(self, lambda_config: LambdaConfig, cache: CacheService):
        self.lambda_config = lambda_config
        self.cache = cache
        self.client = LambdaLabsClient(lambda_config.api_key)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_poll_time: Optional[datetime] = None
        self._cached_types: Dict[str, LambdaInstanceType] = {}
    
    def start(self) -> None:
        """Start the price monitoring loop in a background thread"""
        if self._running:
            logger.warning("Price monitor already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Price monitor started (Lambda Labs)")
    
    def stop(self) -> None:
        """Stop the price monitoring loop"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Price monitor stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        # Initial poll immediately
        self._poll_prices()
        
        while self._running:
            try:
                time.sleep(self.lambda_config.poll_interval_seconds)
                self._poll_prices()
            except Exception as e:
                logger.error(f"Error polling prices: {e}")
                time.sleep(30)  # Wait before retry on error
    
    def _poll_prices(self) -> None:
        """Poll Lambda Labs for current prices and availability"""
        logger.info("Polling Lambda Labs for instance types...")
        
        try:
            instance_types = self.client.list_instance_types()
            self._last_poll_time = datetime.utcnow()
            
            # Filter to GPU instances only
            gpu_types = [
                it for it in instance_types 
                if it.gpus >= self.lambda_config.min_gpus
                and it.gpus <= self.lambda_config.max_gpus
            ]
            
            logger.info(f"Found {len(gpu_types)} GPU instance types ({len(instance_types)} total)")
            
            for it in gpu_types:
                # Store in local cache for quick lookup
                self._cached_types[it.name] = it
                
                # Convert to PriceData format for compatibility
                price_data = self._instance_type_to_price_data(it)
                
                if price_data:
                    self.cache.store_price(price_data)
                    self.cache.append_price_history(price_data)
                    
                    # Store availability
                    self.cache.store_availability(
                        it.name,  # instance type as "gpu_type"
                        it.gpus,
                        it.is_available
                    )
                    
                    status = "✅" if it.is_available else "❌"
                    regions_str = ", ".join(it.regions_available[:3]) if it.regions_available else "none"
                    logger.info(
                        f"{status} {it.name}: {it.gpus} GPUs @ ${it.price_per_gpu_hour:.2f}/gpu/hr "
                        f"(${it.price_per_hour:.2f}/hr) - regions: {regions_str}"
                    )
            
            logger.info("Price poll complete")
            
        except LambdaAPIError as e:
            logger.error(f"Lambda API error: {e}")
        except Exception as e:
            logger.error(f"Error polling Lambda Labs: {e}")
    
    def _instance_type_to_price_data(self, it: LambdaInstanceType) -> Optional[PriceData]:
        """
        Convert Lambda InstanceType to PriceData format.
        
        Lambda Labs uses hourly pricing per instance (not per GPU-hour contracts).
        We normalize to per-GPU-hour for comparison.
        """
        if it.gpus == 0:
            return None
        
        return PriceData(
            gpu_type=it.name,  # Use instance type name as the identifier
            gpu_count=it.gpus,
            duration_hours=1,  # Lambda Labs is hourly (no fixed duration)
            price_per_gpu_hour=it.price_per_gpu_hour,
            available=it.is_available,
            timestamp=datetime.utcnow().isoformat(),
            # Store extra info in metadata
            metadata={
                "instance_type": it.name,
                "description": it.description,
                "gpu_description": it.gpu_description,
                "price_per_hour": it.price_per_hour,
                "regions_available": it.regions_available,
                "vcpus": it.vcpus,
                "memory_gib": it.memory_gib,
                "storage_gib": it.storage_gib,
            }
        )
    
    def get_current_prices(self) -> List[PriceData]:
        """Get all current cached prices"""
        return self.cache.get_all_prices()
    
    def get_best_options(
        self, 
        min_gpus: int = 1, 
        max_gpus: int = 8, 
        max_duration: int = 24
    ) -> List[PriceData]:
        """
        Get available options sorted by price per GPU hour.
        
        Args:
            min_gpus: Minimum GPUs required
            max_gpus: Maximum GPUs to consider
            max_duration: Not used for Lambda (hourly billing)
        
        Returns:
            List of available options sorted by price (cheapest first)
        """
        prices = self.get_current_prices()
        
        # Filter by requirements
        valid = [
            p for p in prices
            if p.available
            and min_gpus <= p.gpu_count <= max_gpus
            and p.price_per_gpu_hour <= self.lambda_config.max_price_per_gpu_hour
        ]
        
        # Sort by price per GPU hour
        return sorted(valid, key=lambda p: p.price_per_gpu_hour)
    
    def get_instance_type(self, name: str) -> Optional[LambdaInstanceType]:
        """Get cached instance type by name"""
        return self._cached_types.get(name)
    
    def get_available_instance_types(self) -> List[LambdaInstanceType]:
        """Get all available instance types from cache"""
        return [it for it in self._cached_types.values() if it.is_available]
    
    def poll_once(self) -> List[PriceData]:
        """Perform a single poll and return results"""
        self._poll_prices()
        return self.get_current_prices()
    
    def get_best_region_for_instance_type(self, instance_type_name: str) -> Optional[str]:
        """
        Get the best available region for a given instance type.
        
        Prefers regions in the configured order.
        """
        it = self._cached_types.get(instance_type_name)
        if not it or not it.regions_available:
            return None
        
        # Check preferred regions first
        for region in self.lambda_config.preferred_regions:
            if region in it.regions_available:
                return region
        
        # Fall back to first available
        return it.regions_available[0] if it.regions_available else None
