"""
Discord Service
Handles Discord webhook integration for buy approval workflow
"""
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Callable, Dict, Any

import requests

from ..config import DiscordConfig
from ..models.job import TrainingJob, ApprovalRequest, ApprovalStatus, BuyOption
from .cache import CacheService

logger = logging.getLogger(__name__)


class DiscordService:
    """
    Service for Discord integration:
    - Sends approval requests via webhook
    - Tracks pending approvals
    - Notifies of buy results
    """
    
    def __init__(self, config: DiscordConfig, cache: CacheService):
        self.config = config
        self.cache = cache
        self._approval_callbacks: Dict[str, Callable[[ApprovalRequest], None]] = {}
        self._lock = threading.Lock()
    
    def is_enabled(self) -> bool:
        """Check if Discord approval is enabled and configured"""
        return self.config.require_approval and self.config.is_configured
    
    def send_approval_request(
        self,
        job: TrainingJob,
        buy_option: BuyOption
    ) -> Optional[ApprovalRequest]:
        """
        Send an approval request to Discord and return the request object.
        
        The Discord message will contain:
        - Job details (ID, data path)
        - Buy option (GPUs, duration, price)
        - Approve/Reject buttons (via callback URL)
        """
        if not self.config.is_configured:
            logger.warning("Discord not configured, skipping approval request")
            return None
        
        # Create approval request
        request = ApprovalRequest(
            job_id=job.job_id,
            buy_option=buy_option.to_dict(),
        )
        
        # Store in cache
        self.cache.store_approval_request(
            request, 
            ttl_seconds=self.config.approval_timeout_seconds + 300  # Extra buffer
        )
        
        # Build the Discord message
        embed = self._build_approval_embed(job, buy_option, request)
        
        # Send to Discord
        success = self._send_webhook_message(embed, request)
        
        if success:
            logger.info(f"Sent approval request {request.request_id} for job {job.job_id}")
            return request
        else:
            logger.error(f"Failed to send approval request for job {job.job_id}")
            self.cache.delete_approval_request(request.request_id)
            return None
    
    def _build_approval_embed(
        self,
        job: TrainingJob,
        buy_option: BuyOption,
        request: ApprovalRequest
    ) -> Dict[str, Any]:
        """Build Discord embed for approval request"""
        
        # Include secret token in callback URLs to bypass GCP auth
        secret_param = f"?token={self.config.callback_secret}" if self.config.callback_secret else ""
        approve_url = f"{self.config.callback_base_url}/discord/approve/{request.request_id}{secret_param}"
        reject_url = f"{self.config.callback_base_url}/discord/reject/{request.request_id}{secret_param}"
        
        return {
            "embeds": [{
                "title": "🚀 SF Compute Buy Approval Required",
                "description": f"A training job is ready to purchase compute resources.",
                "color": 0x5865F2,  # Discord blurple
                "fields": [
                    {
                        "name": "📋 Job ID",
                        "value": f"`{job.job_id}`",
                        "inline": True
                    },
                    {
                        "name": "📁 Data Path",
                        "value": f"`{job.data_gcs_path[:50]}...`" if len(job.data_gcs_path) > 50 else f"`{job.data_gcs_path}`",
                        "inline": True
                    },
                    {
                        "name": "🎯 GPU Configuration",
                        "value": f"**{buy_option.gpu_count}** GPUs for **{buy_option.duration_hours}** hours",
                        "inline": False
                    },
                    {
                        "name": "💰 Price",
                        "value": f"${buy_option.price_per_gpu_hour:.2f}/GPU/hr → **${buy_option.total_price:.2f}** total",
                        "inline": False
                    },
                    {
                        "name": "⏱️ Max Spend Limit",
                        "value": f"${job.max_spend:.2f}/GPU/hr" if job.max_spend else "No limit",
                        "inline": True
                    },
                    {
                        "name": "🆔 Request ID",
                        "value": f"`{request.request_id}`",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": f"Expires in {self.config.approval_timeout_seconds // 60} minutes"
                },
                "timestamp": datetime.utcnow().isoformat()
            }],
            "components": [{
                "type": 1,  # Action row
                "components": [
                    {
                        "type": 2,  # Button
                        "style": 3,  # Green (Success)
                        "label": "✅ Approve",
                        "custom_id": f"approve_{request.request_id}",
                        "url": approve_url
                    },
                    {
                        "type": 2,  # Button
                        "style": 4,  # Red (Danger)
                        "label": "❌ Reject",
                        "custom_id": f"reject_{request.request_id}",
                        "url": reject_url
                    }
                ]
            }]
        }
    
    def _send_webhook_message(self, payload: Dict[str, Any], request: ApprovalRequest) -> bool:
        """Send message via Discord webhook"""
        if not self.config.webhook_url:
            logger.error("Discord webhook URL not configured")
            return False
        
        try:
            # Note: Discord webhooks don't support interactive buttons directly
            # We'll use link buttons that redirect to our API
            # Remove the components section for webhooks (they don't support it)
            # Instead, add the links to the message
            
            # Include secret token in callback URLs to bypass GCP auth
            secret_param = f"?token={self.config.callback_secret}" if self.config.callback_secret else ""
            approve_url = f"{self.config.callback_base_url}/discord/approve/{request.request_id}{secret_param}"
            reject_url = f"{self.config.callback_base_url}/discord/reject/{request.request_id}{secret_param}"
            
            # Add action links to the embed
            webhook_payload = {
                "embeds": payload["embeds"],
                "content": f"**[✅ APPROVE]({approve_url})** | **[❌ REJECT]({reject_url})**"
            }
            
            response = requests.post(
                self.config.webhook_url,
                json=webhook_payload,
                timeout=10
            )
            
            if response.status_code in [200, 204]:
                return True
            else:
                logger.error(f"Discord webhook failed: {response.status_code} - {response.text}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Failed to send Discord webhook: {e}")
            return False
    
    def send_buy_result(
        self,
        job: TrainingJob,
        success: bool,
        message: str = ""
    ) -> bool:
        """
        Send notification about buy result (success or failure).
        Called after an approved buy is attempted.
        """
        if not self.config.webhook_url:
            return False
        
        try:
            if success:
                embed = {
                    "title": "✅ Buy Successful",
                    "description": f"Job `{job.job_id}` has successfully purchased compute.",
                    "color": 0x57F287,  # Green
                    "fields": [
                        {
                            "name": "Status",
                            "value": job.status.value,
                            "inline": True
                        }
                    ],
                    "timestamp": datetime.utcnow().isoformat()
                }
                if job.buy_option:
                    embed["fields"].append({
                        "name": "Configuration",
                        "value": f"{job.buy_option.get('gpu_count', '?')} GPUs, ${job.buy_option.get('total_price', '?')}",
                        "inline": True
                    })
            else:
                embed = {
                    "title": "❌ Buy Failed",
                    "description": f"Job `{job.job_id}` failed to purchase compute.",
                    "color": 0xED4245,  # Red
                    "fields": [
                        {
                            "name": "Reason",
                            "value": message or job.error_message or "Unknown error",
                            "inline": False
                        },
                        {
                            "name": "Next Steps",
                            "value": "Job remains at front of queue for retry",
                            "inline": False
                        }
                    ],
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            response = requests.post(
                self.config.webhook_url,
                json={"embeds": [embed]},
                timeout=10
            )
            
            return response.status_code in [200, 204]
            
        except requests.RequestException as e:
            logger.error(f"Failed to send buy result notification: {e}")
            return False
    
    def send_rejection_notification(self, job: TrainingJob, request: ApprovalRequest) -> bool:
        """Send notification that a buy was rejected"""
        if not self.config.webhook_url:
            return False
        
        try:
            embed = {
                "title": "🚫 Buy Rejected",
                "description": f"Job `{job.job_id}` buy request was rejected.",
                "color": 0xFEE75C,  # Yellow
                "fields": [
                    {
                        "name": "Action",
                        "value": "Job moved to back of queue",
                        "inline": False
                    }
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = requests.post(
                self.config.webhook_url,
                json={"embeds": [embed]},
                timeout=10
            )
            
            return response.status_code in [200, 204]
            
        except requests.RequestException as e:
            logger.error(f"Failed to send rejection notification: {e}")
            return False
    
    def handle_approval_response(
        self,
        request_id: str,
        approved: bool
    ) -> tuple[bool, Optional[ApprovalRequest], str]:
        """
        Handle an approval response from Discord callback.
        
        Returns:
            (success, request, message)
        """
        request = self.cache.get_approval_request(request_id)
        
        if not request:
            return False, None, "Approval request not found or expired"
        
        if request.status != ApprovalStatus.PENDING:
            return False, request, f"Request already {request.status.value}"
        
        # Update the request
        request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        request.responded_at = datetime.utcnow().isoformat()
        self.cache.update_approval_request(request)
        
        action = "approved" if approved else "rejected"
        logger.info(f"Approval request {request_id} {action}")
        
        return True, request, f"Request {action} successfully"
    
    def wait_for_approval(
        self,
        request: ApprovalRequest,
        check_interval: float = 2.0
    ) -> ApprovalStatus:
        """
        Wait for an approval response with timeout.
        
        Returns the final status (APPROVED, REJECTED, or EXPIRED)
        """
        start_time = time.time()
        timeout = self.config.approval_timeout_seconds
        
        while time.time() - start_time < timeout:
            # Check current status
            current = self.cache.get_approval_request(request.request_id)
            
            if not current:
                logger.warning(f"Approval request {request.request_id} disappeared")
                return ApprovalStatus.EXPIRED
            
            if current.status != ApprovalStatus.PENDING:
                return current.status
            
            time.sleep(check_interval)
        
        # Timeout reached - mark as expired
        request.status = ApprovalStatus.EXPIRED
        request.responded_at = datetime.utcnow().isoformat()
        self.cache.update_approval_request(request)
        
        logger.warning(f"Approval request {request.request_id} expired after {timeout}s")
        return ApprovalStatus.EXPIRED

