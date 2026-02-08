#!/usr/bin/env python3
"""
InnatePolicy: Diffusion-based policy with DINOv2 vision encoder and flow matching loss.

Architecture:
- Vision Encoder: DINOv2 Small with learned attention pooling
- Action Decoder: 1D UNet with conditional residual blocks
- Training: Flow matching loss instead of traditional diffusion loss
"""

import math
import torch
import torch.nn as nn
from typing import Union, Optional, Dict, Tuple


# ============================================================================
# Vision Encoder Components
# ============================================================================

class LearnedSpatialPooling(nn.Module):
    """
    Learned attention pooling over patch tokens.
    Introduces K learned query tokens that attend over spatial patch features.
    """
    def __init__(self, embed_dim: int = 384, num_queries: int = 8, num_heads: int = 6):
        """
        Args:
            embed_dim: Dimension of patch embeddings (384 for DINOv2 small)
            num_queries: Number of learned query tokens (K)
            num_heads: Number of attention heads
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        
        # Learned query tokens
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, N, embed_dim] where N is number of patches
            
        Returns:
            pooled: [B, num_queries * embed_dim] flattened pooled features
        """
        B = patch_tokens.shape[0]
        
        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, embed_dim]
        
        # Apply attention: queries attend over patch tokens
        pooled, _ = self.attention(
            queries,  # query: [B, num_queries, embed_dim]
            patch_tokens,  # key: [B, N, embed_dim]
            patch_tokens   # value: [B, N, embed_dim]
        )  # output: [B, num_queries, embed_dim]
        
        # Normalize
        pooled = self.norm(pooled)
        
        # Flatten to [B, num_queries * embed_dim]
        pooled = pooled.reshape(B, -1)
        
        return pooled


class DINOv2VisionEncoder(nn.Module):
    """
    DINOv2 Small vision encoder.
    Extracts patch-level features with camera embeddings.
    Note: Pooling is done separately in MultiCameraAttentionPooling.
    """
    def __init__(self, 
                 pretrained: bool = True,
                 freeze_backbone: bool = True,
                 num_cameras: int = 1):
        """
        Args:
            pretrained: Whether to use pretrained DINOv2 weights
            freeze_backbone: Whether to freeze the DINOv2 backbone
            num_cameras: Number of cameras (for camera embeddings)
        """
        super().__init__()
        
        self.num_cameras = num_cameras
        self.embed_dim = 384  # DINOv2 small embedding dimension
        
        # Load DINOv2 small (384 dim embeddings, 16x16 patches for 224x224 images)
        if pretrained:
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        else:
            # For testing without internet connection
            from torchvision.models import vision_transformer as vit
            self.backbone = vit.vit_b_16(pretrained=False)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Camera embeddings (like positional embeddings, but for cameras)
        # This is ESSENTIAL for multi-camera setups!
        if num_cameras > 1:
            self.camera_embeddings = nn.Parameter(
                torch.randn(num_cameras, self.embed_dim) * 0.02
            )
        else:
            self.camera_embeddings = None
        
    def get_patch_tokens(self, images: torch.Tensor, camera_id: Optional[int] = None) -> torch.Tensor:
        """
        Extract patch tokens from images, optionally adding camera embedding.
        
        Args:
            images: [B, 3, H, W] input images
            camera_id: Optional camera index for adding camera embedding
            
        Returns:
            patch_tokens: [B, num_patches, 384] patch tokens
        """
        # Extract patch tokens from DINOv2
        output = self.backbone.get_intermediate_layers(
            images, n=1, return_class_token=True
        )
        patch_tokens = output[0][0]  # [B, num_patches, 384]
        
        # Add camera embedding if specified
        if camera_id is not None and self.camera_embeddings is not None:
            cam_embed = self.camera_embeddings[camera_id]
            patch_tokens = patch_tokens + cam_embed.unsqueeze(0).unsqueeze(0)
        
        return patch_tokens


# ============================================================================
# Diffusion Network Components (1D UNet)
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Positional encoding for the diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    """Strided convolution to reduce temporal resolution."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """Transposed convolution to increase temporal resolution."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish activation."""
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    Conditional residual block with FiLM conditioning.
    Takes input x and conditioning cond, applies FiLM modulation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels)
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        Args:
            x: [batch_size, in_channels, horizon]
            cond: [batch_size, cond_dim]

        Returns:
            out: [batch_size, out_channels, horizon]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        
        # Reshape to [B, 2, out_channels, 1] for scale and bias
        B = embed.shape[0]
        embed = embed.view(B, 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """
    1D UNet for action sequence prediction with FiLM conditioning.
    """
    def __init__(self,
                 input_dim,
                 global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=5,
                 n_groups=8):
        """
        Args:
            input_dim: Dimension of actions
            global_cond_dim: Dimension of global conditioning (visual features)
            diffusion_step_embed_dim: Size of positional encoding for timestep
            down_dims: Channel sizes for each UNet level
            kernel_size: Conv kernel size
            n_groups: Number of groups for GroupNorm
        """
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # All up blocks upsample to restore temporal resolution
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in)
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("ConditionalUnet1D parameters: {:e}".format(
            sum(p.numel() for p in self.parameters())
        ))

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                global_cond=None):
        """
        Args:
            sample: [B, T, input_dim] action sequence
            timestep: [B,] or scalar, flow timestep in [0, 1]
            global_cond: [B, global_cond_dim] visual features

        Returns:
            output: [B, T, input_dim] predicted velocity field
        """
        # (B, T, C) -> (B, C, T)
        sample = sample.moveaxis(-1, -2)

        # Process timestep - KEEP AS FLOAT for flow matching!
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device, dtype=torch.float32)
        timesteps = timesteps.expand(sample.shape[0]).to(dtype=torch.float32)

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        # UNet forward pass
        x = sample
        h = []
        
        # Downsampling
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Middle
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Upsampling
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B, C, T) -> (B, T, C)
        x = x.moveaxis(-1, -2)
        return x


# ============================================================================
# Flow Matching Loss
# ============================================================================

class FlowMatchingLoss(nn.Module):
    """
    Flow matching loss for continuous normalizing flows.
    
    For the straight interpolation path x_t = t*x_1 + (1-t)*x_0,
    the target velocity is simply: v* = x_1 - x_0
    
    Note: This uses independent random noise coupling (rectified flow),
    not optimal transport coupling.
    
    Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2023)
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, 
                model_output: torch.Tensor,
                target_velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        Args:
            model_output: [B, T, D] predicted velocity field from model
            target_velocity: [B, T, D] target velocity (x_1 - x_0)
            
        Returns:
            loss: scalar loss value
        """
        return torch.nn.functional.mse_loss(model_output, target_velocity, reduction='mean')


# ============================================================================
# Multi-View Fusion: Token Concatenation + Attention Pooling
# ============================================================================

class MultiCameraAttentionPooling(nn.Module):
    """
    Multi-camera fusion via token concatenation + attention pooling.
    
    Steps:
    1. Extract patch tokens from each camera
    2. Add camera-specific learned embeddings to tokens
    3. Concatenate all tokens: [B, C*N, D]
    4. Attention pool with learned queries
    """
    def __init__(self,
                 embed_dim: int = 384,
                 num_queries: int = 16,
                 num_heads: int = 6):
        """
        Args:
            embed_dim: Dimension of patch embeddings (384 for DINOv2 small)
            num_queries: Number of learned pooling queries
            num_heads: Number of attention heads
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        
        # Learned query tokens that attend over ALL camera tokens
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        
        # Multi-head cross-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, all_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_tokens: [B, C*N, embed_dim] concatenated tokens from all cameras
            
        Returns:
            pooled: [B, num_queries * embed_dim] pooled features
        """
        B = all_tokens.shape[0]
        
        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, embed_dim]
        
        # Cross-attention: queries attend over all camera tokens
        pooled, _ = self.attention(queries, all_tokens, all_tokens)
        
        # Normalize
        pooled = self.norm(pooled)
        
        # Flatten
        pooled = pooled.reshape(B, -1)  # [B, num_queries * embed_dim]
        
        return pooled


# ============================================================================
# Main Policy Network
# ============================================================================

class InnatePolicy(nn.Module):
    """
    InnatePolicy: Diffusion-based imitation learning policy.
    
    Architecture:
    - Vision encoder: DINOv2 small with learned attention pooling
    - Proprioception encoder: MLP for robot state
    - Action decoder: Conditional 1D UNet
    - Training: Flow matching loss
    
    Similar to ACT but using diffusion/flow matching instead of transformers.
    """
    def __init__(self,
                 # Vision encoder params
                 num_queries: int = 8,
                 freeze_vision_backbone: bool = True,
                 
                 # Multi-camera params
                 num_cameras: int = 2,
                 
                 # Proprioception params
                 state_dim: int = 6,
                 proprio_hidden_dim: int = 256,
                 
                 # Action space params
                 action_dim: int = 8,
                 action_horizon: int = 16,
                 
                 # UNet params
                 diffusion_step_embed_dim: int = 256,
                 down_dims: list = [256, 512, 1024],
                 kernel_size: int = 5,
                 n_groups: int = 8,
                 
                 # Training params
                 num_inference_steps: int = 10):
        """
        Args:
            num_queries: Number of learned attention queries for pooling
            freeze_vision_backbone: Whether to freeze DINOv2 weights
            num_cameras: Number of cameras (e.g., 2 for wrist+static)
            state_dim: Dimension of proprioceptive state
            proprio_hidden_dim: Hidden dimension for proprioception encoder
            action_dim: Dimension of action space
            action_horizon: Length of action sequence to predict
            diffusion_step_embed_dim: Embedding dim for flow timestep
            down_dims: UNet channel dimensions
            kernel_size: Conv kernel size
            n_groups: GroupNorm groups
            num_inference_steps: Number of sampling steps during inference
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_inference_steps = num_inference_steps
        self.state_dim = state_dim
        self.num_cameras = num_cameras
        
        # Vision encoder with camera embeddings
        self.vision_encoder = DINOv2VisionEncoder(
            pretrained=True,
            freeze_backbone=freeze_vision_backbone,
            num_cameras=num_cameras
        )
        
        # Multi-camera attention pooling
        self.multi_camera_pooling = MultiCameraAttentionPooling(
            embed_dim=384,  # DINOv2 small
            num_queries=num_queries,
            num_heads=6
        )
        vision_output_dim = 384 * num_queries
        
        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(state_dim, proprio_hidden_dim),
            nn.ReLU(),
            nn.Linear(proprio_hidden_dim, proprio_hidden_dim),
            nn.ReLU(),
        )
        
        # Total conditioning dimension
        global_cond_dim = vision_output_dim + proprio_hidden_dim
        
        # Action decoder (conditional UNet)
        self.action_decoder = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups
        )
        
        # Flow matching loss
        self.flow_loss = FlowMatchingLoss()
        
        print(f"\nInnatePolicy initialized:")
        print(f"  Number of cameras: {num_cameras}")
        print(f"  Vision: DINOv2 small + camera embeddings + attention pooling")
        print(f"  Vision output dim: {vision_output_dim} ({num_queries} queries × 384)")
        print(f"  Proprio output dim: {proprio_hidden_dim}")
        print(f"  Total conditioning dim: {global_cond_dim}")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}, Action horizon: {action_horizon}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images with multi-camera token concatenation + attention pooling.
        
        Process:
        1. Extract patch tokens from each camera
        2. Add camera-specific embeddings
        3. Concatenate all tokens
        4. Attention pool with learned queries
        
        Args:
            images: [B, C, 3, H, W] multi-camera images (C = num_cameras)
            
        Returns:
            features: [B, num_queries * 384] pooled visual features
        """
        if images.ndim == 5:
            B, C, _, H, W = images.shape
            assert C == self.num_cameras, f"Expected {self.num_cameras} cameras, got {C}"
            
            # Process each camera and add camera embeddings
            all_tokens = []
            for cam_idx in range(C):
                cam_images = images[:, cam_idx]  # [B, 3, H, W]
                
                # Extract patch tokens with camera embedding
                patch_tokens = self.vision_encoder.get_patch_tokens(
                    cam_images, camera_id=cam_idx
                )  # [B, num_patches, 384]
                
                all_tokens.append(patch_tokens)
            
            # Concatenate tokens from all cameras
            all_tokens = torch.cat(all_tokens, dim=1)  # [B, C*num_patches, 384]
            
            # Attention pool over all tokens
            features = self.multi_camera_pooling(all_tokens)  # [B, num_queries * 384]
            
        elif images.ndim == 4:
            # Single camera - still works
            patch_tokens = self.vision_encoder.get_patch_tokens(
                images, camera_id=0 if self.num_cameras > 1 else None
            )
            features = self.multi_camera_pooling(patch_tokens)
        else:
            raise ValueError(f"Expected images shape [B, C, 3, H, W] or [B, 3, H, W], got {images.shape}")
            
        return features
    
    def encode_proprio(self, robot_state: torch.Tensor) -> torch.Tensor:
        """
        Encode proprioceptive robot state.
        
        Args:
            robot_state: [B, state_dim] robot state (joint positions, velocities, etc.)
            
        Returns:
            features: [B, proprio_hidden_dim] proprioceptive features
        """
        return self.proprio_encoder(robot_state)
    
    def forward(self,
                images: torch.Tensor,
                robot_state: torch.Tensor,
                actions: torch.Tensor,
                training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            images: [B, 3, H, W] input images
            robot_state: [B, state_dim] proprioceptive robot state
            actions: [B, T, action_dim] ground truth actions
            training: whether in training mode
            
        Returns:
            dict with 'loss' and 'predictions'
        """
        B = images.shape[0]
        device = images.device
        
        # Encode visual features
        visual_features = self.encode_images(images)  # [B, vision_dim]
        
        # Encode proprioceptive features
        proprio_features = self.encode_proprio(robot_state)  # [B, proprio_dim]
        
        # Concatenate visual and proprioceptive features
        global_cond = torch.cat([visual_features, proprio_features], dim=-1)  # [B, total_cond_dim]
        
        if training:
            # Sample random timesteps t ~ Uniform(0, 1)
            t = torch.rand(B, device=device)
            
            # Sample noise x_0 ~ N(0, I)
            noise = torch.randn_like(actions)
            
            # Interpolate: x_t = t * x_1 + (1 - t) * x_0
            t_expanded = t.reshape(-1, 1, 1)  # [B, 1, 1]
            noisy_actions = t_expanded * actions + (1 - t_expanded) * noise
            
            # Predict velocity field
            predicted_velocity = self.action_decoder(
                sample=noisy_actions,
                timestep=t,
                global_cond=global_cond
            )
            
            # Target velocity: v* = x_1 - x_0 (straight path from noise to data)
            target_velocity = actions - noise
            
            # Compute flow matching loss
            loss = self.flow_loss(predicted_velocity, target_velocity)
            
            return {
                'loss': loss,
                'predictions': predicted_velocity
            }
        else:
            # Inference: sample actions using flow matching
            predicted_actions = self.sample_actions(global_cond)
            return {
                'predictions': predicted_actions
            }
    
    @torch.no_grad()
    def sample_actions(self, global_cond: torch.Tensor, use_heun: bool = True) -> torch.Tensor:
        """
        Sample actions using flow matching with Euler or Heun integration.
        
        Args:
            global_cond: [B, cond_dim] concatenated visual + proprioceptive features
            use_heun: If True, use Heun's method (RK2) for better quality
            
        Returns:
            actions: [B, action_horizon, action_dim] sampled actions
        """
        B = global_cond.shape[0]
        device = global_cond.device
        
        # Start from noise x_0 ~ N(0, I)
        x_t = torch.randn(B, self.action_horizon, self.action_dim, device=device)
        
        # Create timesteps from 0 to 1
        steps = self.num_inference_steps
        timesteps = torch.linspace(0.0, 1.0, steps, device=device)
        
        for i in range(steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            
            t_batch = t.expand(B)
            
            if use_heun:
                # Heun's method (RK2) - more accurate
                # Predict velocity at current point
                v1 = self.action_decoder(x_t, t_batch, global_cond)
                
                # Euler step to get intermediate point
                x_euler = x_t + dt * v1
                
                # Predict velocity at intermediate point
                t_next_batch = t_next.expand(B)
                v2 = self.action_decoder(x_euler, t_next_batch, global_cond)
                
                # Average velocities for final step
                x_t = x_t + 0.5 * dt * (v1 + v2)
            else:
                # Simple Euler method
                v_t = self.action_decoder(x_t, t_batch, global_cond)
                x_t = x_t + dt * v_t
        
        return x_t
    
    def get_action(self, 
                   images: torch.Tensor,
                   robot_state: torch.Tensor,
                   **kwargs) -> torch.Tensor:
        """
        Get action for inference (convenience method).
        
        Args:
            images: [B, 3, H, W] input images
            robot_state: [B, state_dim] proprioceptive robot state
            
        Returns:
            actions: [B, action_horizon, action_dim] predicted actions
        """
        self.eval()
        visual_features = self.encode_images(images)
        proprio_features = self.encode_proprio(robot_state)
        global_cond = torch.cat([visual_features, proprio_features], dim=-1)
        actions = self.sample_actions(global_cond)
        return actions


# ============================================================================
# Testing and Example Usage
# ============================================================================

def test_innate_policy():
    """Test InnatePolicy with dummy data."""
    print("Testing InnatePolicy...")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Create policy
    policy = InnatePolicy(
        num_queries=16,
        freeze_vision_backbone=True,
        num_cameras=2,  # wrist + static camera
        state_dim=14,
        proprio_hidden_dim=256,
        action_dim=7,
        action_horizon=16,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        num_inference_steps=10
    ).to(device)
    
    # Dummy data
    batch_size = 4
    robot_state = torch.randn(batch_size, 14, device=device)
    actions = torch.randn(batch_size, 16, 7, device=device)
    
    # Test with multi-camera (the main use case)
    print("\n[Test 1] Multi-camera input (2 cameras):")
    multi_cam_images = torch.randn(batch_size, 2, 3, 224, 224, device=device)
    print(f"  Images shape: {multi_cam_images.shape}")
    print(f"  Robot state: {robot_state.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"\n  Process:")
    print(f"    1. Extract tokens from camera 0 (static): [4, 256, 384]")
    print(f"    2. Add camera 0 embedding → [4, 256, 384]")
    print(f"    3. Extract tokens from camera 1 (wrist): [4, 256, 384]")
    print(f"    4. Add camera 1 embedding → [4, 256, 384]")
    print(f"    5. Concatenate: [4, 512, 384]")
    print(f"    6. Attention pool with 16 queries → [4, 16*384] = [4, 6144]")
    
    policy.train()
    output = policy(multi_cam_images, robot_state, actions, training=True)
    print(f"\n  Loss: {output['loss'].item():.4f}")
    print(f"  Predictions shape: {output['predictions'].shape}")
    
    # Test inference
    print("\n[Test 2] Inference mode:")
    policy.eval()
    predicted_actions = policy.get_action(multi_cam_images, robot_state)
    print(f"  Predicted actions shape: {predicted_actions.shape}")
    
    # Test with single camera (should still work)
    print("\n[Test 3] Single camera input:")
    single_cam_images = torch.randn(batch_size, 3, 224, 224, device=device)
    print(f"  Images shape: {single_cam_images.shape}")
    policy.train()
    output = policy(single_cam_images, robot_state, actions, training=True)
    print(f"  Loss: {output['loss'].item():.4f}")
    
    # Test with different number of queries
    print("\n[Test 4] Effect of num_queries:")
    for num_q in [4, 8, 16, 32]:
        policy_test = InnatePolicy(
            num_queries=num_q,
            freeze_vision_backbone=True,
            num_cameras=2,
            state_dim=14,
            proprio_hidden_dim=256,
            action_dim=7,
            action_horizon=16,
            num_inference_steps=10
        ).to(device)
        
        num_params = sum(p.numel() for p in policy_test.parameters())
        vision_dim = num_q * 384
        print(f"  num_queries={num_q:2d} → vision_dim={vision_dim:5d}, total_params={num_params:,}")
    
    print("\n" + "=" * 70)
    print("✓ InnatePolicy test completed successfully!")
    print("\nArchitecture summary:")
    print("  1. Extract patch tokens from each camera (256 patches × 384 dim)")
    print("  2. Add camera-specific learned embeddings")
    print("  3. Concatenate all tokens (2 cameras × 256 = 512 tokens)")
    print("  4. Attention pool with learned queries")
    print("=" * 70)


if __name__ == '__main__':
    test_innate_policy()
