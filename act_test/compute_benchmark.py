#!/usr/bin/env python3
"""
Compute Benchmark for ACT Model
Measures forward and backward pass timing on GPU 0

This script generates random batches matching your training configuration and
repeatedly runs forward and backward passes to measure compute performance.

Batch Structure (with batch_size=96):
- observation.image_camera_1: [96, 3, 224, 224]  # Camera 1 RGB images (224x224)
- observation.image_camera_2: [96, 3, 224, 224]  # Camera 2 RGB images (224x224)
- observation.state:          [96, 6]             # Robot joint states (qpos)
- action (target):            [96, 30, 10]        # Action sequences (chunk_size=30)

Note: 
- Unlike typical ACT configurations, the dataloader produces single-timestep
  observations (no n_obs_steps dimension), not temporal sequences.
- Uses BF16 (bfloat16) mixed precision training for faster computation

The script provides detailed timing statistics for:
- Forward pass
- Backward pass
- Total iteration time
- Throughput (samples/sec, batches/sec)
- GPU memory usage
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from ACT import ACTPolicy, ACTConfig


def generate_random_batch(batch_size: int, config: ACTConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Generate a random batch of data matching the model's expected input format.
    
    Note: The dataloader produces batches WITHOUT n_obs_steps dimension:
    - Images: (batch_size, channels, height, width)
    - State: (batch_size, state_dim)
    - Action: (batch_size, chunk_size, action_dim)
    """
    batch = {}
    
    # Generate image observations if present
    # Shape: (batch_size, channels, height, width) - NO n_obs_steps dimension
    for img_key in config.image_input_keys:
        img_shape = config.input_shapes[img_key]  # e.g., [3, 480, 640]
        batch[img_key] = torch.randn(
            batch_size, img_shape[0], img_shape[1], img_shape[2],
            device=device, requires_grad=False
        )
    
    # Generate state observation if present
    # Shape: (batch_size, state_dim) - NO n_obs_steps dimension
    if "observation.state" in config.input_shapes:
        state_dim = config.input_shapes["observation.state"][0]
        batch["observation.state"] = torch.randn(
            batch_size, state_dim,
            device=device, requires_grad=False
        )
    
    # Generate action targets for training (backward pass)
    # Shape: (batch_size, chunk_size, action_dim)
    if "action" in config.output_shapes:
        action_dim = config.output_shapes["action"][0]
        batch["action"] = torch.randn(
            batch_size, config.chunk_size, action_dim,
            device=device, requires_grad=False
        )
    
    return batch


def benchmark_model(
    model: nn.Module,
    batch_size: int,
    config: ACTConfig,
    device: torch.device,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    use_bf16: bool = False
):
    """
    Benchmark forward and backward pass timing.
    
    Args:
        model: The model to benchmark
        batch_size: Number of samples per batch
        config: Model configuration
        device: Device to run on
        num_iterations: Number of iterations to benchmark
        warmup_iterations: Number of warmup iterations (not counted)
        use_bf16: Whether to use bfloat16 mixed precision
    """
    print(f"\n{'='*80}")
    print(f"COMPUTE BENCHMARK - ACT Model")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Precision: {'BF16 (mixed)' if use_bf16 else 'FP32'}")
    print(f"Warmup Iterations: {warmup_iterations}")
    print(f"Benchmark Iterations: {num_iterations}")
    print(f"{'='*80}\n")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    forward_times = []
    backward_times = []
    total_times = []
    
    # Warmup phase
    print("🔥 Warming up...")
    for i in range(warmup_iterations):
        batch = generate_random_batch(batch_size, config, device)
        optimizer.zero_grad()
        
        if use_bf16:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = model(batch)
        else:
            loss, loss_dict = model(batch)
        
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 5 == 0:
            print(f"  Warmup iteration {i+1}/{warmup_iterations}")
    
    # Ensure all warmup operations are complete
    torch.cuda.synchronize()
    
    print("\n📊 Running benchmark...")
    
    # Benchmark phase
    for i in range(num_iterations):
        # Generate fresh batch
        batch = generate_random_batch(batch_size, config, device)
        optimizer.zero_grad()
        
        # Measure forward pass
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        
        if use_bf16:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = model(batch)
        else:
            loss, loss_dict = model(batch)
        
        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        forward_time = forward_end - forward_start
        
        # Measure backward pass
        torch.cuda.synchronize()
        backward_start = time.perf_counter()
        
        loss.backward()
        
        torch.cuda.synchronize()
        backward_end = time.perf_counter()
        backward_time = backward_end - backward_start
        
        # Update weights
        optimizer.step()
        
        # Record times
        total_time = forward_time + backward_time
        forward_times.append(forward_time)
        backward_times.append(backward_time)
        total_times.append(total_time)
        
        # Progress update every 10 iterations
        if (i + 1) % 10 == 0:
            avg_total = np.mean(total_times[-10:])
            l1_loss = loss_dict.get('l1_loss', 0)
            kld_loss = loss_dict.get('kld_loss', 0)
            print(f"  Iteration {i+1}/{num_iterations} | "
                  f"Avg time (last 10): {avg_total*1000:.2f} ms | "
                  f"Loss: {loss.item():.4f} (L1: {l1_loss:.4f}, KLD: {kld_loss:.4f})")
    
    # Calculate statistics
    forward_times = np.array(forward_times) * 1000  # Convert to ms
    backward_times = np.array(backward_times) * 1000
    total_times = np.array(total_times) * 1000
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*80}")
    
    print(f"\n📈 Forward Pass Statistics (ms):")
    print(f"  Mean:       {np.mean(forward_times):.3f}")
    print(f"  Median:     {np.median(forward_times):.3f}")
    print(f"  Std Dev:    {np.std(forward_times):.3f}")
    print(f"  Min:        {np.min(forward_times):.3f}")
    print(f"  Max:        {np.max(forward_times):.3f}")
    print(f"  P95:        {np.percentile(forward_times, 95):.3f}")
    print(f"  P99:        {np.percentile(forward_times, 99):.3f}")
    
    print(f"\n📉 Backward Pass Statistics (ms):")
    print(f"  Mean:       {np.mean(backward_times):.3f}")
    print(f"  Median:     {np.median(backward_times):.3f}")
    print(f"  Std Dev:    {np.std(backward_times):.3f}")
    print(f"  Min:        {np.min(backward_times):.3f}")
    print(f"  Max:        {np.max(backward_times):.3f}")
    print(f"  P95:        {np.percentile(backward_times, 95):.3f}")
    print(f"  P99:        {np.percentile(backward_times, 99):.3f}")
    
    print(f"\n🔄 Total (Forward + Backward) Statistics (ms):")
    print(f"  Mean:       {np.mean(total_times):.3f}")
    print(f"  Median:     {np.median(total_times):.3f}")
    print(f"  Std Dev:    {np.std(total_times):.3f}")
    print(f"  Min:        {np.min(total_times):.3f}")
    print(f"  Max:        {np.max(total_times):.3f}")
    print(f"  P95:        {np.percentile(total_times, 95):.3f}")
    print(f"  P99:        {np.percentile(total_times, 99):.3f}")
    
    print(f"\n⚡ Throughput:")
    samples_per_sec = (batch_size * num_iterations) / (np.sum(total_times) / 1000)
    print(f"  Samples/sec:     {samples_per_sec:.2f}")
    print(f"  Batches/sec:     {num_iterations / (np.sum(total_times) / 1000):.2f}")
    print(f"  Time per sample: {np.mean(total_times) / batch_size:.3f} ms")
    
    print(f"\n💾 Memory Usage:")
    print(f"  Allocated:  {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    print(f"  Reserved:   {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    print(f"  Max Allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
    
    print(f"\n{'='*80}\n")
    
    return {
        'forward_times': forward_times,
        'backward_times': backward_times,
        'total_times': total_times,
    }


def create_dummy_dataset_stats(config: ACTConfig, device: torch.device) -> Dict:
    """
    Create dummy dataset statistics for benchmarking.
    Uses zeros for mean and ones for std to avoid normalization issues.
    
    Matches the exact structure from calculate_webdataset_stats in data_utils.py:
    - Image stats: mean and std are (channels, 1, 1) for broadcasting
    - State/Action stats: mean and std are (dim,) 1D tensors
    """
    dataset_stats = {}
    
    # Image statistics - shape (channels, 1, 1) for broadcasting
    for img_key in config.image_input_keys:
        img_shape = config.input_shapes[img_key]
        channels = img_shape[0]
        dataset_stats[img_key] = {
            "mean": torch.zeros(channels, 1, 1, device=device),
            "std": torch.ones(channels, 1, 1, device=device),
        }
    
    # State statistics - shape (state_dim,) 
    if "observation.state" in config.input_shapes:
        state_dim = config.input_shapes["observation.state"][0]
        dataset_stats["observation.state"] = {
            "mean": torch.zeros(state_dim, device=device),
            "std": torch.ones(state_dim, device=device),
        }
    
    # Action statistics - shape (action_dim,)
    if "action" in config.output_shapes:
        action_dim = config.output_shapes["action"][0]
        dataset_stats["action"] = {
            "mean": torch.zeros(action_dim, device=device),
            "std": torch.ones(action_dim, device=device),
        }
    
    return dataset_stats


def main():
    # Configuration - matching train_dist.py exactly
    BATCH_SIZE = 96
    NUM_ITERATIONS = 100
    WARMUP_ITERATIONS = 10
    
    # Image and action dimensions - using 224x224 for faster training
    IMAGE_H = 224
    IMAGE_W = 224
    IMAGE_C = 3
    QPOS_DIM = 6
    ACTION_DIM = 10
    CHUNK_SIZE = 30  # Default from train_dist.py
    
    # Training precision
    USE_BF16 = True  # Use bfloat16 mixed precision training
    
    # Use GPU 0
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA is not available!")
        sys.exit(1)
    
    device = torch.device("cuda:0")
    print(f"🎮 Using device: {device}")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"   Precision: {'BF16 (mixed)' if USE_BF16 else 'FP32'}")
    
    # Create model configuration (matching train_dist.py exactly)
    config = ACTConfig(
        # Input/output structure
        n_obs_steps=1,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        input_shapes={
            "observation.image_camera_1": [IMAGE_C, IMAGE_H, IMAGE_W],
            "observation.image_camera_2": [IMAGE_C, IMAGE_H, IMAGE_W],
            "observation.state": [QPOS_DIM],
        },
        output_shapes={
            "action": [ACTION_DIM],
        },
        
        # Architecture
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        
        # Transformer
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        n_encoder_layers=4,
        n_decoder_layers=4,
        
        # VAE
        use_vae=True,
        latent_dim=32,
        n_vae_encoder_layers=4,
        
        # Training
        dropout=0.1,
        kl_weight=10.0,
    )
    
    print("\n🏗️  Creating dummy dataset statistics for benchmarking...")
    dataset_stats = create_dummy_dataset_stats(config, device)
    
    print("🏗️  Building model...")
    model = ACTPolicy(config, dataset_stats=dataset_stats)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Compile the model for optimized performance
    print("\n⚡ Compiling model with torch.compile()...")
    print("   Mode: default (with inductor backend)")
    compile_start = time.perf_counter()
    try:
        # Use default inductor backend without suppressing errors
        model = torch.compile(model, mode='default')
        compile_time = time.perf_counter() - compile_start
        print(f"   ✓ Compilation setup completed in {compile_time:.2f}s")
        print("   Note: First forward/backward pass will trigger actual compilation")
    except Exception as e:
        print(f"   ⚠️  Compilation failed: {e}")
        print("   Falling back to uncompiled model")
        compile_time = 0
    
    print(f"\n📦 Batch Structure (batch_size={BATCH_SIZE}):")
    print(f"   observation.image_camera_1: [{BATCH_SIZE}, {IMAGE_C}, {IMAGE_H}, {IMAGE_W}]")
    print(f"   observation.image_camera_2: [{BATCH_SIZE}, {IMAGE_C}, {IMAGE_H}, {IMAGE_W}]")
    print(f"   observation.state:          [{BATCH_SIZE}, {QPOS_DIM}]")
    print(f"   action (target):            [{BATCH_SIZE}, {CHUNK_SIZE}, {ACTION_DIM}]")
    
    # Run benchmark
    results = benchmark_model(
        model=model,
        batch_size=BATCH_SIZE,
        config=config,
        device=device,
        num_iterations=NUM_ITERATIONS,
        warmup_iterations=WARMUP_ITERATIONS,
        use_bf16=USE_BF16
    )
    
    print("✅ Benchmark completed successfully!")


if __name__ == "__main__":
    main()

