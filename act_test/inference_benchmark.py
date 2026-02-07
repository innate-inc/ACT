#!/usr/bin/env python3
"""
Inference Benchmark for ACT Model
Measures forward pass timing during inference on GPU 0

This script generates random batches matching your training configuration and
repeatedly runs forward passes in inference mode to measure inference performance.

Batch Structure (with batch_size=1):
- observation.image_camera_1: [1, 3, 224, 224]  # Camera 1 RGB images (224x224)
- observation.image_camera_2: [1, 3, 224, 224]  # Camera 2 RGB images (224x224)
- observation.state:          [1, 6]             # Robot joint states (qpos)

Note: 
- Unlike training, inference does not require action targets
- Uses model.eval() to ensure no gradients are computed
- Uses torch.compile() for optimized inference performance
- Measures latency (time per sample/batch)

The script provides detailed timing statistics for:
- Forward pass (inference)
- Latency statistics (percentiles)
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


def create_act_config(action_dim=8):
    """Create ACT configuration matching the behavior server setup."""
    input_shapes = {
        "observation.image_camera_1": [3, 224, 224],
        "observation.image_camera_2": [3, 224, 224],
        "observation.state": [6],
    }

    output_shapes = {
        "action": [action_dim],
    }

    return ACTConfig(
        n_obs_steps=1,
        chunk_size=30,
        n_action_steps=1,
        speed=1.0,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        vision_backbone="resnet18",
        replace_final_stride_with_dilation=False,
        pre_norm=False,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        n_encoder_layers=4,
        n_decoder_layers=4,
        use_vae=True,
        dropout=0.1,
        kl_weight=10.0,
        temporal_ensemble_coeff=0.01,
        optimizer_lr=1e-5,
        optimizer_weight_decay=1e-4,
        optimizer_lr_backbone=1e-5,
    )


def generate_random_batch(batch_size: int, config: ACTConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Generate a random batch of data matching the model's expected input format for inference.
    
    Note: Only input observations are needed for inference, no action targets.
    - Images: (batch_size, channels, height, width)
    - State: (batch_size, state_dim)
    """
    batch = {}
    
    # Generate image observations if present
    # Shape: (batch_size, channels, height, width)
    for img_key in config.image_input_keys:
        img_shape = config.input_shapes[img_key]  # e.g., [3, 480, 640]
        batch[img_key] = torch.randn(
            batch_size, img_shape[0], img_shape[1], img_shape[2],
            device=device, requires_grad=False
        )
    
    # Generate state observation if present
    # Shape: (batch_size, state_dim)
    if "observation.state" in config.input_shapes:
        state_dim = config.input_shapes["observation.state"][0]
        batch["observation.state"] = torch.randn(
            batch_size, state_dim,
            device=device, requires_grad=False
        )
    
    return batch


def benchmark_inference(
    model: nn.Module,
    batch_size: int,
    config: ACTConfig,
    device: torch.device,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    use_compile: bool = True
):
    """
    Benchmark forward pass timing for inference.
    
    Args:
        model: The model to benchmark
        batch_size: Number of samples per batch
        config: Model configuration
        device: Device to run on
        num_iterations: Number of iterations to benchmark
        warmup_iterations: Number of warmup iterations (not counted)
        use_compile: Whether to use torch.compile for optimization
    """
    print(f"\n{'='*80}")
    print(f"INFERENCE BENCHMARK - ACT Model")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Model Mode: eval (inference)")
    print(f"Compilation: {'enabled' if use_compile else 'disabled'}")
    print(f"Warmup Iterations: {warmup_iterations}")
    print(f"Benchmark Iterations: {num_iterations}")
    print(f"{'='*80}\n")
    
    # Set model to inference mode (no gradients, no dropout)
    model.eval()
    
    forward_times = []
    
    # Warmup phase
    print("🔥 Warming up...")
    for i in range(warmup_iterations):
        batch = generate_random_batch(batch_size, config, device)
        
        with torch.no_grad():  # No gradient computation for inference
            _ = model.select_action(batch)  # Use select_action for inference
        
        if (i + 1) % 5 == 0:
            print(f"  Warmup iteration {i+1}/{warmup_iterations}")
    
    # Ensure all warmup operations are complete
    torch.cuda.synchronize()
    
    print("\n📊 Running benchmark...")
    
    # Benchmark phase
    for i in range(num_iterations):
        # Generate fresh batch
        batch = generate_random_batch(batch_size, config, device)
        
        # Measure forward pass (inference)
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        
        with torch.no_grad():  # No gradient computation for inference
            output_action = model.select_action(batch)
        
        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        forward_time = forward_end - forward_start
        
        # Record times
        forward_times.append(forward_time)
        
        # Progress update every 10 iterations
        if (i + 1) % 10 == 0:
            avg_total = np.mean(forward_times[-10:])
            print(f"  Iteration {i+1}/{num_iterations} | "
                  f"Avg time (last 10): {avg_total*1000:.2f} ms")
    
    # Calculate statistics
    forward_times = np.array(forward_times) * 1000  # Convert to ms
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*80}")
    
    print(f"\n📈 Inference Latency Statistics (ms):")
    print(f"  Mean:       {np.mean(forward_times):.3f}")
    print(f"  Median:     {np.median(forward_times):.3f}")
    print(f"  Std Dev:    {np.std(forward_times):.3f}")
    print(f"  Min:        {np.min(forward_times):.3f}")
    print(f"  Max:        {np.max(forward_times):.3f}")
    print(f"  P50:        {np.percentile(forward_times, 50):.3f}")
    print(f"  P75:        {np.percentile(forward_times, 75):.3f}")
    print(f"  P90:        {np.percentile(forward_times, 90):.3f}")
    print(f"  P95:        {np.percentile(forward_times, 95):.3f}")
    print(f"  P99:        {np.percentile(forward_times, 99):.3f}")
    
    print(f"\n⚡ Throughput:")
    samples_per_sec = (batch_size * num_iterations) / (np.sum(forward_times) / 1000)
    print(f"  Samples/sec:     {samples_per_sec:.2f}")
    print(f"  Batches/sec:     {num_iterations / (np.sum(forward_times) / 1000):.2f}")
    print(f"  Time per sample: {np.mean(forward_times) / batch_size:.3f} ms")
    print(f"  Time per batch:  {np.mean(forward_times):.3f} ms")
    
    print(f"\n💾 Memory Usage:")
    print(f"  Allocated:  {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    print(f"  Reserved:   {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    print(f"  Max Allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
    
    print(f"\n{'='*80}\n")
    
    return {
        'forward_times': forward_times,
        'latency_stats': {
            'mean': np.mean(forward_times),
            'median': np.median(forward_times),
            'std': np.std(forward_times),
            'min': np.min(forward_times),
            'max': np.max(forward_times),
            'p50': np.percentile(forward_times, 50),
            'p75': np.percentile(forward_times, 75),
            'p90': np.percentile(forward_times, 90),
            'p95': np.percentile(forward_times, 95),
            'p99': np.percentile(forward_times, 99),
        },
        'throughput': {
            'samples_per_sec': samples_per_sec,
            'batches_per_sec': num_iterations / (np.sum(forward_times) / 1000),
            'time_per_sample_ms': np.mean(forward_times) / batch_size,
            'time_per_batch_ms': np.mean(forward_times),
        }
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
    # Configuration - inference with batch size 1 (online inference)
    BATCH_SIZE = 1
    NUM_ITERATIONS = 1000
    WARMUP_ITERATIONS = 10
    USE_COMPILE = True  # Use torch.compile for optimized inference
    
    # Action dimension from behavior metadata/config
    ACTION_DIM = 10
    
    # Use GPU 0
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA is not available!")
        sys.exit(1)
    
    device = torch.device("cuda:0")
    print(f"🎮 Using device: {device}")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create model configuration matching behavior server
    config = create_act_config(action_dim=ACTION_DIM)
    
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
    if USE_COMPILE:
        print("\n⚡ Compiling model with torch.compile()...")
        print("   Mode: default (with inductor backend)")
        compile_start = time.perf_counter()
        try:
            # Use default inductor backend without suppressing errors
            model = torch.compile(model, mode='default')
            compile_time = time.perf_counter() - compile_start
            print(f"   ✓ Compilation setup completed in {compile_time:.2f}s")
            print("   Note: First forward pass will trigger actual compilation")
        except Exception as e:
            print(f"   ⚠️  Compilation failed: {e}")
            print("   Falling back to uncompiled model")
            compile_time = 0
    else:
        print("\n⚡ Skipping model compilation")
    
    print(f"\n📦 Batch Structure (batch_size={BATCH_SIZE}):")
    print(f"   observation.image_camera_1: [{BATCH_SIZE}, 3, 224, 224]")
    print(f"   observation.image_camera_2: [{BATCH_SIZE}, 3, 224, 224]")
    print(f"   observation.state:          [{BATCH_SIZE}, 6]")
    
    # Run benchmark
    results = benchmark_inference(
        model=model,
        batch_size=BATCH_SIZE,
        config=config,
        device=device,
        num_iterations=NUM_ITERATIONS,
        warmup_iterations=WARMUP_ITERATIONS,
        use_compile=USE_COMPILE
    )
    
    print("✅ Inference benchmark completed successfully!")


if __name__ == "__main__":
    main()

