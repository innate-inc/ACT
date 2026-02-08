#!/usr/bin/env python3
"""
Inference Benchmark for InnatePolicy
Measures forward pass timing during inference on GPU 0

This script generates random batches matching your training configuration and
repeatedly runs forward passes in inference mode to measure inference performance.

Batch Structure (with batch_size=1):
- images (multi-camera): [1, 2, 3, 224, 224]  # 2 cameras, RGB, 224x224
- robot_state:           [1, 6]                # Robot joint states (qpos)

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
from torch.profiler import profile, record_function, ProfilerActivity

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from innate_policy import InnatePolicy


def create_innate_policy_config(
    num_cameras: int = 2,
    state_dim: int = 6,
    action_dim: int = 8,
    action_horizon: int = 16,
    num_queries: int = 16,
    freeze_vision_backbone: bool = True,
    num_inference_steps: int = 10
):
    """Create InnatePolicy configuration matching the behavior server setup."""
    return {
        'num_queries': num_queries,
        'freeze_vision_backbone': freeze_vision_backbone,
        'num_cameras': num_cameras,
        'state_dim': state_dim,
        'proprio_hidden_dim': 256,
        'action_dim': action_dim,
        'action_horizon': action_horizon,
        'diffusion_step_embed_dim': 256,
        'down_dims': [256, 512, 1024],
        'kernel_size': 5,
        'n_groups': 8,
        'num_inference_steps': num_inference_steps,
    }


def count_parameters_by_component(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters for each component of the model.
    
    Returns:
        Dictionary mapping component names to parameter counts
    """
    def count_params(module_or_param):
        """Count parameters, handling both modules and Parameter objects."""
        if isinstance(module_or_param, nn.Parameter):
            return module_or_param.numel()
        else:
            return sum(p.numel() for p in module_or_param.parameters())
    
    def count_trainable_params(module_or_param):
        """Count trainable parameters, handling both modules and Parameter objects."""
        if isinstance(module_or_param, nn.Parameter):
            return module_or_param.numel() if module_or_param.requires_grad else 0
        else:
            return sum(p.numel() for p in module_or_param.parameters() if p.requires_grad)
    
    param_counts = {}
    
    # Vision encoder components
    if hasattr(model, 'vision_encoder'):
        param_counts['vision_encoder_total'] = count_params(model.vision_encoder)
        param_counts['vision_encoder_trainable'] = count_trainable_params(model.vision_encoder)
        
        # DINOv2 backbone
        if hasattr(model.vision_encoder, 'backbone'):
            param_counts['dinov2_backbone'] = count_params(model.vision_encoder.backbone)
            param_counts['dinov2_backbone_trainable'] = count_trainable_params(model.vision_encoder.backbone)
        
        # Camera embeddings
        if hasattr(model.vision_encoder, 'camera_embeddings'):
            param_counts['camera_embeddings'] = count_params(model.vision_encoder.camera_embeddings)
    
    # Multi-camera attention pooling
    if hasattr(model, 'multi_camera_pooling'):
        param_counts['multi_camera_pooling'] = count_params(model.multi_camera_pooling)
    
    # Proprioception encoder
    if hasattr(model, 'proprio_encoder'):
        param_counts['proprio_encoder'] = count_params(model.proprio_encoder)
    
    # Action decoder (UNet)
    if hasattr(model, 'action_decoder'):
        param_counts['action_decoder'] = count_params(model.action_decoder)
    
    # Total
    param_counts['total'] = sum(p.numel() for p in model.parameters())
    param_counts['total_trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return param_counts


def print_parameter_breakdown(param_counts: Dict[str, int]):
    """Print parameter counts in a formatted table."""
    print(f"\n{'='*80}")
    print(f"DETAILED PARAMETER BREAKDOWN")
    print(f"{'='*80}\n")
    
    # Vision encoder
    print("📸 Vision Encoder:")
    if 'dinov2_backbone' in param_counts:
        print(f"  DINOv2 Backbone:         {param_counts['dinov2_backbone']:>12,} params")
        print(f"    (trainable):           {param_counts['dinov2_backbone_trainable']:>12,} params")
    if 'camera_embeddings' in param_counts:
        print(f"  Camera Embeddings:       {param_counts['camera_embeddings']:>12,} params")
    if 'vision_encoder_total' in param_counts:
        print(f"  Vision Encoder Total:    {param_counts['vision_encoder_total']:>12,} params")
        print(f"    (trainable):           {param_counts['vision_encoder_trainable']:>12,} params")
    
    print()
    
    # Pooling
    if 'multi_camera_pooling' in param_counts:
        print("🔍 Multi-Camera Attention Pooling:")
        print(f"  Learned Queries:         {param_counts['multi_camera_pooling']:>12,} params")
        print()
    
    # Proprio encoder
    if 'proprio_encoder' in param_counts:
        print("🤖 Proprioception Encoder:")
        print(f"  MLP:                     {param_counts['proprio_encoder']:>12,} params")
        print()
    
    # Action decoder
    if 'action_decoder' in param_counts:
        print("🎯 Action Decoder (UNet):")
        print(f"  Conditional UNet:        {param_counts['action_decoder']:>12,} params")
        print()
    
    # Total
    print("📊 TOTALS:")
    print(f"  Total Parameters:        {param_counts['total']:>12,} params")
    print(f"  Trainable Parameters:    {param_counts['total_trainable']:>12,} params")
    frozen = param_counts['total'] - param_counts['total_trainable']
    print(f"  Frozen Parameters:       {frozen:>12,} params")
    print(f"  Trainable Percentage:    {100.0 * param_counts['total_trainable'] / param_counts['total']:>12.1f}%")
    
    print(f"\n{'='*80}\n")


def generate_random_batch(
    batch_size: int,
    num_cameras: int,
    state_dim: int,
    image_size: int = 224,
    device: torch.device = torch.device('cuda')
) -> Dict[str, torch.Tensor]:
    """
    Generate a random batch of data matching the model's expected input format for inference.
    
    Note: Only input observations are needed for inference, no action targets.
    - Images: (batch_size, num_cameras, channels, height, width)
    - State: (batch_size, state_dim)
    """
    batch = {}
    
    # Generate multi-camera image observations
    # Shape: (batch_size, num_cameras, 3, height, width)
    batch['images'] = torch.randn(
        batch_size, num_cameras, 3, image_size, image_size,
        device=device, requires_grad=False
    )
    
    # Generate state observation
    # Shape: (batch_size, state_dim)
    batch['robot_state'] = torch.randn(
        batch_size, state_dim,
        device=device, requires_grad=False
    )
    
    return batch


def profile_model_components(
    model: nn.Module,
    batch_size: int,
    num_cameras: int,
    state_dim: int,
    device: torch.device,
    num_iterations: int = 100,
    warmup_iterations: int = 10
):
    """
    Profile inference with granular timing for each model component.
    
    This profiles:
    - Vision encoding (per camera + pooling)
    - Proprioception encoding
    - Action sampling (diffusion sampling loop)
      - Individual UNet forward passes
    
    Args:
        model: The model to profile
        batch_size: Number of samples per batch
        num_cameras: Number of cameras
        state_dim: Dimension of robot state
        device: Device to run on
        num_iterations: Number of iterations to profile
        warmup_iterations: Number of warmup iterations
    """
    print(f"\n{'='*80}")
    print(f"GRANULAR COMPONENT PROFILING")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Profiling {num_iterations} iterations after {warmup_iterations} warmup iterations")
    print(f"{'='*80}\n")
    
    model.eval()
    
    # Storage for timing results
    timings = {
        'vision_encoding': [],
        'vision_per_camera': [],
        'vision_pooling': [],
        'proprio_encoding': [],
        'action_sampling': [],
        'unet_forward_passes': [],
        'total': []
    }
    
    # Warmup
    print("🔥 Warming up...")
    for i in range(warmup_iterations):
        batch = generate_random_batch(batch_size, num_cameras, state_dim, device=device)
        with torch.no_grad():
            _ = model.get_action(batch['images'], batch['robot_state'])
    
    torch.cuda.synchronize()
    
    # Profile with manual timing
    print("📊 Running granular profiling...\n")
    
    for i in range(num_iterations):
        batch = generate_random_batch(batch_size, num_cameras, state_dim, device=device)
        images = batch['images']
        robot_state = batch['robot_state']
        
        with torch.no_grad():
            # Total time start
            torch.cuda.synchronize()
            total_start = time.perf_counter()
            
            # ============================================================
            # 1. VISION ENCODING
            # ============================================================
            torch.cuda.synchronize()
            vision_start = time.perf_counter()
            
            # Process each camera
            B, C, _, H, W = images.shape
            all_tokens = []
            camera_times = []
            
            for cam_idx in range(C):
                torch.cuda.synchronize()
                cam_start = time.perf_counter()
                
                cam_images = images[:, cam_idx]
                patch_tokens = model.vision_encoder.get_patch_tokens(
                    cam_images, camera_id=cam_idx
                )
                all_tokens.append(patch_tokens)
                
                torch.cuda.synchronize()
                cam_end = time.perf_counter()
                camera_times.append(cam_end - cam_start)
            
            # Concatenate tokens
            all_tokens = torch.cat(all_tokens, dim=1)
            
            # Attention pooling
            torch.cuda.synchronize()
            pooling_start = time.perf_counter()
            
            visual_features = model.multi_camera_pooling(all_tokens)
            
            torch.cuda.synchronize()
            pooling_end = time.perf_counter()
            pooling_time = pooling_end - pooling_start
            
            torch.cuda.synchronize()
            vision_end = time.perf_counter()
            vision_time = vision_end - vision_start
            
            # ============================================================
            # 2. PROPRIOCEPTION ENCODING
            # ============================================================
            torch.cuda.synchronize()
            proprio_start = time.perf_counter()
            
            proprio_features = model.encode_proprio(robot_state)
            
            torch.cuda.synchronize()
            proprio_end = time.perf_counter()
            proprio_time = proprio_end - proprio_start
            
            # ============================================================
            # 3. ACTION SAMPLING (Diffusion Loop)
            # ============================================================
            global_cond = torch.cat([visual_features, proprio_features], dim=-1)
            
            torch.cuda.synchronize()
            sampling_start = time.perf_counter()
            
            # Manually run sampling to time UNet calls
            x_t = torch.randn(B, model.action_horizon, model.action_dim, device=device)
            steps = model.num_inference_steps
            timesteps = torch.linspace(0.0, 1.0, steps, device=device)
            
            unet_times = []
            
            for step_idx in range(steps - 1):
                t = timesteps[step_idx]
                t_next = timesteps[step_idx + 1]
                dt = t_next - t
                t_batch = t.expand(B)
                
                # First UNet call (Heun's method)
                torch.cuda.synchronize()
                unet_start = time.perf_counter()
                
                v1 = model.action_decoder(x_t, t_batch, global_cond)
                
                torch.cuda.synchronize()
                unet_end = time.perf_counter()
                unet_times.append(unet_end - unet_start)
                
                # Euler step
                x_euler = x_t + dt * v1
                
                # Second UNet call (Heun's method)
                torch.cuda.synchronize()
                unet_start = time.perf_counter()
                
                t_next_batch = t_next.expand(B)
                v2 = model.action_decoder(x_euler, t_next_batch, global_cond)
                
                torch.cuda.synchronize()
                unet_end = time.perf_counter()
                unet_times.append(unet_end - unet_start)
                
                # Average velocities
                x_t = x_t + 0.5 * dt * (v1 + v2)
            
            torch.cuda.synchronize()
            sampling_end = time.perf_counter()
            sampling_time = sampling_end - sampling_start
            
            # Total time end
            torch.cuda.synchronize()
            total_end = time.perf_counter()
            total_time = total_end - total_start
            
            # Store timings
            timings['vision_encoding'].append(vision_time)
            timings['vision_per_camera'].append(np.mean(camera_times))
            timings['vision_pooling'].append(pooling_time)
            timings['proprio_encoding'].append(proprio_time)
            timings['action_sampling'].append(sampling_time)
            timings['unet_forward_passes'].extend(unet_times)
            timings['total'].append(total_time)
        
        if (i + 1) % 20 == 0:
            print(f"  Profiled {i+1}/{num_iterations} iterations...")
    
    # Convert to ms and compute statistics
    def compute_stats(times_list):
        times = np.array(times_list) * 1000  # Convert to ms
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
    
    stats = {key: compute_stats(values) for key, values in timings.items()}
    
    # Print results
    print(f"\n{'='*80}")
    print(f"GRANULAR COMPONENT TIMING RESULTS")
    print(f"{'='*80}\n")
    
    total_mean = stats['total']['mean']
    
    print("🎯 Per-Iteration Breakdown (averaged over all iterations):\n")
    
    print(f"1. VISION ENCODING:              {stats['vision_encoding']['mean']:>8.3f} ms  ({100*stats['vision_encoding']['mean']/total_mean:>5.1f}%)")
    print(f"   ├─ Per Camera (avg):          {stats['vision_per_camera']['mean']:>8.3f} ms")
    print(f"   │  (DINOv2 forward pass)      ")
    print(f"   └─ Multi-Camera Pooling:      {stats['vision_pooling']['mean']:>8.3f} ms")
    print()
    
    print(f"2. PROPRIOCEPTION ENCODING:      {stats['proprio_encoding']['mean']:>8.3f} ms  ({100*stats['proprio_encoding']['mean']/total_mean:>5.1f}%)")
    print(f"   (MLP forward pass)")
    print()
    
    print(f"3. ACTION SAMPLING:              {stats['action_sampling']['mean']:>8.3f} ms  ({100*stats['action_sampling']['mean']/total_mean:>5.1f}%)")
    print(f"   (Diffusion sampling loop)")
    unet_mean = stats['unet_forward_passes']['mean']
    unet_count = len(timings['unet_forward_passes']) // num_iterations
    print(f"   ├─ UNet Forward (single):     {unet_mean:>8.3f} ms")
    print(f"   └─ Total UNet calls:          {unet_count} per iteration")
    print(f"      (Heun method: 2× per step)")
    print()
    
    print(f"{'─'*80}")
    print(f"TOTAL TIME PER ITERATION:        {total_mean:>8.3f} ms")
    print(f"{'─'*80}\n")
    
    # Detailed statistics table
    print(f"{'='*80}")
    print(f"DETAILED STATISTICS (all times in ms)")
    print(f"{'='*80}\n")
    
    print(f"{'Component':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*80}")
    
    for name, stat in stats.items():
        if name == 'unet_forward_passes':
            display_name = "UNet Forward (single)"
        elif name == 'vision_per_camera':
            display_name = "DINOv2 (per camera)"
        elif name == 'vision_pooling':
            display_name = "Multi-Camera Pooling"
        elif name == 'vision_encoding':
            display_name = "Vision Encoding (total)"
        elif name == 'proprio_encoding':
            display_name = "Proprio Encoding"
        elif name == 'action_sampling':
            display_name = "Action Sampling (total)"
        elif name == 'total':
            display_name = "TOTAL"
        else:
            display_name = name
        
        print(f"{display_name:<30} {stat['mean']:>10.3f} {stat['std']:>10.3f} {stat['min']:>10.3f} {stat['max']:>10.3f}")
    
    print(f"\n{'='*80}\n")
    
    return timings, stats


def benchmark_inference(
    model: nn.Module,
    batch_size: int,
    num_cameras: int,
    state_dim: int,
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
        num_cameras: Number of cameras
        state_dim: Dimension of robot state
        device: Device to run on
        num_iterations: Number of iterations to benchmark
        warmup_iterations: Number of warmup iterations (not counted)
        use_compile: Whether to use torch.compile for optimization
    """
    print(f"\n{'='*80}")
    print(f"INFERENCE BENCHMARK - InnatePolicy")
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
        batch = generate_random_batch(batch_size, num_cameras, state_dim, device=device)
        
        with torch.no_grad():  # No gradient computation for inference
            _ = model.get_action(batch['images'], batch['robot_state'])
        
        if (i + 1) % 5 == 0:
            print(f"  Warmup iteration {i+1}/{warmup_iterations}")
    
    # Ensure all warmup operations are complete
    torch.cuda.synchronize()
    
    print("\n📊 Running benchmark...")
    
    # Benchmark phase
    for i in range(num_iterations):
        # Generate fresh batch
        batch = generate_random_batch(batch_size, num_cameras, state_dim, device=device)
        
        # Measure forward pass (inference)
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        
        with torch.no_grad():  # No gradient computation for inference
            output_action = model.get_action(batch['images'], batch['robot_state'])
        
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


def main():
    # Configuration - inference with batch size 1 (online inference)
    BATCH_SIZE = 1
    NUM_ITERATIONS = 1000
    WARMUP_ITERATIONS = 10
    USE_COMPILE = True  # Use torch.compile for optimized inference
    RUN_COMPONENT_PROFILER = True  # Run granular component profiler
    PROFILER_ITERATIONS = 100  # Iterations for component profiler
    
    # Model configuration
    NUM_CAMERAS = 2
    STATE_DIM = 6
    ACTION_DIM = 10
    ACTION_HORIZON = 16
    NUM_QUERIES = 16
    NUM_INFERENCE_STEPS = 4  # Number of diffusion sampling steps (4 steps × 2 Heun calls = 8 UNet forwards)
    
    # Use GPU 0
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA is not available!")
        sys.exit(1)
    
    device = torch.device("cuda:0")
    print(f"🎮 Using device: {device}")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create model configuration
    config = create_innate_policy_config(
        num_cameras=NUM_CAMERAS,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        num_queries=NUM_QUERIES,
        freeze_vision_backbone=True,
        num_inference_steps=NUM_INFERENCE_STEPS
    )
    
    print("\n🏗️  Building model...")
    model = InnatePolicy(**config)
    model = model.to(device)
    
    # Detailed parameter breakdown
    param_counts = count_parameters_by_component(model)
    print_parameter_breakdown(param_counts)
    
    # Run granular component profiler first (before compilation)
    if RUN_COMPONENT_PROFILER:
        print("\n" + "="*80)
        print("PHASE 1: GRANULAR COMPONENT PROFILING (Before Compilation)")
        print("="*80)
        profile_model_components(
            model=model,
            batch_size=BATCH_SIZE,
            num_cameras=NUM_CAMERAS,
            state_dim=STATE_DIM,
            device=device,
            num_iterations=PROFILER_ITERATIONS,
            warmup_iterations=10
        )
    
    # Compile the model for optimized performance
    if USE_COMPILE:
        print("\n" + "="*80)
        print("PHASE 2: COMPILING MODEL")
        print("="*80)
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
    print(f"   images (multi-camera): [{BATCH_SIZE}, {NUM_CAMERAS}, 3, 224, 224]")
    print(f"   robot_state:           [{BATCH_SIZE}, {STATE_DIM}]")
    print(f"\n🔄 Diffusion Sampling:")
    print(f"   Inference steps: {NUM_INFERENCE_STEPS}")
    print(f"   Integration method: Heun (RK2)")
    
    # Run benchmark
    print("\n" + "="*80)
    print("PHASE 3: PERFORMANCE BENCHMARK")
    print("="*80)
    results = benchmark_inference(
        model=model,
        batch_size=BATCH_SIZE,
        num_cameras=NUM_CAMERAS,
        state_dim=STATE_DIM,
        device=device,
        num_iterations=NUM_ITERATIONS,
        warmup_iterations=WARMUP_ITERATIONS,
        use_compile=USE_COMPILE
    )
    
    print("✅ Inference benchmark completed successfully!")



if __name__ == "__main__":
    main()
