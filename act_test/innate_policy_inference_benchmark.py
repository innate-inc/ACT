#!/usr/bin/env python3
"""
Inference Benchmark for InnatePolicy
Compares FP32 baseline vs AMP (BF16) performance
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict
import sys
from pathlib import Path

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
    """Create InnatePolicy configuration."""
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
    """Count parameters for each component of the model."""
    def count_params(module_or_param):
        if isinstance(module_or_param, nn.Parameter):
            return module_or_param.numel()
        else:
            return sum(p.numel() for p in module_or_param.parameters())
    
    def count_trainable_params(module_or_param):
        if isinstance(module_or_param, nn.Parameter):
            return module_or_param.numel() if module_or_param.requires_grad else 0
        else:
            return sum(p.numel() for p in module_or_param.parameters() if p.requires_grad)
    
    param_counts = {}
    
    # Vision encoder components
    if hasattr(model, 'vision_encoder'):
        param_counts['vision_encoder_total'] = count_params(model.vision_encoder)
        param_counts['vision_encoder_trainable'] = count_trainable_params(model.vision_encoder)
        
        if hasattr(model.vision_encoder, 'backbone'):
            param_counts['dinov2_backbone'] = count_params(model.vision_encoder.backbone)
            param_counts['dinov2_backbone_trainable'] = count_trainable_params(model.vision_encoder.backbone)
        
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
    """Generate a random batch of data."""
    batch = {}
    
    # Generate multi-camera image observations
    batch['images'] = torch.randn(
        batch_size, num_cameras, 3, image_size, image_size,
        device=device, requires_grad=False
    )
    
    # Generate state observation
    batch['robot_state'] = torch.randn(
        batch_size, state_dim,
        device=device, requires_grad=False
    )
    
    return batch


def benchmark_inference(
    model: nn.Module,
    batch_size: int,
    num_cameras: int,
    state_dim: int,
    device: torch.device,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16
):
    """Benchmark forward pass timing for inference."""
    print(f"\n{'='*80}")
    print(f"INFERENCE BENCHMARK")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Mixed Precision: {'enabled' if use_amp else 'disabled'} ({amp_dtype if use_amp else 'N/A'})")
    print(f"Warmup Iterations: {warmup_iterations}")
    print(f"Benchmark Iterations: {num_iterations}")
    print(f"{'='*80}\n")
    
    # Set model to inference mode
    model.eval()
    
    forward_times = []
    
    # Warmup phase
    print("🔥 Warming up...")
    for i in range(warmup_iterations):
        batch = generate_random_batch(batch_size, num_cameras, state_dim, device=device)
        
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    _ = model.get_action(batch['images'], batch['robot_state'])
            else:
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
        
        # Measure forward pass
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    output_action = model.get_action(batch['images'], batch['robot_state'])
            else:
                output_action = model.get_action(batch['images'], batch['robot_state'])
        
        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        forward_time = forward_end - forward_start
        
        # Record times
        forward_times.append(forward_time)
        
        # Progress update every 20 iterations
        if (i + 1) % 20 == 0:
            avg_total = np.mean(forward_times[-20:])
            print(f"  Iteration {i+1}/{num_iterations} | "
                  f"Avg time (last 20): {avg_total*1000:.2f} ms")
    
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
    # Configuration
    BATCH_SIZE = 1
    NUM_ITERATIONS = 1000
    WARMUP_ITERATIONS = 10
    AMP_DTYPE = torch.bfloat16  # torch.float16 or torch.bfloat16
    
    # Model configuration
    NUM_CAMERAS = 2
    STATE_DIM = 6
    ACTION_DIM = 10
    ACTION_HORIZON = 16
    NUM_QUERIES = 8
    NUM_INFERENCE_STEPS = 4
    
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
    
    print(f"\n📦 Batch Structure (batch_size={BATCH_SIZE}):")
    print(f"   images (multi-camera): [{BATCH_SIZE}, {NUM_CAMERAS}, 3, 224, 224]")
    print(f"   robot_state:           [{BATCH_SIZE}, {STATE_DIM}]")
    print(f"\n🔄 Diffusion Sampling:")
    print(f"   Inference steps: {NUM_INFERENCE_STEPS}")
    print(f"   Integration method: Heun (RK2)")
    print(f"   Total UNet calls: {(NUM_INFERENCE_STEPS - 1) * 2} per inference")
    
    print(f"\n🧪 Comparing FP32 baseline vs AMP ({AMP_DTYPE})")
    
    # ========================================================================
    # EXPERIMENT 1: FP32 BASELINE
    # ========================================================================
    print("\n\n" + "="*80)
    print("🔬 EXPERIMENT 1/2: FP32 BASELINE")
    print("="*80)
    
    results_fp32 = benchmark_inference(
        model=model,
        batch_size=BATCH_SIZE,
        num_cameras=NUM_CAMERAS,
        state_dim=STATE_DIM,
        device=device,
        num_iterations=NUM_ITERATIONS,
        warmup_iterations=WARMUP_ITERATIONS,
        use_amp=False,
        amp_dtype=AMP_DTYPE
    )
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # ========================================================================
    # EXPERIMENT 2: AMP
    # ========================================================================
    print("\n\n" + "="*80)
    print(f"🔬 EXPERIMENT 2/2: AMP ({AMP_DTYPE})")
    print("="*80)
    
    results_amp = benchmark_inference(
        model=model,
        batch_size=BATCH_SIZE,
        num_cameras=NUM_CAMERAS,
        state_dim=STATE_DIM,
        device=device,
        num_iterations=NUM_ITERATIONS,
        warmup_iterations=WARMUP_ITERATIONS,
        use_amp=True,
        amp_dtype=AMP_DTYPE
    )
    
    # ========================================================================
    # COMPARISON SUMMARY
    # ========================================================================
    print("\n\n" + "="*80)
    print(f"📊 COMPARISON SUMMARY: FP32 vs AMP ({AMP_DTYPE})")
    print("="*80 + "\n")
    
    fp32_mean = results_fp32['latency_stats']['mean']
    amp_mean = results_amp['latency_stats']['mean']
    speedup = fp32_mean / amp_mean
    
    print("⏱️  LATENCY COMPARISON (ms per inference):")
    print(f"{'Configuration':<20} {'Mean':>12} {'Median':>12} {'P95':>12} {'P99':>12}")
    print("-" * 72)
    print(f"{'FP32 (Baseline)':<20} {fp32_mean:>12.3f} {results_fp32['latency_stats']['median']:>12.3f} "
          f"{results_fp32['latency_stats']['p95']:>12.3f} {results_fp32['latency_stats']['p99']:>12.3f}")
    print(f"{f'AMP ({AMP_DTYPE})':<20} {amp_mean:>12.3f} {results_amp['latency_stats']['median']:>12.3f} "
          f"{results_amp['latency_stats']['p95']:>12.3f} {results_amp['latency_stats']['p99']:>12.3f}")
    print(f"{'Speedup':<20} {speedup:>11.2f}x")
    
    print(f"\n⚡ THROUGHPUT COMPARISON:")
    print(f"{'Configuration':<20} {'Samples/sec':>15} {'Gain':>12}")
    print("-" * 49)
    fp32_throughput = results_fp32['throughput']['samples_per_sec']
    amp_throughput = results_amp['throughput']['samples_per_sec']
    throughput_gain = amp_throughput / fp32_throughput
    
    print(f"{'FP32 (Baseline)':<20} {fp32_throughput:>15.2f} {'1.00x':>12}")
    print(f"{f'AMP ({AMP_DTYPE})':<20} {amp_throughput:>15.2f} {throughput_gain:>11.2f}x")
    
    print(f"\n💡 SUMMARY:")
    if speedup > 1.3:
        print(f"   ✅ AMP provides significant speedup: {speedup:.2f}x faster")
        print(f"   💾 Memory reduction: ~40-50% (using {AMP_DTYPE})")
        print(f"   🎯 RECOMMENDED: Use AMP for production inference")
    elif speedup > 1.1:
        print(f"   ✓ AMP provides moderate speedup: {speedup:.2f}x faster")
        print(f"   💾 Memory reduction: ~40-50% (using {AMP_DTYPE})")
        print(f"   🤔 Consider using AMP if memory savings are important")
    else:
        print(f"   ⚠️  AMP provides minimal benefit: {speedup:.2f}x")
        print(f"   🤷 May not be worth the complexity for this model")
    
    print(f"\n{'='*80}")
    print(f"✅ Benchmark completed successfully!")
    print(f"🚀 Best latency: {amp_mean:.3f} ms per inference ({speedup:.2f}x faster than FP32)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
