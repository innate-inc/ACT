"""
TensorRT Benchmark Script

Compares inference performance between:
1. Plain PyTorch ACTPolicy model
2. TensorRT-optimized ONNX model

Usage:
    python act_test/tensort_benchmark.py --checkpoint_dir /path/to/checkpoint/dir
"""

import argparse
import os
import time
import numpy as np
import torch
from pathlib import Path

# TensorRT imports
import tensorrt as trt

from ACT import ACTConfig, ACTPolicy


def create_dummy_dataset_stats(config: ACTConfig, device: torch.device):
    """Create dummy dataset statistics for model initialization."""
    dataset_stats = {}
    
    # Image statistics (mean and std per channel)
    for img_key in config.image_input_keys:
        img_shape = config.input_shapes[img_key]  # [C, H, W]
        c = img_shape[0]
        dataset_stats[img_key] = {
            "mean": torch.zeros(c, device=device),
            "std": torch.ones(c, device=device),
        }
    
    # State statistics
    if "observation.state" in config.input_shapes:
        state_dim = config.input_shapes["observation.state"][0]
        dataset_stats["observation.state"] = {
            "mean": torch.zeros(state_dim, device=device),
            "std": torch.ones(state_dim, device=device),
        }
    
    # Action statistics
    if "action" in config.output_shapes:
        action_dim = config.output_shapes["action"][0]
        dataset_stats["action"] = {
            "mean": torch.zeros(action_dim, device=device),
            "std": torch.ones(action_dim, device=device),
        }
    
    return dataset_stats


class TensorRTInference:
    """Wrapper for TensorRT inference engine using PyTorch tensors."""
    
    def __init__(self, onnx_path, device='cuda:0', use_fp16=True):
        """Initialize TensorRT engine from ONNX file.
        
        Args:
            onnx_path: Path to ONNX model file
            device: CUDA device to use
            use_fp16: Whether to enable FP16 precision (faster but less accurate)
        """
        self.device = torch.device(device)
        self.use_fp16 = use_fp16
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Build TensorRT engine
        print(f"Building TensorRT engine from {onnx_path}...")
        print("This may take a few minutes on first run...")
        
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX file
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX file")
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # Enable FP16 if requested and supported
        if self.use_fp16 and builder.platform_has_fast_fp16:
            print("  Enabling FP16 precision for TensorRT (faster but less accurate)")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("  Using FP32 precision for TensorRT (more accurate but slower)")
        
        # Create optimization profile for dynamic batch size
        # The ONNX model has dynamic batch dimension, so we need to specify the range
        profile = builder.create_optimization_profile()
        
        # For each input, set min/opt/max shapes (we'll use batch_size=1 for all)
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_name = input_tensor.name
            input_shape = input_tensor.shape
            
            # Replace -1 (dynamic dimension) with actual batch size
            # min, opt, max shapes for optimization (all batch_size=1)
            min_shape = [1 if dim == -1 else dim for dim in input_shape]
            opt_shape = [1 if dim == -1 else dim for dim in input_shape]
            max_shape = [1 if dim == -1 else dim for dim in input_shape]
            
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            print(f"  Setting optimization profile for '{input_name}': {opt_shape}")
        
        config.add_optimization_profile(profile)
        
        # Build engine
        print("  Building engine (this may take a while)...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Deserialize engine
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()
        
        # Get input and output shapes
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}
        self.output_shapes = {}
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_shapes[name] = shape
            else:
                self.output_names.append(name)
                self.output_shapes[name] = shape
        
        print(f"✅ TensorRT engine built successfully!")
        print(f"   Inputs: {self.input_names}")
        print(f"   Outputs: {self.output_names}")
        
        # Allocate device memory using PyTorch
        self.allocate_buffers()
    
    def allocate_buffers(self):
        """Allocate GPU memory for inputs and outputs using PyTorch tensors."""
        self.input_tensors = {}
        self.output_tensors = {}
        
        for name in self.input_names:
            shape = self.input_shapes[name]
            # Replace -1 (dynamic dimension) with 1 for buffer allocation
            shape = tuple([1 if dim == -1 else dim for dim in shape])
            # Allocate CUDA tensor
            tensor = torch.zeros(shape, dtype=torch.float32, device=self.device)
            self.input_tensors[name] = tensor
        
        for name in self.output_names:
            shape = self.output_shapes[name]
            # Replace -1 (dynamic dimension) with 1 for buffer allocation
            shape = tuple([1 if dim == -1 else dim for dim in shape])
            # Allocate CUDA tensor
            tensor = torch.zeros(shape, dtype=torch.float32, device=self.device)
            self.output_tensors[name] = tensor
    
    def infer(self, img_cam1, img_cam2, robot_state):
        """Run inference with TensorRT using PyTorch tensors.
        
        Args:
            img_cam1: torch.Tensor [1, 3, 224, 224] on CUDA
            img_cam2: torch.Tensor [1, 3, 224, 224] on CUDA
            robot_state: torch.Tensor [1, 6] on CUDA
        
        Returns:
            predicted_actions: torch.Tensor [1, chunk_size, action_dim] on CUDA
        """
        # Copy inputs to buffers
        self.input_tensors['image_camera_1'].copy_(img_cam1)
        self.input_tensors['image_camera_2'].copy_(img_cam2)
        self.input_tensors['robot_state'].copy_(robot_state)
        
        # Set input shapes for dynamic dimensions (required for optimization profiles)
        for name in self.input_names:
            self.context.set_input_shape(name, tuple(self.input_tensors[name].shape))
        
        # Set tensor addresses (get data pointer from PyTorch tensors)
        for name in self.input_names:
            self.context.set_tensor_address(name, self.input_tensors[name].data_ptr())
        
        for name in self.output_names:
            self.context.set_tensor_address(name, self.output_tensors[name].data_ptr())
        
        # Run inference
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        
        # Synchronize to ensure inference is complete
        torch.cuda.synchronize()
        
        return self.output_tensors['predicted_actions']


def benchmark_pytorch(policy, img_cam1, img_cam2, robot_state, num_iterations=100, warmup=10):
    """Benchmark PyTorch model inference."""
    policy.eval()
    
    print("\n" + "="*80)
    print("PYTORCH INFERENCE BENCHMARK")
    print("="*80)
    
    # Warmup
    print(f"Warming up for {warmup} iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            batch = {
                "observation.image_camera_1": img_cam1,
                "observation.image_camera_2": img_cam2,
                "observation.state": robot_state,
            }
            batch = policy.normalize_inputs(batch)
            model_batch = policy._prepare_batch_for_model(batch)
            actions_normalized, _ = policy.model(model_batch)
            actions = policy.unnormalize_outputs({"action": actions_normalized})["action"]
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {num_iterations} iterations...")
    times = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            start = time.perf_counter()
            
            batch = {
                "observation.image_camera_1": img_cam1,
                "observation.image_camera_2": img_cam2,
                "observation.state": robot_state,
            }
            batch = policy.normalize_inputs(batch)
            model_batch = policy._prepare_batch_for_model(batch)
            actions_normalized, _ = policy.model(model_batch)
            actions = policy.unnormalize_outputs({"action": actions_normalized})["action"]
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    print(f"\n📊 PyTorch Results:")
    print(f"   Mean: {times.mean():.3f} ms")
    print(f"   Std:  {times.std():.3f} ms")
    print(f"   Min:  {times.min():.3f} ms")
    print(f"   Max:  {times.max():.3f} ms")
    print(f"   Median: {np.median(times):.3f} ms")
    print(f"   FPS: {1000.0 / times.mean():.2f}")
    
    return times, actions


def benchmark_tensorrt(trt_engine, img_cam1, img_cam2, robot_state, num_iterations=100, warmup=10):
    """Benchmark TensorRT model inference."""
    
    print("\n" + "="*80)
    print("TENSORRT INFERENCE BENCHMARK")
    print("="*80)
    
    # Warmup
    print(f"Warming up for {warmup} iterations...")
    for _ in range(warmup):
        actions = trt_engine.infer(img_cam1, img_cam2, robot_state)
    
    # Benchmark
    print(f"Running {num_iterations} iterations...")
    times = []
    
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        actions = trt_engine.infer(img_cam1, img_cam2, robot_state)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    print(f"\n📊 TensorRT Results:")
    print(f"   Mean: {times.mean():.3f} ms")
    print(f"   Std:  {times.std():.3f} ms")
    print(f"   Min:  {times.min():.3f} ms")
    print(f"   Max:  {times.max():.3f} ms")
    print(f"   Median: {np.median(times):.3f} ms")
    print(f"   FPS: {1000.0 / times.mean():.2f}")
    
    return times, actions


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs TensorRT inference")
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to checkpoint directory containing final checkpoint and ONNX file')
    parser.add_argument('--chunk_size', type=int, default=30,
                        help='Chunk size (default: 30)')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of benchmark iterations (default: 100)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations (default: 10)')
    parser.add_argument('--use_fp16', action='store_true', default=False,
                        help='Enable FP16 precision in TensorRT (faster but less accurate)')
    parser.add_argument('--use_fp32', dest='use_fp16', action='store_false',
                        help='Use FP32 precision in TensorRT (more accurate but slower) [default]')
    
    args = parser.parse_args()
    
    # Check if checkpoint directory exists
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"❌ Error: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Find the final checkpoint (try multiple naming patterns)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        checkpoints = sorted(checkpoint_dir.glob("act_policy_step_*.pth"))
    if not checkpoints:
        checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        checkpoints = sorted(checkpoint_dir.glob("*.pth"))
    
    if not checkpoints:
        print(f"❌ Error: No checkpoint files found in {checkpoint_dir}")
        return
    
    final_checkpoint = checkpoints[-1]
    print(f"Using checkpoint: {final_checkpoint}")
    
    # Check if ONNX file exists
    onnx_path = checkpoint_dir / "act_policy_final.onnx"
    if not onnx_path.exists():
        print(f"❌ Error: ONNX file not found: {onnx_path}")
        return
    
    print(f"Using ONNX file: {onnx_path}")
    
    # Configuration (must match train_dist.py)
    IMAGE_H = 224
    IMAGE_W = 224
    IMAGE_C = 3
    QPOS_DIM = 6
    ACTION_DIM = 10
    CHUNK_SIZE = args.chunk_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎮 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create ACT configuration
    print("\n📝 Creating ACT configuration...")
    config = ACTConfig(
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
        dim_model=512,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=4,
        kl_weight=10.0,
        use_vae=True,
    )
    
    # Load or create dataset stats
    dataset_stats_path = checkpoint_dir / "dataset_stats.pt"
    if dataset_stats_path.exists():
        print(f"Loading dataset statistics from {dataset_stats_path}...")
        dataset_stats = torch.load(dataset_stats_path, map_location=device)
    else:
        print("Creating dummy dataset statistics...")
        dataset_stats = create_dummy_dataset_stats(config, device)
    
    # Load PyTorch model
    print("\n🔄 Loading PyTorch model...")
    policy = ACTPolicy(config=config, dataset_stats=dataset_stats).to(device)
    
    checkpoint = torch.load(final_checkpoint, map_location=device)
    # Handle both formats: direct state_dict or dict with 'model_state_dict' key
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove "_orig_mod." prefix added by torch.compile if present
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("  Removing '_orig_mod.' prefix from compiled model checkpoint...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    policy.load_state_dict(state_dict)
    policy.eval()
    print("✅ PyTorch model loaded successfully!")
    
    # Create dummy inputs (batch size 1)
    print("\n🎲 Creating dummy inputs (batch size 1)...")
    img_cam1 = torch.randn(1, IMAGE_C, IMAGE_H, IMAGE_W, device=device)
    img_cam2 = torch.randn(1, IMAGE_C, IMAGE_H, IMAGE_W, device=device)
    robot_state = torch.randn(1, QPOS_DIM, device=device)
    
    # Benchmark PyTorch
    pytorch_times, pytorch_output = benchmark_pytorch(
        policy, img_cam1, img_cam2, robot_state,
        num_iterations=args.num_iterations,
        warmup=args.warmup
    )
    
    # Load TensorRT engine
    print("\n🔄 Loading TensorRT engine...")
    trt_engine = TensorRTInference(str(onnx_path), device=str(device), use_fp16=args.use_fp16)
    
    # Benchmark TensorRT (using same PyTorch tensors)
    tensorrt_times, tensorrt_output = benchmark_tensorrt(
        trt_engine, img_cam1, img_cam2, robot_state,
        num_iterations=args.num_iterations,
        warmup=args.warmup
    )
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    pytorch_mean = pytorch_times.mean()
    tensorrt_mean = tensorrt_times.mean()
    speedup = pytorch_mean / tensorrt_mean
    
    print(f"\n📈 Performance Comparison:")
    print(f"   PyTorch:   {pytorch_mean:.3f} ms ({1000.0/pytorch_mean:.2f} FPS)")
    print(f"   TensorRT:  {tensorrt_mean:.3f} ms ({1000.0/tensorrt_mean:.2f} FPS)")
    print(f"   Speedup:   {speedup:.2f}x")
    
    # Check output similarity (both are PyTorch tensors)
    diff = torch.abs(pytorch_output - tensorrt_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Calculate action value statistics for context
    action_min = pytorch_output.min().item()
    action_max = pytorch_output.max().item()
    action_mean = pytorch_output.mean().item()
    action_std = pytorch_output.std().item()
    action_range = action_max - action_min
    
    # Calculate relative error as percentage of action range (more meaningful)
    max_relative_to_range = (max_diff / action_range) * 100 if action_range > 0 else 0
    mean_relative_to_range = (mean_diff / action_range) * 100 if action_range > 0 else 0
    
    print(f"\n🔍 Output Comparison:")
    print(f"   Action Value Range:")
    print(f"     Min:  {action_min:.4f}")
    print(f"     Max:  {action_max:.4f}")
    print(f"     Range: {action_range:.4f}")
    print(f"     Mean: {action_mean:.4f} ± {action_std:.4f}")
    print(f"\n   Absolute Difference:")
    print(f"     Max:  {max_diff:.6f}")
    print(f"     Mean: {mean_diff:.6f}")
    print(f"\n   Relative to Action Range:")
    print(f"     Max:  {max_relative_to_range:.2f}% of range")
    print(f"     Mean: {mean_relative_to_range:.2f}% of range")
    
    # Assessment based on relative error to range
    if max_relative_to_range < 1.0:
        print("\n   ✅ Excellent accuracy (< 1% of action range)")
    elif max_relative_to_range < 5.0:
        print("\n   ✅ Good accuracy (< 5% of action range)")
    elif max_relative_to_range < 10.0:
        print("\n   ⚠️  Acceptable accuracy (< 10% of action range)")
    else:
        print(f"\n   ⚠️  Poor accuracy ({max_relative_to_range:.1f}% of action range - consider using FP32 or checking ONNX export)")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

