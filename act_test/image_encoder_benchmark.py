#!/usr/bin/env python3
"""
Benchmark script to compare ResNet-18 and ViT-Small-14 inference speeds.
Tests both models with 224x224 images and reports throughput and latency.
"""

import torch
import torchvision.models as models
import time
import numpy as np
from typing import Dict, List, Tuple


class ImageEncoderBenchmark:
    """Benchmark image encoder models for inference speed."""
    
    def __init__(self, device: str = None, warmup_iterations: int = 10, 
                 benchmark_iterations: int = 100, batch_size: int = 1):
        """
        Initialize the benchmark.
        
        Args:
            device: Device to run on ('cuda' or 'cpu'). Auto-detects if None.
            warmup_iterations: Number of warmup iterations before benchmarking.
            benchmark_iterations: Number of iterations for benchmarking.
            batch_size: Batch size for inference.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.batch_size = batch_size
        self.image_size = (224, 224)
        
        print(f"Running benchmark on: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.image_size}")
        print(f"Warmup iterations: {self.warmup_iterations}")
        print(f"Benchmark iterations: {self.benchmark_iterations}")
        print("-" * 70)
    
    def load_resnet18(self) -> torch.nn.Module:
        """Load ResNet-18 model."""
        print("Loading ResNet-18...")
        model = models.resnet18(pretrained=False)
        model = model.to(self.device)
        model.eval()
        return model
    
    def load_vit_small_14(self) -> torch.nn.Module:
        """Load DINOv2 ViT-Small with patch size 14."""
        print("Loading DINOv2 ViT-Small-14...")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model = model.to(self.device)
        model.eval()
        return model
    
    def create_dummy_input(self) -> torch.Tensor:
        """Create dummy input tensor."""
        return torch.randn(
            self.batch_size, 3, *self.image_size, 
            device=self.device
        )
    
    def benchmark_model(self, model: torch.nn.Module, model_name: str, 
                       get_patch_embeddings: bool = False) -> Dict[str, float]:
        """
        Benchmark a model and return performance metrics.
        
        Args:
            model: The model to benchmark.
            model_name: Name of the model for display.
            get_patch_embeddings: If True, extract patch embeddings for ViT models.
            
        Returns:
            Dictionary containing performance metrics.
        """
        print(f"\nBenchmarking {model_name}...")
        
        # Create input
        dummy_input = self.create_dummy_input()
        
        # Determine if this is a DINOv2 model
        is_dinov2 = hasattr(model, 'get_intermediate_layers')
        
        # Get output shape
        with torch.no_grad():
            if is_dinov2 and get_patch_embeddings:
                # For DINOv2, use get_intermediate_layers to get patch tokens
                output = model.get_intermediate_layers(dummy_input, n=1, return_class_token=True)
                # output is a list of tuples (patch_tokens, class_token)
                patch_tokens = output[0][0]  # (batch, num_patches, dim)
                cls_token = output[0][1]      # (batch, dim)
                output_shape = tuple(patch_tokens.shape)
                print(f"Patch embeddings shape: {output_shape}")
                print(f"CLS token shape: {tuple(cls_token.shape)}")
                print(f"Number of patches: {patch_tokens.shape[1]}")
            else:
                output = model(dummy_input)
                output_shape = tuple(output.shape)
                print(f"Output shape: {output_shape}")
        
        # Create the forward function for benchmarking
        if is_dinov2 and get_patch_embeddings:
            def forward_fn(x):
                return model.get_intermediate_layers(x, n=1, return_class_token=True)
        else:
            def forward_fn(x):
                return model(x)
        
        # Warmup
        print(f"Warming up ({self.warmup_iterations} iterations)...")
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = forward_fn(dummy_input)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
        
        # Benchmark
        print(f"Running benchmark ({self.benchmark_iterations} iterations)...")
        times = []
        
        with torch.no_grad():
            for _ in range(self.benchmark_iterations):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    
                    start.record()
                    _ = forward_fn(dummy_input)
                    end.record()
                    
                    torch.cuda.synchronize()
                    elapsed_time = start.elapsed_time(end) / 1000.0  # Convert to seconds
                else:
                    start = time.perf_counter()
                    _ = forward_fn(dummy_input)
                    end = time.perf_counter()
                    elapsed_time = end - start
                
                times.append(elapsed_time)
        
        # Calculate statistics
        times_array = np.array(times)
        mean_time = np.mean(times_array)
        std_time = np.std(times_array)
        min_time = np.min(times_array)
        max_time = np.max(times_array)
        median_time = np.median(times_array)
        
        # Calculate throughput
        throughput = self.batch_size / mean_time
        
        results = {
            'mean_latency_ms': mean_time * 1000,
            'std_latency_ms': std_time * 1000,
            'min_latency_ms': min_time * 1000,
            'max_latency_ms': max_time * 1000,
            'median_latency_ms': median_time * 1000,
            'throughput_fps': throughput,
            'output_shape': output_shape,
        }
        
        return results
    
    def print_results(self, results: Dict[str, Dict[str, float]], mode: str = ""):
        """Print benchmark results in a formatted table."""
        title = f"BENCHMARK RESULTS - {mode}" if mode else "BENCHMARK RESULTS"
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Output Shape:   {metrics['output_shape']}")
            print(f"  Mean Latency:   {metrics['mean_latency_ms']:.3f} ms")
            print(f"  Std Latency:    {metrics['std_latency_ms']:.3f} ms")
            print(f"  Min Latency:    {metrics['min_latency_ms']:.3f} ms")
            print(f"  Max Latency:    {metrics['max_latency_ms']:.3f} ms")
            print(f"  Median Latency: {metrics['median_latency_ms']:.3f} ms")
            print(f"  Throughput:     {metrics['throughput_fps']:.2f} images/sec")
        
        # Compare results
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        
        if len(results) == 2:
            models = list(results.keys())
            model1, model2 = models[0], models[1]
            
            speedup = results[model2]['mean_latency_ms'] / results[model1]['mean_latency_ms']
            throughput_ratio = results[model1]['throughput_fps'] / results[model2]['throughput_fps']
            
            print(f"\n{model1} vs {model2}:")
            print(f"  Latency ratio: {speedup:.2f}x")
            if speedup > 1:
                print(f"  {model1} is {speedup:.2f}x faster than {model2}")
            else:
                print(f"  {model2} is {1/speedup:.2f}x faster than {model1}")
            
            print(f"\n  Throughput ratio: {throughput_ratio:.2f}x")
            if throughput_ratio > 1:
                print(f"  {model1} has {throughput_ratio:.2f}x higher throughput than {model2}")
            else:
                print(f"  {model2} has {1/throughput_ratio:.2f}x higher throughput than {model1}")
    
    def print_compilation_comparison(self, results_no_compile: Dict[str, Dict[str, float]], 
                                     results_compiled: Dict[str, Dict[str, float]]):
        """Print comparison between compiled and non-compiled models."""
        print("\n" + "=" * 70)
        print("COMPILATION SPEEDUP ANALYSIS")
        print("=" * 70)
        
        for model_name in results_no_compile.keys():
            if model_name in results_compiled:
                no_compile = results_no_compile[model_name]
                compiled = results_compiled[model_name]
                
                latency_speedup = no_compile['mean_latency_ms'] / compiled['mean_latency_ms']
                throughput_speedup = compiled['throughput_fps'] / no_compile['throughput_fps']
                
                print(f"\n{model_name}:")
                print(f"  Without compilation: {no_compile['mean_latency_ms']:.3f} ms")
                print(f"  With compilation:    {compiled['mean_latency_ms']:.3f} ms")
                print(f"  Latency speedup:     {latency_speedup:.2f}x")
                print(f"  Throughput speedup:  {throughput_speedup:.2f}x")
                
                if latency_speedup > 1:
                    improvement = (latency_speedup - 1) * 100
                    print(f"  Improvement:         {improvement:.1f}% faster with compilation")
                else:
                    degradation = (1 - latency_speedup) * 100
                    print(f"  Degradation:         {degradation:.1f}% slower with compilation")
    
    def run(self):
        """Run the complete benchmark."""
        all_results = {
            'without_compilation': {},
            'with_compilation': {}
        }
        
        print("\n" + "=" * 70)
        print("PHASE 1: BENCHMARKING WITHOUT COMPILATION")
        print("=" * 70)
        
        # Benchmark ResNet-18 without compilation
        resnet18 = self.load_resnet18()
        all_results['without_compilation']['ResNet-18'] = self.benchmark_model(
            resnet18, 'ResNet-18 (No Compilation)', get_patch_embeddings=False
        )
        
        # Benchmark ViT-Small-14 without compilation (with patch embeddings)
        vit_small = self.load_vit_small_14()
        all_results['without_compilation']['ViT-Small-14'] = self.benchmark_model(
            vit_small, 'ViT-Small-14 (No Compilation)', get_patch_embeddings=True
        )
        
        # Print results for non-compiled models
        self.print_results(all_results['without_compilation'], "WITHOUT COMPILATION")
        
        # Clean up
        del resnet18, vit_small
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        print("\n" + "=" * 70)
        print("PHASE 2: BENCHMARKING WITH TORCH.COMPILE()")
        print("=" * 70)
        
        # Benchmark ResNet-18 with compilation
        resnet18_compiled = self.load_resnet18()
        print("Compiling ResNet-18 with torch.compile()...")
        resnet18_compiled = torch.compile(resnet18_compiled)
        all_results['with_compilation']['ResNet-18'] = self.benchmark_model(
            resnet18_compiled, 'ResNet-18 (Compiled)', get_patch_embeddings=False
        )
        
        # Benchmark ViT-Small-14 with compilation (with patch embeddings)
        vit_small_compiled = self.load_vit_small_14()
        print("Compiling ViT-Small-14 with torch.compile()...")
        vit_small_compiled = torch.compile(vit_small_compiled)
        all_results['with_compilation']['ViT-Small-14'] = self.benchmark_model(
            vit_small_compiled, 'ViT-Small-14 (Compiled)', get_patch_embeddings=True
        )
        
        # Print results for compiled models
        self.print_results(all_results['with_compilation'], "WITH COMPILATION")
        
        # Print compilation comparison
        self.print_compilation_comparison(all_results['without_compilation'], all_results['with_compilation'])
        
        # Clean up
        del resnet18_compiled, vit_small_compiled
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return all_results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark image encoder models')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to run on (cuda/cpu). Auto-detects if not specified.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations (default: 10)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of benchmark iterations (default: 100)')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = ImageEncoderBenchmark(
        device=args.device,
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        batch_size=args.batch_size
    )
    
    results = benchmark.run()
    
    print("\n" + "=" * 70)
    print("Benchmark completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
