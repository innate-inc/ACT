import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import os
import time
import argparse
import numpy as np
from datetime import datetime
import json
import shutil

from data_utils import initialize_webdataset_data
from data_tools.webdataset import convert_hdf5_to_webdataset

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # Different port from training
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the GPU for the current process
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def detect_and_convert_dataset(data_dir, force_reconvert=False, shard_size=1000):
    """
    Detect dataset format and convert HDF5 to WebDataset if needed.
    
    Args:
        data_dir: Path to dataset directory (either HDF5 or WebDataset)
        force_reconvert: If True, reconvert even if WebDataset already exists
        shard_size: Number of samples per shard (default: 1000)
        
    Returns:
        tuple: (success, webdataset_dir_path)
    """
    # Check if it's already a WebDataset directory (contains .tar files)
    tar_files = [f for f in os.listdir(data_dir) if f.endswith('.tar')]
    
    if tar_files and not force_reconvert:
        print(f"✅ Found {len(tar_files)} WebDataset .tar files in {data_dir}")
        print("Skipping conversion (dataset already in WebDataset format)")
        return True, data_dir
    
    # Check if it's an HDF5 directory (contains .h5 files and dataset_metadata.json)
    h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    metadata_path = os.path.join(data_dir, "dataset_metadata.json")
    
    if h5_files and os.path.exists(metadata_path):
        print(f"📦 Found {len(h5_files)} HDF5 files in {data_dir}")
        print("🔄 Converting HDF5 to WebDataset format...")
        
        # Create WebDataset directory
        webd_dir = os.path.join(data_dir, "webdataset")
        
        # Remove existing WebDataset directory if force_reconvert
        if os.path.exists(webd_dir) and force_reconvert:
            print(f"🗑️  Removing existing WebDataset directory: {webd_dir}")
            shutil.rmtree(webd_dir)
        
        if os.path.exists(webd_dir):
            print(f"✅ WebDataset directory already exists: {webd_dir}")
            print("Using existing WebDataset (use --force-reconvert to recreate)")
            return True, webd_dir
        
        print("=" * 80)
        print(f"📁 HDF5 source: {data_dir}")
        print(f"📁 WebDataset target: {webd_dir}")
        print(f"📦 Shard size: {shard_size}")
        print("=" * 80)
        
        # Perform conversion
        success = convert_hdf5_to_webdataset(
            hdf5_directory=data_dir,
            webd_directory=webd_dir,
            shard_size=shard_size
        )
        
        if success:
            print("✅ Data conversion completed successfully!")
            return True, webd_dir
        else:
            print("❌ Data conversion failed!")
            return False, None
    
    # Neither HDF5 nor WebDataset format detected
    print(f"❌ Error: Could not detect dataset format in {data_dir}")
    print("Expected either:")
    print("  - WebDataset: Directory with .tar files")
    print("  - HDF5: Directory with .h5 files and dataset_metadata.json")
    return False, None

def benchmark_data_loading(rank, world_size, args, webd_dir):
    """Benchmark data loading for a single GPU in DDP setup."""
    setup(rank, world_size)
    
    # --- Configuration ---
    DATA_DIR = webd_dir  # Use the WebDataset directory
    CHUNK_SIZE = args.chunk_size
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    NUM_BATCHES = args.num_batches
    WARMUP_BATCHES = args.warmup_batches
    TRAIN_VAL_SPLIT = 0.9
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    
    # Only print on rank 0
    if rank == 0:
        print("=" * 80)
        print("DATA LOADING BENCHMARK - Multi-GPU DDP Setup")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  World size (GPUs): {world_size}")
        print(f"  Data directory: {DATA_DIR}")
        print(f"  Chunk size: {CHUNK_SIZE}")
        print(f"  Batch size per GPU: {BATCH_SIZE}")
        print(f"  Total effective batch size: {BATCH_SIZE * world_size}")
        print(f"  Num workers per GPU: {NUM_WORKERS}")
        print(f"  Warmup batches: {WARMUP_BATCHES}")
        print(f"  Benchmark batches: {NUM_BATCHES}")
        print(f"  Device: {device}")
        print("=" * 80)
    
    # --- Initialize DataLoaders ---
    if rank == 0:
        print("Initializing WebDataset data loaders...")
    
    try:
        train_dataloader, val_dataloader, dataset_stats = initialize_webdataset_data(
            data_dir=DATA_DIR,
            chunk_size=CHUNK_SIZE,
            batch_size=BATCH_SIZE,
            train_val_split=TRAIN_VAL_SPLIT,
            num_workers=NUM_WORKERS,
            prefetch_factor=2,
            seed=42 + rank  # Different seed per rank for different data
        )
    except (FileNotFoundError, ValueError) as e:
        if rank == 0:
            print(f"Error initializing WebDataset: {e}")
            print(f"Please ensure your data directory '{DATA_DIR}' contains WebDataset .tar files.")
        cleanup()
        return None
    
    if rank == 0:
        print("WebDataset dataloaders initialized successfully!")
        print("=" * 80)
    
    # Synchronize all ranks before starting benchmark
    dist.barrier()
    
    # --- Benchmark Data Loading ---
    train_iter = iter(train_dataloader)
    
    # Storage for timing data (per rank)
    batch_load_times = []
    data_to_device_times = []
    total_times = []
    
    # Warmup phase
    if rank == 0:
        print(f"Warming up with {WARMUP_BATCHES} batches...")
    
    for i in range(WARMUP_BATCHES):
        try:
            batch_start = time.time()
            batch = next(train_iter)
            
            # Move batch to device (simulating what happens in training)
            device_start = time.time()
            batch_device = {}
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor):
                    batch_device[key] = tensor.to(device, non_blocking=True)
                else:
                    batch_device[key] = tensor
            
            # Synchronize to ensure all data is on device
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
    
    # Synchronize all ranks after warmup
    dist.barrier()
    
    if rank == 0:
        print(f"Warmup complete. Starting benchmark...")
        print("=" * 80)
        pbar = tqdm(total=NUM_BATCHES, desc=f"GPU {rank} Benchmark", position=rank)
    
    # Actual benchmark
    benchmark_start = time.time()
    
    for i in range(NUM_BATCHES):
        try:
            # Time the entire batch loading process
            batch_start = time.time()
            
            # Get batch from dataloader
            batch = next(train_iter)
            batch_load_time = time.time() - batch_start
            
            # Time moving data to device
            device_start = time.time()
            batch_device = {}
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor):
                    batch_device[key] = tensor.to(device, non_blocking=True)
                else:
                    batch_device[key] = tensor
            
            # Synchronize to ensure all data is on device
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            data_to_device_time = time.time() - device_start
            total_time = time.time() - batch_start
            
            # Store timing data
            batch_load_times.append(batch_load_time)
            data_to_device_times.append(data_to_device_time)
            total_times.append(total_time)
            
            if rank == 0:
                pbar.update(1)
                if (i + 1) % 10 == 0:
                    avg_total = np.mean(total_times[-10:]) * 1000
                    pbar.set_postfix({'avg_ms': f'{avg_total:.1f}'})
            
        except StopIteration:
            # Reset iterator if dataset is exhausted
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
            if rank == 0:
                pbar.write(f"Dataset exhausted at batch {i}, resetting iterator...")
    
    benchmark_end = time.time()
    total_benchmark_time = benchmark_end - benchmark_start
    
    if rank == 0:
        pbar.close()
    
    # Synchronize all ranks after benchmark
    dist.barrier()
    
    # --- Compute Statistics per Rank ---
    rank_stats = {
        'rank': rank,
        'num_batches': NUM_BATCHES,
        'batch_size': BATCH_SIZE,
        'total_samples': NUM_BATCHES * BATCH_SIZE,
        'total_time_sec': total_benchmark_time,
        'avg_batch_load_ms': np.mean(batch_load_times) * 1000,
        'std_batch_load_ms': np.std(batch_load_times) * 1000,
        'avg_data_to_device_ms': np.mean(data_to_device_times) * 1000,
        'std_data_to_device_ms': np.std(data_to_device_times) * 1000,
        'avg_total_ms': np.mean(total_times) * 1000,
        'std_total_ms': np.std(total_times) * 1000,
        'min_total_ms': np.min(total_times) * 1000,
        'max_total_ms': np.max(total_times) * 1000,
        'throughput_samples_per_sec': (NUM_BATCHES * BATCH_SIZE) / total_benchmark_time,
        'throughput_batches_per_sec': NUM_BATCHES / total_benchmark_time,
    }
    
    # Gather statistics from all ranks to rank 0
    all_stats = [None] * world_size
    dist.all_gather_object(all_stats, rank_stats)
    
    # --- Print Results on Rank 0 ---
    if rank == 0:
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        
        # Per-GPU statistics
        print("\nPer-GPU Statistics:")
        print("-" * 80)
        for stats in all_stats:
            print(f"\nGPU {stats['rank']}:")
            print(f"  Batches processed: {stats['num_batches']}")
            print(f"  Samples processed: {stats['total_samples']}")
            print(f"  Total time: {stats['total_time_sec']:.2f} sec")
            print(f"  Avg batch load time: {stats['avg_batch_load_ms']:.2f} ± {stats['std_batch_load_ms']:.2f} ms")
            print(f"  Avg data->device time: {stats['avg_data_to_device_ms']:.2f} ± {stats['std_data_to_device_ms']:.2f} ms")
            print(f"  Avg total time: {stats['avg_total_ms']:.2f} ± {stats['std_total_ms']:.2f} ms")
            print(f"  Min/Max total time: {stats['min_total_ms']:.2f} / {stats['max_total_ms']:.2f} ms")
            print(f"  Throughput: {stats['throughput_samples_per_sec']:.2f} samples/sec ({stats['throughput_batches_per_sec']:.2f} batches/sec)")
        
        # Aggregate statistics
        print("\n" + "=" * 80)
        print("AGGREGATE STATISTICS (All GPUs)")
        print("=" * 80)
        
        avg_time_per_batch = np.mean([s['avg_total_ms'] for s in all_stats])
        total_throughput_samples = sum([s['throughput_samples_per_sec'] for s in all_stats])
        total_throughput_batches = sum([s['throughput_batches_per_sec'] for s in all_stats])
        
        print(f"  Number of GPUs: {world_size}")
        print(f"  Batch size per GPU: {BATCH_SIZE}")
        print(f"  Total effective batch size: {BATCH_SIZE * world_size}")
        print(f"  Average time per batch (across GPUs): {avg_time_per_batch:.2f} ms")
        print(f"  Total throughput: {total_throughput_samples:.2f} samples/sec")
        print(f"  Total throughput: {total_throughput_batches:.2f} batches/sec")
        print(f"  Effective global batch time: {1000.0 / total_throughput_batches:.2f} ms")
        
        # Calculate percentiles
        all_total_times = []
        for _ in range(world_size):
            all_total_times.extend(total_times)
        
        p50 = np.percentile(all_total_times, 50) * 1000
        p90 = np.percentile(all_total_times, 90) * 1000
        p95 = np.percentile(all_total_times, 95) * 1000
        p99 = np.percentile(all_total_times, 99) * 1000
        
        print(f"\n  Latency Percentiles:")
        print(f"    P50: {p50:.2f} ms")
        print(f"    P90: {p90:.2f} ms")
        print(f"    P95: {p95:.2f} ms")
        print(f"    P99: {p99:.2f} ms")
        
        # Save results to JSON
        results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'world_size': world_size,
                'data_dir': DATA_DIR,
                'chunk_size': CHUNK_SIZE,
                'batch_size_per_gpu': BATCH_SIZE,
                'total_effective_batch_size': BATCH_SIZE * world_size,
                'num_workers': NUM_WORKERS,
                'num_batches': NUM_BATCHES,
                'warmup_batches': WARMUP_BATCHES,
            },
            'per_gpu_stats': all_stats,
            'aggregate_stats': {
                'num_gpus': world_size,
                'avg_time_per_batch_ms': avg_time_per_batch,
                'total_throughput_samples_per_sec': total_throughput_samples,
                'total_throughput_batches_per_sec': total_throughput_batches,
                'effective_global_batch_time_ms': 1000.0 / total_throughput_batches,
                'latency_percentiles': {
                    'p50_ms': p50,
                    'p90_ms': p90,
                    'p95_ms': p95,
                    'p99_ms': p99,
                }
            }
        }
        
        results_file = f"dataloading_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n  Results saved to: {results_file}")
        print("=" * 80)
    
    cleanup()
    return rank_stats

def main():
    parser = argparse.ArgumentParser(description='Data Loading Benchmark for Multi-GPU DDP')
    parser.add_argument('--world_size', type=int, default=None, 
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory (HDF5 or WebDataset format)')
    parser.add_argument('--chunk_size', type=int, default=30,
                        help='Action sequence length / chunk size')
    parser.add_argument('--batch_size', type=int, default=96,
                        help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers per GPU')
    parser.add_argument('--num_batches', type=int, default=100,
                        help='Number of batches to benchmark')
    parser.add_argument('--warmup_batches', type=int, default=10,
                        help='Number of warmup batches before benchmark')
    parser.add_argument('--force-reconvert', action='store_true',
                        help='Force reconversion of HDF5 to WebDataset even if already exists')
    parser.add_argument('--shard_size', type=int, default=500,
                        help='Number of samples per WebDataset shard (default: 500)')
    args = parser.parse_args()
    
    # Get number of available GPUs
    if args.world_size is None:
        world_size = torch.cuda.device_count()
    else:
        world_size = args.world_size
    
    if world_size == 0:
        print("No CUDA devices available. Exiting.")
        return
    
    print("=" * 80)
    print("DATA LOADING BENCHMARK - Multi-GPU DDP Setup")
    print("=" * 80)
    print(f"Original data directory: {args.data_dir}")
    print(f"Target GPUs: {world_size}")
    print("=" * 80)
    print()
    
    # Detect dataset format and convert if needed
    force_reconvert = getattr(args, 'force_reconvert', False)
    shard_size = getattr(args, 'shard_size', 500)
    success, webd_dir = detect_and_convert_dataset(args.data_dir, force_reconvert=force_reconvert, shard_size=shard_size)
    
    if not success:
        print("❌ Failed to prepare dataset. Exiting...")
        return
    
    print()
    print("=" * 80)
    print(f"Using WebDataset directory: {webd_dir}")
    print(f"Starting data loading benchmark on {world_size} GPUs...")
    print("=" * 80)
    print()
    
    # Spawn processes for distributed benchmark
    mp.spawn(benchmark_data_loading, args=(world_size, args, webd_dir), nprocs=world_size, join=True)
    
    print("\n" + "=" * 80)
    print("✅ Benchmark complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

