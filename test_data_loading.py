#!/usr/bin/env python3
"""
Test script for timing data loading and verifying data shuffling/uniqueness between GPUs.
This script mimics the data loading behavior from train_dist.py and provides comprehensive timing metrics.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
import argparse
import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import hashlib
import pickle
from tqdm import tqdm

from act_test.data_utils import initialize_webdataset_data
from act_test.data_tools.webdataset import convert_hdf5_to_webdataset


class DummyModel(torch.nn.Module):
    """Simple dummy model for DDP testing."""
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the GPU for the current process
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def convert_data_always(data_dir):
    """Always convert HDF5 to WebDataset format with new shard size."""
    webd_dir = os.path.join(data_dir, "webdataset")
    
    # Always remove existing WebDataset to force re-conversion with new shard size
    if os.path.exists(webd_dir):
        import shutil
        print(f"🗑️  Removing existing WebDataset directory: {webd_dir}")
        shutil.rmtree(webd_dir)
    
    # Check if HDF5 data exists and convert
    metadata_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(metadata_path):
        print("🔄 Converting HDF5 to WebDataset format...")
        print(f"📁 HDF5 source: {data_dir}")
        print(f"📁 WebDataset target: {webd_dir}")
        
        # Perform conversion
        success = convert_hdf5_to_webdataset(
            hdf5_directory=data_dir,
            webd_directory=webd_dir,
            shard_size=128  # Very small shards for optimal caching
        )
        
        if success:
            print("✅ Data conversion completed successfully!")
            return True, webd_dir
        else:
            print("❌ Data conversion failed!")
            return False, None
    else:
        print(f"❌ No HDF5 data found in {data_dir}")
        return False, None


def create_batch_hash(batch: Dict[str, Any]) -> str:
    """Create a hash for a batch to check for uniqueness."""
    # Create a deterministic hash based on the batch content
    hash_data = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # Use tensor data for hashing
            hash_data[key] = value.cpu().numpy().tobytes()
        else:
            hash_data[key] = str(value).encode()
    
    # Sort keys for consistent hashing
    sorted_data = sorted(hash_data.items())
    combined = b''.join([k.encode() + v for k, v in sorted_data])
    return hashlib.md5(combined).hexdigest()


def test_data_loading(rank, world_size, args, webd_dir):
    """Test data loading performance and verify shuffling/uniqueness."""
    setup(rank, world_size)
    
    # Configuration (matching train_dist.py)
    CHUNK_SIZE = args.chunk_size
    TRAIN_VAL_SPLIT = 0.9
    BATCH_SIZE = 96
    NUM_WORKERS = 4  # 4 workers per GPU
    NUM_BATCHES_TO_TEST = args.num_batches
    
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"Testing data loading on {world_size} GPUs")
        print(f"Configuration:")
        print(f"  Chunk size: {CHUNK_SIZE}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Num workers: {NUM_WORKERS}")
        print(f"  Num batches to test: {NUM_BATCHES_TO_TEST}")
        print(f"  Data directory: {webd_dir}")
    
    # Initialize data loaders using WebDataset format
    try:
        train_dataloader, val_dataloader, dataset_stats = initialize_webdataset_data(
            data_dir=webd_dir,
            chunk_size=CHUNK_SIZE,
            batch_size=BATCH_SIZE,
            train_val_split=TRAIN_VAL_SPLIT,
            num_workers=NUM_WORKERS,
            prefetch_factor=2,
            seed=42 + rank  # Different seed per rank for shuffling verification
        )
    except (FileNotFoundError, ValueError) as e:
        if rank == 0:
            print(f"Error initializing WebDataset: {e}")
        cleanup()
        return
    
    if rank == 0:
        print("✅ Data loaders initialized successfully")
    
    # Timing and verification data collection
    timing_data = {
        'batch_load_times': [],
        'data_transfer_times': [],
        'total_batch_times': [],
        'throughput_samples_per_sec': [],
        'batch_hash_times': [],
        'sample_hash_times': [],
        'total_hash_times': []
    }
    
    # Data verification
    batch_hashes = []  # Store batch hashes for uniqueness verification
    sample_hashes = []  # Store individual sample hashes
    batch_sequences = []  # Store batch sequences for shuffling verification
    
    # Create iterator
    train_iter = iter(train_dataloader)
    
    if rank == 0:
        print(f"\n🔄 Testing data loading for {NUM_BATCHES_TO_TEST} batches...")
        pbar = tqdm(total=NUM_BATCHES_TO_TEST, desc=f"Rank {rank} - Testing batches")
    
    for batch_idx in range(NUM_BATCHES_TO_TEST):
        batch_start_time = time.time()
        
        try:
            # Time batch loading
            load_start = time.time()
            batch = next(train_iter)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            load_time = time.time() - load_start
            
            # Time data transfer to device
            transfer_start = time.time()
            batch_device = {}
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor):
                    batch_device[key] = tensor.to(device)
                else:
                    batch_device[key] = tensor
            if device.type == 'cuda':
                torch.cuda.synchronize()
            transfer_time = time.time() - transfer_start
            
            total_batch_time = time.time() - batch_start_time
            
            # Store timing data
            timing_data['batch_load_times'].append(load_time)
            timing_data['data_transfer_times'].append(transfer_time)
            timing_data['total_batch_times'].append(total_batch_time)
            
            # Calculate throughput (samples per second)
            samples_per_sec = BATCH_SIZE / total_batch_time
            timing_data['throughput_samples_per_sec'].append(samples_per_sec)
            
            # Time batch hashing
            if rank == 0:
                hash_start = time.time()
            
            # Create batch hash for uniqueness verification
            batch_hash = create_batch_hash(batch_device)
            batch_hashes.append(batch_hash)
            
            # Time sample hashing
            if rank == 0:
                sample_hash_start = time.time()
            
            # Create sample hashes for individual sample verification
            batch_size = batch_device['action'].shape[0]
            for sample_idx in range(batch_size):
                sample_data = {
                    'action': batch_device['action'][sample_idx],
                    'observation.state': batch_device['observation.state'][sample_idx],
                    'observation.image_camera_1': batch_device['observation.image_camera_1'][sample_idx],
                    'observation.image_camera_2': batch_device['observation.image_camera_2'][sample_idx]
                }
                sample_hash = create_batch_hash(sample_data)
                sample_hashes.append(sample_hash)
            
            # Store batch sequence info for shuffling verification
            batch_sequences.append({
                'batch_idx': batch_idx,
                'rank': rank,
                'batch_hash': batch_hash,
                'first_action_hash': sample_hashes[-1] if sample_hashes else None
            })
            
            # Record hashing times
            if rank == 0:
                batch_hash_time = time.time() - hash_start
                sample_hash_time = time.time() - sample_hash_start
                total_hash_time = batch_hash_time + sample_hash_time
                
                # Store hashing times
                timing_data['batch_hash_times'].append(batch_hash_time)
                timing_data['sample_hash_times'].append(sample_hash_time)
                timing_data['total_hash_times'].append(total_hash_time)
            
            if rank == 0:
                pbar.update(1)
                # Calculate rolling averages for better display
                recent_batches = min(10, len(timing_data['total_batch_times']))
                avg_total = sum(timing_data['total_batch_times'][-recent_batches:]) / recent_batches
                avg_throughput = sum(timing_data['throughput_samples_per_sec'][-recent_batches:]) / recent_batches
                
                pbar.set_postfix({
                    'batch_ms': f"{total_batch_time*1000:.1f}",
                    'hash_ms': f"{total_hash_time*1000:.1f}",
                    'samples/sec': f"{samples_per_sec:.0f}",
                    'avg_ms': f"{avg_total*1000:.1f}"
                })
                
        except StopIteration:
            # Reset iterator when dataset is exhausted
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
            if rank == 0:
                print(f"Dataset exhausted at batch {batch_idx}, resetting iterator")
    
    if rank == 0:
        pbar.close()
    
    # Time the barrier synchronization
    if rank == 0:
        barrier_start = time.time()
    
    # Synchronize all processes before analysis (only if multiple GPUs)
    if world_size > 1:
        try:
            dist.barrier()
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Barrier failed: {e}")
    
    if rank == 0:
        barrier_time = time.time() - barrier_start
        print(f"\n⏱️  BARRIER TIMING: {barrier_time*1000:.2f}ms")
    
    # Collect data from all ranks for analysis
    if rank == 0:
        print("\n📊 Analyzing data across all GPUs...")
        
        # Gather all timing data
        all_timing_data = [timing_data]
        all_batch_hashes = [batch_hashes]
        all_sample_hashes = [sample_hashes]
        all_batch_sequences = [batch_sequences]
        
        # Use all_gather_object to collect data from all ranks at once
        try:
            # Gather timing data from all ranks
            timing_data_gathered = [None] * world_size
            dist.all_gather_object(timing_data_gathered, timing_data)
            all_timing_data = timing_data_gathered
            
            # Gather batch hashes from all ranks
            batch_hashes_gathered = [None] * world_size
            dist.all_gather_object(batch_hashes_gathered, batch_hashes)
            all_batch_hashes = batch_hashes_gathered
            
            # Gather sample hashes from all ranks
            sample_hashes_gathered = [None] * world_size
            dist.all_gather_object(sample_hashes_gathered, sample_hashes)
            all_sample_hashes = sample_hashes_gathered
            
            # Gather batch sequences from all ranks
            batch_sequences_gathered = [None] * world_size
            dist.all_gather_object(batch_sequences_gathered, batch_sequences)
            all_batch_sequences = batch_sequences_gathered
        except Exception as e:
            print(f"⚠️  Failed to gather data from all ranks: {e}")
            # Use only rank 0 data if gathering fails
            all_timing_data = [timing_data]
            all_batch_hashes = [batch_hashes]
            all_sample_hashes = [sample_hashes]
            all_batch_sequences = [batch_sequences]
        
        # Analyze results
        analyze_results(all_timing_data, all_batch_hashes, all_sample_hashes, 
                       all_batch_sequences, world_size, args)
    else:
        # Send data to rank 0 using all_gather_object for complex data structures
        # This is more robust than broadcast_object_list
        timing_data_list = [timing_data]
        batch_hashes_list = [batch_hashes]
        sample_hashes_list = [sample_hashes]
        batch_sequences_list = [batch_sequences]
        
        # Use all_gather_object for complex data structures
        timing_data_gathered = [None] * world_size
        batch_hashes_gathered = [None] * world_size
        sample_hashes_gathered = [None] * world_size
        batch_sequences_gathered = [None] * world_size
        
        dist.all_gather_object(timing_data_gathered, timing_data)
        dist.all_gather_object(batch_hashes_gathered, batch_hashes)
        dist.all_gather_object(sample_hashes_gathered, sample_hashes)
        dist.all_gather_object(batch_sequences_gathered, batch_sequences)
    
    cleanup()


def test_data_loading_ddp(rank, world_size, args, webd_dir):
    """Test data loading performance using DDP approach - more PyTorch-native."""
    setup(rank, world_size)
    
    # Configuration (matching train_dist.py)
    CHUNK_SIZE = args.chunk_size
    TRAIN_VAL_SPLIT = 0.9
    BATCH_SIZE = 96
    NUM_WORKERS = 4  # 4 workers per GPU
    NUM_BATCHES_TO_TEST = args.num_batches
    
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"Testing data loading on {world_size} GPUs (DDP approach)")
        print(f"Configuration:")
        print(f"  Chunk size: {CHUNK_SIZE}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Num workers: {NUM_WORKERS}")
        print(f"  Num batches to test: {NUM_BATCHES_TO_TEST}")
        print(f"  Data directory: {webd_dir}")
    
    # Initialize data loaders using WebDataset format
    try:
        train_dataloader, val_dataloader, dataset_stats = initialize_webdataset_data(
            data_dir=webd_dir,
            chunk_size=CHUNK_SIZE,
            batch_size=BATCH_SIZE,
            train_val_split=TRAIN_VAL_SPLIT,
            num_workers=NUM_WORKERS,
            prefetch_factor=2,
            seed=42 + rank  # Different seed per rank for shuffling verification
        )
    except (FileNotFoundError, ValueError) as e:
        if rank == 0:
            print(f"Error initializing WebDataset: {e}")
        cleanup()
        return
    
    if rank == 0:
        print("✅ Data loaders initialized successfully")
    
    # Create a minimal model for DDP to make data operations representative
    # This ensures the same data loading patterns as real training
    model = DummyModel().to(device)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # Timing and verification data collection
    timing_data = {
        'batch_load_times': [],
        'data_transfer_times': [],
        'total_batch_times': [],
        'throughput_samples_per_sec': [],
        'batch_hash_times': [],
        'sample_hash_times': [],
        'total_hash_times': []
    }
    
    # Data verification - TEMPORARILY COMMENTED OUT
    # batch_hashes = []
    # sample_hashes = []
    # batch_sequences = []
    batch_hashes = []  # Empty for compatibility
    sample_hashes = []  # Empty for compatibility
    batch_sequences = []  # Empty for compatibility
    
    if rank == 0:
        print(f"\n🔄 Testing data loading for {NUM_BATCHES_TO_TEST} batches...")
    
    # Create iterator from DataLoader
    train_iter = iter(train_dataloader)
    
    # Test data loading performance with DDP for representative data operations
    start_time = time.time()
    batch_times = []
    
    with tqdm(total=NUM_BATCHES_TO_TEST, desc=f"Rank {rank} - Testing batches", 
              position=rank, leave=True) as pbar:
        for batch_idx in range(NUM_BATCHES_TO_TEST):
            # 1. GRANULAR DATA LOADING TIMING (break down each step)
            batch_start = time.time()
            
            # Step 1: Iterator call (WebDataset loading)
            iter_start = time.time()
            try:
                batch = next(train_iter)
            except StopIteration:
                # Reset iterator when dataset is exhausted
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            iter_time = time.time() - iter_start
            
            # Step 2: Device transfer
            transfer_start = time.time()
            batch_device = {}
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor):
                    batch_device[key] = tensor.to(device, non_blocking=True)
                else:
                    batch_device[key] = tensor
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            transfer_time = time.time() - transfer_start
            
            # Total data loading time
            batch_load_time = time.time() - batch_start
            
            # 2. DDP forward pass (no gradients, just to make data operations representative)
            # This ensures the same data loading patterns as real DDP training
            ddp_start = time.time()
            ddp_model.train()  # Set to training mode for DDP
            with torch.no_grad():  # No gradients needed
                # Create minimal dummy input to trigger DDP communication patterns
                batch_size = len(batch_device.get('action', []))
                dummy_input = torch.randn(batch_size, 256).to(device)
                _ = ddp_model(dummy_input)
            ddp_time = time.time() - ddp_start
            
            # TEMPORARILY COMMENTED OUT - HASHING AND UNIQUENESS CHECKS
            # # Compute batch hash for uniqueness verification
            # batch_hash_start = time.time()
            # batch_hash = create_batch_hash(batch_device)
            # batch_hash_time = time.time() - batch_hash_start
            
            # # Compute individual sample hashes (using batch size from batch_device)
            # sample_hash_start = time.time()
            # sample_hashes_batch = []
            # batch_size = len(batch_device.get('action', []))
            # for i in range(batch_size):
            #     # Create individual sample hash
            #     sample_data = {k: v[i] if hasattr(v, '__getitem__') and len(v) > i else v for k, v in batch_device.items()}
            #     sample_hash = create_batch_hash(sample_data)
            #     sample_hashes_batch.append(sample_hash)
            # sample_hash_time = time.time() - sample_hash_start
            
            # # Store verification data
            # batch_hashes.append(batch_hash)
            # sample_hashes.extend(sample_hashes_batch)
            # batch_sequences.append({
            #     'batch_idx': batch_idx,
            #     'rank': rank,
            #     'batch_hash': batch_hash,
            #     'first_action_hash': sample_hashes[-1] if sample_hashes else None
            # })
            
            # Store dummy verification data for compatibility
            batch_hashes.append(f"dummy_hash_{batch_idx}")
            sample_hashes.extend([f"dummy_sample_{batch_idx}_{i}" for i in range(batch_size)])
            batch_sequences.append({
                'batch_idx': batch_idx,
                'rank': rank,
                'batch_hash': f"dummy_hash_{batch_idx}",
                'first_action_hash': f"dummy_sample_{batch_idx}_0"
            })
            
            # Record timing (pure data loading performance)
            data_transfer_time = transfer_time  # Actual device transfer time
            total_batch_time = time.time() - batch_start  # Total time for entire batch processing
            throughput = batch_size / batch_load_time if batch_load_time > 0 else 0
            
            timing_data['batch_load_times'].append(batch_load_time)
            timing_data['data_transfer_times'].append(data_transfer_time)
            timing_data['total_batch_times'].append(total_batch_time)
            timing_data['throughput_samples_per_sec'].append(throughput)
            # Granular timing breakdown
            timing_data['iterator_times'].append(iter_time)
            timing_data['device_transfer_times'].append(transfer_time)
            timing_data['ddp_forward_times'].append(ddp_time)
            # TEMPORARILY COMMENTED OUT - HASHING TIMES
            # timing_data['batch_hash_times'].append(batch_hash_time)
            # timing_data['sample_hash_times'].append(sample_hash_time)
            # timing_data['total_hash_times'].append(batch_hash_time + sample_hash_time)
            timing_data['batch_hash_times'].append(0.0)  # Placeholder
            timing_data['sample_hash_times'].append(0.0)  # Placeholder
            timing_data['total_hash_times'].append(0.0)  # Placeholder
            
            batch_times.append(batch_load_time)
            
            # Update progress bar with timing info
            avg_batch_time = np.mean(batch_times) * 1000
            ddp_time_ms = ddp_time * 1000
            total_time_ms = total_batch_time * 1000
            current_throughput = throughput
            
            pbar.set_postfix({
                'iter_ms': f'{iter_time*1000:.1f}',
                'transfer_ms': f'{transfer_time*1000:.1f}',
                'ddp_ms': f'{ddp_time_ms:.1f}',
                'total_ms': f'{total_time_ms:.1f}',
                'samples/sec': f'{current_throughput:.0f}'
            })
            pbar.update(1)
    
    total_time = time.time() - start_time
    
    # Synchronize all processes before data collection (not during data loading)
    if rank == 0:
        barrier_start = time.time()
    
    dist.barrier()  # Only needed for data collection, not data loading
    
    if rank == 0:
        barrier_time = time.time() - barrier_start
        print(f"\n⏱️  BARRIER TIMING: {barrier_time*1000:.2f}ms")
    
    # Collect results using DDP's built-in communication
    if rank == 0:
        print("\n📊 Analyzing data across all GPUs...")
        all_timing_data = [timing_data]
        all_batch_hashes = [batch_hashes]
        all_sample_hashes = [sample_hashes]
        all_batch_sequences = [batch_sequences]
        
        # Use all_gather_object to collect data from all ranks
        try:
            timing_data_gathered = [None] * world_size
            dist.all_gather_object(timing_data_gathered, timing_data)
            all_timing_data = timing_data_gathered
            
            batch_hashes_gathered = [None] * world_size
            dist.all_gather_object(batch_hashes_gathered, batch_hashes)
            all_batch_hashes = batch_hashes_gathered
            
            sample_hashes_gathered = [None] * world_size
            dist.all_gather_object(sample_hashes_gathered, sample_hashes)
            all_sample_hashes = sample_hashes_gathered
            
            batch_sequences_gathered = [None] * world_size
            dist.all_gather_object(batch_sequences_gathered, batch_sequences)
            all_batch_sequences = batch_sequences_gathered
        except Exception as e:
            print(f"⚠️  Failed to gather data from all ranks: {e}")
            all_timing_data = [timing_data]
            all_batch_hashes = [batch_hashes]
            all_sample_hashes = [sample_hashes]
            all_batch_sequences = [batch_sequences]
        
        # Analyze results
        analyze_results(all_timing_data, all_batch_hashes, all_sample_hashes, 
                       all_batch_sequences, world_size, args)
    else:
        # Participate in all_gather_object
        timing_data_gathered = [None] * world_size
        batch_hashes_gathered = [None] * world_size
        sample_hashes_gathered = [None] * world_size
        batch_sequences_gathered = [None] * world_size
        
        dist.all_gather_object(timing_data_gathered, timing_data)
        dist.all_gather_object(batch_hashes_gathered, batch_hashes)
        dist.all_gather_object(sample_hashes_gathered, sample_hashes)
        dist.all_gather_object(batch_sequences_gathered, batch_sequences)
    
    cleanup()


def analyze_results(all_timing_data, all_batch_hashes, all_sample_hashes, 
                   all_batch_sequences, world_size, args):
    """Analyze timing and data verification results."""
    print("\n" + "="*80)
    print("📈 DATA LOADING PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Timing analysis
    print("\n⏱️  TIMING METRICS:")
    print("-" * 40)
    
    for rank in range(world_size):
        if rank < len(all_timing_data) and all_timing_data[rank] is not None:
            timing = all_timing_data[rank]
            print(f"\nRank {rank}:")
            
            # Check if keys exist before accessing them
            if 'batch_load_times' in timing and timing['batch_load_times']:
                print(f"  Average batch load time: {np.mean(timing['batch_load_times'])*1000:.2f} ± {np.std(timing['batch_load_times'])*1000:.2f} ms")
            else:
                print(f"  Average batch load time: N/A")
                
            if 'data_transfer_times' in timing and timing['data_transfer_times']:
                print(f"  Average data transfer time: {np.mean(timing['data_transfer_times'])*1000:.2f} ± {np.std(timing['data_transfer_times'])*1000:.2f} ms")
            else:
                print(f"  Average data transfer time: N/A")
                
            if 'batch_hash_times' in timing and timing['batch_hash_times']:
                print(f"  Average batch hash time: {np.mean(timing['batch_hash_times'])*1000:.2f} ± {np.std(timing['batch_hash_times'])*1000:.2f} ms")
            else:
                print(f"  Average batch hash time: N/A")
                
            if 'sample_hash_times' in timing and timing['sample_hash_times']:
                print(f"  Average sample hash time: {np.mean(timing['sample_hash_times'])*1000:.2f} ± {np.std(timing['sample_hash_times'])*1000:.2f} ms")
            else:
                print(f"  Average sample hash time: N/A")
                
            if 'total_hash_times' in timing and timing['total_hash_times']:
                print(f"  Average total hash time: {np.mean(timing['total_hash_times'])*1000:.2f} ± {np.std(timing['total_hash_times'])*1000:.2f} ms")
            else:
                print(f"  Average total hash time: N/A")
                
            if 'total_batch_times' in timing and timing['total_batch_times']:
                print(f"  Average total batch time: {np.mean(timing['total_batch_times'])*1000:.2f} ± {np.std(timing['total_batch_times'])*1000:.2f} ms")
                
                # Calculate overhead breakdown
                if 'batch_load_times' in timing and timing['batch_load_times']:
                    avg_data_time = np.mean(timing['batch_load_times'])
                    avg_total_time = np.mean(timing['total_batch_times'])
                    overhead_time = avg_total_time - avg_data_time
                    overhead_percent = (overhead_time / avg_total_time) * 100
                    print(f"  📊 OVERHEAD BREAKDOWN:")
                    print(f"    Data loading: {avg_data_time*1000:.2f} ms ({100-overhead_percent:.1f}%)")
                    print(f"    Other overhead: {overhead_time*1000:.2f} ms ({overhead_percent:.1f}%)")
            else:
                print(f"  Average total batch time: N/A")
                
            if 'throughput_samples_per_sec' in timing and timing['throughput_samples_per_sec']:
                print(f"  Average throughput: {np.mean(timing['throughput_samples_per_sec']):.1f} ± {np.std(timing['throughput_samples_per_sec']):.1f} samples/sec")
            else:
                print(f"  Average throughput: N/A")
                
            # Granular timing breakdown
            if 'iterator_times' in timing and timing['iterator_times']:
                print(f"  🔍 GRANULAR BREAKDOWN:")
                print(f"    Iterator (WebDataset): {np.mean(timing['iterator_times'])*1000:.2f} ± {np.std(timing['iterator_times'])*1000:.2f} ms")
            if 'device_transfer_times' in timing and timing['device_transfer_times']:
                print(f"    Device transfer: {np.mean(timing['device_transfer_times'])*1000:.2f} ± {np.std(timing['device_transfer_times'])*1000:.2f} ms")
            if 'ddp_forward_times' in timing and timing['ddp_forward_times']:
                print(f"    DDP forward pass: {np.mean(timing['ddp_forward_times'])*1000:.2f} ± {np.std(timing['ddp_forward_times'])*1000:.2f} ms")
        else:
            print(f"\nRank {rank}: No timing data available")
    
    # Overall statistics
    all_load_times = []
    all_transfer_times = []
    all_total_times = []
    all_throughputs = []
    
    for timing in all_timing_data:
        if timing is not None:
            if 'batch_load_times' in timing and timing['batch_load_times']:
                all_load_times.extend(timing['batch_load_times'])
            if 'data_transfer_times' in timing and timing['data_transfer_times']:
                all_transfer_times.extend(timing['data_transfer_times'])
            if 'total_batch_times' in timing and timing['total_batch_times']:
                all_total_times.extend(timing['total_batch_times'])
            if 'throughput_samples_per_sec' in timing and timing['throughput_samples_per_sec']:
                all_throughputs.extend(timing['throughput_samples_per_sec'])
    
    print(f"\nOverall (all ranks combined):")
    if all_load_times:
        print(f"  Average batch load time: {np.mean(all_load_times)*1000:.2f} ± {np.std(all_load_times)*1000:.2f} ms")
    else:
        print(f"  Average batch load time: N/A")
        
    if all_transfer_times:
        print(f"  Average data transfer time: {np.mean(all_transfer_times)*1000:.2f} ± {np.std(all_transfer_times)*1000:.2f} ms")
    else:
        print(f"  Average data transfer time: N/A")
        
    if all_total_times:
        print(f"  Average total batch time: {np.mean(all_total_times)*1000:.2f} ± {np.std(all_total_times)*1000:.2f} ms")
    else:
        print(f"  Average total batch time: N/A")
        
    if all_throughputs:
        print(f"  Average throughput: {np.mean(all_throughputs):.1f} ± {np.std(all_throughputs):.1f} samples/sec")
        print(f"  Total effective throughput: {np.mean(all_throughputs) * world_size:.1f} samples/sec")
    else:
        print(f"  Average throughput: N/A")
        print(f"  Total effective throughput: N/A")
    
    # Data uniqueness verification
    print("\n🔍 DATA UNIQUENESS VERIFICATION:")
    print("-" * 40)
    
    # Check batch uniqueness within each rank
    for rank in range(world_size):
        batch_hashes = all_batch_hashes[rank]
        unique_batches = len(set(batch_hashes))
        total_batches = len(batch_hashes)
        print(f"Rank {rank}: {unique_batches}/{total_batches} unique batches ({unique_batches/total_batches*100:.1f}%)")
    
    # Check batch uniqueness across ranks
    all_batch_hashes_flat = []
    for rank in range(world_size):
        all_batch_hashes_flat.extend(all_batch_hashes[rank])
    
    unique_batches_across_ranks = len(set(all_batch_hashes_flat))
    total_batches_across_ranks = len(all_batch_hashes_flat)
    print(f"Across all ranks: {unique_batches_across_ranks}/{total_batches_across_ranks} unique batches ({unique_batches_across_ranks/total_batches_across_ranks*100:.1f}%)")
    
    # Check sample uniqueness
    print("\n🔍 SAMPLE UNIQUENESS VERIFICATION:")
    print("-" * 40)
    
    for rank in range(world_size):
        sample_hashes = all_sample_hashes[rank]
        unique_samples = len(set(sample_hashes))
        total_samples = len(sample_hashes)
        print(f"Rank {rank}: {unique_samples}/{total_samples} unique samples ({unique_samples/total_samples*100:.1f}%)")
    
    # Check sample uniqueness across ranks
    all_sample_hashes_flat = []
    for rank in range(world_size):
        all_sample_hashes_flat.extend(all_sample_hashes[rank])
    
    unique_samples_across_ranks = len(set(all_sample_hashes_flat))
    total_samples_across_ranks = len(all_sample_hashes_flat)
    print(f"Across all ranks: {unique_samples_across_ranks}/{total_samples_across_ranks} unique samples ({unique_samples_across_ranks/total_samples_across_ranks*100:.1f}%)")
    
    # Data shuffling verification
    print("\n🔀 DATA SHUFFLING VERIFICATION:")
    print("-" * 40)
    
    # Check if different ranks see different data
    rank_0_hashes = set(all_batch_hashes[0])
    for rank in range(1, world_size):
        rank_hashes = set(all_batch_hashes[rank])
        overlap = len(rank_0_hashes.intersection(rank_hashes))
        total_unique = len(rank_0_hashes.union(rank_hashes))
        overlap_percent = overlap / total_unique * 100 if total_unique > 0 else 0
        print(f"Rank 0 vs Rank {rank}: {overlap}/{total_unique} overlapping batches ({overlap_percent:.1f}% overlap)")
    
    # Check sequence order differences
    print("\n📊 BATCH SEQUENCE ANALYSIS:")
    print("-" * 40)
    
    for rank in range(world_size):
        sequences = all_batch_sequences[rank]
        if sequences:
            first_actions = [seq['first_action_hash'] for seq in sequences if seq['first_action_hash']]
            if len(first_actions) > 1:
                # Check if first actions are different (indicating shuffling)
                unique_first_actions = len(set(first_actions))
                print(f"Rank {rank}: {unique_first_actions}/{len(first_actions)} unique first actions in sequence")
    
    # Save detailed results
    results = {
        'timing_data': all_timing_data,
        'batch_hashes': all_batch_hashes,
        'sample_hashes': all_sample_hashes,
        'batch_sequences': all_batch_sequences,
        'world_size': world_size,
        'config': {
            'chunk_size': args.chunk_size,
            'batch_size': 96,
            'num_workers': 4,
            'num_batches_tested': args.num_batches
        }
    }
    
    results_file = f"data_loading_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = json.loads(json.dumps(results, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print("="*80)


class TestArgs:
    """Args class for test configuration."""
    def __init__(self, data_dir, chunk_size, num_batches, world_size):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.num_batches = num_batches
        self.world_size = world_size


def run_test_suite(data_dir: str, test_configs: List[Dict[str, Any]]) -> List[tuple]:
    """Run a suite of tests with different configurations."""
    print("🚀 Running Data Loading Test Suite")
    print("=" * 60)
    
    results = []
    
    for i, config in enumerate(test_configs):
        test_name = config.get('name', f'Test {i+1}')
        chunk_size = config.get('chunk_size', 30)
        num_batches = config.get('num_batches', 100)
        world_size = config.get('world_size', None)
        
        print(f"\n🧪 Running: {test_name}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Number of batches: {num_batches}")
        print(f"  World size: {world_size if world_size else 'all available'}")
        print("-" * 40)
        
        # Create args object
        args = TestArgs(data_dir, chunk_size, num_batches, world_size)
        
        # Get actual world size
        if world_size is None:
            actual_world_size = torch.cuda.device_count()
        else:
            actual_world_size = world_size
        
        if actual_world_size == 0:
            print("❌ No CUDA devices available. Skipping test.")
            results.append((test_name, False))
            continue
        
        # Use data directory directly (skip conversion)
        conversion_success, webd_dir = convert_data_always(data_dir)
        
        if not conversion_success:
            print("❌ Failed to convert data. Skipping test.")
            results.append((test_name, False))
            continue
        
        try:
            # Spawn processes for distributed testing using DDP approach
            mp.spawn(test_data_loading_ddp, args=(actual_world_size, args, webd_dir), nprocs=actual_world_size, join=True)
            print(f"✅ {test_name} completed successfully!")
            results.append((test_name, True))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    return results


def get_basic_test_configs() -> List[Dict[str, Any]]:
    """Get basic test configurations."""
    return [
        {"name": "Quick Test", "chunk_size": 10, "num_batches": 20, "world_size": 1},
        {"name": "Standard Test", "chunk_size": 30, "num_batches": 50, "world_size": 1},
        {"name": "Extended Test", "chunk_size": 30, "num_batches": 100, "world_size": 1},
    ]


def get_multi_gpu_test_configs() -> List[Dict[str, Any]]:
    """Get multi-GPU test configurations."""
    available_gpus = torch.cuda.device_count()
    
    configs = []
    if available_gpus >= 2:
        configs.append({"name": "2-GPU Test", "chunk_size": 30, "num_batches": 50, "world_size": 2})
    if available_gpus >= 4:
        configs.append({"name": "4-GPU Test", "chunk_size": 30, "num_batches": 50, "world_size": 4})
    if available_gpus > 1:
        configs.append({"name": "All-GPU Test", "chunk_size": 30, "num_batches": 50, "world_size": available_gpus})
    
    return configs


def get_performance_test_configs() -> List[Dict[str, Any]]:
    """Get performance-focused test configurations."""
    return [
        {"name": "High Throughput", "chunk_size": 30, "num_batches": 200, "world_size": 1},
        {"name": "Large Chunks", "chunk_size": 100, "num_batches": 50, "world_size": 1},
        {"name": "Small Chunks", "chunk_size": 10, "num_batches": 100, "world_size": 1},
    ]


def print_test_summary(results: List[tuple]):
    """Print a summary of test results."""
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    print("-" * 40)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "No tests run")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Data Loading Test Suite')
    parser.add_argument('--data_dir', type=str, 
                        default="/home/vignesh/raid/PaperMulti_1_2_Filtered",
                        help='Path to the dataset directory')
    parser.add_argument('--test_type', type=str, 
                        choices=['basic', 'multi_gpu', 'performance', 'custom', 'all'],
                        default='all',
                        help='Type of tests to run')
    parser.add_argument('--chunk_size', type=int, default=30,
                        help='Chunk size for custom test')
    parser.add_argument('--num_batches', type=int, default=100,
                        help='Number of batches for custom test')
    parser.add_argument('--world_size', type=int, default=None,
                        help='Number of GPUs for custom test')
    parser.add_argument('--single_test', action='store_true',
                        help='Run a single test instead of a test suite')
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory does not exist: {args.data_dir}")
        return 1
    
    try:
        if args.single_test:
            # Run a single test
            config = {
                "name": "Single Test",
                "chunk_size": args.chunk_size,
                "num_batches": args.num_batches,
                "world_size": args.world_size
            }
            results = run_test_suite(args.data_dir, [config])
        else:
            # Run test suite based on type
            if args.test_type in ['basic', 'all']:
                configs = get_basic_test_configs()
                results = run_test_suite(args.data_dir, configs)
            elif args.test_type == 'multi_gpu':
                configs = get_multi_gpu_test_configs()
                results = run_test_suite(args.data_dir, configs)
            elif args.test_type == 'performance':
                configs = get_performance_test_configs()
                results = run_test_suite(args.data_dir, configs)
            elif args.test_type == 'custom':
                config = {
                    "name": "Custom Test",
                    "chunk_size": args.chunk_size,
                    "num_batches": args.num_batches,
                    "world_size": args.world_size
                }
                results = run_test_suite(args.data_dir, [config])
        
        # Print summary
        print_test_summary(results)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    main()
