#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import hashlib
from collections import defaultdict
import glob
import re
from functools import partial

# Add the current directory to path so we can import local modules
sys.path.append('/home/vignesh/act_test/act_test')

from data_utils import train_split_filter, val_split_filter, WebDatasetDecoder
import webdataset as wds

def hash_tensor(tensor):
    """Create a hash of a tensor for comparison."""
    if isinstance(tensor, torch.Tensor):
        return hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()
    return None

def hash_batch(batch):
    """Create a hash of the entire batch for comparison."""
    batch_str = ""
    for key in sorted(batch.keys()):
        if isinstance(batch[key], torch.Tensor):
            batch_str += f"{key}:{hash_tensor(batch[key])}"
    return hashlib.md5(batch_str.encode()).hexdigest()

def create_rank_dataloader(rank, world_size, webd_dir, chunk_size=30, batch_size=8):
    """Create a dataloader for a specific rank, mimicking the training setup."""
    print(f"\n🔧 Setting up dataloader for rank {rank}")
    
    # Use the same seed strategy as in training: seed = 42 + rank
    seed = 42 + rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Find all .tar files and create pattern (same as in training)
    dataset_pattern = os.path.join(webd_dir, "train-*.tar")
    all_files = sorted(glob.glob(dataset_pattern))
    
    if len(all_files) == 0:
        raise FileNotFoundError(f"No WebDataset .tar files found in {webd_dir}")
    
    print(f"   Found {len(all_files)} WebDataset files for rank {rank}")
    
    # Create pattern for all files (same logic as training)
    if len(all_files) == 1:
        full_pattern = all_files[0]
    else:
        first_file = all_files[0]
        last_file = all_files[-1]
        
        match_first = re.search(r'train-(\d+)\.tar', first_file)
        match_last = re.search(r'train-(\d+)\.tar', last_file)
        
        if match_first and match_last:
            start_num = int(match_first.group(1))
            end_num = int(match_last.group(1))
            base_path = first_file.replace(match_first.group(0), "train-{%05d..%05d}.tar" % (start_num, end_num))
            full_pattern = base_path
        else:
            full_pattern = "{" + ",".join(all_files) + "}"
    
    print(f"   Using dataset pattern: {full_pattern}")
    
    # Create decode function
    decode_fn = WebDatasetDecoder(chunk_size)
    
    # Create split function (same as training)
    train_split_ratio = 0.8
    train_split_fn = partial(train_split_filter, split_ratio=train_split_ratio)
    
    # Create train dataset (same setup as training)
    train_dataset = (
        wds.WebDataset(full_pattern, shardshuffle=True)
        .decode("pil")
        .to_tuple("cam1.png", "cam2.png", "qpos.npy", "actions.npy")
        .select(train_split_fn)
        .map(decode_fn)
    )
    
    # Create dataloader (same parameters as training)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,  # Same as training
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    return train_dataloader

def test_batch_distribution():
    """Test if different ranks are getting different batches."""
    
    webd_dir = "/media/vignesh/External4TB/socks1wed_socks2wed_filt_merged_webdataset"
    num_ranks = 4
    num_batches_to_test = 5
    chunk_size = 30
    batch_size = 8
    
    print("🧪 TESTING BATCH DISTRIBUTION ACROSS RANKS")
    print("=" * 60)
    print(f"📁 WebDataset directory: {webd_dir}")
    print(f"🎯 Testing {num_ranks} ranks with {num_batches_to_test} batches each")
    print(f"📦 Batch size: {batch_size}, Chunk size: {chunk_size}")
    
    # Create dataloaders for each rank
    dataloaders = {}
    iterators = {}
    
    for rank in range(num_ranks):
        try:
            dataloaders[rank] = create_rank_dataloader(
                rank=rank, 
                world_size=num_ranks, 
                webd_dir=webd_dir,
                chunk_size=chunk_size,
                batch_size=batch_size
            )
            iterators[rank] = iter(dataloaders[rank])
            print(f"✅ Rank {rank} dataloader created successfully")
        except Exception as e:
            print(f"❌ Failed to create dataloader for rank {rank}: {e}")
            return False
    
    print(f"\n🔍 SAMPLING {num_batches_to_test} BATCHES FROM EACH RANK")
    print("-" * 60)
    
    # Store batches for comparison
    rank_batches = defaultdict(list)
    batch_hashes = defaultdict(list)
    qpos_hashes = defaultdict(list)
    action_hashes = defaultdict(list)
    
    # Sample batches from each rank
    for batch_idx in range(num_batches_to_test):
        print(f"\n📊 Sampling batch {batch_idx + 1}/{num_batches_to_test}")
        
        for rank in range(num_ranks):
            try:
                batch = next(iterators[rank])
                rank_batches[rank].append(batch)
                
                # Create various hashes for comparison
                batch_hash = hash_batch(batch)
                qpos_hash = hash_tensor(batch['observation.state'])
                action_hash = hash_tensor(batch['action'])
                
                batch_hashes[rank].append(batch_hash)
                qpos_hashes[rank].append(qpos_hash)
                action_hashes[rank].append(action_hash)
                
                print(f"   Rank {rank}: batch_hash={batch_hash[:8]}..., "
                      f"qpos_hash={qpos_hash[:8]}..., action_hash={action_hash[:8]}...")
                
            except StopIteration:
                print(f"   ⚠️  Rank {rank}: Iterator exhausted at batch {batch_idx}")
                return False
            except Exception as e:
                print(f"   ❌ Rank {rank}: Error sampling batch: {e}")
                return False
    
    print(f"\n🔍 ANALYZING BATCH DISTRIBUTION")
    print("-" * 60)
    
    # Check for duplicates across ranks
    all_batch_hashes = []
    all_qpos_hashes = []
    all_action_hashes = []
    
    for rank in range(num_ranks):
        all_batch_hashes.extend([(rank, h) for h in batch_hashes[rank]])
        all_qpos_hashes.extend([(rank, h) for h in qpos_hashes[rank]])
        all_action_hashes.extend([(rank, h) for h in action_hashes[rank]])
    
    # Check for exact batch duplicates
    batch_hash_counts = defaultdict(list)
    for rank, h in all_batch_hashes:
        batch_hash_counts[h].append(rank)
    
    duplicate_batches = {h: ranks for h, ranks in batch_hash_counts.items() if len(ranks) > 1}
    
    # Check for qpos duplicates
    qpos_hash_counts = defaultdict(list)
    for rank, h in all_qpos_hashes:
        qpos_hash_counts[h].append(rank)
    
    duplicate_qpos = {h: ranks for h, ranks in qpos_hash_counts.items() if len(ranks) > 1}
    
    # Check for action duplicates
    action_hash_counts = defaultdict(list)
    for rank, h in all_action_hashes:
        action_hash_counts[h].append(rank)
    
    duplicate_actions = {h: ranks for h, ranks in action_hash_counts.items() if len(ranks) > 1}
    
    # Print results
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 60)
    
    total_batches = num_ranks * num_batches_to_test
    unique_batch_hashes = len(set(h for _, h in all_batch_hashes))
    unique_qpos_hashes = len(set(h for _, h in all_qpos_hashes))
    unique_action_hashes = len(set(h for _, h in all_action_hashes))
    
    print(f"📦 Total batches sampled: {total_batches}")
    print(f"🔍 Unique full batch hashes: {unique_batch_hashes}/{total_batches}")
    print(f"🎯 Unique qpos hashes: {unique_qpos_hashes}/{total_batches}")
    print(f"🎬 Unique action hashes: {unique_action_hashes}/{total_batches}")
    
    # Check for problems
    has_issues = False
    
    if duplicate_batches:
        print(f"\n❌ FOUND {len(duplicate_batches)} DUPLICATE FULL BATCHES!")
        has_issues = True
        for batch_hash, ranks in list(duplicate_batches.items())[:3]:  # Show first 3
            print(f"   Batch {batch_hash[:8]}... appears in ranks: {ranks}")
        if len(duplicate_batches) > 3:
            print(f"   ... and {len(duplicate_batches) - 3} more duplicates")
    
    if duplicate_qpos:
        print(f"\n⚠️  FOUND {len(duplicate_qpos)} DUPLICATE QPOS STATES!")
        has_issues = True
        for qpos_hash, ranks in list(duplicate_qpos.items())[:3]:  # Show first 3
            print(f"   QPos {qpos_hash[:8]}... appears in ranks: {ranks}")
        if len(duplicate_qpos) > 3:
            print(f"   ... and {len(duplicate_qpos) - 3} more duplicates")
    
    if duplicate_actions:
        print(f"\n⚠️  FOUND {len(duplicate_actions)} DUPLICATE ACTION SEQUENCES!")
        has_issues = True
        for action_hash, ranks in list(duplicate_actions.items())[:3]:  # Show first 3
            print(f"   Action {action_hash[:8]}... appears in ranks: {ranks}")
        if len(duplicate_actions) > 3:
            print(f"   ... and {len(duplicate_actions) - 3} more duplicates")
    
    if not has_issues:
        print(f"\n✅ SUCCESS: All batches appear to be unique across ranks!")
        print(f"✅ No duplicate batches detected.")
        
    # Additional analysis: Check if each rank is getting different data distributions
    print(f"\n📈 RANK-SPECIFIC ANALYSIS")
    print("-" * 60)
    
    for rank in range(num_ranks):
        print(f"\nRank {rank}:")
        
        # Sample some values from the first batch for this rank
        if rank_batches[rank]:
            first_batch = rank_batches[rank][0]
            qpos_sample = first_batch['observation.state'][0]  # First sample in batch
            action_sample = first_batch['action'][0, 0]  # First action of first sample
            
            print(f"   Sample qpos: {qpos_sample[:3].tolist()}")  # First 3 values
            print(f"   Sample action: {action_sample[:3].tolist()}")  # First 3 values
            print(f"   Batch shape - qpos: {first_batch['observation.state'].shape}")
            print(f"   Batch shape - action: {first_batch['action'].shape}")
    
    return not has_issues

if __name__ == "__main__":
    success = test_batch_distribution()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 TEST PASSED: Batches appear to be properly distributed across ranks!")
        print("✅ Your distributed training should work correctly.")
    else:
        print("⚠️  TEST FAILED: Found issues with batch distribution!")
        print("❌ Your distributed training may have duplicate data across GPUs.")
        print("\n💡 POTENTIAL FIXES:")
        print("   1. Use different WebDataset patterns per rank")
        print("   2. Implement proper distributed sampling")
        print("   3. Use DistributedSampler with WebDataset")
        print("   4. Check if shardshuffle is working correctly")
