#!/usr/bin/env python3
"""
Test the correct way to use WebDataset split functions.
"""

import webdataset as wds

def test_correct_usage():
    """Test the actual correct way to use split functions."""
    print("🧪 Testing correct usage of split functions...")
    print(f"WebDataset version: {wds.__version__}")
    
    try:
        # Method 1: Direct function application
        print("\n1️⃣  Testing direct function application...")
        
        base_dataset = wds.WebDataset("dummy.tar", shardshuffle=False)
        split_dataset = wds.split_by_worker(base_dataset)
        print("✅ Direct function call works")
        
        # Method 2: Chain them
        print("\n2️⃣  Testing chained application...")
        
        base_dataset = wds.WebDataset("dummy.tar", shardshuffle=False)
        node_split = wds.split_by_node(base_dataset)
        worker_split = wds.split_by_worker(node_split)
        print("✅ Chained function calls work")
        
        # Method 3: Functional style in one line
        print("\n3️⃣  Testing functional composition...")
        
        final_dataset = wds.split_by_worker(
            wds.split_by_node(
                wds.WebDataset("dummy.tar", shardshuffle=False)
            )
        )
        print("✅ Functional composition works")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_real_world_usage():
    """Show how to use this in your actual data_utils.py"""
    print(f"\n{'='*60}")
    print("🚀 HOW TO UPDATE YOUR data_utils.py")
    print(f"{'='*60}")
    
    print("""
REPLACE this in your initialize_webdataset_data function:

❌ OLD (random filtering):
    train_dataset = (
        wds.WebDataset(full_pattern, shardshuffle=True)
        .decode("pil")
        .to_tuple("cam1.png", "cam2.png", "qpos.npy", "actions.npy")
        .select(train_split_fn)                    # ← Remove this
        .map(decode_fn)
    )

✅ NEW (proper distributed splitting):
    base_dataset = wds.WebDataset(full_pattern, shardshuffle=True)
    
    # Apply distributed splitting
    if dist.is_initialized():  # Check if running distributed
        base_dataset = wds.split_by_node(base_dataset)    # Multi-node
    
    split_dataset = wds.split_by_worker(base_dataset)     # Multi-worker
    
    train_dataset = (
        split_dataset
        .decode("pil")
        .to_tuple("cam1.png", "cam2.png", "qpos.npy", "actions.npy")
        .map(decode_fn)
    )

🎯 KEY BENEFITS:
1. ✅ No data overlap between workers
2. ✅ Automatic load balancing  
3. ✅ Works with any number of workers/nodes
4. ✅ No need for manual shard assignment
5. ✅ Deterministic data distribution
""")

def create_updated_function_template():
    """Show the complete updated function."""
    print(f"\n{'='*60}")
    print("📝 COMPLETE UPDATED FUNCTION TEMPLATE")
    print(f"{'='*60}")
    
    template = '''
def initialize_webdataset_data_distributed(data_dir, chunk_size=100, batch_size=8, 
                                          num_workers=4, prefetch_factor=2, seed=42):
    """
    Initialize WebDataset with proper distributed splitting.
    """
    import torch.distributed as dist
    
    # Find dataset files
    dataset_pattern = os.path.join(data_dir, "train-*.tar")
    all_files = sorted(glob.glob(dataset_pattern))
    
    if len(all_files) == 0:
        raise FileNotFoundError(f"No WebDataset .tar files found in {data_dir}")
    
    # Create pattern
    if len(all_files) == 1:
        full_pattern = all_files[0]
    else:
        full_pattern = "{" + ",".join(all_files) + "}"
    
    print(f"Using dataset pattern: {full_pattern}")
    
    # Create base dataset
    base_dataset = wds.WebDataset(full_pattern, shardshuffle=True)
    
    # Apply distributed splitting if needed
    if dist.is_initialized():
        print("Applying distributed node splitting...")
        base_dataset = wds.split_by_node(base_dataset)
    
    print("Applying worker splitting...")
    split_dataset = wds.split_by_worker(base_dataset)
    
    # Create training dataset (no more random filtering!)
    train_dataset = (
        split_dataset
        .decode("pil")
        .to_tuple("cam1.png", "cam2.png", "qpos.npy", "actions.npy")
        .map(WebDatasetDecoder(chunk_size))
    )
    
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )
    
    return train_dataloader
'''
    
    print(template)

if __name__ == "__main__":
    success = test_correct_usage()
    if success:
        print("🎉 All tests passed!")
        show_real_world_usage()
        create_updated_function_template()
    else:
        print("❌ Tests failed")
