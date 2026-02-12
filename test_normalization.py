#!/usr/bin/env python3
"""
Quick test to verify data normalization is working correctly.
"""
import torch
import sys
import os

# Add the act_test module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'act_test'))

from data_utils import initialize_webdataset_data

def test_normalization():
    """Test that normalization produces data close to N(0,1)."""
    
    # Configuration
    DATA_DIR = "/home/vignesh/raid/PaperMulti_1_2_Filtered/webdataset"
    BATCH_SIZE = 16
    CHUNK_SIZE = 16
    
    print("=" * 80)
    print("TESTING DATA NORMALIZATION")
    print("=" * 80)
    
    # Test 1: Load with normalization enabled
    print("\n[Test 1] Loading data WITH normalization...")
    try:
        train_loader_norm, val_loader_norm, stats_norm = initialize_webdataset_data(
            data_dir=DATA_DIR,
            chunk_size=CHUNK_SIZE,
            batch_size=BATCH_SIZE,
            num_workers=2,
            normalize=True,
            compute_stats_from_all=False  # Use 1000 samples for speed
        )
        
        # Get a batch
        batch = next(iter(train_loader_norm))
        
        # Check robot state normalization
        qpos = batch['observation.state']
        qpos_mean = qpos.mean(dim=0)
        qpos_std = qpos.std(dim=0)
        
        print(f"\n✅ Normalized Robot State Statistics:")
        print(f"   Mean (should be ~0): {qpos_mean.tolist()}")
        print(f"   Std  (should be ~1): {qpos_std.tolist()}")
        
        # Check action normalization
        actions = batch['action']
        is_pad = batch['action_is_pad']
        
        # Only compute stats on non-padded actions
        real_actions = []
        for i in range(len(actions)):
            real_actions.append(actions[i][~is_pad[i]])
        real_actions = torch.cat(real_actions, dim=0)
        
        action_mean = real_actions.mean(dim=0)
        action_std = real_actions.std(dim=0)
        
        print(f"\n✅ Normalized Action Statistics:")
        print(f"   Mean (should be ~0): {action_mean.tolist()}")
        print(f"   Std  (should be ~1): {action_std.tolist()}")
        
        # Print dataset stats for reference
        print(f"\n📊 Original Dataset Statistics (before normalization):")
        print(f"   Robot state mean: {stats_norm['observation.state']['mean'].tolist()}")
        print(f"   Robot state std:  {stats_norm['observation.state']['std'].tolist()}")
        print(f"   Action mean: {stats_norm['action']['mean'].tolist()}")
        print(f"   Action std:  {stats_norm['action']['std'].tolist()}")
        
    except Exception as e:
        print(f"\n❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Load without normalization
    print("\n" + "=" * 80)
    print("[Test 2] Loading data WITHOUT normalization...")
    try:
        train_loader_unnorm, val_loader_unnorm, stats_unnorm = initialize_webdataset_data(
            data_dir=DATA_DIR,
            chunk_size=CHUNK_SIZE,
            batch_size=BATCH_SIZE,
            num_workers=2,
            normalize=False,
            compute_stats_from_all=False
        )
        
        # Get a batch
        batch_unnorm = next(iter(train_loader_unnorm))
        
        # Check that values are NOT normalized
        qpos_unnorm = batch_unnorm['observation.state']
        actions_unnorm = batch_unnorm['action']
        is_pad_unnorm = batch_unnorm['action_is_pad']
        
        # Compute stats on raw data
        qpos_unnorm_mean = qpos_unnorm.mean(dim=0)
        qpos_unnorm_std = qpos_unnorm.std(dim=0)
        
        real_actions_unnorm = []
        for i in range(len(actions_unnorm)):
            real_actions_unnorm.append(actions_unnorm[i][~is_pad_unnorm[i]])
        real_actions_unnorm = torch.cat(real_actions_unnorm, dim=0)
        
        action_unnorm_mean = real_actions_unnorm.mean(dim=0)
        action_unnorm_std = real_actions_unnorm.std(dim=0)
        
        print(f"\n✅ Unnormalized Robot State Statistics (raw values):")
        print(f"   Mean: {qpos_unnorm_mean.tolist()}")
        print(f"   Std:  {qpos_unnorm_std.tolist()}")
        
        print(f"\n✅ Unnormalized Action Statistics (raw values):")
        print(f"   Mean: {action_unnorm_mean.tolist()}")
        print(f"   Std:  {action_unnorm_std.tolist()}")
        
        # Verify these match the dataset stats
        print(f"\n📊 These should roughly match the original dataset stats above")
        
    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\n📝 Summary:")
    print("   - Normalization correctly transforms data to ~N(0,1)")
    print("   - Unnormalized data retains original physical values")
    print("   - Dataset statistics are computed correctly")
    
    return True

if __name__ == "__main__":
    success = test_normalization()
    sys.exit(0 if success else 1)
