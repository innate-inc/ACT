webd_dir='/home/vignesh/data/DropSocks_1_2_webd/'
hdf5_dir='/home/vignesh/data/DropSocks_1_2/'

import os
import h5py
import torch
import numpy as np
from data_utils import EpisodicHDF5DatasetRAM, initialize_data, initialize_webdataset_data

# Set seeds for deterministic behavior
torch.manual_seed(42)
np.random.seed(42)

print("Creating HDF5 dataset...")
# Create HDF5 dataloader with deterministic sampling
try:
    hdf5_train_loader, hdf5_val_loader, hdf5_stats = initialize_data(
        data_dir=hdf5_dir,
        chunk_size=10,
        train_val_split=1.0,  # Use all data for training to ensure we get samples
        batch_size=2,
        num_workers=0,
        use_img_aug_train=False,  # No augmentation for comparison
        seed=42  # Fixed seed
    )
    print("HDF5 dataset created successfully")
except Exception as e:
    print(f"Error creating HDF5 dataset: {e}")
    hdf5_train_loader = None

print("\nCreating WebDataset...")
# Create WebDataset dataloader with deterministic sampling
try:
    webd_train_loader, webd_val_loader, webd_stats = initialize_webdataset_data(
        data_dir=webd_dir,
        chunk_size=10,
        batch_size=2,
        train_val_split=1.0,  # Use all data for training
        use_img_aug_train=False,  # No augmentation for comparison
        use_img_aug_val=False,
        num_workers=0,
        seed=42  # Fixed seed
    )
    print("WebDataset created successfully")
except Exception as e:
    print(f"Error creating WebDataset: {e}")
    webd_train_loader = None

# Compare first batches if both datasets were created successfully
if hdf5_train_loader is not None and webd_train_loader is not None:
    print("\n" + "="*50)
    print("COMPARING FIRST BATCHES")
    print("="*50)
    
    # Get first batch from HDF5 dataset
    print("Getting first batch from HDF5 dataset...")
    hdf5_batch = next(iter(hdf5_train_loader))
    
    # Get first batch from WebDataset
    print("Getting first batch from WebDataset...")
    webd_batch = next(iter(webd_train_loader))
    
    # Compare batch structures
    print(f"\nHDF5 batch keys: {list(hdf5_batch.keys())}")
    print(f"WebDataset batch keys: {list(webd_batch.keys())}")
    
    # Compare each key
    print("\nDetailed comparison:")
    common_keys = set(hdf5_batch.keys()).intersection(set(webd_batch.keys()))
    
    for key in sorted(common_keys):
        print(f"\n--- Comparing '{key}' ---")
        hdf5_tensor = hdf5_batch[key]
        webd_tensor = webd_batch[key]
        
        print(f"HDF5 shape: {hdf5_tensor.shape}, dtype: {hdf5_tensor.dtype}")
        print(f"WebD shape: {webd_tensor.shape}, dtype: {webd_tensor.dtype}")
        
        if hdf5_tensor.shape == webd_tensor.shape:
            # Calculate differences
            if key.startswith('observation.image'):
                # For images, show a few sample values
                diff = torch.abs(hdf5_tensor - webd_tensor)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                print(f"Max pixel difference: {max_diff:.6f}")
                print(f"Mean pixel difference: {mean_diff:.6f}")
                print(f"HDF5 sample values: {hdf5_tensor.flatten()[:5]}")
                print(f"WebD sample values: {webd_tensor.flatten()[:5]}")
            else:
                # For other data, show exact values
                if torch.allclose(hdf5_tensor, webd_tensor, atol=1e-6):
                    print("✓ Tensors are identical (within tolerance)")
                else:
                    print("✗ Tensors differ")
                    print(f"HDF5 values: {hdf5_tensor}")
                    print(f"WebD values: {webd_tensor}")
                    diff = torch.abs(hdf5_tensor - webd_tensor)
                    print(f"Max difference: {torch.max(diff).item():.6f}")
        else:
            print("✗ Shapes don't match - cannot compare values")
    
    # Check for keys that exist in only one dataset
    hdf5_only = set(hdf5_batch.keys()) - set(webd_batch.keys())
    webd_only = set(webd_batch.keys()) - set(hdf5_batch.keys())
    
    if hdf5_only:
        print(f"\nKeys only in HDF5: {hdf5_only}")
    if webd_only:
        print(f"Keys only in WebDataset: {webd_only}")

else:
    if hdf5_train_loader is None:
        print("Failed to create HDF5 dataset - cannot compare")
    if webd_train_loader is None:
        print("Failed to create WebDataset - cannot compare")

print("\nDone!")

