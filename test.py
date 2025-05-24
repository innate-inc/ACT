import torch
from torch.utils.data import DataLoader
import os

# Assuming data_utils.py is in the same directory or in PYTHONPATH
# and EpisodicHDF5DatasetRAM can be imported.
from data_utils import EpisodicHDF5DatasetRAM
from ACT import ACTConfig, ACTPolicy # ADDED: Import ACTPolicy and ACTConfig

def print_sample_info(sample_item, sample_name="Sample"):
    print(f"\n--- {sample_name} ---")
    for key, value in sample_item.items():
        if isinstance(value, torch.Tensor):
            print(f"  Key: '{key}', Shape: {value.shape}, Dtype: {value.dtype}")
        else:
            print(f"  Key: '{key}', Type: {type(value)}, Value: {value}")

def main():
    data_dir = "/media/vignesh/Crucial_X9/PaperThurs/"
    episode_ids_to_load = [0, 1] # Ensure these episodes exist
    test_chunk_size = 100 # ADDED: Define a chunk size for the test

    # Dataset constants (mirroring data_utils.py for config)
    ACTION_DIM = 8
    QPOS_DIM = 6
    IMAGE_H = 480
    IMAGE_W = 640
    IMAGE_C = 3
    POLICY_QPOS_KEY = 'observation.state'
    POLICY_IMG_KEYS = ['observation.image_camera_1', 'observation.image_camera_2']
    POLICY_ACTION_KEY = 'action'

    # Direct instantiation, assuming data_dir is valid and episodes exist.
    dataset = EpisodicHDF5DatasetRAM(
        data_dir=data_dir,
        episode_ids=episode_ids_to_load,
        chunk_size=test_chunk_size, # ADDED: Pass chunk_size
        use_img_aug=False 
    )

    print(f"Dataset initialized. Number of loaded episodes: {len(dataset)}")
    # The attribute 'max_episode_length' was removed from the dataset,
    # so we can't directly print it here anymore.
    # If you need to know the max length across episodes for some reason,
    # you might need to re-calculate it or store it differently if it's crucial for testing.
    # For now, I'll comment out the line that tries to access it.
    # print(f"Max episode length: {dataset.max_episode_length}") 

    # Get and print the first sample.
    # Assumes len(dataset) > 0.
    sample_0 = dataset[0]
    print_sample_info(sample_0, sample_name="Dataset Sample 0")

    # Get and print the second sample if available.
    # Assumes len(dataset) > 1.
    if len(dataset) > 1:
        sample_1 = dataset[1]
        print_sample_info(sample_1, sample_name="Dataset Sample 1")

    # --- Test with DataLoader ---
    # Assumes dataset is not empty for DataLoader.
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    # Get one batch. Assumes dataloader will produce at least one batch.
    first_batch = next(iter(dataloader))
    print_sample_info(first_batch, sample_name="DataLoader First Batch")

    # --- Initialize ACT Policy and Test Forward Pass ---
    print("\n--- ACT Policy Test ---")
    
    # Create ACTConfig based on dataset properties
    act_config = ACTConfig(
        chunk_size=test_chunk_size,
        n_action_steps=test_chunk_size, # Often same as chunk_size for training
        input_shapes={
            POLICY_QPOS_KEY: [QPOS_DIM],
            POLICY_IMG_KEYS[0]: [IMAGE_C, IMAGE_H, IMAGE_W],
            POLICY_IMG_KEYS[1]: [IMAGE_C, IMAGE_H, IMAGE_W],
        },
        output_shapes={
            POLICY_ACTION_KEY: [ACTION_DIM]
        },
        use_vae=True 
    )

    # Get dataset statistics
    dataset_stats = dataset.get_dataset_stats()
    print("\n--- Computed Dataset Statistics ---")
    for key, stats_dict in dataset_stats.items():
        print(f"  Stats for '{key}':")
        if "mean" in stats_dict:
            print(f"    Mean shape: {stats_dict['mean'].shape}, dtype: {stats_dict['mean'].dtype}")
            print(f"    Mean values: {stats_dict['mean']}")
        if "std" in stats_dict:
            print(f"    Std shape: {stats_dict['std'].shape}, dtype: {stats_dict['std'].dtype}")
            print(f"    Std values: {stats_dict['std']}")

    # Instantiate the policy
    policy = ACTPolicy(config=act_config, dataset_stats=dataset_stats)
    policy.train() # Set to train mode for VAE path and loss calculation

    print(f"ACTPolicy initialized with chunk_size: {act_config.chunk_size}")

    # The check for VAE-specific keys has been removed.
    # If 'action' or 'action_is_pad' are missing from first_batch,
    # the policy(first_batch) call will likely raise a KeyError or
    # similar error during the L1 loss calculation or VAE processing,
    # which will be caught by the try-except block below.

    # Pass the batch through the policy's forward method
    try:
        loss, loss_dict = policy(first_batch)
        print(f"\nPolicy Forward Pass Output:")
        print(f"  Total Loss: {loss.item()}")
        print(f"  Loss Dictionary: {loss_dict}")
    except Exception as e:
        print(f"\nError during policy forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
