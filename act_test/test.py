import torch
from torch.utils.data import DataLoader
import os

# Assuming data_utils.py is in the same directory or in PYTHONPATH
# and EpisodicHDF5DatasetRAM can be imported.
from data_utils import (
    initialize_data,
)  # MODIFIED: Import initialize_data instead of EpisodicHDF5DatasetRAM
from ACT import ACTConfig, ACTPolicy  # ADDED: Import ACTPolicy and ACTConfig


def print_sample_info(sample_item, sample_name="Sample"):
    print(f"\n--- {sample_name} ---")
    for key, value in sample_item.items():
        if isinstance(value, torch.Tensor):
            print(f"  Key: '{key}', Shape: {value.shape}, Dtype: {value.dtype}")
        else:
            print(f"  Key: '{key}', Type: {type(value)}, Value: {value}")


def main():
    data_dir = "/media/vignesh/Crucial_X9/PaperThurs/"
    # episode_ids_to_load = [0, 1] # REMOVED: Handled by initialize_data from metadata.json
    CHUNK_SIZE = 100  # RENAMED from test_chunk_size for clarity
    BATCH_SIZE = 2  # ADDED: Define a batch size for DataLoaders

    # Dataset constants (mirroring data_utils.py for config)
    ACTION_DIM = 8
    QPOS_DIM = 6
    IMAGE_H = 480
    IMAGE_W = 640
    IMAGE_C = 3
    POLICY_QPOS_KEY = "observation.state"
    POLICY_IMG_KEYS = ["observation.image_camera_1", "observation.image_camera_2"]
    POLICY_ACTION_KEY = "action"

    # Direct instantiation, assuming data_dir is valid and episodes exist.
    # dataset = EpisodicHDF5DatasetRAM(
    #     data_dir=data_dir,
    #     episode_ids=episode_ids_to_load,
    #     chunk_size=test_chunk_size, # ADDED: Pass chunk_size
    #     use_img_aug=False
    # )
    # print(f"Dataset initialized. Number of loaded episodes: {len(dataset)}")
    # REMOVED ABOVE BLOCK: Replaced by initialize_data

    print("--- Initializing Training and Validation DataLoaders ---")
    try:
        train_dataloader, val_dataloader, dataset_stats = initialize_data(
            data_dir=data_dir,
            chunk_size=CHUNK_SIZE,
            batch_size=BATCH_SIZE,
            train_val_split=0.8,  # Example split, adjust as needed
            use_img_aug_train=False,  # Keep augmentations off for simple testing
            use_img_aug_val=False,
            seed=42,  # For reproducible splits
        )
    except Exception as e:
        print(f"Error during data initialization: {e}")
        import traceback

        traceback.print_exc()
        return

    print(
        f"Train DataLoader: {len(train_dataloader.dataset)} samples, Val DataLoader: {len(val_dataloader.dataset)} samples"
    )

    # The attribute 'max_episode_length' was removed from the dataset,
    # so we can't directly print it here anymore.
    # If you need to know the max length across episodes for some reason,
    # you might need to re-calculate it or store it differently if it's crucial for testing.
    # For now, I'll comment out the line that tries to access it.
    # print(f"Max episode length: {dataset.max_episode_length}")

    # Get and print the first sample.
    # Assumes len(dataset) > 0.
    # sample_0 = dataset[0] # REMOVED: Dataset access is now through DataLoader
    # print_sample_info(sample_0, sample_name="Dataset Sample 0")

    # Get and print the second sample if available.
    # Assumes len(dataset) > 1.
    # if len(dataset) > 1:
    #     sample_1 = dataset[1]
    #     print_sample_info(sample_1, sample_name="Dataset Sample 1")
    # REMOVED ABOVE BLOCK

    # --- Test with DataLoader ---
    # Assumes dataset is not empty for DataLoader.
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0) # REMOVED

    # Get one batch. Assumes dataloader will produce at least one batch.
    # first_batch = next(iter(dataloader)) # REMOVED
    # print_sample_info(first_batch, sample_name="DataLoader First Batch") # REMOVED

    # --- Initialize ACT Policy and Test Forward Pass ---
    print("\n--- ACT Policy Test ---")

    # Create ACTConfig based on dataset properties
    act_config = ACTConfig(
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,  # Often same as chunk_size for training
        input_shapes={
            POLICY_QPOS_KEY: [QPOS_DIM],
            POLICY_IMG_KEYS[0]: [IMAGE_C, IMAGE_H, IMAGE_W],
            POLICY_IMG_KEYS[1]: [IMAGE_C, IMAGE_H, IMAGE_W],
        },
        output_shapes={POLICY_ACTION_KEY: [ACTION_DIM]},
        use_vae=True,
    )

    # Get dataset statistics
    # dataset_stats = dataset.get_dataset_stats() # MOVED: dataset_stats now comes from initialize_data
    print("\n--- Loaded Dataset Statistics from Training Set ---")
    for key, stats_dict in dataset_stats.items():
        print(f"  Stats for '{key}':")
        if "mean" in stats_dict:
            print(
                f"    Mean shape: {stats_dict['mean'].shape}, dtype: {stats_dict['mean'].dtype}"
            )
            print(f"    Mean values: {stats_dict['mean']}")
        if "std" in stats_dict:
            print(
                f"    Std shape: {stats_dict['std'].shape}, dtype: {stats_dict['std'].dtype}"
            )
            print(f"    Std values: {stats_dict['std']}")

    # Instantiate the policy
    policy = ACTPolicy(config=act_config, dataset_stats=dataset_stats)
    # policy.train() # Set to train mode for VAE path and loss calculation # MOVED: Set mode before each operation

    print(f"ACTPolicy initialized with chunk_size: {act_config.chunk_size}")

    # The check for VAE-specific keys has been removed.
    # If 'action' or 'action_is_pad' are missing from first_batch,
    # the policy(first_batch) call will likely raise a KeyError or
    # similar error during the L1 loss calculation or VAE processing,
    # which will be caught by the try-except block below.

    # Pass the batch through the policy's forward method
    # try:
    #     loss, loss_dict = policy(first_batch)
    #     print(f"\nPolicy Forward Pass Output:")
    #     print(f"  Total Loss: {loss.item()}")
    #     print(f"  Loss Dictionary: {loss_dict}")
    # except Exception as e:
    #     print(f"\nError during policy forward pass: {e}")
    #     import traceback
    #     traceback.print_exc()
    # REMOVED: Replaced with new testing structure below

    # --- Test with Training Data ---
    print("\n--- Testing with Training Data Batch ---")
    if len(train_dataloader) > 0:
        train_batch = next(iter(train_dataloader))
        print_sample_info(train_batch, sample_name="Train DataLoader First Batch")

        # Test forward pass in training mode
        policy.train()
        print("\nPolicy in TRAIN mode:")
        try:
            loss, loss_dict = policy(train_batch)
            print(f"  Forward Pass (Train Mode) - Total Loss: {loss.item()}")
            print(f"  Loss Dictionary: {loss_dict}")
        except Exception as e:
            print(f"  Error during policy forward pass (Train Mode): {e}")
            import traceback

            traceback.print_exc()

        # Test select_action in evaluation mode
        policy.eval()
        policy.reset()  # Reset any internal state like action queues before select_action
        print("\nPolicy in EVAL mode (using train_batch for select_action):")
        try:
            # select_action expects a batch and returns actions for that batch
            # For a single step prediction based on current obs of the batch
            # current_obs_batch = {
            #     key: (
            #         val[:, 0] if key == POLICY_QPOS_KEY and val.ndim > 2 else val
            #     )  # Select first qpos if it's a sequence
            #     for key, val in train_batch.items()
            #     if key.startswith("observation") or key == POLICY_QPOS_KEY
            # }
            # Ensure correct shapes for select_action if it expects non-sequenced obs
            # The current select_action in ACT.py takes the full batch (which includes sequences)
            # and its internal model call will handle it.
            # The main thing is that the batch keys match what normalize_inputs expects.

            # For a simple test, let's use the full train_batch.
            # select_action will internally use the first observation step from the chunk
            # if it's designed for single-step inference from a sequence.
            # The provided select_action takes the whole batch, model predicts a chunk,
            # and select_action manages an action queue.

            # Re-slicing train_batch to be more like what `select_action` might expect for a single step
            # For ACT, `select_action` processes the observation part of the batch.
            # It doesn't need 'action' or 'action_is_pad' typically.
            inference_batch = {
                k: v
                for k, v in train_batch.items()
                if k != POLICY_ACTION_KEY and k != "action_is_pad"
            }

            action_selected = policy.select_action(inference_batch)
            print(
                f"  select_action output shape: {action_selected.shape}, dtype: {action_selected.dtype}"
            )
            # print(f"  select_action output value (first item in batch): {action_selected[0]}")
        except Exception as e:
            print(
                f"  Error during policy select_action (Eval Mode with train_batch): {e}"
            )
            import traceback

            traceback.print_exc()
    else:
        print("Train DataLoader is empty. Skipping training data tests.")

    # --- Test with Validation Data ---
    print("\n--- Testing with Validation Data Batch ---")
    if len(val_dataloader) > 0:
        val_batch = next(iter(val_dataloader))
        print_sample_info(val_batch, sample_name="Val DataLoader First Batch")

        policy.eval()  # Ensure policy is in eval mode
        policy.reset()  # Reset action queue
        print("\nPolicy in EVAL mode (using val_batch for select_action):")
        try:
            inference_val_batch = {
                k: v
                for k, v in val_batch.items()
                if k != POLICY_ACTION_KEY and k != "action_is_pad"
            }
            action_selected_val = policy.select_action(inference_val_batch)
            print(
                f"  select_action output shape: {action_selected_val.shape}, dtype: {action_selected_val.dtype}"
            )
            # print(f"  select_action output value (first item in batch): {action_selected_val[0]}")

            # Optionally, if you also want to see "loss" on validation data (without gradients)
            # This is not standard validation but can be a sanity check.
            # Ensure 'action' and 'action_is_pad' are present if policy.forward() needs them for loss.
            # The policy.forward() includes VAE components that use 'action'.
            # If val_batch doesn't have 'action', policy(val_batch) might fail or behave unexpectedly.
            # The standard use of val_dataloader is for inference/select_action.
            # For this test, we will only call select_action.

        except Exception as e:
            print(
                f"  Error during policy select_action (Eval Mode with val_batch): {e}"
            )
            import traceback

            traceback.print_exc()
    else:
        print("Validation DataLoader is empty. Skipping validation data tests.")


if __name__ == "__main__":
    main()
