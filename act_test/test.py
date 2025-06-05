import torch
from torch.utils.data import DataLoader
import os

# Import WebDataset functionality
from data_utils import initialize_webdataset_data
from ACT import ACTConfig, ACTPolicy

def print_sample_info(sample_item, sample_name="Sample"):
    print(f"\n--- {sample_name} ---")
    for key, value in sample_item.items():
        if isinstance(value, torch.Tensor):
            print(f"  Key: '{key}', Shape: {value.shape}, Dtype: {value.dtype}")
        else:
            print(f"  Key: '{key}', Type: {type(value)}, Value: {value}")

def main():
    webdataset_dir = "/home/vignesh/raid/DropSocks_1_2_webd/"
    CHUNK_SIZE = 100 
    BATCH_SIZE = 2 

    # Dataset constants (mirroring data_utils.py for config)
    ACTION_DIM = 8
    QPOS_DIM = 6
    IMAGE_H = 480
    IMAGE_W = 640
    IMAGE_C = 3
    POLICY_QPOS_KEY = 'observation.state'
    POLICY_IMG_KEYS = ['observation.image_camera_1', 'observation.image_camera_2']
    POLICY_ACTION_KEY = 'action'

    print("=== Testing WebDataset ===")
    try:
        train_dataloader, val_dataloader, dataset_stats = initialize_webdataset_data(
            data_dir=webdataset_dir,
            chunk_size=CHUNK_SIZE,
            batch_size=BATCH_SIZE,
            train_val_split=0.8,
            use_img_aug_train=False,
            use_img_aug_val=False,
            num_workers=4,
            seed=42
        )
        
        test_dataloader_and_policy(train_dataloader, val_dataloader, dataset_stats,
                                 CHUNK_SIZE, ACTION_DIM, QPOS_DIM, IMAGE_H, IMAGE_W, IMAGE_C)
    except Exception as e:
        print(f"Error during WebDataset data initialization: {e}")
        import traceback
        traceback.print_exc()

def test_dataloader_and_policy(train_dataloader, val_dataloader, dataset_stats, 
                              chunk_size, action_dim, qpos_dim, image_h, image_w, image_c):
    """Test dataloader and policy with WebDataset."""
    
    POLICY_QPOS_KEY = 'observation.state'
    POLICY_IMG_KEYS = ['observation.image_camera_1', 'observation.image_camera_2']
    POLICY_ACTION_KEY = 'action'
    
    print(f"\n--- WebDataset DataLoader Test ---")
    
    # Check for None explicitly to avoid __len__() calls
    if train_dataloader is not None:
        print(f"Train DataLoader: Created (streaming dataset - no fixed length)")
    if val_dataloader is not None:
        print(f"Val DataLoader: Created (streaming dataset - no fixed length)")

    # Print dataset statistics
    print(f"\n--- WebDataset Dataset Statistics ---")
    for key, stats_dict in dataset_stats.items():
        print(f"  Stats for '{key}':")
        if "mean" in stats_dict:
            print(f"    Mean shape: {stats_dict['mean'].shape}, dtype: {stats_dict['mean'].dtype}")
            print(f"    Mean values: {stats_dict['mean']}")
        if "std" in stats_dict:
            print(f"    Std shape: {stats_dict['std'].shape}, dtype: {stats_dict['std'].dtype}")
            print(f"    Std values: {stats_dict['std']}")

    # Initialize ACT Policy
    act_config = ACTConfig(
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        input_shapes={
            POLICY_QPOS_KEY: [qpos_dim],
            POLICY_IMG_KEYS[0]: [image_c, image_h, image_w],
            POLICY_IMG_KEYS[1]: [image_c, image_h, image_w],
        },
        output_shapes={
            POLICY_ACTION_KEY: [action_dim]
        },
        use_vae=True 
    )

    policy = ACTPolicy(config=act_config, dataset_stats=dataset_stats)
    print(f"ACTPolicy initialized for WebDataset with chunk_size: {act_config.chunk_size}")

    # Test with Training Data
    print(f"\n--- Testing WebDataset Training Data ---")
    if train_dataloader is not None:
        try:
            train_batch = next(iter(train_dataloader))
            print_sample_info(train_batch, sample_name="WebDataset Train Batch")

            # Test forward pass in training mode
            policy.train()
            print(f"\nWebDataset Policy in TRAIN mode:")
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
            policy.reset()
            print(f"\nWebDataset Policy in EVAL mode:")
            try:
                inference_batch = {k: v for k, v in train_batch.items() if k != POLICY_ACTION_KEY and k != "action_is_pad"}
                action_selected = policy.select_action(inference_batch)
                print(f"  select_action output shape: {action_selected.shape}, dtype: {action_selected.dtype}")
            except Exception as e:
                print(f"  Error during policy select_action (Eval Mode): {e}")
                import traceback
                traceback.print_exc()
        except StopIteration:
            print(f"WebDataset Train DataLoader is empty or exhausted.")
        except Exception as e:
            print(f"Error getting training batch: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"No training dataloader provided.")

    # Test with Validation Data
    print(f"\n--- Testing WebDataset Validation Data ---")
    if val_dataloader is not None:
        try:
            val_batch = next(iter(val_dataloader))
            print_sample_info(val_batch, sample_name="WebDataset Val Batch")

            policy.eval()
            policy.reset()
            print(f"\nWebDataset Policy in EVAL mode (validation):")
            try:
                inference_val_batch = {k: v for k, v in val_batch.items() if k != POLICY_ACTION_KEY and k != "action_is_pad"}
                action_selected_val = policy.select_action(inference_val_batch)
                print(f"  select_action output shape: {action_selected_val.shape}, dtype: {action_selected_val.dtype}")
            except Exception as e:
                print(f"  Error during policy select_action (Eval Mode with val_batch): {e}")
                import traceback
                traceback.print_exc()
        except StopIteration:
            print(f"WebDataset Validation DataLoader is empty or exhausted.")
        except Exception as e:
            print(f"Error getting validation batch: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"No validation dataloader provided.")

if __name__ == "__main__":
    main()
