import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
import json

# Assuming torchvision.transforms.v2 will be available if default augmentation is used.
# If not, and use_img_aug=True with img_aug_transforms=None, a NameError/ImportError will occur.
from torchvision.transforms import v2

class EpisodicHDF5DatasetRAM(Dataset):
    # --- Hardcoded Values Based on User Guarantees ---
    ACTION_DIM = 10
    QPOS_DIM = 6
    IMAGE_H = 480
    IMAGE_W = 640
    IMAGE_C = 3 

    H5_ACTION_KEY = 'action'
    H5_QPOS_KEY = 'observations/qpos'
    H5_CAMERA_NAMES = ['camera_1', 'camera_2'] 

    POLICY_ACTION_KEY = 'action'
    POLICY_QPOS_KEY = 'observation.state'
    POLICY_IMG_KEYS = ['observation.image_camera_1', 'observation.image_camera_2']
    # --- End Hardcoded Values ---

    def __init__(self,
                 data_dir: str,
                 episode_ids: List[int],
                 chunk_size: int,
                 use_img_aug: bool = False,
                 img_aug_transforms: Optional[torch.nn.Module] = None):
        super().__init__()
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.use_img_aug = use_img_aug
        self.transforms = img_aug_transforms

        if self.use_img_aug and self.transforms is None:
            self.transforms = v2.Compose([v2.RandomPhotometricDistort(p=0.5), v2.RandomHorizontalFlip(p=0.5)])

        self.loaded_episodes_data: List[Dict[str, Any]] = []
        self.episode_info: List[Dict[str, Any]] = []
        
        for ep_id in episode_ids:
            file_path = os.path.join(self.data_dir, f"episode_{ep_id}.h5")
            with h5py.File(file_path, 'r') as f:
                current_ep_len = f[self.H5_ACTION_KEY].shape[0]
                
                ep_data_ram = {}
                ep_data_ram[self.POLICY_ACTION_KEY] = torch.from_numpy(f[self.H5_ACTION_KEY][:]).float()
                ep_data_ram[self.POLICY_QPOS_KEY] = torch.from_numpy(f[self.H5_QPOS_KEY][:]).float()

                for i, h5_cam_name in enumerate(self.H5_CAMERA_NAMES):
                    policy_img_key = self.POLICY_IMG_KEYS[i]
                    h5_img_path = f'observations/images/{h5_cam_name}'
                    img_thwc_uint8 = f[h5_img_path][:]
                    img_tchw_float = torch.from_numpy(img_thwc_uint8).float().permute(0, 3, 1, 2) / 255.0
                    ep_data_ram[policy_img_key] = img_tchw_float
                
                self.loaded_episodes_data.append(ep_data_ram)
                self.episode_info.append({"id": ep_id, "length": current_ep_len})
        
        # Compute statistics using the "collect references then concatenate" method
        self.computed_stats = self._compute_stats_via_concat()

    def _compute_stats_via_concat(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Computes mean and standard deviation by first collecting references to
        all relevant tensors from self.loaded_episodes_data, then concatenating
        them once per modality, and finally computing stats.
        """
        stats: Dict[str, Dict[str, torch.Tensor]] = {}

        if not self.loaded_episodes_data: # Handle case with no episodes
            if self.POLICY_QPOS_KEY:
                 stats[self.POLICY_QPOS_KEY] = {"mean": torch.zeros(self.QPOS_DIM), "std": torch.ones(self.QPOS_DIM)}
            if self.POLICY_ACTION_KEY:
                stats[self.POLICY_ACTION_KEY] = {"mean": torch.zeros(self.ACTION_DIM), "std": torch.ones(self.ACTION_DIM)}
            for key in self.POLICY_IMG_KEYS:
                 stats[key] = {"mean": torch.zeros(self.IMAGE_C,1,1), "std": torch.ones(self.IMAGE_C,1,1)}
            return stats

        # Collect tensor references for stats from self.loaded_episodes_data
        qpos_tensors_for_stats = [ep_data[self.POLICY_QPOS_KEY] for ep_data in self.loaded_episodes_data if self.POLICY_QPOS_KEY in ep_data]
        action_tensors_for_stats = [ep_data[self.POLICY_ACTION_KEY] for ep_data in self.loaded_episodes_data if self.POLICY_ACTION_KEY in ep_data]
        
        img_tensors_for_stats: Dict[str, List[torch.Tensor]] = {key: [] for key in self.POLICY_IMG_KEYS}
        for key in self.POLICY_IMG_KEYS:
            for ep_data in self.loaded_episodes_data:
                if key in ep_data:
                    img_tensors_for_stats[key].append(ep_data[key])

        # Compute stats for QPOS (State)
        if qpos_tensors_for_stats:
            qpos_tensor_all = torch.cat(qpos_tensors_for_stats, dim=0)
            stats[self.POLICY_QPOS_KEY] = {
                "mean": torch.mean(qpos_tensor_all, dim=0),
                "std": torch.clamp(torch.std(qpos_tensor_all, dim=0), min=1e-6)
            }
        else: # Default if no qpos data found
            stats[self.POLICY_QPOS_KEY] = {"mean": torch.zeros(self.QPOS_DIM), "std": torch.ones(self.QPOS_DIM)}
        
        # Compute stats for Action
        if action_tensors_for_stats:
            action_tensor_all = torch.cat(action_tensors_for_stats, dim=0)
            stats[self.POLICY_ACTION_KEY] = {
                "mean": torch.mean(action_tensor_all, dim=0),
                "std": torch.clamp(torch.std(action_tensor_all, dim=0), min=1e-6)
            }
        else: # Default if no action data
            stats[self.POLICY_ACTION_KEY] = {"mean": torch.zeros(self.ACTION_DIM), "std": torch.ones(self.ACTION_DIM)}

        # Compute stats for Images
        for policy_img_key in self.POLICY_IMG_KEYS:
            if img_tensors_for_stats[policy_img_key]:
                img_tensor_all = torch.cat(img_tensors_for_stats[policy_img_key], dim=0)
                stats[policy_img_key] = {
                    "mean": torch.mean(img_tensor_all, dim=(0, 2, 3)).reshape(-1, 1, 1),
                    "std": torch.clamp(torch.std(img_tensor_all, dim=(0, 2, 3)).reshape(-1, 1, 1), min=1e-6)
                }
            else: # Default if no image data for this key
                stats[policy_img_key] = {"mean": torch.zeros(self.IMAGE_C,1,1), "std": torch.ones(self.IMAGE_C,1,1)}
        
        return stats

    def get_dataset_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Returns the computed normalization statistics for the dataset."""
        return self.computed_stats

    def __len__(self) -> int:
        return len(self.loaded_episodes_data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ep_data = self.loaded_episodes_data[index]
        ep_info = self.episode_info[index]
        ep_len = ep_info["length"]

        start_ts = random.randint(0, ep_len - 1) 
        item: Dict[str, Any] = {}

        actions_full_episode = ep_data[self.POLICY_ACTION_KEY]
        actions_raw = actions_full_episode[start_ts : start_ts + self.chunk_size]
        actions_raw_len = actions_raw.shape[0]

        padded_actions = torch.zeros((self.chunk_size, self.ACTION_DIM), dtype=torch.float32)
        if actions_raw_len > 0:
            padded_actions[:actions_raw_len] = actions_raw
        item[self.POLICY_ACTION_KEY] = padded_actions
        
        action_is_pad = torch.ones(self.chunk_size, dtype=torch.bool)
        if actions_raw_len > 0:
            action_is_pad[:actions_raw_len] = False
        item["action_is_pad"] = action_is_pad

        item[self.POLICY_QPOS_KEY] = ep_data[self.POLICY_QPOS_KEY][start_ts]

        images_to_augment = []
        policy_keys_for_img_aug = []
        for i, _ in enumerate(self.H5_CAMERA_NAMES): 
            policy_img_key = self.POLICY_IMG_KEYS[i]
            img_chw = ep_data[policy_img_key][start_ts] 
            if self.use_img_aug and self.transforms:
                images_to_augment.append(img_chw)
                policy_keys_for_img_aug.append(policy_img_key)
            else:
                item[policy_img_key] = img_chw

        if self.use_img_aug and self.transforms and images_to_augment:
            for i, img_to_aug in enumerate(images_to_augment):
                item[policy_keys_for_img_aug[i]] = self.transforms(img_to_aug)
        
        return item

def initialize_data(
    data_dir: str,
    chunk_size: int,
    train_val_split: float = 0.9,
    batch_size: int = 32,
    num_workers: int = 0,
    use_img_aug_train: bool = False,
    use_img_aug_val: bool = False,
    img_aug_transforms_train: Optional[torch.nn.Module] = None,
    img_aug_transforms_val: Optional[torch.nn.Module] = None,
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Initializes training and validation DataLoaders from HDF5 episodes listed in metadata.json.

    Args:
        data_dir: Directory containing 'metadata.json' and HDF5 episode files.
        chunk_size: The chunk size for loading sequences from episodes.
        train_val_split: Fraction of episodes to use for training (0.0 to 1.0).
        batch_size: Batch size for the DataLoaders.
        num_workers: Number of worker processes for data loading. Defaults to 0.
        use_img_aug_train: Whether to use image augmentation for the training set.
        use_img_aug_val: Whether to use image augmentation for the validation set.
                       (Typically False for validation).
        img_aug_transforms_train: Custom torchvision transforms for training images.
                                  If None and use_img_aug_train is True, default is used.
        img_aug_transforms_val: Custom torchvision transforms for validation images.
                                If None and use_img_aug_val is True, default is used.
        seed: Optional random seed for reproducible train/val split.

    Returns:
        A tuple containing:
            - train_dataloader: DataLoader for the training set.
            - val_dataloader: DataLoader for the validation set.
            - dataset_stats: Normalization statistics computed from the training set.
    """
    metadata_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {data_dir}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    all_episode_ids = [ep_info["episode_id"] for ep_info in metadata.get("episodes", [])]
    if not all_episode_ids:
        raise ValueError("No episodes found in metadata.json or 'episodes' key is missing/empty.")
    
    print(f"Found a total of {len(all_episode_ids)} episodes: {sorted(all_episode_ids)}")

    if seed is not None:
        random.seed(seed)
    random.shuffle(all_episode_ids)

    num_total_episodes = len(all_episode_ids)
    num_train_episodes = int(num_total_episodes * train_val_split)
    
    train_episode_ids = all_episode_ids[:num_train_episodes]
    val_episode_ids = all_episode_ids[num_train_episodes:]

    print(f"Using {len(train_episode_ids)} episodes for training: {sorted(train_episode_ids)}")
    print(f"Using {len(val_episode_ids)} episodes for validation: {sorted(val_episode_ids)}")

    if not train_episode_ids:
        raise ValueError(
            f"Train/validation split resulted in 0 training episodes. "
            f"Total episodes: {num_total_episodes}, requested train split: {train_val_split}. "
            f"Ensure 'train_val_split' is > 0 or there are enough episodes."
        )
    
    if not val_episode_ids:
        print(
            f"Warning: Train/validation split resulted in 0 validation episodes. "
            f"Total episodes: {num_total_episodes}, train split: {train_val_split} "
            f"(Num train: {len(train_episode_ids)}, Num val: {len(val_episode_ids)}). "
            f"Validation DataLoader will be empty."
        )

    # Create training dataset
    train_dataset = EpisodicHDF5DatasetRAM(
        data_dir=data_dir,
        episode_ids=train_episode_ids,
        chunk_size=chunk_size,
        use_img_aug=use_img_aug_train,
        img_aug_transforms=img_aug_transforms_train
    )

    # Compute stats from the training dataset
    dataset_stats = train_dataset.get_dataset_stats()

    # Create validation dataset
    # Handles val_episode_ids being empty, resulting in an empty dataset
    val_dataset = EpisodicHDF5DatasetRAM(
        data_dir=data_dir,
        episode_ids=val_episode_ids, 
        chunk_size=chunk_size,
        use_img_aug=use_img_aug_val,
        img_aug_transforms=img_aug_transforms_val
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() 
    )
    
    # For validation, shuffle is usually False.
    # DataLoader handles empty val_dataset (if val_episode_ids was empty) correctly,
    # it will simply yield no batches.
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_dataloader, val_dataloader, dataset_stats

