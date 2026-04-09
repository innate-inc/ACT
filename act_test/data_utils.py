import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
import json
import time
from tqdm import tqdm

# Assuming torchvision.transforms.v2 will be available if default augmentation is used.
# If not, and use_img_aug=True with img_aug_transforms=None, a NameError/ImportError will occur.
from torchvision.transforms import v2

# Add WebDataset imports
import webdataset as wds
import glob
from torchvision import transforms
from functools import partial
import re
import io

class EpisodicHDF5DatasetRAM(Dataset):
    # --- Hardcoded Values Based on User Guarantees ---
    ACTION_DIM = 10
    QPOS_DIM = 6
    IMAGE_H = 224  # Updated from 480 to 224
    IMAGE_W = 224  # Updated from 640 to 224
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
                 chunk_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.chunk_size = chunk_size

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

        for i, _ in enumerate(self.H5_CAMERA_NAMES): 
            policy_img_key = self.POLICY_IMG_KEYS[i]
            img_chw = ep_data[policy_img_key][start_ts] 
            item[policy_img_key] = img_chw
        
        return item

def initialize_data(
    data_dir: str,
    chunk_size: int,
    train_val_split: float = 0.9,
    batch_size: int = 32,
    num_workers: int = 0,
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
        seed: Optional random seed for reproducible train/val split.

    Returns:
        A tuple containing:
            - train_dataloader: DataLoader for the training set.
            - val_dataloader: DataLoader for the validation set.
            - dataset_stats: Normalization statistics computed from the training set.
    """
    # Check for metadata file (try both naming conventions)
    metadata_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(data_dir, "dataset_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json or dataset_metadata.json not found in {data_dir}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    all_episode_ids = [ep_info["episode_id"] for ep_info in metadata.get("episodes", [])]
    if not all_episode_ids:
        raise ValueError("No episodes found in metadata file or 'episodes' key is missing/empty.")
    
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
        chunk_size=chunk_size
    )

    # Compute stats from the training dataset
    dataset_stats = train_dataset.get_dataset_stats()

    # Create validation dataset
    val_dataset = EpisodicHDF5DatasetRAM(
        data_dir=data_dir,
        episode_ids=val_episode_ids, 
        chunk_size=chunk_size
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

class WebDatasetStreaming:
    """WebDataset implementation for ACT policy training with streaming data loading."""
    
    def __init__(self, dataset_path, chunk_size=100):
        self.dataset_path = dataset_path
        self.chunk_size = chunk_size
        
        # Check if dataset files exist - handle WebDataset patterns properly
        if "{" in dataset_path and "}" in dataset_path:
            # For WebDataset patterns like train-{00000..00015}.tar, extract directory and check files
            dir_path = os.path.dirname(dataset_path)
            
            # Check if it's a range pattern like {00000..00015}
            range_match = re.search(r'\{(\d+)\.\.(\d+)\}', dataset_path)
            if range_match:
                start_num = int(range_match.group(1))
                end_num = int(range_match.group(2))
                # Check if files exist in the range
                base_pattern = re.sub(r'\{.*\}', '*', dataset_path)
                pattern_files = glob.glob(base_pattern)
                
                if len(pattern_files) == 0:
                    raise FileNotFoundError(f"No WebDataset files found matching pattern: {base_pattern}")
            else:
                # For comma-separated patterns like {file1,file2,file3}
                pattern_files = glob.glob(dataset_path.replace("{*}", "*"))
                if len(pattern_files) == 0:
                    raise FileNotFoundError(f"No WebDataset files found at: {dataset_path}")
        else:
            # Simple glob pattern
            pattern_files = glob.glob(dataset_path)
            if len(pattern_files) == 0:
                raise FileNotFoundError(f"No WebDataset files found at: {dataset_path}")
        
        print(f"WebDataset pattern validated: {dataset_path}")
    
    def decode_sample(self, cam1, cam2, qpos, actions):
        """Decode a single sample from WebDataset and convert to ACT policy format."""
        # Images are now uint8 torch tensors (HWC format: 224x224x3)
        # Convert to float32, HWC to CHW, and normalize to [0, 1]
        cam1_tensor = cam1.float().permute(2, 0, 1) * (1.0/255.0)
        cam2_tensor = cam2.float().permute(2, 0, 1) * (1.0/255.0)
        
        # Handle action chunking and padding
        # actions is already a PyTorch tensor (float16 from .pth file)
        original_length = len(actions)
        
        if original_length >= self.chunk_size:
            # Take the first chunk_size actions (next actions from current obs)
            padded_actions = actions[:self.chunk_size].float()
            is_pad_tensor = torch.zeros(self.chunk_size, dtype=torch.bool)
        else:
            # Pad to chunk_size if we have fewer actions than needed
            padded_actions = torch.zeros((self.chunk_size, actions.shape[1]), dtype=torch.float32)
            padded_actions[:original_length] = actions.float()
            is_pad_tensor = torch.ones(self.chunk_size, dtype=torch.bool)
            is_pad_tensor[:original_length] = False
        
        # Convert to ACT policy expected format
        sample = {
            'observation.image_camera_1': cam1_tensor,  # (3, H, W)
            'observation.image_camera_2': cam2_tensor,  # (3, H, W)
            'observation.state': qpos.float(),  # (6,) - already a tensor
            'action': padded_actions,  # (chunk_size, action_dim)
            'action_is_pad': is_pad_tensor  # (chunk_size,)
        }
        
        return sample
    
    def create_webdataset(self):
        """Create the WebDataset pipeline."""
        # Note: Images are now stored as .pth files (PyTorch tensors)
        dataset = (
            wds.WebDataset(self.dataset_path)
            .decode("torch")
            .to_tuple("cam1.pth", "cam2.pth", "qpos.pth", "actions.pth")
            .map(lambda x: self.decode_sample(*x))
        )
        return dataset

# Global picklable functions for WebDataset
def decode_npy(data):
    """Picklable function to decode .npy files."""
    return np.load(io.BytesIO(data))

def train_split_filter(x, split_ratio=0.8):
    """Picklable filter function for training split."""
    return np.random.random() < split_ratio

def val_split_filter(x, split_ratio=0.8):
    """Picklable filter function for validation split."""
    return np.random.random() >= split_ratio

class WebDatasetDecoder:
    """Picklable decoder class for WebDataset."""
    
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
    
    def __call__(self, sample_tuple):
        """Decode a single sample from WebDataset and convert to ACT policy format."""
        cam1, cam2, qpos, actions = sample_tuple
        
        # Images are now uint8 torch tensors (HWC format: 224x224x3)
        # Convert to float32, HWC to CHW, and normalize to [0, 1]
        cam1_tensor = cam1.float().permute(2, 0, 1) * (1.0/255.0)
        cam2_tensor = cam2.float().permute(2, 0, 1) * (1.0/255.0)
        
        # Handle action chunking and padding
        # actions is already a PyTorch tensor (float16 from .pth file)
        original_length = len(actions)
        
        if original_length >= self.chunk_size:
            # Take the first chunk_size actions (next actions from current obs)
            padded_actions = actions[:self.chunk_size].float()
            is_pad_tensor = torch.zeros(self.chunk_size, dtype=torch.bool)
        else:
            # Pad to chunk_size if we have fewer actions than needed
            padded_actions = torch.zeros((self.chunk_size, actions.shape[1]), dtype=torch.float32)
            padded_actions[:original_length] = actions.float()
            is_pad_tensor = torch.ones(self.chunk_size, dtype=torch.bool)
            is_pad_tensor[:original_length] = False
        
        # Convert to ACT policy expected format
        sample = {
            'observation.image_camera_1': cam1_tensor,  # (3, H, W)
            'observation.image_camera_2': cam2_tensor,  # (3, H, W)
            'observation.state': qpos.float(),  # (6,) - already a tensor
            'action': padded_actions,  # (chunk_size, action_dim)
            'action_is_pad': is_pad_tensor  # (chunk_size,)
        }
        
        return sample

def calculate_webdataset_stats(dataloader, max_samples=None):
    """Calculate dataset statistics from WebDataset dataloader using online computation."""
    if max_samples is None:
        print("Computing dataset statistics from ALL samples...")
    else:
        print(f"Computing dataset statistics from up to {max_samples} samples...")
    
    # Online statistics accumulators
    qpos_sum = None
    qpos_sum_sq = None
    qpos_count = 0
    
    action_sum = None
    action_sum_sq = None
    action_count = 0
    
    cam1_sum = None
    cam1_sum_sq = None
    cam1_count = 0
    
    cam2_sum = None
    cam2_sum_sq = None
    cam2_count = 0
    
    sample_count = 0
    
    # Wrap the dataloader with tqdm
    pbar = tqdm(dataloader, desc="Computing stats", unit="batch")
    
    for batch in pbar:
        batch_size = batch['observation.state'].shape[0]
        
        # Process qpos data
        qpos_batch = batch['observation.state']  # (B, qpos_dim)
        if qpos_sum is None:
            qpos_sum = torch.zeros_like(qpos_batch[0])
            qpos_sum_sq = torch.zeros_like(qpos_batch[0])
        
        qpos_sum += qpos_batch.sum(dim=0)
        qpos_sum_sq += (qpos_batch ** 2).sum(dim=0)
        qpos_count += batch_size
        
        # Process action data (excluding padded actions)
        actions_batch = batch['action']  # (B, chunk_size, action_dim)
        is_pad_batch = batch['action_is_pad']  # (B, chunk_size)
        
        for i in range(batch_size):
            real_actions = actions_batch[i][~is_pad_batch[i]]  # (real_length, action_dim)
            if len(real_actions) > 0:
                if action_sum is None:
                    action_sum = torch.zeros_like(real_actions[0])
                    action_sum_sq = torch.zeros_like(real_actions[0])
                
                action_sum += real_actions.sum(dim=0)
                action_sum_sq += (real_actions ** 2).sum(dim=0)
                action_count += len(real_actions)
        
        # Process camera 1 images
        cam1_batch = batch['observation.image_camera_1']  # (B, 3, H, W)
        if cam1_sum is None:
            cam1_sum = torch.zeros(cam1_batch.shape[1])  # (3,)
            cam1_sum_sq = torch.zeros(cam1_batch.shape[1])  # (3,)
        
        # Sum over batch, height, width dimensions
        cam1_batch_flat = cam1_batch.view(batch_size, cam1_batch.shape[1], -1)  # (B, 3, H*W)
        cam1_sum += cam1_batch_flat.sum(dim=(0, 2))  # Sum over batch and spatial
        cam1_sum_sq += (cam1_batch_flat ** 2).sum(dim=(0, 2))
        cam1_count += batch_size * cam1_batch.shape[2] * cam1_batch.shape[3]  # B * H * W
        
        # Process camera 2 images
        cam2_batch = batch['observation.image_camera_2']  # (B, 3, H, W)
        if cam2_sum is None:
            cam2_sum = torch.zeros(cam2_batch.shape[1])  # (3,)
            cam2_sum_sq = torch.zeros(cam2_batch.shape[1])  # (3,)
        
        cam2_batch_flat = cam2_batch.view(batch_size, cam2_batch.shape[1], -1)
        cam2_sum += cam2_batch_flat.sum(dim=(0, 2))
        cam2_sum_sq += (cam2_batch_flat ** 2).sum(dim=(0, 2))
        cam2_count += batch_size * cam2_batch.shape[2] * cam2_batch.shape[3]
        
        sample_count += batch_size
        
        # Update progress bar
        pbar.set_postfix({
            'samples': sample_count,
            'batch_size': batch_size
        })
        
        # Break if we've reached max_samples
        if max_samples is not None and sample_count >= max_samples:
            break
    
    pbar.close()
    
    # Compute final statistics using online formulas
    # mean = sum / count
    # var = (sum_sq / count) - (mean ** 2)
    # std = sqrt(var)
    
    # QPos statistics
    qpos_mean = qpos_sum / qpos_count
    qpos_var = (qpos_sum_sq / qpos_count) - (qpos_mean ** 2)
    # Clamp variance to non-negative before sqrt to prevent NaN from floating point errors
    qpos_var = torch.clamp(qpos_var, min=0)
    qpos_std = torch.clamp(torch.sqrt(qpos_var), min=1e-6)
    
    # Action statistics
    action_mean = action_sum / action_count
    action_var = (action_sum_sq / action_count) - (action_mean ** 2)
    # Clamp variance to non-negative before sqrt to prevent NaN from floating point errors
    action_var = torch.clamp(action_var, min=0)
    action_std = torch.clamp(torch.sqrt(action_var), min=1e-6)
    
    # Camera statistics
    cam1_mean = (cam1_sum / cam1_count).reshape(-1, 1, 1)  # (3, 1, 1)
    cam1_var = (cam1_sum_sq / cam1_count) - (cam1_sum / cam1_count) ** 2
    cam1_var = torch.clamp(cam1_var, min=0)  # Prevent negative variance from float precision
    cam1_std = torch.clamp(torch.sqrt(cam1_var), min=1e-6).reshape(-1, 1, 1)
    
    cam2_mean = (cam2_sum / cam2_count).reshape(-1, 1, 1)  # (3, 1, 1)
    cam2_var = (cam2_sum_sq / cam2_count) - (cam2_sum / cam2_count) ** 2
    cam2_var = torch.clamp(cam2_var, min=0)  # Prevent negative variance from float precision
    cam2_std = torch.clamp(torch.sqrt(cam2_var), min=1e-6).reshape(-1, 1, 1)
    
    dataset_stats = {
        'observation.state': {
            'mean': qpos_mean,
            'std': qpos_std
        },
        'action': {
            'mean': action_mean,
            'std': action_std
        },
        'observation.image_camera_1': {
            'mean': cam1_mean,
            'std': cam1_std
        },
        'observation.image_camera_2': {
            'mean': cam2_mean,
            'std': cam2_std
        }
    }
    
    print("Dataset statistics computed successfully!")
    print(f"Processed {sample_count} samples, {qpos_count} qpos, {action_count} actions")
    
    return dataset_stats

def initialize_webdataset_data(data_dir, chunk_size=100, batch_size=8, 
                              train_val_split=0.8, num_workers=4, 
                              prefetch_factor=2, seed=42, compute_stats_from_all=True):
    """
    Initialize WebDataset-based training and validation dataloaders.
    Uses WebDataset's built-in splitting mechanism.
    """
    
    # Set random seed for reproducible splits
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Find all .tar files and create pattern
    dataset_pattern = os.path.join(data_dir, "train-*.tar")
    all_files = sorted(glob.glob(dataset_pattern))
    
    if len(all_files) == 0:
        raise FileNotFoundError(f"No WebDataset .tar files found in {data_dir}")
    
    num_shards = len(all_files)
    print(f"Found {num_shards} WebDataset files")
    
    # Auto-reduce num_workers if we have fewer shards than workers
    # Each worker needs at least one shard to avoid "No samples found" error
    if num_workers > num_shards:
        old_workers = num_workers
        num_workers = max(1, num_shards)
        print(f"⚠️  Reducing num_workers from {old_workers} to {num_workers} (only {num_shards} shards available)")
    
    # Create pattern for all files
    if len(all_files) == 1:
        full_pattern = all_files[0]
    else:
        # Create webdataset range pattern
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
    
    print(f"Using dataset pattern: {full_pattern}")
    
    # Create decode functions - no augmentation parameters
    decode_fn = WebDatasetDecoder(chunk_size)
    
    # Create split functions using functional approach
    train_split_fn = partial(train_split_filter, split_ratio=train_val_split)
    val_split_fn = partial(val_split_filter, split_ratio=train_val_split)
    
    # Create train dataset
    # Note: Images are now stored as .pth files (PyTorch tensors, uint8)
    # WebDataset has built-in support for .pth files with decode("torch")
    train_dataset = (
        wds.WebDataset(full_pattern, shardshuffle=True)
        .decode("torch")
        .to_tuple("cam1.pth", "cam2.pth", "qpos.pth", "actions.pth")
        .select(train_split_fn)
        .map(decode_fn)
    )
    
    # Create val dataset
    val_dataset = (
        wds.WebDataset(full_pattern, shardshuffle=True)
        .decode("torch")
        .to_tuple("cam1.pth", "cam2.pth", "qpos.pth", "actions.pth")
        .select(val_split_fn)
        .map(decode_fn)
    )
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        drop_last=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        drop_last=False
    )
    
    # Calculate dataset statistics from training data
    print("Calculating dataset statistics...")
    max_samples = None if compute_stats_from_all else 1000
    dataset_stats = calculate_webdataset_stats(train_dataloader, max_samples=max_samples)
    
    return train_dataloader, val_dataloader, dataset_stats

