import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional

# Assuming torchvision.transforms.v2 will be available if default augmentation is used.
# If not, and use_img_aug=True with img_aug_transforms=None, a NameError/ImportError will occur.
from torchvision.transforms import v2

class EpisodicHDF5DatasetRAM(Dataset):
    # --- Hardcoded Values Based on User Guarantees ---
    ACTION_DIM = 8
    QPOS_DIM = 6
    IMAGE_H = 480
    IMAGE_W = 640
    IMAGE_C = 3 

    H5_ACTION_KEY = 'action'
    H5_QPOS_KEY = 'observations/qpos'
    H5_CAMERA_NAMES = ['camera_1', 'camera_2'] 

    POLICY_ACTION_KEY = 'action'
    POLICY_QPOS_KEY = 'observation.qpos' 
    POLICY_IMG_KEYS = ['observation.image_camera_1', 'observation.image_camera_2']
    # --- End Hardcoded Values ---

    def __init__(self,
                 data_dir: str,
                 episode_ids: List[int],
                 use_img_aug: bool = False,
                 img_aug_transforms: Optional[torch.nn.Module] = None):
        super().__init__()
        self.data_dir = data_dir
        self.use_img_aug = use_img_aug
        self.transforms = img_aug_transforms

        if self.use_img_aug and self.transforms is None:
            # Directly uses v2, will fail if not imported/available.
            self.transforms = v2.Compose([v2.RandomPhotometricDistort(p=0.5), v2.RandomHorizontalFlip(p=0.5)])
        # If use_img_aug is False, or if img_aug_transforms is provided, this block is skipped or uses provided one.

        self.loaded_episodes_data: List[Dict[str, Any]] = []
        self.episode_info: List[Dict[str, Any]] = []
        self.max_episode_length = 0
        
        for ep_id in episode_ids:
            file_path = os.path.join(self.data_dir, f"episode_{ep_id}.h5")
            with h5py.File(file_path, 'r') as f:
                current_ep_len = f[self.H5_ACTION_KEY].shape[0]
                self.max_episode_length = max(self.max_episode_length, current_ep_len)
                
                ep_data_ram = {}
                ep_data_ram[self.POLICY_ACTION_KEY] = torch.from_numpy(f[self.H5_ACTION_KEY][:]).float()
                ep_data_ram[self.POLICY_QPOS_KEY] = torch.from_numpy(f[self.H5_QPOS_KEY][:]).float()

                for i, h5_cam_name in enumerate(self.H5_CAMERA_NAMES):
                    policy_img_key = self.POLICY_IMG_KEYS[i]
                    h5_img_path = f'observations/images/{h5_cam_name}'
                    img_thwc_uint8 = f[h5_img_path][:]
                    ep_data_ram[policy_img_key] = torch.from_numpy(img_thwc_uint8).float().permute(0, 3, 1, 2) / 255.0
                
                self.loaded_episodes_data.append(ep_data_ram)
                self.episode_info.append({"id": ep_id, "length": current_ep_len})

    def __len__(self) -> int:
        return len(self.loaded_episodes_data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ep_data = self.loaded_episodes_data[index]
        ep_info = self.episode_info[index]
        ep_len = ep_info["length"]

        start_ts = random.randint(0, ep_len - 1) 
        item: Dict[str, Any] = {}

        action_start_idx = max(0, start_ts - 1)
        actions_raw = ep_data[self.POLICY_ACTION_KEY][action_start_idx:]
        actions_raw_len = actions_raw.shape[0]

        padded_actions = torch.zeros((self.max_episode_length, self.ACTION_DIM), dtype=torch.float32)
        padded_actions[:actions_raw_len] = actions_raw
        item[self.POLICY_ACTION_KEY] = padded_actions
        
        action_is_pad = torch.ones(self.max_episode_length, dtype=torch.bool)
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

