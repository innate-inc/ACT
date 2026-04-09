import h5py
import json
import os
import tarfile
import numpy as np
import cv2
import io
from pathlib import Path
from tqdm import tqdm
import torch


def convert_episode_to_samples(hdf5_path, episode_id, target_size=(224, 224)):
    """
    Convert a single HDF5 episode file to WebDataset samples.
    
    Args:
        hdf5_path (str): Path to the episode HDF5 file
        episode_id (int): Episode ID for naming
        target_size (tuple): Target image size (height, width) for resizing. Default: (224, 224)
        
    Returns:
        list: List of sample dictionaries, each containing the 4 files as bytes
    """
    samples = []
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Extract data
            actions = f['action'][:]  # Shape: (timesteps, action_dim)
            camera1_images = f['observations/images/camera_1'][:]  # Shape: (timesteps, H, W, 3)
            camera2_images = f['observations/images/camera_2'][:]  # Shape: (timesteps, H, W, 3)
            qpos = f['observations/qpos'][:]  # Shape: (timesteps, 6)
            
            num_timesteps = actions.shape[0]
            
            # Create a sample for each timestep (no progress bar here)
            for timestep in range(num_timesteps):
                sample_key = f"episode_{episode_id:04d}_{timestep:04d}"
                
                # Resize and convert camera images to uint8 torch tensors
                cam1_img = camera1_images[timestep].astype(np.uint8)
                cam1_img_resized = cv2.resize(cam1_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
                # Store as PyTorch tensor (uint8, HWC format)
                cam1_tensor = torch.from_numpy(cam1_img_resized)
                cam1_buffer = io.BytesIO()
                torch.save(cam1_tensor, cam1_buffer)
                cam1_bytes = cam1_buffer.getvalue()
                
                cam2_img = camera2_images[timestep].astype(np.uint8)
                cam2_img_resized = cv2.resize(cam2_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
                # Store as PyTorch tensor (uint8, HWC format)
                cam2_tensor = torch.from_numpy(cam2_img_resized)
                cam2_buffer = io.BytesIO()
                torch.save(cam2_tensor, cam2_buffer)
                cam2_bytes = cam2_buffer.getvalue()
                
                # Convert qpos to PyTorch tensor
                qpos_tensor = torch.from_numpy(qpos[timestep].astype(np.float16))
                qpos_buffer = io.BytesIO()
                torch.save(qpos_tensor, qpos_buffer)
                qpos_bytes = qpos_buffer.getvalue()
                
                # Convert actions from current timestep to end
                actions_future = actions[timestep:]
                actions_tensor = torch.from_numpy(actions_future.astype(np.float16))
                actions_buffer = io.BytesIO()
                torch.save(actions_tensor, actions_buffer)
                actions_bytes = actions_buffer.getvalue()
                
                # Create sample dictionary
                sample = {
                    'key': sample_key,
                    'cam1.pth': cam1_bytes,
                    'cam2.pth': cam2_bytes,
                    'qpos.pth': qpos_bytes,
                    'actions.pth': actions_bytes
                }
                
                samples.append(sample)
                
        return samples, num_timesteps  # Return both samples and timestep count
        
    except Exception as e:
        print(f"  ❌ Error processing episode {episode_id:04d}: {e}")
        return [], 0


def write_samples_to_tar(samples, tar_path, shard_idx):
    """
    Write samples to a tar file (WebDataset shard).
    
    Args:
        samples (list): List of sample dictionaries
        tar_path (str): Output tar file path
        shard_idx (int): Shard index for progress reporting
    """
    try:
        with tarfile.open(tar_path, 'w') as tar:
            for sample in samples:
                sample_key = sample['key']
                
                # Add each file to the tar
                for ext, data in sample.items():
                    if ext == 'key':
                        continue
                    
                    filename = f"{sample_key}.{ext}"
                    tarinfo = tarfile.TarInfo(name=filename)
                    tarinfo.size = len(data)
                    
                    tar.addfile(tarinfo, io.BytesIO(data))
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error creating shard {shard_idx:05d}: {e}")
        return False


def convert_hdf5_to_webdataset(hdf5_directory, webd_directory, shard_size=1000, target_size=(224, 224)):
    """
    Convert HDF5 episode files to WebDataset format.
    
    Args:
        hdf5_directory (str): Directory containing HDF5 episode files and metadata.json
        webd_directory (str): Output directory for WebDataset shards
        shard_size (int): Number of samples per shard (default: 1000)
        target_size (tuple): Target image size (height, width) for resizing. Default: (224, 224)
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    # Check for metadata file (try both naming conventions)
    metadata_file = os.path.join(hdf5_directory, "metadata.json")
    if not os.path.exists(metadata_file):
        metadata_file = os.path.join(hdf5_directory, "dataset_metadata.json")
    
    if not os.path.exists(metadata_file):
        print(f"❌ Error: metadata.json or dataset_metadata.json not found in {hdf5_directory}")
        return False
    
    # Create output directory
    os.makedirs(webd_directory, exist_ok=True)
    
    try:
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print("🔄 CONVERTING HDF5 TO WEBDATASET FORMAT")
        print("=" * 60)
        print(f"📋 Task: {metadata.get('task_name', 'N/A')}")
        print(f"📁 Input: {hdf5_directory}")
        print(f"📁 Output: {webd_directory}")
        print(f"📦 Samples per shard: {shard_size}")
        print(f"🖼️  Image resize: {target_size[0]}x{target_size[1]}")
        print(f"💾 Image format: uint8 numpy arrays")
        
        episodes = metadata.get('episodes', [])
        if not episodes:
            print("❌ No episodes found in metadata")
            return False
        
        # Calculate total expected timesteps for progress bar
        print("📊 Calculating total timesteps...")
        total_timesteps = 0
        valid_episodes = []
        
        for episode in episodes:
            episode_id = episode.get('episode_id', episodes.index(episode))
            file_name = episode.get('file_name', '')
            hdf5_path = os.path.join(hdf5_directory, file_name)
            
            if not os.path.exists(hdf5_path):
                print(f"  ⚠️  File not found: {file_name}")
                continue
            
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    ep_timesteps = f['action'].shape[0]
                    total_timesteps += ep_timesteps
                    valid_episodes.append((episode, episode_id, hdf5_path, ep_timesteps))
            except Exception as e:
                print(f"  ⚠️  Error reading {file_name}: {e}")
                continue
        
        if not valid_episodes:
            print("❌ No valid episodes found")
            return False
        
        print(f"📈 Found {len(valid_episodes)} valid episodes with {total_timesteps:,} total timesteps")
        
        # Initialize streaming variables
        current_shard_samples = []
        current_shard_idx = 0
        total_samples = 0
        successful_shards = 0
        
        def write_current_shard():
            nonlocal current_shard_samples, current_shard_idx, successful_shards
            if current_shard_samples:
                tar_filename = f"train-{current_shard_idx:05d}.tar"
                tar_path = os.path.join(webd_directory, tar_filename)
                
                if write_samples_to_tar(current_shard_samples, tar_path, current_shard_idx):
                    successful_shards += 1
                    tqdm.write(f"  ✅ Created shard {current_shard_idx:05d}: {tar_filename} ({len(current_shard_samples)} samples)")
                
                current_shard_samples = []
                current_shard_idx += 1
        
        # Process episodes with single progress bar
        with tqdm(total=total_timesteps, desc="🔄 Converting timesteps", unit="samples") as pbar:
            for episode_info, episode_id, hdf5_path, expected_timesteps in valid_episodes:
                # Process episode and get samples
                episode_samples, actual_timesteps = convert_episode_to_samples(hdf5_path, episode_id, target_size=target_size)
                total_samples += len(episode_samples)
                
                # Update progress bar
                pbar.update(actual_timesteps)
                pbar.set_postfix({
                    'episode': f"{episode_id:04d}",
                    'samples': len(episode_samples),
                    'shards': successful_shards
                })
                
                # Add samples to current shard, writing when full
                for sample in episode_samples:
                    current_shard_samples.append(sample)
                    
                    # Write shard when it reaches the target size
                    if len(current_shard_samples) >= shard_size:
                        write_current_shard()
        
        # Write the final partial shard if it has samples
        write_current_shard()
        
        # Create dataset info file
        dataset_info = {
            'task_name': metadata.get('task_name', 'N/A'),
            'original_episodes': len(episodes),
            'valid_episodes': len(valid_episodes),
            'total_samples': total_samples,
            'samples_per_shard': shard_size,
            'num_shards': successful_shards,
            'successful_shards': successful_shards,
            'image_size': target_size,
            'sample_format': {
                'cam1.pth': f'RGB image from camera 1 (torch.uint8, {target_size[0]}x{target_size[1]}x3)',
                'cam2.pth': f'RGB image from camera 2 (torch.uint8, {target_size[0]}x{target_size[1]}x3)', 
                'qpos.pth': 'Joint positions (torch.float16)',
                'actions.pth': 'Future actions from current timestep (torch.float16)'
            }
        }
        
        info_path = os.path.join(webd_directory, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n📈 CONVERSION SUMMARY")
        print("=" * 60)
        print(f"✅ Successfully created {successful_shards} shards")
        print(f"🔢 Total samples: {total_samples:,}")
        print(f"📄 Dataset info saved: {info_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting HDF5 to WebDataset: {e}")
        return False
