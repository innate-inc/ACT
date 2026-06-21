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


def _torch_save_bytes(tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def _box_to_sock_state(result, conf_threshold, width, height):
    empty = np.zeros((5,), dtype=np.float32)
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return empty

    confs = boxes.conf.detach().cpu().numpy()
    keep = np.where(confs >= conf_threshold)[0]
    if keep.size == 0:
        return empty

    best_idx = int(keep[np.argmax(confs[keep])])
    x1, y1, x2, y2 = boxes.xyxy[best_idx].detach().cpu().numpy().astype(np.float32)
    state = np.array(
        [
            1.0,
            ((x1 + x2) * 0.5) / max(width, 1),
            ((y1 + y2) * 0.5) / max(height, 1),
            (x2 - x1) / max(width, 1),
            (y2 - y1) / max(height, 1),
        ],
        dtype=np.float32,
    )
    return np.clip(state, 0.0, 1.0)


def _predict_sock_states(model, frames, *, width, height, conf_threshold, imgsz, batch_size, device):
    if model is None:
        return None
    if not frames:
        return np.zeros((0, 5), dtype=np.float32)

    states = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start:start + batch_size]
        results = model.predict(
            batch,
            imgsz=imgsz,
            conf=conf_threshold,
            device=device,
            verbose=False,
            max_det=3,
        )
        for result in results:
            states.append(_box_to_sock_state(result, conf_threshold, width, height))
    return np.stack(states).astype(np.float32)


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


def convert_mp4_episode_to_samples(
    hdf5_path,
    video_paths,
    episode_id,
    target_size=(224, 224),
    sock_yolo_model=None,
    sock_yolo_conf=0.25,
    sock_yolo_imgsz=640,
    sock_yolo_batch=32,
    sock_yolo_device=None,
):
    """
    Convert an episode with MP4 video files + HDF5 (actions/qpos only) to WebDataset samples.
    
    Args:
        hdf5_path (str): Path to the episode HDF5 file (contains action and qpos, no images)
        video_paths (list): List of two MP4 file paths [camera_1.mp4, camera_2.mp4]
        episode_id (int): Episode ID for naming
        target_size (tuple): Target image size (height, width) for resizing. Default: (224, 224)
        
    Returns:
        tuple: (list of sample dicts, number of timesteps used)
    """
    samples = []
    caps = []

    try:
        with h5py.File(hdf5_path, 'r') as f:
            actions = f['action'][:]
            qpos = f['observations/qpos'][:]
            num_h5_timesteps = actions.shape[0]

        for vp in video_paths:
            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                print(f"  ❌ Cannot open video: {vp}")
                for c in caps:
                    c.release()
                return [], 0
            caps.append(cap)

        video_frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
        min_video_frames = min(video_frame_counts)
        num_timesteps = min(num_h5_timesteps, min_video_frames)

        if num_h5_timesteps != min_video_frames:
            print(f"  ⚠️  Episode {episode_id:04d}: H5 has {num_h5_timesteps} timesteps, "
                  f"videos have {video_frame_counts} frames. Using {num_timesteps}.")

        camera_frame_bytes = [[], []]
        camera_yolo_frames = [[], []]
        camera_sizes = []

        for cap_idx, cap in enumerate(caps):
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            camera_sizes.append((width, height))

            for timestep in range(num_timesteps):
                ret, frame = cap.read()
                if not ret:
                    print(f"  ❌ Failed to read frame {timestep} from camera {cap_idx + 1} (episode {episode_id:04d})")
                    for c in caps:
                        c.release()
                    return samples, timestep

                if sock_yolo_model is not None:
                    camera_yolo_frames[cap_idx].append(frame)

                frame_resized = cv2.resize(frame, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(frame_rgb)
                camera_frame_bytes[cap_idx].append(_torch_save_bytes(tensor))

            cap.release()

        camera_sock_states = None
        if sock_yolo_model is not None:
            if sock_yolo_device is None:
                sock_yolo_device = "0" if torch.cuda.is_available() else "cpu"
            camera_sock_states = []
            for frames, (width, height) in zip(camera_yolo_frames, camera_sizes):
                camera_sock_states.append(
                    _predict_sock_states(
                        sock_yolo_model,
                        frames,
                        width=width,
                        height=height,
                        conf_threshold=sock_yolo_conf,
                        imgsz=sock_yolo_imgsz,
                        batch_size=sock_yolo_batch,
                        device=sock_yolo_device,
                    )
                )

        for timestep in range(num_timesteps):
            sample_key = f"episode_{episode_id:04d}_{timestep:04d}"

            qpos_tensor = torch.from_numpy(qpos[timestep].astype(np.float16))

            actions_future = actions[timestep:]
            actions_tensor = torch.from_numpy(actions_future.astype(np.float16))

            sample = {
                'key': sample_key,
                'cam1.pth': camera_frame_bytes[0][timestep],
                'cam2.pth': camera_frame_bytes[1][timestep],
                'qpos.pth': _torch_save_bytes(qpos_tensor),
                'actions.pth': _torch_save_bytes(actions_tensor)
            }
            if camera_sock_states is not None:
                sock_state = np.concatenate(
                    [states[timestep] for states in camera_sock_states],
                    axis=0,
                ).astype(np.float16)
                sample['sock_state.pth'] = _torch_save_bytes(torch.from_numpy(sock_state))
            samples.append(sample)

        return samples, num_timesteps

    except Exception as e:
        print(f"  ❌ Error processing episode {episode_id:04d}: {e}")
        for cap in caps:
            cap.release()
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


def convert_to_webdataset(
    data_directory,
    webd_directory,
    shard_size=1000,
    target_size=(224, 224),
    use_sock_state=False,
    sock_yolo_weights=None,
    sock_yolo_conf=0.25,
    sock_yolo_imgsz=640,
    sock_yolo_batch=32,
    sock_yolo_device=None,
):
    """
    Convert episode files to WebDataset format.
    
    Supports two dataset types based on metadata `dataset_type` field:
      - Default / "h5": images stored inside HDF5 files
      - "h264": images stored as separate MP4 video files, HDF5 contains only actions/qpos
    
    Args:
        data_directory (str): Directory containing episode files and metadata.json
        webd_directory (str): Output directory for WebDataset shards
        shard_size (int): Number of samples per shard (default: 1000)
        target_size (tuple): Target image size (height, width) for resizing. Default: (224, 224)
        use_sock_state (bool): If True, add a YOLO-derived sock_state.pth tensor to each sample.
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    # Check for metadata file (try both naming conventions)
    metadata_file = os.path.join(data_directory, "metadata.json")
    if not os.path.exists(metadata_file):
        metadata_file = os.path.join(data_directory, "dataset_metadata.json")
    
    if not os.path.exists(metadata_file):
        print(f"❌ Error: metadata.json or dataset_metadata.json not found in {data_directory}")
        return False
    
    # Create output directory
    os.makedirs(webd_directory, exist_ok=True)
    
    try:
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        dataset_type = metadata.get('dataset_type', 'h5')
        is_h264 = dataset_type == 'h264'
        sock_yolo_model = None

        if use_sock_state:
            if not is_h264:
                print("❌ Sock-state conversion currently requires h264 datasets with MP4 video files")
                return False
            if not sock_yolo_weights or not os.path.exists(sock_yolo_weights):
                print(f"❌ Sock YOLO weights not found: {sock_yolo_weights}")
                return False
            from ultralytics import YOLO
            print(f"🧦 Loading sock YOLO checkpoint: {sock_yolo_weights}")
            sock_yolo_model = YOLO(sock_yolo_weights)
            if sock_yolo_device is None:
                sock_yolo_device = "0" if torch.cuda.is_available() else "cpu"

        print("🔄 CONVERTING EPISODES TO WEBDATASET FORMAT")
        print("=" * 60)
        print(f"📋 Task: {metadata.get('task_name', 'N/A')}")
        print(f"📋 Dataset type: {dataset_type}")
        print(f"📁 Input: {data_directory}")
        print(f"📁 Output: {webd_directory}")
        print(f"📦 Samples per shard: {shard_size}")
        print(f"🖼️  Image resize: {target_size[0]}x{target_size[1]}")
        if is_h264:
            print(f"🎬 Image source: MP4 video files")
        else:
            print(f"💾 Image source: HDF5 arrays")
        print(f"🧦 Sock state: {'enabled' if use_sock_state else 'disabled'}")
        if use_sock_state:
            print(f"   YOLO conf/imgsz/batch/device: {sock_yolo_conf}/{sock_yolo_imgsz}/{sock_yolo_batch}/{sock_yolo_device}")
        
        episodes = metadata.get('episodes', [])
        if not episodes:
            print("❌ No episodes found in metadata")
            return False
        
        # Validate episodes and calculate total expected timesteps
        print("📊 Calculating total timesteps...")
        total_timesteps = 0
        valid_episodes = []
        
        for episode in episodes:
            episode_id = episode.get('episode_id', episodes.index(episode))
            file_name = episode.get('file_name', '')
            hdf5_path = os.path.join(data_directory, file_name)
            
            if not os.path.exists(hdf5_path):
                print(f"  ⚠️  H5 file not found: {file_name}")
                continue
            
            # For h264, also check that the video files exist
            video_abs_paths = []
            if is_h264:
                video_files = episode.get('video_files', [])
                if len(video_files) < 2:
                    print(f"  ⚠️  Episode {episode_id}: expected 2 video_files, got {len(video_files)}")
                    continue
                missing = False
                for vf in video_files[:2]:
                    vp = os.path.join(data_directory, vf)
                    if not os.path.exists(vp):
                        print(f"  ⚠️  Video file not found: {vf}")
                        missing = True
                        break
                    video_abs_paths.append(vp)
                if missing:
                    continue
            
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    ep_timesteps = f['action'].shape[0]
                    total_timesteps += ep_timesteps
                    valid_episodes.append((episode, episode_id, hdf5_path, ep_timesteps, video_abs_paths))
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
            for episode_info, episode_id, hdf5_path, expected_timesteps, video_abs_paths in valid_episodes:
                if is_h264:
                    episode_samples, actual_timesteps = convert_mp4_episode_to_samples(
                        hdf5_path,
                        video_abs_paths,
                        episode_id,
                        target_size=target_size,
                        sock_yolo_model=sock_yolo_model,
                        sock_yolo_conf=sock_yolo_conf,
                        sock_yolo_imgsz=sock_yolo_imgsz,
                        sock_yolo_batch=sock_yolo_batch,
                        sock_yolo_device=sock_yolo_device,
                    )
                else:
                    episode_samples, actual_timesteps = convert_episode_to_samples(
                        hdf5_path, episode_id, target_size=target_size)

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
            'dataset_type': dataset_type,
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
            },
            'sock_state_enabled': use_sock_state,
        }
        if use_sock_state:
            dataset_info['sample_format']['sock_state.pth'] = (
                'YOLO sock state for camera 1 and camera 2: '
                '[valid, cx, cy, w, h] x 2, normalized torch.float16'
            )
            dataset_info['sock_yolo'] = {
                'weights': os.path.basename(sock_yolo_weights),
                'conf': sock_yolo_conf,
                'imgsz': sock_yolo_imgsz,
                'batch': sock_yolo_batch,
                'device': sock_yolo_device,
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
        print(f"❌ Error converting to WebDataset: {e}")
        return False
