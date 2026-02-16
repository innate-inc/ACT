#!/usr/bin/env python3
"""
Distributed training script for InnatePolicy.
Similar to train_dist.py but adapted for diffusion-based policy with flow matching.
"""

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb
import os
from datetime import datetime
import socket
import time
import argparse
import math

from innate_policy import InnatePolicy
from data_utils import initialize_webdataset_data
from data_tools.webdataset import convert_hdf5_to_webdataset


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # Different port from train_dist
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the GPU for the current process
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def convert_data_always(data_dir, shard_size=500, force_reconvert=True):
    """Always convert HDF5 data to WebDataset format (before distributed training)."""
    DATA_DIR = data_dir
    WEBD_DIR = os.path.join(DATA_DIR, "webdataset")
    
    print("🔄 CONVERTING HDF5 TO WEBDATASET FORMAT")
    print("=" * 50)
    
    # Remove existing WebDataset directory if it exists and force_reconvert is True
    if os.path.exists(WEBD_DIR) and force_reconvert:
        import shutil
        print(f"🗑️  Removing existing WebDataset directory: {WEBD_DIR}")
        shutil.rmtree(WEBD_DIR)
    
    # Check if already converted
    if os.path.exists(WEBD_DIR) and not force_reconvert:
        print(f"✅ WebDataset directory already exists: {WEBD_DIR}")
        print("   Skipping conversion (use --force-reconvert to recreate)")
        return True, WEBD_DIR
    
    print(f"🔄 Converting HDF5 data to WebDataset format...")
    print(f"📁 HDF5 source: {DATA_DIR}")
    print(f"📁 WebDataset target: {WEBD_DIR}")
    print(f"📦 Shard size: {shard_size}")
    print(f"🖼️  Image format: uint8 PyTorch tensors, 224x224")
    
    # Perform conversion with optimized settings
    success = convert_hdf5_to_webdataset(
        hdf5_directory=DATA_DIR,
        webd_directory=WEBD_DIR,
        shard_size=shard_size,
        target_size=(224, 224)  # Resize images to 224x224
    )
    
    if success:
        print("✅ Data conversion completed successfully!")
        return True, WEBD_DIR
    else:
        print("❌ Data conversion failed!")
        return False, None


def train_ddp(rank, world_size, args, webd_dir):
    """Main distributed training function for InnatePolicy."""
    setup(rank, world_size)
    
    # --- Configuration ---
    # Data parameters
    DATA_DIR = args.data_dir
    CHUNK_SIZE = args.chunk_size
    TRAIN_VAL_SPLIT = 0.9
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    # WebDataset conversion parameters
    SHARD_SIZE = args.shard_size
    
    # Training optimization parameters
    USE_BF16 = args.use_bf16  # BF16 mixed precision training
    USE_COMPILE = args.use_compile  # torch.compile() optimization
    NORMALIZE_DATA = args.normalize_data  # Normalize actions and robot state

    # Task name and automatic checkpoint directory generation
    TASK_NAME = os.path.basename(DATA_DIR.rstrip('/'))
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use custom wandb run name if provided, otherwise auto-generate
    if args.wandb_run_name:
        RUN_NAME = args.wandb_run_name
    else:
        RUN_NAME = f"{TASK_NAME}_{TIMESTAMP}_innate_ddp"
    
    CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints", RUN_NAME)

    # Use the WebDataset directory passed from main
    WEBD_DIR = webd_dir

    # Image dimensions (DINOv2 expects 224x224)
    IMAGE_H = 224
    IMAGE_W = 224
    IMAGE_C = 3
    
    # State and action dimensions
    STATE_DIM = args.state_dim
    ACTION_DIM = args.action_dim

    INPUT_SHAPES = {
        "observation.image_camera_1": [IMAGE_C, IMAGE_H, IMAGE_W],
        "observation.image_camera_2": [IMAGE_C, IMAGE_H, IMAGE_W],
        "observation.state": [STATE_DIM]
    }
    OUTPUT_SHAPES = {
        "action": [ACTION_DIM]
    }

    # InnatePolicy Hyperparameters
    NUM_CAMERAS = 2
    NUM_QUERIES = args.num_queries
    FREEZE_VISION_BACKBONE = args.freeze_vision_backbone
    PROPRIO_HIDDEN_DIM = args.proprio_hidden_dim
    ACTION_HORIZON = CHUNK_SIZE
    
    # UNet parameters
    DIFFUSION_STEP_EMBED_DIM = args.diffusion_step_embed_dim
    DOWN_DIMS = args.down_dims
    KERNEL_SIZE = args.kernel_size
    N_GROUPS = args.n_groups
    
    # Inference parameters
    NUM_INFERENCE_STEPS = args.num_inference_steps

    # Training parameters
    MAX_STEPS = args.max_steps
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    LEARNING_RATE_BACKBONE = args.learning_rate_backbone
    # Default warmup to 5% of total steps if not specified
    WARMUP_STEPS = args.warmup_steps if args.warmup_steps is not None else int(0.05 * MAX_STEPS)
    MIN_LR_RATIO = 0.1  # Decay to 1/10th of original LR
    
    # Calculate checkpoint interval for exactly 10 checkpoints
    CHECKPOINT_INTERVAL = MAX_STEPS // 10

    # W&B Configuration
    WANDB_PROJECT = args.wandb_project
    WANDB_ENTITY = args.wandb_entity
    WANDB_API_KEY = args.wandb_api_key

    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    
    # Only print and create directories on rank 0
    if rank == 0:
        print(f"Starting distributed training on {world_size} GPUs")
        print(f"Using device: {device}")
        print(f"Using WebDataset directory: {WEBD_DIR}")
        
        # Create checkpoint directory
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        print(f"Checkpoint directory: {CHECKPOINT_DIR}")

        # Initialize W&B only on rank 0
        if WANDB_API_KEY and not os.getenv("WANDB_API_KEY"):
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=RUN_NAME, config={
            # Environment
            "data_dir": DATA_DIR,
            "webd_dir": WEBD_DIR,
            "task_name": TASK_NAME,
            "timestamp": TIMESTAMP,
            "run_name": RUN_NAME,
            "hostname": socket.gethostname(),
            "checkpoint_dir": CHECKPOINT_DIR,
            
            # Data & Batch
            "chunk_size": CHUNK_SIZE,
            "batch_size_per_gpu": BATCH_SIZE,
            "total_effective_batch_size": BATCH_SIZE * world_size,
            "num_workers": NUM_WORKERS,
            "shard_size": SHARD_SIZE,
            
            # Hardware
            "world_size": world_size,
            
            # Optimization
            "use_bf16": USE_BF16,
            "use_compile": USE_COMPILE,
            "normalize_data": NORMALIZE_DATA,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "lr_backbone": LEARNING_RATE_BACKBONE,
            "max_steps": MAX_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "min_lr_ratio": MIN_LR_RATIO,
            
            # Model Architecture
            "num_cameras": NUM_CAMERAS,
            "num_queries": NUM_QUERIES,
            "freeze_vision_backbone": FREEZE_VISION_BACKBONE,
            "state_dim": STATE_DIM,
            "proprio_hidden_dim": PROPRIO_HIDDEN_DIM,
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "diffusion_step_embed_dim": DIFFUSION_STEP_EMBED_DIM,
            "down_dims": DOWN_DIMS,
            "kernel_size": KERNEL_SIZE,
            "n_groups": N_GROUPS,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            
            # Input/Output Shapes
            "image_h": IMAGE_H,
            "image_w": IMAGE_W,
            "input_shapes": INPUT_SHAPES,
            "output_shapes": OUTPUT_SHAPES,
        })

    # --- 1. Initialize DataLoaders and get dataset_stats ---
    if rank == 0:
        print("Initializing WebDataset data loaders...")
    
    try:
        train_dataloader, val_dataloader, dataset_stats = initialize_webdataset_data(
            data_dir=WEBD_DIR,
            chunk_size=CHUNK_SIZE,
            batch_size=BATCH_SIZE,
            train_val_split=TRAIN_VAL_SPLIT,
            num_workers=NUM_WORKERS,
            prefetch_factor=2,
            seed=42 + rank,
            normalize=NORMALIZE_DATA  # Use the normalization flag
        )
    except (FileNotFoundError, ValueError) as e:
        if rank == 0:
            print(f"Error initializing WebDataset: {e}")
            print(f"Please ensure your data directory '{WEBD_DIR}' contains WebDataset .tar files.")
            wandb.finish(exit_code=1)
        cleanup()
        return

    # Save dataset_stats only on rank 0
    if rank == 0 and dataset_stats:
        stats_path = os.path.join(CHECKPOINT_DIR, "dataset_stats.pt")
        try:
            torch.save(dataset_stats, stats_path)
            print(f"Saved dataset_stats to {stats_path}")
            wandb.config.update({"dataset_stats_path": stats_path})
        except Exception as e:
            print(f"Error saving dataset_stats: {e}")

    if rank == 0:
        print("WebDataset dataloaders initialized (streaming datasets - no fixed length)")

    # --- 2. Initialize InnatePolicy and Optimizer ---
    if rank == 0:
        print("Initializing InnatePolicy...")
    
    policy = InnatePolicy(
        num_queries=NUM_QUERIES,
        freeze_vision_backbone=FREEZE_VISION_BACKBONE,
        num_cameras=NUM_CAMERAS,
        state_dim=STATE_DIM,
        proprio_hidden_dim=PROPRIO_HIDDEN_DIM,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        diffusion_step_embed_dim=DIFFUSION_STEP_EMBED_DIM,
        down_dims=DOWN_DIMS,
        kernel_size=KERNEL_SIZE,
        n_groups=N_GROUPS,
        num_inference_steps=NUM_INFERENCE_STEPS
    ).to(device)
    
    # Apply torch.compile() if enabled (before DDP wrapping)
    if USE_COMPILE:
        if rank == 0:
            print("⚡ Compiling model with torch.compile()...")
            print("   Mode: default (with inductor backend)")
        try:
            policy = torch.compile(policy, mode='default')
            if rank == 0:
                print("   ✓ Model compilation setup completed")
                print("   Note: First forward/backward pass will trigger actual compilation")
        except Exception as e:
            if rank == 0:
                print(f"   ⚠️  Compilation failed: {e}")
                print("   Falling back to uncompiled model")
    
    # Wrap model with DDP
    policy = DDP(policy, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # Setup optimizer with different learning rates for vision backbone vs rest
    # Separate parameters into vision backbone and rest
    vision_params = []
    other_params = []
    
    for name, param in policy.module.named_parameters():
        if 'vision_encoder.backbone' in name:
            vision_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': other_params, 'lr': LEARNING_RATE},
        {'params': vision_params, 'lr': LEARNING_RATE_BACKBONE}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    
    # Create learning rate scheduler: Linear warmup + Cosine annealing to min_lr
    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            # Linear warmup from 0 to 1
            return float(current_step) / float(max(1, WARMUP_STEPS))
        else:
            # Cosine annealing from 1.0 to MIN_LR_RATIO after warmup
            progress = float(current_step - WARMUP_STEPS) / float(max(1, MAX_STEPS - WARMUP_STEPS))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine_decay
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    if rank == 0:
        print(f"Learning rate scheduler: Linear warmup ({WARMUP_STEPS} steps, {WARMUP_STEPS/MAX_STEPS*100:.1f}%) + Cosine annealing to {MIN_LR_RATIO*100:.0f}% of max LR")

    # Only watch model on rank 0
    if rank == 0:
        wandb.watch(policy.module, log="all", log_freq=100)

    # --- 3. Step-based Training Loop ---
    if rank == 0:
        print("Starting step-based distributed training...")
    
    step = 0
    
    # Initialize tqdm only on rank 0
    if rank == 0:
        overall_pbar = tqdm(total=MAX_STEPS, unit="step", desc="Training Progress")

    # Initialize placeholders for validation metrics
    latest_val_avg_loss = "N/A"

    # Create infinite iterator from dataloader
    train_iter = iter(train_dataloader)
    
    # Create persistent validation iterator (only on rank 0)
    val_iter = None
    if rank == 0:
        val_iter = iter(val_dataloader)
    
    # Profiling variables (only track on rank 0)
    if rank == 0:
        batch_load_times = []
        forward_times = []
        backward_times = []
        optimizer_times = []
        total_step_times = []
    
    while step < MAX_STEPS:
        if rank == 0:
            step_start_time = time.time()
        
        policy.train()
        
        # 1. Batch Loading
        if rank == 0:
            batch_load_start = time.time()
        
        try:
            batch = next(train_iter)
        except StopIteration:
            # Reset iterator when dataset is exhausted
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
            if rank == 0:
                overall_pbar.write(f"Completed one pass through dataset at step {step}")
        
        # Move batch to device and prepare for InnatePolicy
        # Expected keys: observation.image_camera_1, observation.image_camera_2, observation.state, action
        images = torch.stack([
            batch["observation.image_camera_1"],
            batch["observation.image_camera_2"]
        ], dim=1).to(device)  # [B, 2, 3, H, W]
        
        robot_state = batch["observation.state"].to(device)  # [B, state_dim]
        actions = batch["action"].to(device)  # [B, chunk_size, action_dim]
        
        if rank == 0:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_load_time = time.time() - batch_load_start
        
        # 2. Forward Pass (with BF16 if enabled)
        if rank == 0:
            forward_start = time.time()
        
        if USE_BF16:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = policy(images, robot_state, actions, training=True)
                loss = output['loss']
        else:
            output = policy(images, robot_state, actions, training=True)
            loss = output['loss']
        
        if rank == 0:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            forward_time = time.time() - forward_start
        
        # 3. Backward Pass
        if rank == 0:
            backward_start = time.time()
        
        optimizer.zero_grad()
        loss.backward()
        
        if rank == 0:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            backward_time = time.time() - backward_start
        
        # 4. Optimizer Update
        if rank == 0:
            optimizer_start = time.time()
        
        optimizer.step()
        scheduler.step()  # Update learning rate
        
        if rank == 0:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            optimizer_time = time.time() - optimizer_start
            
            total_step_time = time.time() - step_start_time
            
            # Store timing data
            batch_load_times.append(batch_load_time)
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            optimizer_times.append(optimizer_time)
            total_step_times.append(total_step_time)

            # Log metrics to WandB
            wandb.log({
                "train/flow_matching_loss": loss.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/lr_backbone": optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr'],
                "timing/batch_load_ms": batch_load_time * 1000,
                "timing/forward_ms": forward_time * 1000,
                "timing/backward_ms": backward_time * 1000,
                "timing/optimizer_ms": optimizer_time * 1000,
                "timing/total_step_ms": total_step_time * 1000,
                "step": step
            })
        
        step += 1
        
        if rank == 0:
            overall_pbar.update(1)
            
            # Update progress bar postfix with timing info
            overall_pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                step_ms=f"{total_step_time*1000:.1f}",
                val_loss=latest_val_avg_loss,
                refresh=True
            )
            
            # Print detailed timing every 100 steps
            if step % 100 == 0:
                avg_batch_load = sum(batch_load_times[-100:]) / len(batch_load_times[-100:]) * 1000
                avg_forward = sum(forward_times[-100:]) / len(forward_times[-100:]) * 1000
                avg_backward = sum(backward_times[-100:]) / len(backward_times[-100:]) * 1000
                avg_optimizer = sum(optimizer_times[-100:]) / len(optimizer_times[-100:]) * 1000
                avg_total = sum(total_step_times[-100:]) / len(total_step_times[-100:]) * 1000
                
                overall_pbar.write(
                    f"Step {step} - Avg timing (last 100 steps): "
                    f"Batch: {avg_batch_load:.1f}ms, "
                    f"Forward: {avg_forward:.1f}ms, "
                    f"Backward: {avg_backward:.1f}ms, "
                    f"Optimizer: {avg_optimizer:.1f}ms, "
                    f"Total: {avg_total:.1f}ms"
                )

        # --- Validation Step (only on rank 0) ---
        if step % 100 == 0:
            # Synchronize all ranks before validation
            dist.barrier()
            
            if rank == 0:
                val_start_time = time.time()
                policy.eval()
                
                with torch.no_grad():
                    try:
                        # Time batch loading
                        batch_load_start = time.time()
                        batch_val = next(val_iter)
                        
                        # Prepare validation batch
                        images_val = torch.stack([
                            batch_val["observation.image_camera_1"],
                            batch_val["observation.image_camera_2"]
                        ], dim=1).to(device)
                        robot_state_val = batch_val["observation.state"].to(device)
                        actions_gt = batch_val["action"].to(device)
                        
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        batch_load_time = time.time() - batch_load_start
                        
                        # === 1. Flow Matching Loss (velocity prediction) ===
                        forward_start = time.time()
                        if USE_BF16:
                            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                                output_val = policy(images_val, robot_state_val, actions_gt, training=True)
                                loss_val = output_val['loss']
                        else:
                            output_val = policy(images_val, robot_state_val, actions_gt, training=True)
                            loss_val = output_val['loss']
                        
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        forward_time = time.time() - forward_start
                        
                        # === 2. Action Prediction Error (sampled actions) ===
                        sampling_start = time.time()
                        
                        # Get visual and proprioceptive features
                        if USE_BF16:
                            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                                visual_features = policy.module.encode_images(images_val)
                                proprio_features = policy.module.encode_proprio(robot_state_val)
                        else:
                            visual_features = policy.module.encode_images(images_val)
                            proprio_features = policy.module.encode_proprio(robot_state_val)
                        
                        global_cond = torch.cat([visual_features, proprio_features], dim=-1)
                        
                        # Sample actions using flow matching (full inference)
                        if USE_BF16:
                            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                                actions_pred = policy.module.sample_actions(global_cond, use_heun=True)
                        else:
                            actions_pred = policy.module.sample_actions(global_cond, use_heun=True)
                        
                        # === PRIMARY METRICS: Normalized errors (percentage-like interpretation) ===
                        # With normalized data ~N(0,1), error of 0.1 ≈ 10% relative error
                        action_l1_error = torch.abs(actions_pred - actions_gt).mean()
                        action_l2_error = torch.sqrt(((actions_pred - actions_gt) ** 2).mean())
                        
                        # Per-dimension errors (normalized, for balanced comparison across dimensions)
                        action_l1_per_dim = torch.abs(actions_pred - actions_gt).mean(dim=(0, 1))  # [action_dim]
                        
                        # Per-timestep errors (normalized, to see if early/late actions are harder)
                        action_l1_per_timestep = torch.abs(actions_pred - actions_gt).mean(dim=(0, 2))  # [T]
                        action_l1_first = action_l1_per_timestep[0].item()
                        action_l1_last = action_l1_per_timestep[-1].item()
                        
                        # === SECONDARY METRICS: Physical units (for real-world interpretation) ===
                        if NORMALIZE_DATA:
                            action_mean = dataset_stats['action']['mean'].to(device)
                            action_std = dataset_stats['action']['std'].to(device)
                            actions_pred_phys = actions_pred * action_std + action_mean
                            actions_gt_phys = actions_gt * action_std + action_mean
                            
                            # Compute physical errors (secondary, for interpretation)
                            action_l1_error_physical = torch.abs(actions_pred_phys - actions_gt_phys).mean()
                            action_l2_error_physical = torch.sqrt(((actions_pred_phys - actions_gt_phys) ** 2).mean())
                        else:
                            # If no normalization, normalized and physical are the same
                            action_l1_error_physical = action_l1_error
                            action_l2_error_physical = action_l2_error
                        
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        sampling_time = time.time() - sampling_start
                        
                        # Single batch validation metrics
                        latest_val_avg_loss = f"{loss_val.item():.4f}"

                        total_val_time = time.time() - val_start_time
                        
                        # Log all metrics to WandB
                        wandb.log({
                            # Flow matching loss (on normalized data)
                            "val/flow_matching_loss": loss_val.item(),
                            
                            # PRIMARY: Action prediction errors (NORMALIZED - percentage-like)
                            # Error of 0.1 ≈ 10% relative error since data is ~N(0,1)
                            "val/action_l1_error": action_l1_error.item(),
                            "val/action_l2_error": action_l2_error.item(),
                            "val/action_l1_first_step": action_l1_first,
                            "val/action_l1_last_step": action_l1_last,
                            
                            # SECONDARY: Physical units (for real-world interpretation)
                            "val/action_l1_error_physical": action_l1_error_physical.item(),
                            "val/action_l2_error_physical": action_l2_error_physical.item(),
                            
                            # Per-dimension errors (normalized, for debugging - first 5 dims)
                            **{f"val/action_l1_dim_{i}": action_l1_per_dim[i].item() 
                               for i in range(min(5, len(action_l1_per_dim)))},
                            
                            # Timing
                            "val_timing/batch_load_ms": batch_load_time * 1000,
                            "val_timing/forward_ms": forward_time * 1000,
                            "val_timing/sampling_ms": sampling_time * 1000,
                            "val_timing/total_val_ms": total_val_time * 1000,
                            "step": step
                        })
                        
                        overall_pbar.write(
                            f"Validation at step {step}: "
                            f"Loss={loss_val.item():.4f}, "
                            f"L1={action_l1_error.item():.4f} (~{action_l1_error.item()*100:.1f}%), "
                            f"L2={action_l2_error.item():.4f}, "
                            f"L1_phys={action_l1_error_physical.item():.4f}, "
                            f"Time={total_val_time*1000:.1f}ms"
                        )
                        
                    except StopIteration:
                        # Reset validation iterator when dataset is exhausted
                        val_iter = iter(val_dataloader)
                        overall_pbar.write(f"Validation dataset exhausted at step {step}, resetting iterator")
                    except Exception as e:
                        overall_pbar.write(f"Validation failed at step {step}: {e}")
                        wandb.log({"val/error": str(e), "step": step})

        # Save model checkpoint (only on rank 0)
        if (step % CHECKPOINT_INTERVAL == 0 or step == MAX_STEPS) and rank == 0:
            checkpoint_name = f"innate_policy_step_{step}.pth"
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
            
            # Save state dict, handling torch.compile if needed
            state_dict = policy.module.state_dict()
            if USE_COMPILE:
                # Remove "_orig_mod." prefix added by torch.compile
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            
            torch.save(state_dict, checkpoint_path)
            overall_pbar.write(f"Saved checkpoint: {checkpoint_path}")

    # Cleanup
    if rank == 0:
        overall_pbar.close()
        print("Training finished.")
        
        # Save final model
        final_checkpoint_path = os.path.join(CHECKPOINT_DIR, "innate_policy_final.pth")
        state_dict = policy.module.state_dict()
        if USE_COMPILE:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        torch.save(state_dict, final_checkpoint_path)
        print(f"✅ Final model saved to: {final_checkpoint_path}")
        
        wandb.finish()
    
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='Distributed InnatePolicy Training')
    
    # Hardware & Distributed
    parser.add_argument('--world_size', type=int, default=None, 
                        help='Number of GPUs to use (default: all available)')
    
    # Data
    parser.add_argument('--data_dir', type=str, 
                        default="/home/vignesh/raid/PaperMulti_1_2_Filtered",
                        help='Path to the HDF5 dataset directory')
    parser.add_argument('--chunk_size', type=int, default=16,
                        help='Action sequence length / chunk size')
    parser.add_argument('--batch_size', type=int, default=96,
                        help='Batch size per GPU (default: 96)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers per GPU (default: 4)')
    parser.add_argument('--shard_size', type=int, default=500,
                        help='Number of samples per WebDataset shard (default: 500)')
    parser.add_argument('--force-reconvert', action='store_true',
                        help='Force reconversion of HDF5 to WebDataset even if already exists')
    
    # Model Architecture
    parser.add_argument('--state_dim', type=int, default=6,
                        help='Dimension of robot state (default: 6)')
    parser.add_argument('--action_dim', type=int, default=10,
                        help='Dimension of action space (default: 10)')
    parser.add_argument('--num_queries', type=int, default=8,
                        help='Number of attention pooling queries (default: 8)')
    parser.add_argument('--freeze-vision-backbone', type=lambda x: x.lower() == 'true', default=True,
                        help='Freeze DINOv2 backbone weights: true or false (default: true)')
    parser.add_argument('--proprio_hidden_dim', type=int, default=256,
                        help='Hidden dimension for proprioception encoder (default: 256)')
    parser.add_argument('--diffusion_step_embed_dim', type=int, default=256,
                        help='Diffusion timestep embedding dimension (default: 256)')
    parser.add_argument('--down_dims', type=int, nargs='+', default=[256, 512, 1024],
                        help='UNet channel dimensions (default: 256 512 1024)')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='Conv kernel size (default: 5)')
    parser.add_argument('--n_groups', type=int, default=8,
                        help='Number of groups for GroupNorm (default: 8)')
    parser.add_argument('--num_inference_steps', type=int, default=10,
                        help='Number of flow matching sampling steps (default: 10)')
    
    # Training
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Maximum number of training steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Main learning rate (default: 1e-4)')
    parser.add_argument('--learning_rate_backbone', type=float, default=1e-5,
                        help='Learning rate for vision backbone (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Number of warmup steps for learning rate scheduler (default: 5%% of max_steps)')
    
    # Optimizations
    parser.add_argument('--use-bf16', type=lambda x: x.lower() == 'true', default=True,
                        help='Use BF16 mixed precision: true or false (default: true)')
    parser.add_argument('--use-compile', type=lambda x: x.lower() == 'true', default=False,
                        help='Use torch.compile() optimization: true or false (default: false)')
    parser.add_argument('--normalize-data', type=lambda x: x.lower() == 'true', default=True,
                        help='Normalize actions and robot state: true or false (default: true)')
    
    # W&B Configuration
    parser.add_argument('--wandb_project', type=str, default='innate-policy',
                        help='W&B project name (default: innate-policy)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity/username (default: None)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name (default: auto-generated from task and timestamp)')
    parser.add_argument('--wandb_api_key', type=str, 
                        default='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',
                        help='W&B API key (default: hardcoded key)')
    
    args = parser.parse_args()
    
    # Get number of available GPUs
    if args.world_size is None:
        world_size = torch.cuda.device_count()
    else:
        world_size = args.world_size
    
    if world_size == 0:
        print("No CUDA devices available. Exiting.")
        return
    
    # Always convert data BEFORE starting distributed training
    force_reconvert = getattr(args, 'force_reconvert', False)
    conversion_success, webd_dir = convert_data_always(
        args.data_dir, 
        shard_size=args.shard_size,
        force_reconvert=force_reconvert
    )
    
    if not conversion_success:
        print("❌ Failed to convert data. Exiting...")
        return
    
    print(f"\n{'='*80}")
    print(f"STARTING DISTRIBUTED INNATE POLICY TRAINING")
    print(f"{'='*80}")
    print(f"Hardware:")
    print(f"  GPUs: {world_size}")
    print(f"  Precision: {'BF16 (mixed)' if args.use_bf16 else 'FP32'}")
    print(f"  torch.compile: {'Enabled' if args.use_compile else 'Disabled'}")
    print(f"\nData:")
    print(f"  Directory: {args.data_dir}")
    print(f"  WebDataset: {webd_dir}")
    print(f"  Shard size: {args.shard_size} samples")
    print(f"  Batch size per GPU: {args.batch_size}")
    print(f"  Total effective batch: {args.batch_size * world_size}")
    print(f"  Workers per GPU: {args.num_workers}")
    print(f"  Normalization: {'✅ ENABLED (actions & state normalized to ~N(0,1))' if args.normalize_data else '⚠️  DISABLED (raw values)'}")
    print(f"\nModel:")
    print(f"  Architecture: InnatePolicy (DINOv2 + Flow Matching)")
    print(f"  State dim: {args.state_dim}")
    print(f"  Action dim: {args.action_dim}")
    print(f"  Action horizon: {args.chunk_size}")
    print(f"  Num queries: {args.num_queries}")
    print(f"  Freeze vision: {args.freeze_vision_backbone}")
    print(f"  UNet dims: {args.down_dims}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"\nTraining:")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LR backbone: {args.learning_rate_backbone}")
    print(f"  Weight decay: {args.weight_decay}")
    warmup_display = args.warmup_steps if args.warmup_steps is not None else f"{int(0.05 * args.max_steps)} (5% of steps)"
    print(f"  Warmup steps: {warmup_display}")
    print(f"  LR schedule: Linear warmup (5%) + Cosine decay to 10% of max LR")
    print(f"{'='*80}\n")
    
    # Spawn processes for distributed training, passing the WebDataset directory
    mp.spawn(train_ddp, args=(world_size, args, webd_dir), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
