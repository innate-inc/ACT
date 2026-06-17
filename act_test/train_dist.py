"""Distributed (multi-GPU) training script for the ACT policy.

This script orchestrates the full training pipeline:

1. **Data conversion** – converts episodes to WebDataset tar shards
   (runs once on the main process before spawning workers).
2. **DDP setup** – spawns one process per GPU via ``torch.multiprocessing.spawn``
   and initializes NCCL-backed ``DistributedDataParallel``.
3. **Training loop** – step-based loop with WebDataset streaming, optional BF16
   mixed precision (``torch.amp.autocast``), optional ``torch.compile()``,
   linear warmup + cosine annealing LR schedule, periodic validation,
   checkpointing, and Weights & Biases logging.
4. **ONNX export** – exports the trained model to ONNX at the end of training
   (rank 0 only).

Usage::

    python -m act_test.train_dist \\
        --data_dir /path/to/data \\
        --world_size 4 \\
        --max_steps 120000 \\
        --batch_size 96
"""

import sys
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

from urllib.parse import urlparse
from ACT import ACTConfig, ACTPolicy
from data_utils import initialize_webdataset_data
from data_tools.webdataset import convert_to_webdataset

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the GPU for the current process
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def convert_data_always(data_dir, shard_size=500, force_reconvert=True):
    """Always convert episode data to WebDataset format (before distributed training)."""
    DATA_DIR = data_dir
    WEBD_DIR = os.path.join(DATA_DIR, "webdataset")
    
    print("🔄 CONVERTING EPISODES TO WEBDATASET FORMAT")
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
    
    print(f"🔄 Converting episode data to WebDataset format...")
    print(f"📁 Source directory: {DATA_DIR}")
    print(f"📁 WebDataset target: {WEBD_DIR}")
    print(f"📦 Shard size: {shard_size}")
    print(f"🖼️  Image format: uint8 PyTorch tensors, 224x224")
    
    # Perform conversion with optimized settings
    success = convert_to_webdataset(
        data_directory=DATA_DIR,
        webd_directory=WEBD_DIR,
        shard_size=shard_size,
        target_size=(224, 224)
    )
    
    if success:
        print("✅ Data conversion completed successfully!")
        return True, WEBD_DIR
    else:
        print("❌ Data conversion failed!")
        return False, None

def train_ddp(rank, world_size, args, webd_dir):
    """Main distributed training function."""
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

    # Task name and automatic checkpoint directory generation
    TASK_NAME = os.path.basename(DATA_DIR.rstrip('/'))
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = f"{TASK_NAME}_{TIMESTAMP}_ddp"

    # Override RUN_NAME with user_id/job_id from JOB_SPEC_URL if available
    JOB_SPEC_URL = os.getenv("JOB_SPEC_URL")
    if JOB_SPEC_URL:
        try:
            path_parts = urlparse(JOB_SPEC_URL).path.strip('/').split('/')
            # URL format: /managed-innate-training/<user_id>/<job_id>/...
            if len(path_parts) >= 3:
                RUN_NAME = f"{path_parts[1]}/{path_parts[2]}"
        except (ValueError, IndexError) as e:
            print(f"Warning: failed to parse JOB_SPEC_URL: {e}")
    CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints", RUN_NAME)

    # Use the WebDataset directory passed from main
    WEBD_DIR = webd_dir

    # ACT Policy parameters
    IMAGE_H = 224  # Updated from 480 to 224 for faster training
    IMAGE_W = 224  # Updated from 640 to 224 for faster training
    IMAGE_C = 3
    QPOS_DIM = 6
    ACTION_DIM = 10

    INPUT_SHAPES = {
        "observation.image_camera_1": [IMAGE_C, IMAGE_H, IMAGE_W],
        "observation.image_camera_2": [IMAGE_C, IMAGE_H, IMAGE_W],
        "observation.state": [QPOS_DIM]
    }
    OUTPUT_SHAPES = {
        "action": [ACTION_DIM]
    }

    # Model Hyperparameters
    N_OBS_STEPS = 1
    N_ACTION_STEPS = CHUNK_SIZE
    DIM_MODEL = 512
    N_HEADS = 8
    N_ENCODER_LAYERS = 4
    N_DECODER_LAYERS = 4
    KL_WEIGHT = 10.0
    USE_VAE = True

    # Training parameters
    MAX_STEPS = args.max_steps
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = 5e-4
    LEARNING_RATE_BACKBONE = args.learning_rate_backbone
    # Default warmup to 5% of total steps if not specified
    WARMUP_STEPS = args.warmup_steps if args.warmup_steps is not None else int(0.05 * MAX_STEPS)
    MIN_LR_RATIO = 0.1  # Decay to 1/10th of original LR
    
    # Calculate checkpoint interval for exactly 10 checkpoints (minimum 1 to avoid division by zero)
    CHECKPOINT_INTERVAL = max(1, MAX_STEPS // 10)

    # W&B Configuration
    WANDB_PROJECT = "act-simple"
    WANDB_ENTITY = None
    WANDB_API_KEY = "f25e8c35a0cd601c2cafcdbfd698ce8cfba25a9c"

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
        if not os.getenv("WANDB_API_KEY"):
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
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "lr_backbone": LEARNING_RATE_BACKBONE,
            "max_steps": MAX_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "min_lr_ratio": MIN_LR_RATIO,
            
            # Model Architecture
            "dim_model": DIM_MODEL,
            "n_heads": N_HEADS,
            "n_encoder_layers": N_ENCODER_LAYERS,
            "n_decoder_layers": N_DECODER_LAYERS,
            "kl_weight": KL_WEIGHT if USE_VAE else 0,
            "use_vae": USE_VAE,
            "n_obs_steps": N_OBS_STEPS,
            "n_action_steps": N_ACTION_STEPS,
            
            # Input/Output Shapes
            "image_h": IMAGE_H,
            "image_w": IMAGE_W,
            "qpos_dim": QPOS_DIM,
            "action_dim": ACTION_DIM,
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
            seed=42 + rank
        )
    except (FileNotFoundError, ValueError) as e:
        if rank == 0:
            print(f"❌ Error initializing WebDataset: {e}")
            print(f"Please ensure your data directory '{WEBD_DIR}' contains WebDataset .tar files.")
            print(f"Note: You need at least as many shards as (num_workers * world_size)")
            wandb.finish(exit_code=1)
        cleanup()
        sys.exit(1)  # Exit with error code so the startup script detects the failure

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

    # --- 2. Initialize ACT Policy and Optimizer ---
    if rank == 0:
        print("Initializing ACT Policy...")
    
    act_config = ACTConfig(
        n_obs_steps=N_OBS_STEPS,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        input_shapes=INPUT_SHAPES,
        output_shapes=OUTPUT_SHAPES,
        dim_model=DIM_MODEL,
        n_heads=N_HEADS,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,
        kl_weight=KL_WEIGHT,
        use_vae=USE_VAE,
        optimizer_lr=LEARNING_RATE,
        optimizer_weight_decay=WEIGHT_DECAY,
        optimizer_lr_backbone=LEARNING_RATE_BACKBONE,
    )

    policy = ACTPolicy(config=act_config, dataset_stats=dataset_stats).to(device)
    
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
    
    optimizer = optim.AdamW(policy.module.get_optim_params(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
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
        warmup_percentage = (WARMUP_STEPS / max(1, MAX_STEPS) * 100) if MAX_STEPS > 0 else 0.0
        print(f"Learning rate scheduler: Linear warmup ({WARMUP_STEPS} steps, {warmup_percentage:.1f}%) + Cosine annealing to {MIN_LR_RATIO*100:.0f}% of max LR")

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
        
        # Move batch to device
        batch_device = {}
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                batch_device[key] = tensor.to(device)
            else:
                batch_device[key] = tensor
        
        if rank == 0:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_load_time = time.time() - batch_load_start
        
        # 2. Forward Pass (with BF16 if enabled)
        if rank == 0:
            forward_start = time.time()
        
        if USE_BF16:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, loss_dict = policy(batch_device)
        else:
            loss, loss_dict = policy(batch_device)
        
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
                "train/total_loss": loss.item(),
                "train/l1_loss": loss_dict.get("l1_loss", 0),
                "train/kld_loss": loss_dict.get("kld_loss", 0),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/lr_backbone": optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr'],
                "timing/batch_load_ms": batch_load_time * 1000,
                "timing/forward_ms": forward_time * 1000,
                "timing/backward_ms": backward_time * 1000,
                "timing/optimizer_ms": optimizer_time * 1000,
                "timing/total_step_ms": total_step_time * 1000,
                "step": step
            })

            # Progress marker for the cloud webapp's step bar/ETA. Quote-free
            # so it survives JSON-escaping in process_output.jsonl.
            if step % 25 == 0 or step >= MAX_STEPS - 1:
                print(f'@@INNATE_PROGRESS step={step} total={MAX_STEPS}', flush=True)

        step += 1
        
        if rank == 0:
            overall_pbar.update(1)
            
            # Update progress bar postfix with timing info
            overall_pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                l1=f"{loss_dict.get('l1_loss', 0):.4f}",
                kld=f"{loss_dict.get('kld_loss', 0):.4f}",
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
                        # Time batch loading (no iterator creation needed)
                        batch_load_start = time.time()
                        batch_val = next(val_iter)
                        batch_device_val = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_val.items()}
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        batch_load_time = time.time() - batch_load_start
                        
                        # Time forward pass (with BF16 if enabled)
                        forward_start = time.time()
                        if USE_BF16:
                            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                                loss, loss_dict = policy(batch_device_val)
                        else:
                            loss, loss_dict = policy(batch_device_val)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        forward_time = time.time() - forward_start
                        
                        # Single batch validation metrics
                        latest_val_avg_loss = f"{loss.item():.4f}"
                        latest_val_avg_l1_loss = f"{loss_dict.get('l1_loss', 0):.4f}"
                        latest_val_avg_kld_loss = f"{loss_dict.get('kld_loss', 0):.4f}"

                        total_val_time = time.time() - val_start_time
                        
                        wandb.log({
                            "val/avg_total_loss": loss.item(),
                            "val/avg_l1_loss": loss_dict.get("l1_loss", 0),
                            "val/avg_kld_loss": loss_dict.get("kld_loss", 0),
                            "val_timing/batch_load_ms": batch_load_time * 1000,
                            "val_timing/forward_ms": forward_time * 1000,
                            "val_timing/total_val_ms": total_val_time * 1000,
                            "step": step
                        })
                        
                        overall_pbar.write(f"Validation at step {step}: Loss={loss.item():.4f}, "
                                         f"Total={total_val_time*1000:.1f}ms (load={batch_load_time*1000:.1f}ms, forward={forward_time*1000:.1f}ms)")
                        
                    except StopIteration:
                        # Reset validation iterator when dataset is exhausted
                        val_iter = iter(val_dataloader)
                        overall_pbar.write(f"Validation dataset exhausted at step {step}, resetting iterator")
                    except Exception as e:
                        overall_pbar.write(f"Validation failed at step {step}: {e}")
                        wandb.log({"val/error": str(e), "step": step})

        # Save model checkpoint (only on rank 0)
        if (step % CHECKPOINT_INTERVAL == 0 or step == MAX_STEPS) and rank == 0:
            checkpoint_name = f"act_policy_step_{step}.pth"
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
            torch.save(policy.module.state_dict(), checkpoint_path)
            overall_pbar.write(f"Saved checkpoint: {checkpoint_path}")

    # Cleanup
    if rank == 0:
        overall_pbar.close()
        print("Training finished.")
        
        # Export to ONNX (only on rank 0)
        print("Exporting model to ONNX format...")
        try:
            # Create a wrapper for inference-only export (no loss computation)
            class ACTPolicyInferenceWrapper(torch.nn.Module):
                def __init__(self, policy):
                    super().__init__()
                    self.policy = policy
                
                def forward(self, img_cam1, img_cam2, robot_state):
                    """Forward pass for inference - returns predicted actions only."""
                    batch = {
                        "observation.image_camera_1": img_cam1,
                        "observation.image_camera_2": img_cam2,
                        "observation.state": robot_state,
                    }
                    # Normalize inputs
                    batch = self.policy.normalize_inputs(batch)
                    # Prepare batch for model (adds latent if VAE is used)
                    model_batch = self.policy._prepare_batch_for_model(batch)
                    # Get predicted actions from model (returns actions and VAE params)
                    actions_normalized, _ = self.policy.model(model_batch)
                    # Unnormalize actions to get real-world values
                    actions = self.policy.unnormalize_outputs({"action": actions_normalized})["action"]
                    return actions
            
            # Create a fresh uncompiled model for ONNX export
            # (torch.compile models cannot be exported to ONNX)
            print("  Creating fresh model without torch.compile()...")
            onnx_policy = ACTPolicy(config=act_config, dataset_stats=dataset_stats).to(device)
            
            # Load the trained weights from the DDP-wrapped compiled model
            # If torch.compile was used, strip the "_orig_mod." prefix from state dict keys
            state_dict = policy.module.state_dict()
            if USE_COMPILE:
                # Remove "_orig_mod." prefix added by torch.compile
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            
            onnx_policy.load_state_dict(state_dict)
            onnx_policy.eval()
            
            # Wrap for inference
            inference_wrapper = ACTPolicyInferenceWrapper(onnx_policy).to(device)
            inference_wrapper.eval()
            
            # Create dummy inputs (just observations, no actions needed for inference)
            dummy_img_cam1 = torch.randn(1, IMAGE_C, IMAGE_H, IMAGE_W, device=device)
            dummy_img_cam2 = torch.randn(1, IMAGE_C, IMAGE_H, IMAGE_W, device=device)
            dummy_robot_state = torch.randn(1, QPOS_DIM, device=device)
            
            # Define ONNX export path
            onnx_path = os.path.join(CHECKPOINT_DIR, "act_policy_final.onnx")
            
            print(f"  Exporting to {onnx_path}...")
            # Export to ONNX for inference
            torch.onnx.export(
                inference_wrapper,
                (dummy_img_cam1, dummy_img_cam2, dummy_robot_state),
                onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['image_camera_1', 'image_camera_2', 'robot_state'],
                output_names=['predicted_actions'],
                dynamic_axes={
                    'image_camera_1': {0: 'batch_size'},
                    'image_camera_2': {0: 'batch_size'},
                    'robot_state': {0: 'batch_size'},
                    'predicted_actions': {0: 'batch_size'}
                }
            )
            
            print(f"✅ ONNX model saved to: {onnx_path}")
            print(f"   Model inputs: image_camera_1, image_camera_2, robot_state")
            print(f"   Model output: predicted_actions (shape: [batch, {CHUNK_SIZE}, {ACTION_DIM}])")
            wandb.save(onnx_path)
            
            # Clean up the temporary models
            del onnx_policy, inference_wrapper
            
        except Exception as e:
            print(f"❌ Failed to export ONNX model: {e}")
            import traceback
            traceback.print_exc()
        
        wandb.finish()
    
    cleanup()

def main():
    parser = argparse.ArgumentParser(description='Distributed ACT Training with Optimizations')
    
    # Hardware & Distributed
    parser.add_argument('--world_size', type=int, default=None, 
                        help='Number of GPUs to use (default: all available)')
    
    # Data
    parser.add_argument('--data_dir', type=str, 
                        default="/home/vignesh/raid/PaperMulti_1_2_Filtered",
                        help='Path to the episode dataset directory')
    parser.add_argument('--chunk_size', type=int, default=30,
                        help='Action sequence length / chunk size')
    parser.add_argument('--batch_size', type=int, default=96,
                        help='Batch size per GPU (default: 96)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers per GPU (default: 4)')
    parser.add_argument('--shard_size', type=int, default=500,
                        help='Number of samples per WebDataset shard (default: 500)')
    parser.add_argument('--force-reconvert', action='store_true',
                        help='Force reconversion of HDF5 to WebDataset even if already exists')
    
    # Training
    parser.add_argument('--max_steps', type=int, default=120000,
                        help='Maximum number of training steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Main learning rate')
    parser.add_argument('--learning_rate_backbone', type=float, default=5e-5,
                        help='Learning rate for vision backbone')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Number of warmup steps for learning rate scheduler (default: 5%% of max_steps)')
    
    # Optimizations
    parser.add_argument('--use-bf16', action='store_true', default=True,
                        help='Use BF16 mixed precision training (default: True)')
    parser.add_argument('--no-bf16', dest='use_bf16', action='store_false',
                        help='Disable BF16 mixed precision training')
    parser.add_argument('--use-compile', action='store_true', default=True,
                        help='Use torch.compile() for model optimization (default: True)')
    parser.add_argument('--no-compile', dest='use_compile', action='store_false',
                        help='Disable torch.compile() optimization')
    
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
        sys.exit(1)  # Exit with error code so orchestrator detects failure
    
    print(f"\n{'='*80}")
    print(f"STARTING DISTRIBUTED TRAINING")
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
    print(f"\nTraining:")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LR backbone: {args.learning_rate_backbone}")
    warmup_display = args.warmup_steps if args.warmup_steps is not None else f"{int(0.05 * args.max_steps)} (5% of steps)"
    print(f"  Warmup steps: {warmup_display}")
    print(f"  LR schedule: Linear warmup (5%) + Cosine decay to 10% of max LR")
    print(f"{'='*80}\n")
    
    # Spawn processes for distributed training, passing the WebDataset directory
    mp.spawn(train_ddp, args=(world_size, args, webd_dir), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
