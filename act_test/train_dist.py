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

from ACT import ACTConfig, ACTPolicy
from data_utils import initialize_webdataset_data

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

def train_ddp(rank, world_size, args):
    """Main distributed training function."""
    setup(rank, world_size)
    
    # --- Configuration ---
    # Data parameters
    DATA_DIR = "/home/vignesh/raid/DropSocks_1_2_webd/"
    CHUNK_SIZE = 30
    TRAIN_VAL_SPLIT = 0.9
    BATCH_SIZE = 96  # Keep original batch size per GPU for 4x effective batch size
    NUM_WORKERS = 4
    USE_IMG_AUG_TRAIN = False
    USE_IMG_AUG_VAL = False

    # Task name and automatic checkpoint directory generation
    TASK_NAME = os.path.basename(DATA_DIR.rstrip('/'))
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = f"{TASK_NAME}_{TIMESTAMP}_ddp"
    CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints", RUN_NAME)

    # ACT Policy parameters
    IMAGE_H = 480
    IMAGE_W = 640
    IMAGE_C = 3
    QPOS_DIM = 6
    ACTION_DIM = 8

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
    MAX_STEPS = 12000  # 20000 / 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 5e-4
    LEARNING_RATE_BACKBONE = 1e-5
    
    # Calculate checkpoint interval for exactly 10 checkpoints
    CHECKPOINT_INTERVAL = MAX_STEPS // 10

    # W&B Configuration
    WANDB_PROJECT = "wandb_test_ddp"
    WANDB_ENTITY = None
    WANDB_API_KEY = "f25e8c35a0cd601c2cafcdbfd698ce8cfba25a9c"

    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    
    # Only print and create directories on rank 0
    if rank == 0:
        print(f"Starting distributed training on {world_size} GPUs")
        print(f"Using device: {device}")
        
        # Create checkpoint directory
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        print(f"Checkpoint directory: {CHECKPOINT_DIR}")

        # Initialize W&B only on rank 0
        if not os.getenv("WANDB_API_KEY"):
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=RUN_NAME, config={
            "data_dir": DATA_DIR,
            "task_name": TASK_NAME,
            "timestamp": TIMESTAMP,
            "run_name": RUN_NAME,
            "hostname": socket.gethostname(),
            "checkpoint_dir": CHECKPOINT_DIR,
            "chunk_size": CHUNK_SIZE,
            "batch_size_per_gpu": BATCH_SIZE,
            "total_effective_batch_size": BATCH_SIZE * world_size,
            "world_size": world_size,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "lr_backbone": LEARNING_RATE_BACKBONE,
            "max_steps": MAX_STEPS,
            "dim_model": DIM_MODEL,
            "n_heads": N_HEADS,
            "n_encoder_layers": N_ENCODER_LAYERS,
            "n_decoder_layers": N_DECODER_LAYERS,
            "kl_weight": KL_WEIGHT if USE_VAE else 0,
            "use_vae": USE_VAE,
            "n_obs_steps": N_OBS_STEPS,
            "n_action_steps": N_ACTION_STEPS,
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
            data_dir=DATA_DIR,
            chunk_size=CHUNK_SIZE,
            batch_size=BATCH_SIZE,
            train_val_split=TRAIN_VAL_SPLIT,
            use_img_aug_train=USE_IMG_AUG_TRAIN,
            use_img_aug_val=USE_IMG_AUG_VAL,
            num_workers=NUM_WORKERS,
            prefetch_factor=2,
            seed=42 + rank  # Different seed for each rank to ensure different data
        )
    except (FileNotFoundError, ValueError) as e:
        if rank == 0:
            print(f"Error initializing WebDataset: {e}")
            print(f"Please ensure your data directory '{DATA_DIR}' contains WebDataset .tar files.")
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
    
    # Wrap model with DDP
    policy = DDP(policy, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    optimizer = optim.AdamW(policy.module.get_optim_params(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

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
        
        # 2. Forward Pass
        if rank == 0:
            forward_start = time.time()
        
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
        if step % 10 == 0 and rank == 0:
            policy.eval()
            val_loss_sum = 0.0
            val_l1_loss_sum = 0.0
            val_kld_loss_sum = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                val_iter = iter(val_dataloader)
                # Run validation for a fixed number of batches
                for _ in range(50):
                    try:
                        batch_val = next(val_iter)
                        batch_device_val = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_val.items()}
                        loss, loss_dict = policy(batch_device_val)
                        val_loss_sum += loss.item()
                        val_l1_loss_sum += loss_dict.get("l1_loss", 0)
                        val_kld_loss_sum += loss_dict.get("kld_loss", 0)
                        val_batch_count += 1
                    except StopIteration:
                        break
            
            # Update validation metrics
            if val_batch_count > 0:
                latest_val_avg_loss = f"{(val_loss_sum / val_batch_count):.4f}"
                latest_val_avg_l1_loss = f"{(val_l1_loss_sum / val_batch_count):.4f}"
                latest_val_avg_kld_loss = f"{(val_kld_loss_sum / val_batch_count):.4f}"

                wandb.log({
                    "val/avg_total_loss": float(latest_val_avg_loss),
                    "val/avg_l1_loss": float(latest_val_avg_l1_loss),
                    "val/avg_kld_loss": float(latest_val_avg_kld_loss),
                    "step": step
                })

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
        wandb.finish()
    
    cleanup()

def main():
    parser = argparse.ArgumentParser(description='Distributed ACT Training')
    parser.add_argument('--world_size', type=int, default=None, 
                        help='Number of GPUs to use (default: all available)')
    args = parser.parse_args()
    
    # Get number of available GPUs
    if args.world_size is None:
        world_size = torch.cuda.device_count()
    else:
        world_size = args.world_size
    
    if world_size == 0:
        print("No CUDA devices available. Exiting.")
        return
    
    print(f"Starting distributed training on {world_size} GPUs")
    
    # Spawn processes for distributed training
    mp.spawn(train_ddp, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
