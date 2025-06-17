import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import os # For WANDB_API_KEY and os.path, os.makedirs
from datetime import datetime
import socket
import time  # Add for profiling

from ACT import ACTConfig, ACTPolicy # Assuming ACT.py is in the same directory or PYTHONPATH
from data_utils import initialize_webdataset_data # Changed from initialize_data to initialize_webdataset_data

# --- Configuration ---
# Data parameters
DATA_DIR = "/home/vignesh/raid/DropSocks_1_2_webd/" # Changed to WebDataset directory
CHUNK_SIZE = 30
TRAIN_VAL_SPLIT = 0.9
BATCH_SIZE = 96 # Adjust based on your GPU memory
NUM_WORKERS = 4 # Increased for WebDataset efficiency
USE_IMG_AUG_TRAIN = False # Example, set as needed
USE_IMG_AUG_VAL = False

# Task name and automatic checkpoint directory generation
TASK_NAME = os.path.basename(DATA_DIR.rstrip('/'))  # Extract directory name from DATA_DIR
# Generate timestamp for consistent naming with wandb
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"{TASK_NAME}_{TIMESTAMP}"
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints", RUN_NAME)

# ACT Policy parameters (Example - these should match your dataset and desired model complexity)
# These are just placeholders and should be configured based on data_utils.py constants
# and your specific task requirements.
IMAGE_H = 480 # From data_utils.py
IMAGE_W = 640 # From data_utils.py
IMAGE_C = 3   # From data_utils.py
QPOS_DIM = 6  # From data_utils.py (State dimension)
ACTION_DIM = 8 # From data_utils.py

# Define input_shapes based on your data_utils.py and dataset structure
# This is a critical part and needs to be accurate.
INPUT_SHAPES = {
    "observation.image_camera_1": [IMAGE_C, IMAGE_H, IMAGE_W],
    "observation.image_camera_2": [IMAGE_C, IMAGE_H, IMAGE_W],
    "observation.state": [QPOS_DIM]
}
OUTPUT_SHAPES = {
    "action": [ACTION_DIM]
}

# Model Hyperparameters
N_OBS_STEPS = 1 # Number of observation steps to use from the chunk
N_ACTION_STEPS = CHUNK_SIZE # Number of action steps to predict (typically same as chunk_size for ACT)
DIM_MODEL = 512
N_HEADS = 8
N_ENCODER_LAYERS = 4
N_DECODER_LAYERS = 4 # Often larger for ACT
KL_WEIGHT = 10.0 # If using VAE
USE_VAE = True # Or False, depending on your choice

# Training parameters
MAX_STEPS = 60000  # Changed from NUM_EPOCHS to MAX_STEPS
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
LEARNING_RATE_BACKBONE = 1e-5

# Calculate checkpoint interval for exactly 10 checkpoints
CHECKPOINT_INTERVAL = MAX_STEPS // 10

# W&B Configuration
WANDB_PROJECT = "wandb_test"  # Changed from "act-simple"
WANDB_ENTITY = None # Replace with your W&B username or team name if desired

# Your W&B API Key
WANDB_API_KEY = "f25e8c35a0cd601c2cafcdbfd698ce8cfba25a9c"

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")

    # Initialize W&B
    # Try to use environment variable first, then fallback to the hardcoded key
    if not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY not found in environment. Using key from script.")
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    
    if not os.getenv("WANDB_API_KEY"): # Double check if it's set now
        print("Warning: WANDB_API_KEY is still not set. WandB logging will likely fail.")
        print("Please ensure the key is valid or set it in your environment.")

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=RUN_NAME, config={
        "data_dir": DATA_DIR,
        "task_name": TASK_NAME,
        "timestamp": TIMESTAMP,
        "run_name": RUN_NAME,
        "hostname": socket.gethostname(),
        "checkpoint_dir": CHECKPOINT_DIR,
        "chunk_size": CHUNK_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "lr_backbone": LEARNING_RATE_BACKBONE,
        "max_steps": MAX_STEPS,  # Changed from num_epochs
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
    print("Initializing WebDataset data loaders...")
    try:
        train_dataloader, val_dataloader, dataset_stats = initialize_webdataset_data(
            data_dir=DATA_DIR,
            chunk_size=CHUNK_SIZE,
            batch_size=BATCH_SIZE,
            train_val_split=TRAIN_VAL_SPLIT,
            num_workers=NUM_WORKERS,
            prefetch_factor=2,  # WebDataset-specific parameter
            seed=42 # for reproducibility
        )
    except FileNotFoundError as e:
        print(f"Error initializing WebDataset: {e}")
        print(f"Please ensure your data directory '{DATA_DIR}' contains WebDataset .tar files.")
        wandb.finish(exit_code=1)
        return
    except ValueError as e:
        print(f"Error initializing WebDataset: {e}")
        wandb.finish(exit_code=1)
        return

    # Save dataset_stats
    if dataset_stats:
        # Note: CHECKPOINT_DIR is already created above
        stats_path = os.path.join(CHECKPOINT_DIR, "dataset_stats.pt")
        try:
            torch.save(dataset_stats, stats_path)
            print(f"Saved dataset_stats to {stats_path}")
            # Log dataset_stats path to wandb for easy access
            wandb.config.update({"dataset_stats_path": stats_path})
        except Exception as e:
            print(f"Error saving dataset_stats: {e}")
            # Decide if this is a critical error. For now, just print and continue.

    # Note: WebDataset dataloaders don't have len(), so we'll skip length printing
    print("WebDataset dataloaders initialized (streaming datasets - no fixed length)")

    # --- 2. Initialize ACT Policy and Optimizer ---
    print("Initializing ACT Policy...")
    act_config = ACTConfig(
        n_obs_steps=N_OBS_STEPS,
        chunk_size=CHUNK_SIZE, # This is the context length for the transformer
        n_action_steps=N_ACTION_STEPS, # This is the prediction horizon
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
        # vision_backbone can be left as default "resnet18" or specified
        # pretrained_backbone_weights default is "ResNet18_Weights.IMAGENET1K_V1"
    )

    policy = ACTPolicy(config=act_config, dataset_stats=dataset_stats).to(device)
    optimizer = optim.AdamW(policy.get_optim_params(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    wandb.watch(policy, log="all", log_freq=100) # Log gradients and model parameters

    # --- 3. Step-based Training Loop ---
    print("Starting step-based training...")
    step = 0
    
    # Initialize tqdm for the entire training process (steps)
    overall_pbar = tqdm(total=MAX_STEPS, unit="step", desc="Training Progress")

    # Initialize placeholders for validation metrics
    latest_val_avg_loss = "N/A"
    latest_val_avg_l1_loss = "N/A"
    latest_val_avg_kld_loss = "N/A"

    # Create infinite iterator from dataloader
    train_iter = iter(train_dataloader)
    
    # Create persistent validation iterator
    val_iter = iter(val_dataloader)
    
    # Profiling variables
    batch_load_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    total_step_times = []
    
    while step < MAX_STEPS:
        step_start_time = time.time()
        policy.train()
        
        # 1. Batch Loading
        batch_load_start = time.time()
        try:
            batch = next(train_iter)
        except StopIteration:
            # Reset iterator when dataset is exhausted (start new "epoch")
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
            overall_pbar.write(f"Completed one pass through dataset at step {step}")
        
        # Move batch to device
        batch_device = {}
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                batch_device[key] = tensor.to(device)
            else:
                batch_device[key] = tensor
        
        if device.type == 'cuda':
            torch.cuda.synchronize()  # Ensure GPU operations are complete
        batch_load_time = time.time() - batch_load_start
        
        # 2. Forward Pass
        forward_start = time.time()
        loss, loss_dict = policy(batch_device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        forward_time = time.time() - forward_start
        
        # 3. Backward Pass
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        backward_time = time.time() - backward_start
        
        # 4. Optimizer Update
        optimizer_start = time.time()
        optimizer.step()
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

        # Log metrics to WandB (including timing)
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

        # --- Validation Step ---
        if step % 100 == 0:  # Validate every 100 steps
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
                    
                    # Time forward pass
                    forward_start = time.time()
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

        # Save model checkpoint
        if step % CHECKPOINT_INTERVAL == 0 or step == MAX_STEPS:
            checkpoint_name = f"act_policy_step_{step}.pth"
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
            torch.save(policy.state_dict(), checkpoint_path)
            overall_pbar.write(f"Saved checkpoint: {checkpoint_path}")

    overall_pbar.close() 
    print("Training finished.")
    wandb.finish()

if __name__ == "__main__":
    # It's generally better to set WANDB_API_KEY as an environment variable
    # For demonstration, you could uncomment and set it here if needed, but it's not recommended for VCS.
    # os.environ["WANDB_API_KEY"] = "YOUR_API_KEY_HERE" 
    main()
