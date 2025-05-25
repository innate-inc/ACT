import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import os # For WANDB_API_KEY and os.path, os.makedirs

from ACT import ACTConfig, ACTPolicy # Assuming ACT.py is in the same directory or PYTHONPATH
from data_utils import initialize_data # Assuming data_utils.py is in the same directory or PYTHONPATH

# --- Configuration ---
# Data parameters
DATA_DIR = "/home/vignesh/raid/PaperBench" # Replace with your actual data directory
CHUNK_SIZE = 30
TRAIN_VAL_SPLIT = 0.9
BATCH_SIZE = 32 # Adjust based on your GPU memory
NUM_WORKERS = 0 # Adjust based on your system
USE_IMG_AUG_TRAIN = False # Example, set as needed
USE_IMG_AUG_VAL = False

# Checkpoint directory
CHECKPOINT_DIR = "/home/vignesh/raid/PaperBench/checkpoints/" # Specify your desired checkpoint directory

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
N_DECODER_LAYERS = 1 # Often larger for ACT
KL_WEIGHT = 10.0 # If using VAE
USE_VAE = True # Or False, depending on your choice

# Training parameters
NUM_EPOCHS = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
LEARNING_RATE_BACKBONE = 1e-5
#GRAD_CLIP_NORM = 1.0 # Optional gradient clipping

# W&B Configuration
WANDB_PROJECT = "act-simple"
WANDB_ENTITY = None # Replace with your W&B username or team name if desired

# Your W&B API Key
WANDB_API_KEY = "f25e8c35a0cd601c2cafcdbfd698ce8cfba25a9c"

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize W&B
    # Try to use environment variable first, then fallback to the hardcoded key
    if not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY not found in environment. Using key from script.")
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    
    if not os.getenv("WANDB_API_KEY"): # Double check if it's set now
        print("Warning: WANDB_API_KEY is still not set. WandB logging will likely fail.")
        print("Please ensure the key is valid or set it in your environment.")

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config={
        "data_dir": DATA_DIR,
        "checkpoint_dir": CHECKPOINT_DIR,
        "chunk_size": CHUNK_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "lr_backbone": LEARNING_RATE_BACKBONE,
        "num_epochs": NUM_EPOCHS,
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
    print("Initializing data loaders...")
    try:
        train_dataloader, val_dataloader, dataset_stats = initialize_data(
            data_dir=DATA_DIR,
            chunk_size=CHUNK_SIZE,
            train_val_split=TRAIN_VAL_SPLIT,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            use_img_aug_train=USE_IMG_AUG_TRAIN,
            use_img_aug_val=USE_IMG_AUG_VAL,
            seed=42 # for reproducibility
        )
    except FileNotFoundError as e:
        print(f"Error initializing data: {e}")
        print(f"Please ensure your data directory '{DATA_DIR}' is correctly set and contains 'metadata.json' and episode files.")
        wandb.finish(exit_code=1)
        return
    except ValueError as e:
        print(f"Error initializing data: {e}")
        wandb.finish(exit_code=1)
        return

    # Save dataset_stats
    if dataset_stats:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        stats_path = os.path.join(CHECKPOINT_DIR, "dataset_stats.pt")
        try:
            torch.save(dataset_stats, stats_path)
            print(f"Saved dataset_stats to {stats_path}")
            # Log dataset_stats path to wandb for easy access
            wandb.config.update({"dataset_stats_path": stats_path})
        except Exception as e:
            print(f"Error saving dataset_stats: {e}")
            # Decide if this is a critical error. For now, just print and continue.

    print(f"Number of training batches: {len(train_dataloader)}")
    if val_dataloader and len(val_dataloader) > 0 :
        print(f"Number of validation batches: {len(val_dataloader)}")
    else:
        print("No validation data or validation split resulted in 0 validation episodes.")

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

    # --- 3. Training Loop ---
    print("Starting training...")
    global_step = 0
    
    # Initialize tqdm for the entire training process (epochs)
    overall_pbar = tqdm(range(NUM_EPOCHS), unit="epoch", desc="Overall Training Progress")

    # Initialize placeholders for latest validation metrics
    latest_val_avg_loss = "N/A"
    latest_val_avg_l1_loss = "N/A"
    latest_val_avg_kld_loss = "N/A"

    for epoch in overall_pbar: # `epoch` is the current epoch index (0, 1, ...)
        policy.train()
        epoch_loss_sum = 0.0
        epoch_l1_loss_sum = 0.0
        epoch_kld_loss_sum = 0.0
        
        # Set the description for the current epoch on the overall progress bar
        # This description will no longer change for validation
        overall_pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(train_dataloader): # Iterate directly, no inner tqdm
            # Move batch to device
            batch_device = {}
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor):
                    batch_device[key] = tensor.to(device)
                else:
                    batch_device[key] = tensor # e.g. 'action_is_pad' could be bool list
            
            # Forward pass
            loss, loss_dict = policy(batch_device)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log metrics to WandB
            wandb.log({
                "train/total_loss": loss.item(),
                "train/l1_loss": loss_dict.get("l1_loss", 0),
                "train/kld_loss": loss_dict.get("kld_loss", 0),
                "epoch": epoch,
                "global_step": global_step
            })
            
            epoch_loss_sum += loss.item()
            epoch_l1_loss_sum += loss_dict.get("l1_loss", 0)
            epoch_kld_loss_sum += loss_dict.get("kld_loss", 0)
            
            global_step += 1
        
        # After all batches in the epoch, calculate averages
        avg_epoch_loss_str = "N/A"
        avg_l1_loss_str = "N/A"
        avg_kld_loss_str = "N/A"

        if len(train_dataloader) > 0:
            avg_epoch_loss = epoch_loss_sum / len(train_dataloader)
            avg_l1_loss = epoch_l1_loss_sum / len(train_dataloader)
            avg_kld_loss = epoch_kld_loss_sum / len(train_dataloader) if USE_VAE else 0.0
            
            avg_epoch_loss_str = f"{avg_epoch_loss:.4f}"
            avg_l1_loss_str = f"{avg_l1_loss:.4f}"
            avg_kld_loss_str = f"{avg_kld_loss:.4f}"
            
            # Log epoch averages to WandB
            wandb.log({
                "train_epoch/avg_total_loss": avg_epoch_loss,
                "train_epoch/avg_l1_loss": avg_l1_loss,
                "train_epoch/avg_kld_loss": avg_kld_loss,
                "epoch": epoch
            })
        elif epoch == 0: 
            overall_pbar.write("Warning: train_dataloader is empty. Cannot compute epoch averages.")

        # --- Optional: Validation Step ---
        if val_dataloader and len(val_dataloader) > 0 and (epoch + 1) % 5 == 0: 
            # original_desc = overall_pbar.desc # No longer needed
            # overall_pbar.set_description(f"Validating Epoch {epoch+1}/{NUM_EPOCHS}") # No longer changing description
            policy.eval()
            val_loss_sum = 0.0
            val_l1_loss_sum = 0.0
            val_kld_loss_sum = 0.0
            with torch.no_grad():
                for batch_val in val_dataloader: 
                    batch_device_val = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_val.items()}
                    loss, loss_dict = policy(batch_device_val)
                    val_loss_sum += loss.item()
                    val_l1_loss_sum += loss_dict.get("l1_loss", 0)
                    val_kld_loss_sum += loss_dict.get("kld_loss", 0)
            
            # Update latest validation metrics
            latest_val_avg_loss = f"{(val_loss_sum / len(val_dataloader)):.4f}"
            latest_val_avg_l1_loss = f"{(val_l1_loss_sum / len(val_dataloader)):.4f}"
            latest_val_avg_kld_loss = f"{(val_kld_loss_sum / len(val_dataloader) if USE_VAE else 0):.4f}"

            wandb.log({
                "val/avg_total_loss": float(latest_val_avg_loss) if latest_val_avg_loss != "N/A" else 0,
                "val/avg_l1_loss": float(latest_val_avg_l1_loss) if latest_val_avg_l1_loss != "N/A" else 0,
                "val/avg_kld_loss": float(latest_val_avg_kld_loss) if latest_val_avg_kld_loss != "N/A" else 0,
                "epoch": epoch
            })
            # overall_pbar.set_description(original_desc) # No longer needed
            
        # Update overall_pbar postfix with epoch averages AND latest validation metrics
        overall_pbar.set_postfix(
            train_loss=avg_epoch_loss_str,
            train_l1=avg_l1_loss_str,
            train_kld=avg_kld_loss_str,
            val_loss=latest_val_avg_loss,
            val_l1=latest_val_avg_l1_loss,
            val_kld=latest_val_avg_kld_loss,
            refresh=True 
        )

        # Save model checkpoint
        if (epoch + 1) % 10000 == 0 or (epoch + 1) == NUM_EPOCHS:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True) 
            checkpoint_name = f"act_policy_epoch_{epoch+1}.pth"
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
