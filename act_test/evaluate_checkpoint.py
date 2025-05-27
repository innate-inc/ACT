#!/usr/bin/env python
import torch
import os
import argparse

# Assuming ACT.py is in the same directory or PYTHONPATH
# ACTConfig uses FeatureType and NormalizationMode,
# which are likely defined in data_utils
# If ACT.py doesn't re-export them or ACTConfig doesn't handle their scope,
# you might need: from data_utils import FeatureType, NormalizationMode
from ACT import ACTConfig, ACTPolicy

# --- Configuration Constants ---
# IMPORTANT: These constants MUST match the configuration of the checkpoint
# being loaded. These are example values and might need adjustment.
CHUNK_SIZE = 30  # Context length for transformer / sequence length
ACTION_DIM = 8  # Dimension of the action space
QPOS_DIM = 6  # Dimension of the robot's proprioceptive state
IMAGE_H = 480  # Height of the image observations
IMAGE_W = 640  # Width of the image observations
IMAGE_C = 3  # Number of channels in image observations (e.g., 3 for RGB)

POLICY_QPOS_KEY = "observation.state"
POLICY_IMG_KEYS = ["observation.image_camera_1", "observation.image_camera_2"]
# For ACTConfig output_shapes, not directly for select_action input
POLICY_ACTION_KEY = "action"

# ACTConfig specific parameters (should match training)
# Number of observation steps input to the policy for a single prediction
# (Note: select_action usually takes a sequence of CHUNK_SIZE,
# and the policy internally might use the last N_OBS_STEPS)
N_OBS_STEPS = 1
DIM_MODEL = 512  # Transformer model dimension
N_HEADS = 8  # Number of attention heads
N_ENCODER_LAYERS = 4  # Number of encoder layers
# Number of decoder layers (ACT paper often suggests more, e.g., 7)
N_DECODER_LAYERS = 4
KL_WEIGHT = 10.0  # KL divergence weight if VAE is used
USE_VAE = True  # Whether the policy uses a VAE


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Dataset Statistics ---
    stats_path = os.path.join(args.checkpoint_dir, "dataset_stats.pt")
    if not os.path.exists(stats_path):
        print(f"Error: dataset_stats.pt not found in {args.checkpoint_dir}")
        print(
            "This file is crucial for correct input normalization and is saved "
            "by train.py."
        )
        print(
            "Please ensure the --checkpoint_dir path is correct and contains "
            "dataset_stats.pt."
        )
        return

    print(f"Loading dataset statistics from: {stats_path}")
    try:
        dataset_stats = torch.load(stats_path, map_location=device)
    except Exception as e:
        print(f"Error loading dataset_stats.pt: {e}")
        return

    # --- 2. Initialize ACT Policy ---
    print("Initializing ACT Policy...")
    act_config = ACTConfig(
        n_obs_steps=N_OBS_STEPS,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,  # Policy predicts actions for entire chunk
        input_shapes={
            POLICY_QPOS_KEY: [QPOS_DIM],
            POLICY_IMG_KEYS[0]: [IMAGE_C, IMAGE_H, IMAGE_W],
            POLICY_IMG_KEYS[1]: [IMAGE_C, IMAGE_H, IMAGE_W],
        },
        output_shapes={POLICY_ACTION_KEY: [ACTION_DIM]},
        dim_model=DIM_MODEL,
        n_heads=N_HEADS,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,
        kl_weight=KL_WEIGHT if USE_VAE else 0,
        use_vae=USE_VAE,
        # Ensure other relevant architectural params from ACTConfig in ACT.py
        # are set if not covered by defaults or the parameters above.
        # Optimizer-specific params in ACTConfig are not needed for inference.
    )

    try:
        policy = ACTPolicy(config=act_config, dataset_stats=dataset_stats).to(device)
    except Exception as e:
        print(f"Error initializing ACTPolicy: {e}")
        print(
            "This might be due to a mismatch in ACTConfig or missing "
            "dependencies (e.g., FeatureType from data_utils)."
        )
        import traceback

        traceback.print_exc()
        return

    # --- 3. Load Checkpoint ---
    checkpoint_full_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
    if not os.path.exists(checkpoint_full_path):
        print(f"Error: Checkpoint file not found at {checkpoint_full_path}")
        return

    print(f"Loading checkpoint from: {checkpoint_full_path}")
    try:
        state_dict = torch.load(checkpoint_full_path, map_location=device)
        policy.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint's state_dict: {e}")
        print(
            "This could be due to a mismatch between the model architecture "
            "defined here (ACTConfig)"
        )
        print("and the architecture of the saved checkpoint.")
        import traceback

        traceback.print_exc()
        return

    policy.eval()  # Set the policy to evaluation mode
    policy.reset()  # Reset any internal state (e.g., action queue)

    # --- 4. Create Random Input Vector (Batch) ---
    batch_size = 1  # Using a batch size of 1 for simplicity
    print(f"\nCreating a random input batch (batch_size={batch_size}):")

    # Observations for ACTPolicy are typically sequences (chunks).
    # qpos: (batch_size, chunk_size, QPOS_DIM)
    # images: (batch_size, chunk_size, IMAGE_C, IMAGE_H, IMAGE_W)
    random_qpos = torch.randn(batch_size, CHUNK_SIZE, QPOS_DIM, device=device)
    print(f"  {POLICY_QPOS_KEY} shape: {random_qpos.shape}")

    random_img1 = torch.randn(
        batch_size, CHUNK_SIZE, IMAGE_C, IMAGE_H, IMAGE_W, device=device
    )
    print(f"  {POLICY_IMG_KEYS[0]} shape: {random_img1.shape}")

    random_img2 = torch.randn(
        batch_size, CHUNK_SIZE, IMAGE_C, IMAGE_H, IMAGE_W, device=device
    )
    print(f"  {POLICY_IMG_KEYS[1]} shape: {random_img2.shape}")

    # Construct the observation batch dictionary for policy.select_action
    # It expects observation keys as defined in ACTConfig.input_shapes.
    random_obs_batch = {
        POLICY_QPOS_KEY: random_qpos,
        POLICY_IMG_KEYS[0]: random_img1,
        POLICY_IMG_KEYS[1]: random_img2,
    }
    # Note: \'is_first\' is not explicitly added here, as policy.reset() is called.
    # If your ACTPolicy\'s select_action expects an \'is_first\' tensor
    # in the batch, you might need to add it:
    # \'is_first\': torch.tensor([True] * batch_size, dtype=torch.bool, device=device)

    # --- 5. Apply Model to Random Vector ---
    print(
        "\nApplying model (policy.select_action) to the random " "observation batch..."
    )
    try:
        with torch.no_grad():  # Ensure no gradients computed during inference
            # policy.select_action processes the observation batch and returns
            # the next action(s). It typically manages an internal queue of
            # actions predicted by the model over n_action_steps.
            selected_action = policy.select_action(random_obs_batch)

        print("\n--- Model Output (Selected Action) ---")
        # Expected shape: (batch_size, ACTION_DIM)
        print(f"  Selected Action Shape: {selected_action.shape}")
        print(
            f"  Selected Action (first item in batch if batch_size > 0): "
            f"{selected_action[0] if batch_size > 0 else 'N/A'}"
        )
        print(f"  Selected Action Dtype: {selected_action.dtype}")

    except Exception as e:
        print(f"Error during policy.select_action: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Load an ACT policy checkpoint and apply it to a random "
            "observation vector."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help=(
            "Directory where the checkpoint .pth file and dataset_stats.pt "
            "are stored. This is typically a path like "
            "\\'.../checkpoints/YOUR_RUN_NAME\\'."
        ),
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        required=True,
        help=(
            "Name of the checkpoint .pth file (e.g., "
            "\\'act_policy_epoch_10000.pth\\'."
        ),
    )
    # You can add more arguments here to override the hardcoded constants
    # if needed, for example, --chunk_size, --qpos_dim, etc.

    args = parser.parse_args()
    main(args)
