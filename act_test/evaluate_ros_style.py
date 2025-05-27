#!/usr/bin/env python
import torch
import os
import argparse
import numpy as np

# Assuming ACT.py is in the same directory or PYTHONPATH
from ACT import ACTConfig, ACTPolicy

# Key names for observations and actions from the ROS script's batch construction
POLICY_QPOS_KEY = "observation.state"
POLICY_IMG_KEY_1 = "observation.image_camera_1"
POLICY_IMG_KEY_2 = "observation.image_camera_2"
# POLICY_ACTION_KEY = "action" # Not used for select_action input


def create_act_config_ros_style():
    # Based on the provided ROS script
    input_shapes = {
        POLICY_IMG_KEY_1: [3, 480, 640],  # [C, H, W]
        POLICY_IMG_KEY_2: [3, 480, 640],  # [C, H, W]
        POLICY_QPOS_KEY: [6],  # state_dim
    }
    output_shapes = {"action": [8]}  # action_dim

    return ACTConfig(
        n_obs_steps=1,
        chunk_size=30,
        n_action_steps=30,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        vision_backbone="resnet18",
        replace_final_stride_with_dilation=False,
        pre_norm=False,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        n_encoder_layers=4,
        # IMPORTANT: Set to 7 based on previous errors with ckpt_vig.pth
        # The ROS script had 1, which caused 'Unexpected key(s)' error
        # with this specific checkpoint
        # Updated based on combined errors: Missing beyond layer 3, Unexpected for layer 1 with n_dec=1
        n_decoder_layers=4,
        use_vae=True,
        dropout=0.1,
        kl_weight=10.0,
        temporal_ensemble_coeff=None,
        optimizer_lr=1e-5,
        optimizer_weight_decay=1e-4,
        optimizer_lr_backbone=1e-5,
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 0. Get ACT Configuration ---
    act_config = create_act_config_ros_style()
    print("Using ACTConfig (ROS-style inspired):")
    print(f"  chunk_size: {act_config.chunk_size}")
    print(f"  n_obs_steps: {act_config.n_obs_steps}")
    print(f"  n_action_steps: {act_config.n_action_steps}")
    print(f"  dim_model: {act_config.dim_model}")
    print(f"  n_decoder_layers: {act_config.n_decoder_layers}")  # Verify this
    print(f"  use_vae: {act_config.use_vae}")

    # --- 1. Load Dataset Statistics ---
    stats_path = os.path.join(args.checkpoint_dir, "dataset_stats.pt")
    if not os.path.exists(stats_path):
        print(f"Error: dataset_stats.pt not found in {args.checkpoint_dir}")
        return
    print(f"Loading dataset statistics from: {stats_path}")
    try:
        # ROS script loads to CPU, then model is moved to device
        dataset_stats = torch.load(stats_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading dataset_stats.pt: {e}")
        return

    # --- 2. Initialize ACT Policy ---
    print("Initializing ACT Policy...")
    try:
        policy = ACTPolicy(config=act_config, dataset_stats=dataset_stats).to(device)
    except Exception as e:
        print(f"Error initializing ACTPolicy: {e}")
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
        policy.load_state_dict(state_dict)  # strict=True by default
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint's state_dict: {e}")
        print(
            "This could be due to a mismatch between model architecture "
            "(ACTConfig) and saved checkpoint."
        )
        import traceback

        traceback.print_exc()
        return

    policy.eval()
    policy.reset()  # As per typical inference setup

    # --- 4. Create Random Input Vector (Batch) - ROS Style (single step) ---
    batch_size = 1  # ROS script processes one observation at a time
    print(f"\\nCreating random input batch (bs={batch_size}, ROS-style single step):")

    qpos_dim = act_config.input_shapes[POLICY_QPOS_KEY][0]
    action_dim_expected = act_config.output_shapes["action"][0]

    img_c, img_h, img_w = 0, 0, 0
    has_img1 = POLICY_IMG_KEY_1 in act_config.input_shapes
    has_img2 = POLICY_IMG_KEY_2 in act_config.input_shapes

    if has_img1:
        img_c, img_h, img_w = act_config.input_shapes[POLICY_IMG_KEY_1]

    # Prepare observation batch
    # IMPORTANT: ROS script feeds single, unsqueezed observations (batch_size, C, H, W) and (batch_size, D_qpos)
    # This implies ACTPolicy's normalize_inputs and _prepare_batch_for_model handle this.
    # The `n_obs_steps=1` in ACTConfig should align with this.

    # qpos: (batch_size, QPOS_DIM)
    random_qpos_np = np.random.randn(batch_size, qpos_dim).astype(np.float32)
    random_qpos = torch.tensor(random_qpos_np).to(device)
    print(f"  {POLICY_QPOS_KEY} shape: {random_qpos.shape}")

    observation_batch = {POLICY_QPOS_KEY: random_qpos}

    if has_img1 and img_c > 0:
        img_shape = (batch_size, img_c, img_h, img_w)
        random_img1_np = np.random.randn(*img_shape).astype(np.float32)
        random_img1 = torch.tensor(random_img1_np).to(device)
        print(f"  {POLICY_IMG_KEY_1} shape: {random_img1.shape}")
        observation_batch[POLICY_IMG_KEY_1] = random_img1

    if has_img2 and img_c > 0:  # Assuming img2 has same C,H,W as img1 if present
        img_shape = (batch_size, img_c, img_h, img_w)
        random_img2_np = np.random.randn(*img_shape).astype(np.float32)
        random_img2 = torch.tensor(random_img2_np).to(device)
        print(f"  {POLICY_IMG_KEY_2} shape: {random_img2.shape}")
        observation_batch[POLICY_IMG_KEY_2] = random_img2

    # --- 5. Apply Model to Random Vector ---
    print("\\nApplying model (policy.select_action) to random batch...")
    try:
        with torch.no_grad():
            # policy.select_action is expected to handle this single-step batch
            selected_action = policy.select_action(observation_batch)

        print("\\n--- Model Output (Selected Action) ---")
        action_np = selected_action.cpu().numpy()
        # Squeeze batch dimension if present (ROS script does this)
        if action_np.ndim > 1 and action_np.shape[0] == 1:
            action_np = action_np.squeeze(0)

        print(f"  Selected Action Shape (squeezed): {action_np.shape}")
        print(f"  Selected Action Content: {action_np}")
        if action_np.shape[-1] != action_dim_expected:
            print(
                f"  WARNING: Action dim {action_np.shape[-1]}, expected {action_dim_expected}"
            )
        print(f"  Selected Action Dtype: {selected_action.dtype}")

    except Exception as e:
        print(f"Error during policy.select_action: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load ACT policy (ROS-style) & apply to random vector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Dir with checkpoint .pth and dataset_stats.pt.",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        required=True,
        help="Checkpoint .pth file name (e.g., 'act_policy_epoch_50000.pth').",
    )
    args = parser.parse_args()
    main(args)
