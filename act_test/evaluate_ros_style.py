#!/usr/bin/env python
import torch
import os
import argparse
import numpy as np
import datetime
import subprocess
import time

# Assuming ACT.py is in the same directory or PYTHONPATH
from ACT import ACTConfig, ACTPolicy

# Key names for observations and actions from the ROS script's batch construction
POLICY_QPOS_KEY = "observation.state"
POLICY_IMG_KEY_1 = "observation.image_camera_1"
POLICY_IMG_KEY_2 = "observation.image_camera_2"
# POLICY_ACTION_KEY = "action" # Not used for select_action input


def get_git_commit_hash():
    try:
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


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
        # Updated based on combined errors:
        # Missing beyond layer 3, Unexpected for layer 1 with n_dec=1
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

    print("Compiling policy...")
    policy = torch.compile(policy)

    print("Policy compiled successfully.")

    policy.eval()
    policy.reset()  # As per typical inference setup

    # --- Setup Logging ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    commit_hash = get_git_commit_hash()
    log_dir = "benchmark_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{timestamp}_{commit_hash}_benchmark_log.txt")
    print(f"Logging benchmark data to: {log_filename}")

    with open(log_filename, "w") as log_file:
        log_file.write(f"Benchmark Run: {timestamp}\\n")
        log_file.write(f"Git Commit: {commit_hash}\\n")
        log_file.write(f"Checkpoint: {checkpoint_full_path}\\n\\n")

        # --- 4. Create Standard Input Vector (Batch) - ROS Style (single step) ---
        batch_size = 1  # ROS script processes one observation at a time
        print_msg = (
            f"\\nCreating standard input batch (bs={batch_size}, "
            f"ROS-style single step):"
        )
        print(print_msg)
        log_file.write(f"{print_msg.strip()}\\n")

        qpos_dim = act_config.input_shapes[POLICY_QPOS_KEY][0]
        action_dim_expected = act_config.output_shapes["action"][0]

        img_c, img_h, img_w = 0, 0, 0
        has_img1 = POLICY_IMG_KEY_1 in act_config.input_shapes
        has_img2 = POLICY_IMG_KEY_2 in act_config.input_shapes

        if has_img1:
            img_c, img_h, img_w = act_config.input_shapes[POLICY_IMG_KEY_1]

        # Prepare observation batch
        # IMPORTANT: ROS script feeds single, unsqueezed observations
        # (batch_size, C, H, W) and (batch_size, D_qpos)
        # This implies ACTPolicy's normalize_inputs and
        # _prepare_batch_for_model handle this.
        # The `n_obs_steps=1` in ACTConfig should align with this.

        # qpos: (batch_size, QPOS_DIM)
        # Use np.ones for deterministic input, can be changed to other fixed values
        standard_qpos_np = np.ones((batch_size, qpos_dim), dtype=np.float32)
        standard_qpos = torch.tensor(standard_qpos_np).to(device)
        print(f"  {POLICY_QPOS_KEY} shape: {standard_qpos.shape}")
        print(f"  {POLICY_QPOS_KEY} sample value: {standard_qpos_np[0, :3]}")
        log_file.write(f"  {POLICY_QPOS_KEY} shape: {standard_qpos.shape}\\n")
        log_file.write(
            f"  {POLICY_QPOS_KEY} (first 3 values): "
            f"{standard_qpos_np[0, :3].tolist()}\\n"
        )

        observation_batch = {POLICY_QPOS_KEY: standard_qpos}

        if has_img1 and img_c > 0:
            img_shape = (batch_size, img_c, img_h, img_w)
            # Use np.ones for deterministic input, can be changed
            standard_img1_np = (
                np.ones(img_shape, dtype=np.float32) * 0.5
            )  # e.g. mid-range values
            standard_img1 = torch.tensor(standard_img1_np).to(device)
            print(f"  {POLICY_IMG_KEY_1} shape: {standard_img1.shape}")
            # Print a small part of the image for verification
            print(
                f"  {POLICY_IMG_KEY_1} sample value "
                f"(first pixel, all channels): "
                f"{standard_img1_np[0, :, 0, 0]}"
            )
            log_file.write(f"  {POLICY_IMG_KEY_1} shape: {standard_img1.shape}\\n")
            log_file.write(
                f"  {POLICY_IMG_KEY_1} sample value "
                f"(first pixel, all channels): "
                f"{standard_img1_np[0, :, 0, 0].tolist()}\\n"
            )
            observation_batch[POLICY_IMG_KEY_1] = standard_img1

        if has_img2 and img_c > 0:  # Assuming img2 has same C,H,W as img1
            img_shape = (batch_size, img_c, img_h, img_w)
            # Use np.ones for deterministic input, can be changed
            standard_img2_np = (
                np.ones(img_shape, dtype=np.float32) * 0.7
            )  # e.g. different mid-range values
            standard_img2 = torch.tensor(standard_img2_np).to(device)
            print(f"  {POLICY_IMG_KEY_2} shape: {standard_img2.shape}")
            # Print a small part of the image for verification
            print(
                f"  {POLICY_IMG_KEY_2} sample value "
                f"(first pixel, all channels): "
                f"{standard_img2_np[0, :, 0, 0]}"
            )
            log_file.write(f"  {POLICY_IMG_KEY_2} shape: {standard_img2.shape}\\n")
            log_file.write(
                f"  {POLICY_IMG_KEY_2} sample value "
                f"(first pixel, all channels): "
                f"{standard_img2_np[0, :, 0, 0].tolist()}\\n"
            )
            observation_batch[POLICY_IMG_KEY_2] = standard_img2

        log_file.write("\\n--- Full Input Observation Batch (NumPy) ---\\n")
        for key, tensor_val in observation_batch.items():
            log_file.write(f"Key: {key}\n")
            # For brevity, log only a slice or summary of large image arrays
            if "image" in key:
                # Log a small slice e.g. first channel, 5x5 top-left corner
                sample_slice = tensor_val.cpu().numpy()[0, 0, :5, :5]
                log_file.write(f"  Shape: {tensor_val.shape}\n")
                log_file.write(f"  Dtype: {tensor_val.dtype}\n")
                log_file.write(f"  Sample Slice (e.g., [0,0,:5,:5]):\n{sample_slice}\n")
            else:
                log_file.write(f"  Value:\n{tensor_val.cpu().numpy()}\n")
        log_file.write("\n")

        # --- 5. Apply Model to Standard Vector ---
        print_msg_model = "\nApplying model (policy.select_action) to standard batch..."
        print(print_msg_model)
        log_file.write(f"{print_msg_model.strip()}\n")
        try:
            with torch.no_grad():
                selected_action = policy.select_action(observation_batch)

            print_msg_output = "\n--- Model Output (Selected Action) ---"
            print(print_msg_output)
            log_file.write(f"{print_msg_output.strip()}\n")

            action_np = selected_action.cpu().numpy()
            if action_np.ndim > 1 and action_np.shape[0] == 1:
                action_np_squeezed = action_np.squeeze(0)
            else:
                action_np_squeezed = action_np

            print(f"  Selected Action Shape (squeezed): {action_np_squeezed.shape}")
            print(f"  Selected Action Content: {action_np_squeezed}")
            log_file.write(
                f"  Selected Action Shape (original): {selected_action.shape}\n"
            )
            log_file.write(
                f"  Selected Action Shape (squeezed): {action_np_squeezed.shape}\n"
            )
            log_file.write(
                "  Selected Action Content (NumPy):\n"
                f"{action_np_squeezed.tolist()}\n"
            )

            if action_np_squeezed.shape[-1] != action_dim_expected:
                warning_msg = (
                    f"  WARNING: Action dim {action_np_squeezed.shape[-1]}, "
                    f"expected {action_dim_expected}"
                )
                print(warning_msg)
                log_file.write(f"{warning_msg}\n")
            print(f"  Selected Action Dtype: {selected_action.dtype}")
            log_file.write(f"  Selected Action Dtype: {selected_action.dtype}\n")

        except Exception as e:
            error_msg = f"Error during policy.select_action: {e}"
            print(error_msg)
            log_file.write(f"{error_msg}\n")
            import traceback

            traceback.print_exc(file=log_file)
            traceback.print_exc()

        # --- 6. Profiling ---
        log_file.write("\n\n--- Profiling Inference Speed ---\n")
        num_inferences = 0
        profiling_duration = 30.0  # seconds (Increased)
        warmup_iterations = 5  # Added

        # Warm-up inference
        if policy and observation_batch:
            print("Starting warm-up inferences...")  # Added print
            for _ in range(warmup_iterations):
                try:
                    with torch.no_grad():
                        _ = policy.select_action(observation_batch)
                        # if device.type == "cuda": # Removed for CPU focus
                        #     torch.cuda.synchronize() # Ensure warm-up is complete
                except Exception as e:
                    print(f"Error during warm-up inference: {e}")
                    log_file.write(f"Error during warm-up inference: {e}\n")
                    break  # Stop if warmup fails
            print("Warm-up complete.")  # Added print

        print_msg_profiling = "\nStarting profiling loop..."
        print(print_msg_profiling)
        log_file.write(f"{print_msg_profiling.strip()}\\n")

        # if device.type == "cuda": # Removed for CPU focus
        #     torch.cuda.synchronize() # Synchronize before starting timer
        start_time = time.time()
        while (time.time() - start_time) < profiling_duration:
            try:
                with torch.no_grad():
                    # Use the same observation_batch from before
                    _ = policy.select_action(observation_batch)
                num_inferences += 1
            except Exception as e:
                error_msg = f"Error during profiling inference: {e}"
                print(error_msg)
                log_file.write(f"{error_msg}\\n")
                break  # Stop profiling on error
        # if device.type == "cuda": # Removed for CPU focus
        #    torch.cuda.synchronize() # Synchronize before ending timer
        end_time = time.time()
        actual_duration = end_time - start_time
        inferences_per_second = (
            num_inferences / actual_duration if actual_duration > 0 else 0
        )

        profiling_result_msg = (
            f"Performed {num_inferences} inferences "
            f"in {actual_duration:.2f} seconds."
        )
        ips_msg = f"Inferences per second: {inferences_per_second:.2f}"

        print(profiling_result_msg)
        print(ips_msg)
        log_file.write(f"{profiling_result_msg}\n")
        log_file.write(f"{ips_msg}\n")

    print(f"\nBenchmark data saved to {log_filename}")


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
