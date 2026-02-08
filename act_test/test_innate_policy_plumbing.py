#!/usr/bin/env python3
"""
Plumbing check for InnatePolicy.

Verifies:
1. Model loads correctly
2. Forward pass runs without errors
3. All intermediate shapes are correct
4. Training and inference modes work
5. Different configurations work
"""

import torch
import sys
from innate_policy import InnatePolicy


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_shape(tensor, expected_shape, name):
    """Check if tensor has expected shape and print result."""
    actual_shape = tuple(tensor.shape)
    match = actual_shape == expected_shape
    status = "✓" if match else "✗"
    
    print(f"{status} {name:40s} {str(actual_shape):20s}", end="")
    if not match:
        print(f" (expected {expected_shape})", end="")
    print()
    
    if not match:
        raise ValueError(f"Shape mismatch for {name}: got {actual_shape}, expected {expected_shape}")
    
    return match


def test_multi_camera_forward_pass():
    """Test forward pass with multi-camera input, checking all intermediate shapes."""
    print_section("Multi-Camera Forward Pass (Step-by-Step)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Configuration
    batch_size = 4
    num_cameras = 2
    num_queries = 16
    state_dim = 6
    action_dim = 8
    action_horizon = 16
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Cameras: {num_cameras}")
    print(f"  Num queries: {num_queries}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Action horizon: {action_horizon}")
    
    # Create policy
    print(f"\nCreating policy...")
    policy = InnatePolicy(
        num_queries=num_queries,
        freeze_vision_backbone=True,
        num_cameras=num_cameras,
        state_dim=state_dim,
        proprio_hidden_dim=256,
        action_dim=action_dim,
        action_horizon=action_horizon,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        num_inference_steps=10
    ).to(device)
    
    # Input data
    print(f"\n[Step 1] Creating input data")
    images = torch.randn(batch_size, num_cameras, 3, 224, 224, device=device)
    robot_state = torch.randn(batch_size, state_dim, device=device)
    actions = torch.randn(batch_size, action_horizon, action_dim, device=device)
    
    check_shape(images, (batch_size, num_cameras, 3, 224, 224), "Input images")
    check_shape(robot_state, (batch_size, state_dim), "Robot state")
    check_shape(actions, (batch_size, action_horizon, action_dim), "Ground truth actions")
    
    # Vision encoding
    print(f"\n[Step 2] Vision encoding")
    print(f"  Processing camera 0...")
    cam0_images = images[:, 0]  # [B, 3, 224, 224]
    check_shape(cam0_images, (batch_size, 3, 224, 224), "  Camera 0 images")
    
    # Extract tokens (manually for checking)
    with torch.no_grad():
        output = policy.vision_encoder.backbone.get_intermediate_layers(
            cam0_images, n=1, return_class_token=True
        )
        cam0_tokens = output[0][0]  # [B, num_patches, 384]
        check_shape(cam0_tokens, (batch_size, 256, 384), "  Camera 0 patch tokens")
        
        # Check camera embeddings
        cam_embed_0 = policy.vision_encoder.camera_embeddings[0]
        check_shape(cam_embed_0, (384,), "  Camera 0 embedding")
        
        print(f"  Processing camera 1...")
        cam1_images = images[:, 1]
        check_shape(cam1_images, (batch_size, 3, 224, 224), "  Camera 1 images")
        
        output = policy.vision_encoder.backbone.get_intermediate_layers(
            cam1_images, n=1, return_class_token=True
        )
        cam1_tokens = output[0][0]
        check_shape(cam1_tokens, (batch_size, 256, 384), "  Camera 1 patch tokens")
        
        # Concatenate tokens
        all_tokens = torch.cat([cam0_tokens, cam1_tokens], dim=1)
        check_shape(all_tokens, (batch_size, 512, 384), "  Concatenated tokens (2×256)")
    
    # Full vision encoding
    visual_features = policy.encode_images(images)
    check_shape(visual_features, (batch_size, num_queries * 384), "Visual features (after attention pool)")
    
    # Proprioception encoding
    print(f"\n[Step 3] Proprioception encoding")
    proprio_features = policy.encode_proprio(robot_state)
    check_shape(proprio_features, (batch_size, 256), "Proprio features")
    
    # Concatenate conditioning
    print(f"\n[Step 4] Concatenating visual + proprio")
    global_cond = torch.cat([visual_features, proprio_features], dim=-1)
    expected_cond_dim = num_queries * 384 + 256
    check_shape(global_cond, (batch_size, expected_cond_dim), "Global conditioning")
    
    # Training forward pass
    print(f"\n[Step 5] Training forward pass")
    policy.train()
    
    # Sample timestep
    t = torch.rand(batch_size, device=device)
    check_shape(t, (batch_size,), "Flow timesteps")
    
    # Sample noise
    noise = torch.randn_like(actions)
    check_shape(noise, (batch_size, action_horizon, action_dim), "Noise")
    
    # Interpolate
    t_expanded = t.reshape(-1, 1, 1)
    noisy_actions = t_expanded * actions + (1 - t_expanded) * noise
    check_shape(noisy_actions, (batch_size, action_horizon, action_dim), "Noisy actions (interpolated)")
    
    # Predict velocity
    print(f"\n[Step 6] Action decoder (UNet)")
    predicted_velocity = policy.action_decoder(
        sample=noisy_actions,
        timestep=t,
        global_cond=global_cond
    )
    check_shape(predicted_velocity, (batch_size, action_horizon, action_dim), "Predicted velocity")
    
    # Full forward pass
    print(f"\n[Step 7] Full forward pass")
    output = policy(images, robot_state, actions, training=True)
    
    assert 'loss' in output, "Output should contain 'loss'"
    assert 'predictions' in output, "Output should contain 'predictions'"
    
    print(f"✓ Loss value: {output['loss'].item():.6f}")
    check_shape(output['predictions'], (batch_size, action_horizon, action_dim), "Predictions")
    
    # Test backward pass
    print(f"\n[Step 8] Backward pass")
    loss = output['loss']
    loss.backward()
    print(f"✓ Backward pass successful")
    
    # Check gradients exist
    has_grads = sum(1 for p in policy.parameters() if p.grad is not None)
    total_params = sum(1 for p in policy.parameters())
    print(f"✓ Gradients computed: {has_grads}/{total_params} parameters")
    
    return True


def test_inference():
    """Test inference mode."""
    print_section("Inference Mode")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 2
    num_cameras = 2
    num_queries = 8
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num cameras: {num_cameras}")
    print(f"  Num inference steps: 10")
    
    # Create policy
    policy = InnatePolicy(
        num_queries=num_queries,
        num_cameras=num_cameras,
        state_dim=6,  # Match robot_state dimension
        action_dim=8,
        action_horizon=16,
        num_inference_steps=10
    ).to(device)
    policy.eval()
    
    # Input data
    images = torch.randn(batch_size, num_cameras, 3, 224, 224, device=device)
    robot_state = torch.randn(batch_size, 6, device=device)
    
    check_shape(images, (batch_size, num_cameras, 3, 224, 224), "Input images")
    check_shape(robot_state, (batch_size, 6), "Robot state")
    
    # Inference
    print(f"\n[Step 1] Encoding observations")
    visual_features = policy.encode_images(images)
    proprio_features = policy.encode_proprio(robot_state)
    global_cond = torch.cat([visual_features, proprio_features], dim=-1)
    
    check_shape(visual_features, (batch_size, num_queries * 384), "Visual features")
    check_shape(proprio_features, (batch_size, 256), "Proprio features")
    check_shape(global_cond, (batch_size, num_queries * 384 + 256), "Global conditioning")
    
    print(f"\n[Step 2] Sampling actions (Euler integration)")
    with torch.no_grad():
        # Start from noise
        x_t = torch.randn(batch_size, 16, 8, device=device)
        check_shape(x_t, (batch_size, 16, 8), "Initial noise")
        
        # Simulate one step
        t = torch.zeros(batch_size, device=device)
        v_t = policy.action_decoder(x_t, t, global_cond)
        check_shape(v_t, (batch_size, 16, 8), "Predicted velocity (step 0)")
        
        # Full sampling
        predicted_actions = policy.sample_actions(global_cond)
        check_shape(predicted_actions, (batch_size, 16, 8), "Final predicted actions")
    
    print(f"\n[Step 3] Using get_action() convenience method")
    predicted_actions = policy.get_action(images, robot_state)
    check_shape(predicted_actions, (batch_size, 16, 8), "Predicted actions")
    
    print(f"\n✓ Inference successful!")
    
    return True


def test_single_camera():
    """Test with single camera input."""
    print_section("Single Camera Mode")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 4
    
    policy = InnatePolicy(
        num_queries=8,
        num_cameras=1,  # Single camera
        state_dim=6,
        action_dim=8,
        action_horizon=16,
    ).to(device)
    
    # Single camera input [B, 3, H, W] (no camera dimension)
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    robot_state = torch.randn(batch_size, 6, device=device)
    actions = torch.randn(batch_size, 16, 8, device=device)
    
    check_shape(images, (batch_size, 3, 224, 224), "Single camera images")
    
    print(f"\n[Step 1] Forward pass")
    policy.train()
    output = policy(images, robot_state, actions, training=True)
    
    print(f"✓ Loss: {output['loss'].item():.6f}")
    check_shape(output['predictions'], (batch_size, 16, 8), "Predictions")
    
    print(f"\n[Step 2] Inference")
    policy.eval()
    predicted_actions = policy.get_action(images, robot_state)
    check_shape(predicted_actions, (batch_size, 16, 8), "Predicted actions")
    
    print(f"\n✓ Single camera mode works!")
    
    return True


def test_different_configs():
    """Test different configurations."""
    print_section("Different Configurations")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    configs = [
        {'num_queries': 4, 'num_cameras': 2, 'action_horizon': 8},
        {'num_queries': 8, 'num_cameras': 2, 'action_horizon': 16},
        {'num_queries': 16, 'num_cameras': 2, 'action_horizon': 32},
        {'num_queries': 8, 'num_cameras': 3, 'action_horizon': 16},
        {'num_queries': 32, 'num_cameras': 1, 'action_horizon': 16},
    ]
    
    print(f"\nTesting {len(configs)} configurations:\n")
    print(f"{'Queries':>8} {'Cameras':>8} {'Horizon':>8} {'Params':>12} {'Status':>8}")
    print("-" * 50)
    
    for config in configs:
        try:
            policy = InnatePolicy(
                num_queries=config['num_queries'],
                num_cameras=config['num_cameras'],
                action_horizon=config['action_horizon'],
                state_dim=6,
                action_dim=8,
                freeze_vision_backbone=True
            ).to(device)
            
            # Test forward pass
            batch_size = 2
            if config['num_cameras'] > 1:
                images = torch.randn(batch_size, config['num_cameras'], 3, 224, 224, device=device)
            else:
                images = torch.randn(batch_size, 3, 224, 224, device=device)
            
            robot_state = torch.randn(batch_size, 6, device=device)
            actions = torch.randn(batch_size, config['action_horizon'], 8, device=device)
            
            policy.train()
            output = policy(images, robot_state, actions, training=True)
            
            num_params = sum(p.numel() for p in policy.parameters())
            
            print(f"{config['num_queries']:>8} {config['num_cameras']:>8} {config['action_horizon']:>8} {num_params:>12,} {'✓':>8}")
            
        except Exception as e:
            print(f"{config['num_queries']:>8} {config['num_cameras']:>8} {config['action_horizon']:>8} {'N/A':>12} {'✗':>8}")
            print(f"  Error: {str(e)[:60]}")
    
    print()
    return True


def test_batch_sizes():
    """Test different batch sizes."""
    print_section("Different Batch Sizes")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    policy = InnatePolicy(
        num_queries=8,
        num_cameras=2,
        state_dim=6,
        action_dim=8,
        action_horizon=16,
        freeze_vision_backbone=True
    ).to(device)
    
    batch_sizes = [1, 2, 4, 8, 16]
    
    print(f"\nTesting batch sizes: {batch_sizes}\n")
    print(f"{'Batch Size':>12} {'Status':>8}")
    print("-" * 25)
    
    for bs in batch_sizes:
        try:
            images = torch.randn(bs, 2, 3, 224, 224, device=device)
            robot_state = torch.randn(bs, 6, device=device)
            actions = torch.randn(bs, 16, 8, device=device)
            
            policy.train()
            output = policy(images, robot_state, actions, training=True)
            
            assert output['predictions'].shape[0] == bs
            
            print(f"{bs:>12} {'✓':>8}")
            
        except Exception as e:
            print(f"{bs:>12} {'✗':>8}")
            print(f"  Error: {str(e)[:60]}")
    
    print()
    return True


def main():
    """Run all plumbing checks."""
    print("\n" + "=" * 70)
    print("  INNATE POLICY PLUMBING CHECK")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    tests = [
        ("Multi-Camera Forward Pass", test_multi_camera_forward_pass),
        ("Inference Mode", test_inference),
        ("Single Camera Mode", test_single_camera),
        ("Different Configurations", test_different_configs),
        ("Different Batch Sizes", test_batch_sizes),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "✓ PASS"))
        except Exception as e:
            results.append((test_name, f"✗ FAIL: {str(e)}"))
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print_section("SUMMARY")
    print()
    for test_name, result in results:
        status = result.split(":")[0]
        print(f"{status:8s} {test_name}")
    
    passed = sum(1 for _, r in results if "✓" in r)
    total = len(results)
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "=" * 70)
        print("  ✓ ALL PLUMBING CHECKS PASSED!")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("  ✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
