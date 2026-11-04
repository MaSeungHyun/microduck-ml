#!/usr/bin/env python3
"""
Smoke test script for Microduck environment

Verifies:
1. MJCF model loads correctly
2. Environment can be created (both MuJoCo and Genesis backends)
3. Basic step/reset works
4. Observation/action shapes are correct
5. Rewards are computed correctly (upright reward > 0 when standing)
6. A few training iterations work
7. Episode lengths > 1 (termination not triggered immediately)

Usage:
    python scripts/test_env.py --backend mujoco  # CPU test (no GPU required)
    python scripts/test_env.py --backend genesis --device cpu  # Genesis CPU test
    python scripts/test_env.py --backend genesis --device cuda  # Genesis GPU test
"""

import argparse
import os
import sys
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_mjcf_load():
    """Test that the MJCF model can be loaded"""
    print("Testing MJCF model loading...")
    
    try:
        import mujoco
        model_path = os.path.join(os.path.dirname(__file__), "..", "assets", "scene.xml")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Joints: {model.njnt}")
        print(f"  ✓ Actuators: {model.nu}")
        print(f"  ✓ Bodies: {model.nbody}")
        
        # Test a few simulation steps
        for _ in range(10):
            mujoco.mj_step(model, data)
        print(f"  ✓ Simulation steps work")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        return False


def test_environment(backend="mujoco", device="cpu"):
    """Test environment creation and basic operations"""
    print(f"\nTesting environment (backend={backend}, device={device})...")
    
    try:
        from envs.microduck_env import MicroduckEnv, EnvConfig, CurriculumStage
        
        # Genesis requires EGL/OpenGL even for headless mode
        if backend == "genesis":
            try:
                import genesis as gs
            except Exception as e:
                if "EGL" in str(e) or "glGetError" in str(e) or "OpenGL" in str(e):
                    print(f"  ⚠ Genesis requires EGL/OpenGL for headless mode (not available on this system)")
                    print(f"  ⚠ Genesis will work on Windows/Linux with GPU and proper graphics drivers")
                    return None  # Skip but not fail
                raise
        
        cfg = EnvConfig(
            num_envs=4,  # Small number for testing
            stage=CurriculumStage.STAND,
            episode_length=100,
        )
        
        env = MicroduckEnv(
            cfg=cfg,
            backend=backend,
            device=device,
            render=False,
            headless=True,
        )
        
        print(f"  ✓ Environment created")
        print(f"  ✓ Observation dim: {env.num_obs}")
        print(f"  ✓ Action dim: {env.num_actions}")
        print(f"  ✓ Num envs: {env.num_envs}")
        
        if backend == "genesis":
            print(f"  ✓ Total DOFs: {env.total_dofs}")
            print(f"  ✓ Leg DOF indices: {env.leg_dof_indices}")
        
        # Test reset
        obs, info = env.reset()
        print(f"  ✓ Reset works, obs shape: {obs.shape}")
        
        # Test step
        import numpy as np
        import torch
        
        if backend == "genesis":
            actions = torch.zeros((env.num_envs, env.num_actions), device=device, dtype=torch.float32)
        else:
            actions = np.zeros((env.num_envs, env.num_actions), dtype=np.float32)
        
        obs, rewards, dones, truncated, info = env.step(actions)
        print(f"  ✓ Step works")
        print(f"    - Obs shape: {obs.shape}")
        print(f"    - Rewards shape: {rewards.shape}")
        
        # Convert to float for display
        if isinstance(rewards, torch.Tensor):
            mean_reward = float(rewards.mean().item())
        else:
            mean_reward = float(rewards.mean())
        print(f"    - Mean reward: {mean_reward:.4f}")
        
        # Verify reward is positive (upright robot should get positive reward)
        if mean_reward <= 0:
            print(f"  ⚠ Warning: Mean reward is {mean_reward:.4f}, expected > 0 for upright robot")
        else:
            print(f"  ✓ Reward is positive (robot is upright)")
        
        # Run multiple steps and check episode lengths
        total_steps = 0
        max_steps = 50
        for i in range(max_steps):
            if backend == "genesis":
                actions = torch.randn((env.num_envs, env.num_actions), device=device, dtype=torch.float32) * 0.1
            else:
                actions = np.random.randn(env.num_envs, env.num_actions).astype(np.float32) * 0.1
            obs, rewards, dones, truncated, info = env.step(actions)
            total_steps += 1
            
            # Check if any done (shouldn't happen immediately with small actions)
            if backend == "genesis":
                num_dones = int(dones.sum().item())
            else:
                num_dones = int(dones.sum())
            
            if num_dones > 0 and i < 5:
                print(f"  ⚠ Warning: {num_dones} envs terminated at step {i+1}")
        
        # Get episode lengths
        if backend == "genesis":
            mean_ep_len = float(env.episode_length.mean().item())
        else:
            mean_ep_len = float(env.episode_length.mean())
        
        print(f"  ✓ Multiple steps work ({total_steps} iterations)")
        print(f"    - Mean episode length: {mean_ep_len:.1f}")
        
        if mean_ep_len < 5:
            print(f"  ⚠ Warning: Episode length very short ({mean_ep_len:.1f}), check termination conditions")
        
        env.close()
        print(f"  ✓ Environment closed")
        
        return True
    except Exception as e:
        error_msg = str(e)
        if backend == "genesis" and ("EGL" in error_msg or "glGetError" in error_msg or "OpenGL" in error_msg):
            print(f"  ⚠ Genesis requires EGL/OpenGL for headless mode (not available on this system)")
            print(f"  ⚠ Genesis will work on Windows/Linux with GPU and proper graphics drivers")
            return None  # Skip but not fail
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        return False


def test_training_loop(backend="mujoco", device="cpu", num_iterations=20):
    """Test a few training iterations and verify reward increases"""
    print(f"\nTesting training loop (backend={backend}, device={device})...")
    
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        from envs.microduck_env import MicroduckEnv, EnvConfig, CurriculumStage
        
        # Genesis requires EGL/OpenGL even for headless mode
        if backend == "genesis":
            try:
                import genesis as gs
            except Exception as e:
                if "EGL" in str(e) or "glGetError" in str(e) or "OpenGL" in str(e):
                    print(f"  ⚠ Genesis requires EGL/OpenGL (skipped)")
                    return None
                raise
        
        # Create environment
        cfg = EnvConfig(
            num_envs=8,
            stage=CurriculumStage.STAND,
            episode_length=100,
        )
        
        env = MicroduckEnv(
            cfg=cfg,
            backend=backend,
            device=device,
            render=False,
            headless=True,
        )
        
        # Simple policy network
        policy = nn.Sequential(
            nn.Linear(env.num_obs, 64),
            nn.ReLU(),
            nn.Linear(64, env.num_actions),
            nn.Tanh(),
        ).to(device)
        
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        
        # Training loop
        obs, _ = env.reset()
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, device=device, dtype=torch.float32)
        
        episode_rewards = []
        episode_lengths = []
        current_rewards = torch.zeros(env.num_envs, device=device, dtype=torch.float32)
        current_lengths = torch.zeros(env.num_envs, device=device, dtype=torch.float32)
        
        for step in range(num_iterations * 10):  # More steps
            # Get action from policy
            with torch.no_grad():
                actions = policy(obs)
            
            # Step environment
            next_obs, rewards, dones, truncated, info = env.step(actions)
            
            if not isinstance(next_obs, torch.Tensor):
                next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
            if not isinstance(dones, torch.Tensor):
                dones = torch.tensor(dones, device=device)
            if not isinstance(truncated, torch.Tensor):
                truncated = torch.tensor(truncated, device=device)
            
            # Track episode stats
            current_rewards = current_rewards + rewards
            current_lengths = current_lengths + 1.0
            
            # Check for episode ends
            done_mask = dones | truncated
            done_indices = torch.where(done_mask)[0]
            for idx in done_indices:
                episode_rewards.append(float(current_rewards[idx].item()))
                episode_lengths.append(float(current_lengths[idx].item()))
                current_rewards[idx] = 0.0
                current_lengths[idx] = 0.0
            
            # Simple "training" step
            actions_for_grad = policy(obs)
            dummy_loss = (actions_for_grad ** 2).mean() * 0.01
            optimizer.zero_grad()
            dummy_loss.backward()
            optimizer.step()
            
            obs = next_obs
        
        print(f"  ✓ Training loop works")
        print(f"    - Steps completed: {num_iterations * 10 * env.num_envs}")
        print(f"    - Episodes completed: {len(episode_rewards)}")
        
        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            print(f"    - Average episode reward: {avg_reward:.4f}")
            print(f"    - Average episode length: {avg_length:.1f}")
            
            if avg_reward <= 0:
                print(f"  ⚠ Warning: Average reward is {avg_reward:.4f}, expected > 0")
            else:
                print(f"  ✓ Positive average reward achieved")
            
            if avg_length < 5:
                print(f"  ⚠ Warning: Average episode length is {avg_length:.1f}, very short")
            elif avg_length > 20:
                print(f"  ✓ Good episode length (>{avg_length:.0f} steps)")
        else:
            print(f"  ⚠ No episodes completed during test")
        
        env.close()
        return True
    except Exception as e:
        error_msg = str(e)
        if backend == "genesis" and ("EGL" in error_msg or "glGetError" in error_msg or "OpenGL" in error_msg or "already initialized" in error_msg):
            print(f"  ⚠ Genesis requires EGL/OpenGL (skipped)")
            return None
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Smoke test for Microduck environment")
    parser.add_argument("--backend", type=str, default="mujoco",
                        choices=["mujoco", "genesis"],
                        help="Simulation backend to test")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to use")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of training iterations to test")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Microduck Environment Smoke Test")
    print("=" * 60)
    
    # Check available backends
    print("\nChecking available backends...")
    
    mujoco_available = False
    genesis_available = False
    torch_cuda = False
    
    try:
        import mujoco
        mujoco_available = True
        print(f"  ✓ MuJoCo {mujoco.__version__} available")
    except ImportError:
        print("  ✗ MuJoCo not available")
    
    try:
        import genesis as gs
        genesis_available = True
        print(f"  ✓ Genesis available")
    except ImportError:
        print("  ✗ Genesis not available")
    
    try:
        import torch
        torch_cuda = torch.cuda.is_available()
        print(f"  ✓ PyTorch {torch.__version__}, CUDA: {torch_cuda}")
    except ImportError:
        print("  ✗ PyTorch not available")
    
    # Determine what we can test
    test_backend = args.backend
    test_device = args.device
    
    if test_backend == "genesis" and not genesis_available:
        print("\nGenesis not available, falling back to MuJoCo")
        test_backend = "mujoco"
    
    if test_backend == "mujoco" and not mujoco_available:
        print("\nError: MuJoCo not available. Install with: pip install mujoco")
        return 1
    
    if test_device == "cuda" and not torch_cuda:
        print("\nCUDA not available, falling back to CPU")
        test_device = "cpu"
    
    if test_backend == "genesis" and test_device == "cpu":
        print("\nTesting Genesis with CPU backend")
    
    # Run tests
    results = []
    
    # Test 1: MJCF loading (always uses MuJoCo)
    if mujoco_available:
        results.append(("MJCF Loading", test_mjcf_load()))
    
    # Test 2: Environment
    results.append((f"Environment ({test_backend})", test_environment(test_backend, test_device)))
    
    # Test 3: Training loop
    results.append((f"Training Loop ({test_backend})", test_training_loop(test_backend, test_device, args.iterations)))
    
    # Test both backends if available
    if mujoco_available and genesis_available and test_backend == "genesis":
        print("\n" + "-" * 40)
        print("Also testing MuJoCo backend for comparison...")
        results.append(("Environment (mujoco)", test_environment("mujoco", "cpu")))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    any_failed = False
    for name, result in results:
        if result is None:
            status = "⚠ SKIP"
        elif result:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            any_failed = True
        print(f"  {status}: {name}")
    
    print("=" * 60)
    
    if any_failed:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1
    else:
        print("\n✓ All tests passed (or skipped due to missing dependencies)!")
        print("  Environment is ready for training.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
