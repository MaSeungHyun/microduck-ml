#!/usr/bin/env python3
"""
Smoke test script for Microduck environment

Verifies:
1. MJCF model loads correctly
2. Environment can be created
3. Basic step/reset works
4. Observation/action shapes are correct
5. Rewards are computed
6. A few training iterations work

Usage:
    python scripts/test_env.py --backend mujoco  # CPU test (no GPU required)
    python scripts/test_env.py --backend genesis  # GPU test
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
        model_path = os.path.join(os.path.dirname(__file__), "..", "assets", "microduck.xml")
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
        
        # Test reset
        obs, info = env.reset()
        print(f"  ✓ Reset works, obs shape: {obs.shape}")
        
        # Test step
        import numpy as np
        if backend == "genesis":
            import torch
            actions = torch.zeros((env.num_envs, env.num_actions), device=device)
        else:
            actions = np.zeros((env.num_envs, env.num_actions), dtype=np.float32)
        
        obs, rewards, dones, truncated, info = env.step(actions)
        print(f"  ✓ Step works")
        print(f"    - Obs shape: {obs.shape}")
        print(f"    - Rewards shape: {rewards.shape}")
        print(f"    - Mean reward: {float(rewards.mean()):.4f}")
        
        # Run multiple steps
        for i in range(20):
            if backend == "genesis":
                actions = torch.randn((env.num_envs, env.num_actions), device=device) * 0.1
            else:
                actions = np.random.randn(env.num_envs, env.num_actions).astype(np.float32) * 0.1
            obs, rewards, dones, truncated, info = env.step(actions)
        
        print(f"  ✓ Multiple steps work (20 iterations)")
        
        env.close()
        print(f"  ✓ Environment closed")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        return False


def test_training_loop(backend="mujoco", device="cpu"):
    """Test a few training iterations"""
    print(f"\nTesting training loop (backend={backend}, device={device})...")
    
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        from envs.microduck_env import MicroduckEnv, EnvConfig, CurriculumStage
        
        # Create environment
        cfg = EnvConfig(
            num_envs=4,
            stage=CurriculumStage.STAND,
            episode_length=50,
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
            obs = torch.tensor(obs, device=device)
        
        total_reward = 0
        num_steps = 0
        
        for step in range(50):
            # Get action from policy
            with torch.no_grad():
                actions = policy(obs)
            
            # Step environment
            next_obs, rewards, dones, truncated, info = env.step(actions)
            
            if not isinstance(next_obs, torch.Tensor):
                next_obs = torch.tensor(next_obs, device=device)
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, device=device)
            
            # Simple "training" - maximize rewards
            actions_for_grad = policy(obs)
            loss = -rewards.mean()  # Simplified loss
            
            optimizer.zero_grad()
            # Note: This is a simplified gradient - real PPO is more complex
            policy_output = policy(obs)
            dummy_loss = (policy_output ** 2).mean() * 0.01
            dummy_loss.backward()
            optimizer.step()
            
            total_reward += rewards.sum().item()
            num_steps += env.num_envs
            obs = next_obs
        
        avg_reward = total_reward / num_steps
        print(f"  ✓ Training loop works")
        print(f"    - Steps completed: {num_steps}")
        print(f"    - Average reward: {avg_reward:.4f}")
        
        env.close()
        return True
    except Exception as e:
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
    if args.backend == "genesis" and not genesis_available:
        print("\nGenesis not available, falling back to MuJoCo")
        args.backend = "mujoco"
    
    if args.backend == "mujoco" and not mujoco_available:
        print("\nError: MuJoCo not available. Install with: pip install mujoco")
        return 1
    
    if args.device == "cuda" and not torch_cuda:
        print("\nCUDA not available, falling back to CPU")
        args.device = "cpu"
    
    if args.backend == "genesis" and args.device == "cpu":
        print("\nGenesis CPU mode may be slow, consider using 'cuda' if available")
    
    # Run tests
    results = []
    
    # Test 1: MJCF loading (always uses MuJoCo)
    if mujoco_available:
        results.append(("MJCF Loading", test_mjcf_load()))
    
    # Test 2: Environment
    results.append(("Environment", test_environment(args.backend, args.device)))
    
    # Test 3: Training loop
    results.append(("Training Loop", test_training_loop(args.backend, args.device)))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! Environment is ready for training.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
