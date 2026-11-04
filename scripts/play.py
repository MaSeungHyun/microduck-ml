#!/usr/bin/env python3
"""
Microduck Policy Evaluation Script

Loads a trained checkpoint and runs the policy in the viewer.

Usage:
    python scripts/play.py --checkpoint runs/microduck_stand_*/model_final.pt
    python scripts/play.py --checkpoint runs/microduck_walk_*/model_final.pt --stage walk
    python scripts/play.py --checkpoint runs/latest/model_final.pt --num-envs 4 --backend mujoco
"""

import argparse
import os
import sys
import time
import glob

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple

from envs.microduck_env import MicroduckEnv, EnvConfig, CurriculumStage


class ActorCritic(nn.Module):
    """Same network architecture as training script"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        for dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ELU(),
            ])
            prev_dim = dim
        self.shared = nn.Sequential(*layers)
        
        self.actor_mean = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ELU(),
            nn.Linear(hidden_dims[-1], action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ELU(),
            nn.Linear(hidden_dims[-1], 1),
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        action_mean, value = self.forward(obs)
        
        if deterministic:
            return action_mean, value
        
        std = torch.exp(self.actor_logstd)
        noise = torch.randn_like(action_mean) * std
        return action_mean + noise, value


def find_latest_checkpoint(runs_dir: str = "runs") -> str:
    """Find the most recent checkpoint"""
    pattern = os.path.join(runs_dir, "**/model_final.pt")
    checkpoints = glob.glob(pattern, recursive=True)
    
    if not checkpoints:
        pattern = os.path.join(runs_dir, "**/model_*.pt")
        checkpoints = glob.glob(pattern, recursive=True)
    
    if not checkpoints:
        return None
    
    # Sort by modification time, get most recent
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained Microduck policy")
    
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: latest in runs/)")
    parser.add_argument("--stage", type=str, default="stand",
                        choices=["stand", "walk", "command"],
                        help="Environment stage")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of environments to visualize")
    parser.add_argument("--backend", type=str, default="genesis",
                        choices=["genesis", "mujoco"],
                        help="Simulation backend")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic actions (no noise)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run (0 for infinite)")
    parser.add_argument("--record", action="store_true",
                        help="Record video (if supported)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("Error: No checkpoint found. Please train first or specify --checkpoint")
            sys.exit(1)
        print(f"Using latest checkpoint: {checkpoint_path}")
    
    # Handle glob patterns
    if '*' in checkpoint_path:
        matches = glob.glob(checkpoint_path)
        if not matches:
            print(f"Error: No files match pattern: {checkpoint_path}")
            sys.exit(1)
        checkpoint_path = max(matches, key=os.path.getmtime)
        print(f"Using: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # Create environment
    stage_map = {
        "stand": CurriculumStage.STAND,
        "walk": CurriculumStage.WALK,
        "command": CurriculumStage.COMMAND,
    }
    
    env_cfg = EnvConfig(
        num_envs=args.num_envs,
        stage=stage_map[args.stage],
        episode_length=2000,  # Longer episodes for evaluation
    )
    
    print(f"Creating environment...")
    print(f"  Stage: {args.stage}")
    print(f"  Backend: {args.backend}")
    print(f"  Num envs: {args.num_envs}")
    
    try:
        env = MicroduckEnv(
            cfg=env_cfg,
            backend=args.backend,
            device=args.device,
            render=True,
            headless=False,
        )
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create policy network
    policy = ActorCritic(
        obs_dim=env.num_obs,
        action_dim=env.num_actions,
        hidden_dims=[256, 128, 64],
    ).to(args.device)
    
    policy.load_state_dict(checkpoint['policy'])
    policy.eval()
    
    print(f"\nStarting evaluation...")
    print(f"  Deterministic: {args.deterministic}")
    print(f"  Episodes: {args.episodes if args.episodes > 0 else 'infinite'}")
    print("\nPress Ctrl+C to stop\n")
    
    # Run evaluation
    episode_count = 0
    episode_rewards = []
    episode_lengths = []
    
    obs, _ = env.reset()
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, device=args.device)
    
    current_reward = torch.zeros(args.num_envs, device=args.device)
    current_length = torch.zeros(args.num_envs, device=args.device)
    
    try:
        while True:
            with torch.no_grad():
                actions, _ = policy.get_action(obs, deterministic=args.deterministic)
            
            obs, rewards, dones, truncated, _ = env.step(actions)
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, device=args.device)
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, device=args.device)
            if not isinstance(dones, torch.Tensor):
                dones = torch.tensor(dones, device=args.device)
            
            current_reward += rewards
            current_length += 1
            
            # Check for episode ends
            done_indices = (dones | truncated).nonzero(as_tuple=True)[0]
            for idx in done_indices:
                episode_rewards.append(current_reward[idx].item())
                episode_lengths.append(current_length[idx].item())
                current_reward[idx] = 0
                current_length[idx] = 0
                episode_count += 1
                
                print(f"Episode {episode_count}: Reward = {episode_rewards[-1]:.2f}, "
                      f"Length = {episode_lengths[-1]:.0f}")
            
            # Render
            env.render()
            
            # Check episode limit
            if args.episodes > 0 and episode_count >= args.episodes:
                break
            
            # Small delay for visualization
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    # Print summary
    if episode_rewards:
        print("\n" + "=" * 50)
        print("Evaluation Summary")
        print("=" * 50)
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
        print(f"Mean length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
        print(f"Max reward: {np.max(episode_rewards):.2f}")
        print(f"Min reward: {np.min(episode_rewards):.2f}")
    
    env.close()


if __name__ == "__main__":
    main()
