"""
Microduck RL Training Environments

Gymnasium-style vectorized environments for training bipedal locomotion policies.
Supports curriculum learning with three stages:
1. Stand/Balance - Learn to maintain upright posture
2. Walk Forward - Learn to walk at commanded velocity
3. Velocity Tracking - Track velocity commands with domain randomization
"""

from envs.microduck_env import MicroduckEnv

__all__ = ["MicroduckEnv"]
