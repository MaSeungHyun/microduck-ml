"""무작위 행동으로 CartPole을 몇 에피소드 굴려 본다.

저장소 루트에서:

    python -m genesis_cartpole.rollout --episodes 3
"""

from __future__ import annotations

import argparse

import numpy as np

from genesis_cartpole.env import CartPoleEnv


def run_rollout(episodes: int, seed: int, render: bool) -> None:
    env = CartPoleEnv(render=render, seed=seed)
    rng = np.random.default_rng(seed)
    try:
        for episode in range(episodes):
            obs = env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False
            info = {"failure": None, "steps": 0}
            while not (terminated or truncated):
                action = int(rng.integers(0, 2))
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
            print(
                f"episode {episode + 1}: steps={info['steps']} "
                f"return={total_reward:.1f} failure={info['failure']} "
                f"truncated={truncated} final_obs={np.array2string(obs, precision=3)}"
            )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis CartPole을 무작위 행동으로 굴립니다.")
    parser.add_argument("--episodes", type=int, default=3, help="굴릴 에피소드 수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="뷰어를 연다. 화면이 필요하다.")
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes는 1 이상이어야 합니다.")
    run_rollout(args.episodes, args.seed, args.render)


if __name__ == "__main__":
    main()
