"""짧은 REINFORCE로 CartPole 정책을 학습한다.

저장소 루트에서:

    python -m genesis_cartpole.train_reinforce --episodes 200

정책은 이산 행동 2개에 대한 로짓만 내는 작은 MLP다.
에피소드가 끝난 뒤 할인 반환으로 그 에피소드의 로그확률을 한 번에 업데이트한다.
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from genesis_cartpole.env import CartPoleEnv


# 위치는 미터, 각도는 라디안이라 크기가 다르다. 네트워크에 넣기 전에 나눈다.
OBS_SCALE = np.array([2.4, 3.0, 0.21, 3.0], dtype=np.float32)


class Policy(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(obs))


def _discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    returns = []
    discounted = 0.0
    for reward in reversed(rewards):
        discounted = reward + gamma * discounted
        returns.append(discounted)
    returns.reverse()
    tensor = torch.tensor(returns, dtype=torch.float32)
    std = tensor.std(unbiased=False)
    if float(std) > 1e-6:
        tensor = (tensor - tensor.mean()) / (std + 1e-8)
    else:
        tensor = tensor - tensor.mean()
    return tensor


def train(episodes: int, seed: int, learning_rate: float, gamma: float, hidden_size: int, log_every: int) -> None:
    torch.manual_seed(seed)
    env = CartPoleEnv(seed=seed, backend="cpu")
    policy = Policy(hidden_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    recent: List[float] = []

    try:
        for episode in range(1, episodes + 1):
            obs = env.reset()
            log_probs: List[torch.Tensor] = []
            rewards: List[float] = []
            terminated = False
            truncated = False
            while not (terminated or truncated):
                obs_tensor = torch.tensor(obs / OBS_SCALE, dtype=torch.float32)
                distribution = policy(obs_tensor)
                action = distribution.sample()
                log_probs.append(distribution.log_prob(action))
                obs, reward, terminated, truncated, info = env.step(int(action.item()))
                rewards.append(reward)

            returns = _discounted_returns(rewards, gamma)
            loss = -(torch.stack(log_probs) * returns).sum()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            episode_return = float(sum(rewards))
            recent.append(episode_return)
            if len(recent) > log_every:
                del recent[0]
            if episode % log_every == 0 or episode == 1:
                average = sum(recent) / len(recent)
                print(
                    f"episode {episode}: return={episode_return:.1f} "
                    f"avg{log_every}={average:.1f} steps={info['steps']} "
                    f"failure={info['failure']} truncated={truncated}"
                )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis CartPole을 REINFORCE로 학습합니다.")
    parser.add_argument("--episodes", type=int, default=200, help="학습 에피소드 수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes는 1 이상이어야 합니다.")
    train(args.episodes, args.seed, args.lr, args.gamma, args.hidden, args.log_every)


if __name__ == "__main__":
    main()
