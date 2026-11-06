"""보상과 종료를 화면 없이 확인한다.

저장소 루트에서:

    python -m genesis_cartpole.test_cartpole

Genesis 씬은 뷰어 없이 한 번만 만든다. 렌더링 창은 열지 않는다.
"""

from __future__ import annotations

import numpy as np

from genesis_cartpole.env import THETA_THRESHOLD_RADIANS, X_THRESHOLD, CartPoleEnv


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  통과: {message}")


def test_reset_and_reward(env: CartPoleEnv) -> None:
    print("reset과 정상 스텝의 보상")
    obs = env.reset(seed=0)
    _check(obs.shape == (4,), "관측은 4개다")
    _check(obs.dtype == np.float32, "관측 dtype은 float32다")
    _check(bool(np.all(np.isfinite(obs))), "초기 관측이 유한하다")
    _check(bool(np.all(np.abs(obs) <= 0.05 + 1e-5)), "초기 상태는 [-0.05, 0.05] 안이다")

    obs, reward, terminated, truncated, info = env.step(1)
    _check(reward == 1.0, "끝나기 전 보상은 +1이다")
    _check(terminated is False, "곧게 시작한 직후에는 실패로 끝나지 않는다")
    _check(truncated is False, "첫 스텝에 500스텝 제한이 걸리지 않는다")
    _check(info["failure"] is None and info["steps"] == 1, "정보에는 스텝 수와 실패 이유가 있다")
    _check(bool(np.all(np.isfinite(obs))), "스텝 뒤 관측이 유한하다")


def test_pole_termination(env: CartPoleEnv) -> None:
    print("막대가 약 12도를 넘으면 종료")
    env.x_threshold = X_THRESHOLD
    env.theta_threshold_radians = THETA_THRESHOLD_RADIANS
    env.reset(seed=0)
    saw_reward = True
    terminated = False
    truncated = False
    info = {"failure": None, "steps": 0}
    for _ in range(150):
        _, reward, terminated, truncated, info = env.step(1)
        saw_reward = saw_reward and reward == 1.0
        if terminated or truncated:
            break
    _check(saw_reward, "실패로 끝난 스텝을 포함해 보상이 매번 +1이다")
    _check(terminated is True, "막대가 넘어지면 terminated다")
    _check(truncated is False, "막대가 넘어진 에피소드는 500스텝을 채우기 전에 끝난다")
    _check(info["failure"] == "pole", "실패 이유는 막대다")
    _check(1 < info["steps"] < 150, f"몇 걸음 버틴 뒤 넘어진다 (steps={info['steps']})")


def test_cart_termination(env: CartPoleEnv) -> None:
    print("카트가 레일을 벗어나면 종료")
    # 막대 한계를 넓혀 두면, 위치 한계만 남는다.
    env.theta_threshold_radians = 1.0e3
    env.x_threshold = X_THRESHOLD
    env.reset(seed=0)
    terminated = False
    truncated = False
    info = {"failure": None, "steps": 0}
    reward_sum = 0.0
    for _ in range(400):
        _, reward, terminated, truncated, info = env.step(1)
        reward_sum += reward
        if terminated or truncated:
            break
    _check(terminated is True and truncated is False, "레일을 벗어나 실패로 끝난다")
    _check(info["failure"] == "cart", "실패 이유는 카트다")
    _check(reward_sum == float(info["steps"]), "반환값은 버틴 스텝 수와 같다")


def test_time_limit(env: CartPoleEnv) -> None:
    print("500스텝에서 truncated")
    env.theta_threshold_radians = 1.0e6
    env.x_threshold = 1.0e6
    env.reset(seed=0)
    terminated = False
    truncated = False
    info = {"steps": 0}
    for _ in range(499):
        _, reward, terminated, truncated, info = env.step(0)
        if reward != 1.0 or terminated or truncated:
            break
    _check(
        (not terminated) and (not truncated) and info["steps"] == 499,
        "499스텝까지는 시간 제한이 아니다",
    )
    obs, reward, terminated, truncated, info = env.step(0)
    _check(reward == 1.0, "500번째 스텝 보상도 +1이다")
    _check(terminated is False, "한계를 넓히면 500스텝에 실패로 끝나지 않는다")
    _check(truncated is True and info["steps"] == 500, "500스텝에서 truncated다")
    _check(bool(np.all(np.isfinite(obs))), "500스텝 뒤에도 관측이 유한하다")


def test_invalid_action(env: CartPoleEnv) -> None:
    print("이산 행동이 아니면 거절")
    env.reset(seed=0)
    rejected = False
    try:
        env.step(2)
    except ValueError:
        rejected = True
    _check(rejected, "0과 1이 아닌 행동은 ValueError다")


def main() -> None:
    env = CartPoleEnv(render=False, seed=0, backend="cpu")
    try:
        test_reset_and_reward(env)
        test_pole_termination(env)
        test_cart_termination(env)
        test_time_limit(env)
        test_invalid_action(env)
    finally:
        env.close()
    print("모든 검사를 통과했습니다.")


if __name__ == "__main__":
    main()
