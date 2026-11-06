"""
Genesis CartPole 환경.

Gymnasium에 들어 있는 CartPole-v1을 불러 쓰지 않는다.
물리 엔진은 다음 상태만 계산하고, 관측을 고르고 보상을 매기고
끝났는지 판단하는 코드는 이 파일이 직접 가지고 있다.

호출 형태:
    obs = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

action: 0이면 카트를 왼쪽(-x)으로, 1이면 오른쪽(+x)으로 민다.
obs: [카트 위치, 카트 속도, 막대 각도, 막대 각속도]
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import genesis as gs
    import torch
except ImportError as exc:
    raise ImportError(
        "Genesis를 불러오지 못했습니다. torch를 먼저 설치한 뒤 genesis-world를 설치하세요. "
        "예: pip install torch --index-url https://download.pytorch.org/whl/cpu && "
        'pip install "genesis-world==1.4.5"'
    ) from exc


# CartPole-v1과 같은 종료 규칙.
X_THRESHOLD = 2.4
THETA_THRESHOLD_RADIANS = 12.0 * math.pi / 180.0
MAX_EPISODE_STEPS = 500
FORCE_MAGNITUDE = 10.0
INITIAL_STATE_RANGE = 0.05

_ASSET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "cartpole.xml")


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


class CartPoleEnv:
    """카트-막대 하나를 다루는 Gym 형태의 환경."""

    def __init__(
        self,
        render: bool = False,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        x_threshold: float = X_THRESHOLD,
        theta_threshold_radians: float = THETA_THRESHOLD_RADIANS,
        force_magnitude: float = FORCE_MAGNITUDE,
        initial_state_range: float = INITIAL_STATE_RANGE,
        seed: Optional[int] = None,
        dt: float = 0.02,
        gravity: float = 9.8,
        backend: str = "cpu",
    ):
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps는 1 이상이어야 합니다.")
        if backend not in ("cpu", "gpu"):
            raise ValueError('backend는 "cpu" 또는 "gpu"여야 합니다.')

        self.max_episode_steps = int(max_episode_steps)
        self.x_threshold = float(x_threshold)
        self.theta_threshold_radians = float(theta_threshold_radians)
        self.force_magnitude = float(force_magnitude)
        self.initial_state_range = float(initial_state_range)
        self.dt = float(dt)
        self.gravity = float(gravity)
        self.render_enabled = bool(render)
        self.backend = backend
        self.rng = np.random.default_rng(seed)
        self.steps = 0

        self._init_genesis()

    def _init_genesis(self) -> None:
        """씬을 한 번 만들고, 이후에는 reset/step만 반복한다."""
        if not gs._initialized:
            gs.init(
                backend=gs.cpu if self.backend == "cpu" else gs.gpu,
                precision="32",
                logging_level="warning",
            )

        viewer_options = None
        if self.render_enabled:
            viewer_options = gs.options.ViewerOptions(
                camera_pos=(0.0, -3.2, 1.3),
                camera_lookat=(0.0, 0.0, 0.4),
                camera_fov=40,
                max_FPS=60,
            )

        # 뷰어를 꺼도 Genesis는 오프스크린 OpenGL(EGL)로 씬을 만든다.
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                gravity=(0.0, 0.0, -self.gravity),
            ),
            viewer_options=viewer_options,
            show_viewer=self.render_enabled,
        )
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=_ASSET_PATH))
        try:
            self.scene.build()
        except Exception as exc:
            message = str(exc)
            if "EGL" in message or "OpenGL" in message:
                raise RuntimeError(
                    "Genesis가 화면 없는 씬을 만들 때도 EGL이 필요합니다. "
                    "Linux에서는 `sudo apt-get install -y libegl1 libgl1-mesa-dri` 를 실행하세요. "
                    "GPU가 없으면 앞에 `LIBGL_ALWAYS_SOFTWARE=1` 을 붙여 다시 실행해 보세요."
                ) from exc
            raise

        self.slider_dof = self.robot.get_joint("slider").dofs_idx_local
        self.hinge_dof = self.robot.get_joint("hinge").dofs_idx_local

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """에피소드를 새로 시작하고 관측 4개를 반환한다."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.scene.reset()
        # Gymnasium CartPole-v1과 같이 네 성분을 작은 구간에서 뽑는다.
        # 순서는 관측과 같다: 위치, 속도, 각도, 각속도.
        sample = self.rng.uniform(
            -self.initial_state_range,
            self.initial_state_range,
            size=(4,),
        ).astype(np.float32)
        # 조인트 순서는 [슬라이더, 힌지]라서 관측 순서와 다르다.
        self.robot.set_dofs_position(
            torch.tensor([sample[0], sample[2]], dtype=torch.float32),
            zero_velocity=False,
        )
        self.robot.set_dofs_velocity(
            torch.tensor([sample[1], sample[3]], dtype=torch.float32),
        )
        self.steps = 0
        return self._observe()

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """행동 하나를 적용하고 (관측, 보상, 실패 종료, 시간 종료, 정보)를 반환한다."""
        discrete = self._parse_action(action)
        # 0은 왼쪽, 1은 오른쪽. 힘의 크기는 CartPole-v1의 10N.
        force = self.force_magnitude if discrete == 1 else -self.force_magnitude
        self.robot.control_dofs_force(
            torch.tensor([force], dtype=torch.float32),
            self.slider_dof,
        )
        self.scene.step()
        self.steps += 1

        obs = self._observe()
        x = float(obs[0])
        theta = float(obs[2])
        # 물리 엔진은 이 한계를 모른다. 끝났는지는 여기서 정한다.
        pole_fallen = abs(theta) > self.theta_threshold_radians
        cart_left_track = abs(x) > self.x_threshold
        terminated = bool(pole_fallen or cart_left_track)
        truncated = bool(self.steps >= self.max_episode_steps)

        if pole_fallen and cart_left_track:
            failure: Optional[str] = "pole_and_cart"
        elif pole_fallen:
            failure = "pole"
        elif cart_left_track:
            failure = "cart"
        else:
            failure = None

        # 실패로 끝난 스텝도 +1이다. 그래서 에피소드 반환값은 버틴 스텝 수와 같다.
        reward = 1.0
        info = {"steps": self.steps, "failure": failure}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        scene = getattr(self, "scene", None)
        if scene is not None:
            scene.destroy()
            self.scene = None

    def _observe(self) -> np.ndarray:
        qpos = _as_numpy(self.robot.get_qpos())
        qvel = _as_numpy(self.robot.get_dofs_velocity())
        return np.array([qpos[0], qvel[0], qpos[1], qvel[1]], dtype=np.float32)

    @staticmethod
    def _parse_action(action: Any) -> int:
        if isinstance(action, torch.Tensor):
            if action.numel() != 1:
                raise ValueError("action은 0 또는 1 하나여야 합니다.")
            value = float(action.detach().cpu().reshape(-1)[0].item())
        elif isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError("action은 0 또는 1 하나여야 합니다.")
            value = float(action.reshape(-1)[0])
        else:
            value = float(action)
        if value not in (0.0, 1.0):
            raise ValueError("action은 0(왼쪽) 또는 1(오른쪽)이어야 합니다.")
        return int(value)
