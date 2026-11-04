"""
Microduck Bipedal Robot Environment

A Gymnasium-style vectorized environment for training a small bipedal robot.

Curriculum Stages:
1. STAND - Learn to balance and maintain upright posture
2. WALK - Walk forward at commanded velocity
3. COMMAND - Track velocity commands with full domain randomization

Inspired by:
- Open Duck Mini v2: https://github.com/apirrone/Open_Duck_Mini (Apache 2.0)
- Open Duck Playground: https://github.com/apirrone/Open_Duck_Playground
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import os

try:
    import genesis as gs
    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False


class CurriculumStage(Enum):
    """Training curriculum stages"""
    STAND = 1    # Balance in place
    WALK = 2     # Walk forward
    COMMAND = 3  # Velocity command tracking with domain randomization


@dataclass
class EnvConfig:
    """Environment configuration"""
    # Simulation
    num_envs: int = 1024
    episode_length: int = 1000
    dt: float = 0.02  # Control frequency: 50Hz
    sim_dt: float = 0.002  # Simulation timestep
    
    # Curriculum
    stage: CurriculumStage = CurriculumStage.STAND
    
    # Velocity commands (for WALK and COMMAND stages)
    cmd_vel_x_range: Tuple[float, float] = (0.0, 0.5)  # m/s
    cmd_vel_y_range: Tuple[float, float] = (-0.2, 0.2)
    cmd_vel_yaw_range: Tuple[float, float] = (-0.5, 0.5)  # rad/s
    command_resample_time: float = 5.0  # Resample commands every N seconds
    
    # Domain randomization (for COMMAND stage)
    enable_domain_rand: bool = False
    mass_rand_range: Tuple[float, float] = (0.9, 1.1)
    friction_rand_range: Tuple[float, float] = (0.6, 1.0)
    
    # Push perturbations
    enable_push: bool = False
    push_interval: float = 5.0  # seconds
    push_force_range: Tuple[float, float] = (5.0, 20.0)  # Newtons
    
    # Observation noise
    joint_pos_noise: float = 0.01
    joint_vel_noise: float = 0.1
    imu_noise: float = 0.05
    
    # Reward weights
    reward_scales: Dict[str, float] = field(default_factory=lambda: {
        "alive": 2.0,
        "upright": 1.0,
        "height": 1.0,
        "velocity_tracking": 2.0,
        "angular_velocity_tracking": 0.5,
        "action_rate": -0.01,
        "joint_torque": -0.0001,
        "foot_contact": 0.1,
        "symmetry": 0.2,
    })
    
    # Termination
    min_height: float = 0.1  # Terminate if trunk below this height
    max_pitch: float = 1.0   # Terminate if trunk tilted too much
    max_roll: float = 1.0
    
    # Assets
    robot_xml: str = "assets/microduck.xml"


class MicroduckEnv:
    """
    Gymnasium-style vectorized environment for Microduck bipedal robot.
    
    Supports both Genesis (GPU-accelerated) and MuJoCo backends.
    Default uses Genesis for parallel training on GPU.
    """
    
    # Joint configuration (same as Open Duck Mini v2)
    JOINT_NAMES = [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    NUM_JOINTS = 10
    
    # Default standing pose (slightly bent knees for stability)
    DEFAULT_JOINT_POS = np.array([
        0.0, 0.0, 0.3, -0.6, 0.3,   # Left leg: yaw, roll, pitch, knee, ankle
        0.0, 0.0, 0.3, -0.6, 0.3,   # Right leg
    ], dtype=np.float32)
    
    def __init__(
        self,
        cfg: Optional[EnvConfig] = None,
        backend: str = "genesis",
        device: str = "cuda",
        render: bool = False,
        headless: bool = True,
    ):
        """
        Initialize the Microduck environment.
        
        Args:
            cfg: Environment configuration
            backend: "genesis" or "mujoco"
            device: "cuda" or "cpu"
            render: Enable rendering
            headless: Run without display (for training)
        """
        self.cfg = cfg or EnvConfig()
        self.backend = backend.lower()
        self.device = device
        self.render_enabled = render
        self.headless = headless
        
        # Adjust for curriculum stage
        self._setup_curriculum()
        
        # Observation and action dimensions
        # Obs: joint_pos(10) + joint_vel(10) + base_ang_vel(3) + gravity_vec(3) + 
        #      commands(3) + last_actions(10) + phase(2)
        self.num_obs = 10 + 10 + 3 + 3 + 3 + 10 + 2  # = 41
        self.num_actions = self.NUM_JOINTS
        
        # State buffers (will be torch tensors on GPU or numpy on CPU)
        self.num_envs = self.cfg.num_envs
        
        # Initialize backend
        if self.backend == "genesis":
            self._init_genesis()
        else:
            self._init_mujoco()
        
        # Initialize buffers
        self._init_buffers()
    
    def _setup_curriculum(self):
        """Configure environment based on curriculum stage"""
        stage = self.cfg.stage
        
        if stage == CurriculumStage.STAND:
            # Stage 1: Just stand and balance
            self.cfg.cmd_vel_x_range = (0.0, 0.0)
            self.cfg.cmd_vel_y_range = (0.0, 0.0)
            self.cfg.cmd_vel_yaw_range = (0.0, 0.0)
            self.cfg.enable_domain_rand = False
            self.cfg.enable_push = False
            
        elif stage == CurriculumStage.WALK:
            # Stage 2: Walk forward at moderate speed
            self.cfg.cmd_vel_x_range = (0.2, 0.4)
            self.cfg.cmd_vel_y_range = (0.0, 0.0)
            self.cfg.cmd_vel_yaw_range = (0.0, 0.0)
            self.cfg.enable_domain_rand = False
            self.cfg.enable_push = False
            
        elif stage == CurriculumStage.COMMAND:
            # Stage 3: Full velocity tracking with domain randomization
            self.cfg.cmd_vel_x_range = (-0.2, 0.5)
            self.cfg.cmd_vel_y_range = (-0.2, 0.2)
            self.cfg.cmd_vel_yaw_range = (-0.5, 0.5)
            self.cfg.enable_domain_rand = True
            self.cfg.enable_push = True
    
    def _init_genesis(self):
        """Initialize Genesis backend for GPU-accelerated simulation"""
        if not GENESIS_AVAILABLE:
            raise ImportError("Genesis not available. Install with: pip install genesis-world")
        
        # Initialize Genesis
        gs.init(
            backend=gs.gpu if self.device == "cuda" else gs.cpu,
            precision="32",
            logging_level="warning",
        )
        
        # Create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.cfg.sim_dt,
                substeps=2,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                camera_pos=(2.0, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.3),
            ) if self.render_enabled else None,
            show_viewer=self.render_enabled and not self.headless,
        )
        
        # Add ground plane
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        
        # Load robot
        robot_path = os.path.join(os.path.dirname(__file__), "..", self.cfg.robot_xml)
        if not os.path.exists(robot_path):
            robot_path = self.cfg.robot_xml
        
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=robot_path),
        )
        
        # Build scene with parallel environments
        self.scene.build(n_envs=self.num_envs)
        
        # Get joint indices
        self.joint_indices = []
        for name in self.JOINT_NAMES:
            try:
                idx = self.robot.get_joint(name).dof_idx_local
                self.joint_indices.append(idx)
            except:
                pass
        
        self._backend_initialized = True
    
    def _init_mujoco(self):
        """Initialize MuJoCo backend (CPU-based, for debugging)"""
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo not available. Install with: pip install mujoco")
        
        robot_path = os.path.join(os.path.dirname(__file__), "..", self.cfg.robot_xml)
        if not os.path.exists(robot_path):
            robot_path = self.cfg.robot_xml
            
        self.mj_model = mujoco.MjModel.from_xml_path(robot_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # MuJoCo doesn't support vectorized envs natively, 
        # so we simulate sequentially or use a list
        self.mj_models = [self.mj_model]
        self.mj_datas = [mujoco.MjData(self.mj_model) for _ in range(self.num_envs)]
        
        if self.render_enabled:
            self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_datas[0])
        
        self._backend_initialized = True
    
    def _init_buffers(self):
        """Initialize state buffers"""
        import torch
        
        self.use_torch = self.backend == "genesis" and self.device == "cuda"
        
        if self.use_torch:
            device = torch.device(self.device)
            self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=device)
            self.actions = torch.zeros((self.num_envs, self.num_actions), device=device)
            self.last_actions = torch.zeros((self.num_envs, self.num_actions), device=device)
            self.rewards = torch.zeros(self.num_envs, device=device)
            self.dones = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self.episode_length = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
            
            # Velocity commands
            self.commands = torch.zeros((self.num_envs, 3), device=device)
            self.command_time = torch.zeros(self.num_envs, device=device)
            
            # Phase for gait timing (sin/cos representation)
            self.phase = torch.zeros(self.num_envs, device=device)
            self.gait_freq = 2.0  # Hz
            
            # Push timing
            self.push_time = torch.zeros(self.num_envs, device=device)
        else:
            # Numpy fallback for MuJoCo/CPU
            self.obs_buf = np.zeros((self.num_envs, self.num_obs), dtype=np.float32)
            self.actions = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
            self.last_actions = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
            self.rewards = np.zeros(self.num_envs, dtype=np.float32)
            self.dones = np.zeros(self.num_envs, dtype=bool)
            self.episode_length = np.zeros(self.num_envs, dtype=np.int32)
            self.commands = np.zeros((self.num_envs, 3), dtype=np.float32)
            self.command_time = np.zeros(self.num_envs, dtype=np.float32)
            self.phase = np.zeros(self.num_envs, dtype=np.float32)
            self.gait_freq = 2.0
            self.push_time = np.zeros(self.num_envs, dtype=np.float32)
    
    def reset(self, env_ids=None):
        """
        Reset environments.
        
        Args:
            env_ids: Optional indices of environments to reset. If None, reset all.
            
        Returns:
            obs: Observation tensor/array
            info: Additional info dict
        """
        if env_ids is None:
            env_ids = range(self.num_envs) if not self.use_torch else None
        
        if self.backend == "genesis":
            self._reset_genesis(env_ids)
        else:
            self._reset_mujoco(env_ids)
        
        # Resample commands
        self._resample_commands(env_ids)
        
        # Reset phase
        if self.use_torch:
            import torch
            if env_ids is None:
                self.phase.zero_()
            else:
                self.phase[env_ids] = 0
        else:
            if env_ids is None:
                self.phase[:] = 0
            else:
                self.phase[env_ids] = 0
        
        # Compute initial observation
        self._compute_observations()
        
        return self.obs_buf, {}
    
    def _reset_genesis(self, env_ids):
        """Reset Genesis environments"""
        import torch
        
        if env_ids is None:
            # Reset all
            self.robot.set_dofs_position(
                torch.tensor(self.DEFAULT_JOINT_POS, device=self.device).unsqueeze(0).expand(self.num_envs, -1),
            )
            self.robot.set_dofs_velocity(torch.zeros((self.num_envs, self.NUM_JOINTS), device=self.device))
            
            # Reset base position/orientation
            init_pos = torch.tensor([[0.0, 0.0, 0.25]], device=self.device).expand(self.num_envs, -1)
            init_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
            self.robot.set_pos(init_pos)
            self.robot.set_quat(init_quat)
            
            self.episode_length.zero_()
            self.last_actions.zero_()
        else:
            env_ids_tensor = torch.tensor(env_ids, device=self.device)
            n = len(env_ids)
            
            self.robot.set_dofs_position(
                torch.tensor(self.DEFAULT_JOINT_POS, device=self.device).unsqueeze(0).expand(n, -1),
                envs_idx=env_ids_tensor,
            )
            self.robot.set_dofs_velocity(
                torch.zeros((n, self.NUM_JOINTS), device=self.device),
                envs_idx=env_ids_tensor,
            )
            
            init_pos = torch.tensor([[0.0, 0.0, 0.25]], device=self.device).expand(n, -1)
            init_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(n, -1)
            self.robot.set_pos(init_pos, envs_idx=env_ids_tensor)
            self.robot.set_quat(init_quat, envs_idx=env_ids_tensor)
            
            self.episode_length[env_ids_tensor] = 0
            self.last_actions[env_ids_tensor] = 0
    
    def _reset_mujoco(self, env_ids):
        """Reset MuJoCo environments"""
        if env_ids is None:
            env_ids = range(self.num_envs)
        
        for i in env_ids:
            mujoco.mj_resetData(self.mj_model, self.mj_datas[i])
            # Set initial joint positions
            self.mj_datas[i].qpos[:7] = [0, 0, 0.25, 1, 0, 0, 0]  # Free joint
            self.mj_datas[i].qpos[7:7+self.NUM_JOINTS] = self.DEFAULT_JOINT_POS
            mujoco.mj_forward(self.mj_model, self.mj_datas[i])
            
            self.episode_length[i] = 0
            self.last_actions[i] = 0
    
    def _resample_commands(self, env_ids=None):
        """Resample velocity commands for specified environments"""
        import torch
        
        if self.use_torch:
            if env_ids is None:
                n = self.num_envs
                # X velocity (forward/backward)
                self.commands[:, 0] = torch.empty(n, device=self.device).uniform_(
                    self.cfg.cmd_vel_x_range[0], self.cfg.cmd_vel_x_range[1]
                )
                # Y velocity (lateral)
                self.commands[:, 1] = torch.empty(n, device=self.device).uniform_(
                    self.cfg.cmd_vel_y_range[0], self.cfg.cmd_vel_y_range[1]
                )
                # Yaw velocity (turning)
                self.commands[:, 2] = torch.empty(n, device=self.device).uniform_(
                    self.cfg.cmd_vel_yaw_range[0], self.cfg.cmd_vel_yaw_range[1]
                )
                self.command_time.zero_()
            else:
                env_ids_tensor = torch.tensor(env_ids, device=self.device)
                n = len(env_ids)
                self.commands[env_ids_tensor, 0] = torch.empty(n, device=self.device).uniform_(
                    self.cfg.cmd_vel_x_range[0], self.cfg.cmd_vel_x_range[1]
                )
                self.commands[env_ids_tensor, 1] = torch.empty(n, device=self.device).uniform_(
                    self.cfg.cmd_vel_y_range[0], self.cfg.cmd_vel_y_range[1]
                )
                self.commands[env_ids_tensor, 2] = torch.empty(n, device=self.device).uniform_(
                    self.cfg.cmd_vel_yaw_range[0], self.cfg.cmd_vel_yaw_range[1]
                )
                self.command_time[env_ids_tensor] = 0
        else:
            if env_ids is None:
                env_ids = range(self.num_envs)
            for i in env_ids:
                self.commands[i, 0] = np.random.uniform(*self.cfg.cmd_vel_x_range)
                self.commands[i, 1] = np.random.uniform(*self.cfg.cmd_vel_y_range)
                self.commands[i, 2] = np.random.uniform(*self.cfg.cmd_vel_yaw_range)
                self.command_time[i] = 0
    
    def step(self, actions):
        """
        Step the environment.
        
        Args:
            actions: Joint position targets, shape (num_envs, num_actions)
            
        Returns:
            obs: Observations
            rewards: Reward values
            dones: Termination flags
            truncated: Truncation flags (episode length exceeded)
            info: Additional info
        """
        import torch
        
        # Handle torch tensor inputs when using numpy backend
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = actions
        
        # Clip and scale actions
        if self.use_torch:
            self.actions = torch.clamp(actions if isinstance(actions, torch.Tensor) else torch.tensor(actions, device=self.device), -1.0, 1.0)
            # Scale actions to joint range
            action_scale = torch.tensor([0.5, 0.4, 1.2, 2.0, 0.8] * 2, device=self.device)
            default_pos = torch.tensor(self.DEFAULT_JOINT_POS, device=self.device)
            target_positions = default_pos + self.actions * action_scale
        else:
            self.actions = np.clip(actions_np, -1.0, 1.0)
            action_scale = np.array([0.5, 0.4, 1.2, 2.0, 0.8] * 2, dtype=np.float32)
            target_positions = self.DEFAULT_JOINT_POS + self.actions * action_scale
        
        # Step simulation
        if self.backend == "genesis":
            self._step_genesis(target_positions)
        else:
            self._step_mujoco(target_positions)
        
        # Update phase for gait timing
        if self.use_torch:
            self.phase += 2 * np.pi * self.gait_freq * self.cfg.dt
            self.phase = torch.fmod(self.phase, 2 * np.pi)
        else:
            self.phase += 2 * np.pi * self.gait_freq * self.cfg.dt
            self.phase = np.fmod(self.phase, 2 * np.pi)
        
        # Apply push perturbations
        if self.cfg.enable_push:
            self._apply_push()
        
        # Resample commands periodically
        if self.use_torch:
            self.command_time += self.cfg.dt
            resample_mask = self.command_time > self.cfg.command_resample_time
            if resample_mask.any():
                resample_ids = torch.where(resample_mask)[0]
                self._resample_commands(resample_ids.tolist())
        else:
            self.command_time += self.cfg.dt
            resample_ids = np.where(self.command_time > self.cfg.command_resample_time)[0]
            if len(resample_ids) > 0:
                self._resample_commands(resample_ids)
        
        # Compute observations and rewards
        self._compute_observations()
        self._compute_rewards()
        self._check_termination()
        
        # Update episode length and check truncation
        self.episode_length += 1
        if self.use_torch:
            truncated = self.episode_length >= self.cfg.episode_length
        else:
            truncated = self.episode_length >= self.cfg.episode_length
        
        # Save last actions
        if self.use_torch:
            self.last_actions = self.actions.clone()
        else:
            self.last_actions = self.actions.copy()
        
        # Reset done environments
        if self.use_torch:
            reset_ids = torch.where(self.dones | truncated)[0]
            if len(reset_ids) > 0:
                self.reset(reset_ids.tolist())
        else:
            reset_ids = np.where(self.dones | truncated)[0]
            if len(reset_ids) > 0:
                self.reset(reset_ids.tolist())
        
        return self.obs_buf, self.rewards, self.dones, truncated, {}
    
    def _step_genesis(self, target_positions):
        """Step Genesis simulation"""
        self.robot.control_dofs_position(target_positions)
        
        # Step multiple times for control frequency
        substeps = int(self.cfg.dt / self.cfg.sim_dt)
        for _ in range(substeps):
            self.scene.step()
    
    def _step_mujoco(self, target_positions):
        """Step MuJoCo simulation"""
        for i in range(self.num_envs):
            self.mj_datas[i].ctrl[:] = target_positions[i]
            substeps = int(self.cfg.dt / self.cfg.sim_dt)
            for _ in range(substeps):
                mujoco.mj_step(self.mj_model, self.mj_datas[i])
        
        if self.render_enabled and hasattr(self, 'mj_viewer'):
            self.mj_viewer.sync()
    
    def _apply_push(self):
        """Apply random push perturbations to robot"""
        import torch
        
        if self.use_torch:
            self.push_time += self.cfg.dt
            push_mask = self.push_time > self.cfg.push_interval
            
            if push_mask.any():
                push_ids = torch.where(push_mask)[0]
                n = len(push_ids)
                
                # Random push force
                force_mag = torch.empty(n, device=self.device).uniform_(
                    self.cfg.push_force_range[0], self.cfg.push_force_range[1]
                )
                angle = torch.empty(n, device=self.device).uniform_(0, 2 * np.pi)
                
                # Apply horizontal push
                forces = torch.zeros((n, 6), device=self.device)
                forces[:, 0] = force_mag * torch.cos(angle)
                forces[:, 1] = force_mag * torch.sin(angle)
                
                # Apply impulse (simplified - actual implementation depends on Genesis API)
                # self.robot.apply_external_force(forces, envs_idx=push_ids)
                
                self.push_time[push_ids] = 0
    
    def _compute_observations(self):
        """Compute observation vector"""
        import torch
        
        if self.backend == "genesis":
            # Get robot state from Genesis
            joint_pos = self.robot.get_dofs_position()  # (num_envs, num_joints)
            joint_vel = self.robot.get_dofs_velocity()
            base_quat = self.robot.get_quat()  # (num_envs, 4) - wxyz
            base_ang_vel = self.robot.get_ang()  # Angular velocity
            
            # Gravity vector in body frame
            gravity_world = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float32)
            gravity_body = self._quat_rotate_inverse(base_quat, gravity_world.unsqueeze(0).expand(self.num_envs, -1))
            
            # Phase encoding (sin/cos for gait timing)
            phase_sin = torch.sin(self.phase).unsqueeze(1)
            phase_cos = torch.cos(self.phase).unsqueeze(1)
            
            # Add observation noise
            if self.cfg.joint_pos_noise > 0:
                joint_pos = joint_pos + torch.randn_like(joint_pos) * self.cfg.joint_pos_noise
            if self.cfg.joint_vel_noise > 0:
                joint_vel = joint_vel + torch.randn_like(joint_vel) * self.cfg.joint_vel_noise
            
            # Concatenate observations
            self.obs_buf = torch.cat([
                joint_pos,           # 10
                joint_vel,           # 10
                base_ang_vel,        # 3
                gravity_body,        # 3
                self.commands,       # 3
                self.last_actions,   # 10
                phase_sin,           # 1
                phase_cos,           # 1
            ], dim=1)
        
        else:
            # MuJoCo observation (CPU)
            for i in range(self.num_envs):
                d = self.mj_datas[i]
                joint_pos = d.qpos[7:7+self.NUM_JOINTS]
                joint_vel = d.qvel[6:6+self.NUM_JOINTS]
                base_quat = d.qpos[3:7]  # MuJoCo uses wxyz
                base_ang_vel = d.qvel[3:6]
                
                # Gravity in body frame
                gravity_body = self._quat_rotate_inverse_np(base_quat, np.array([0, 0, -1]))
                
                phase_sin = np.sin(self.phase[i])
                phase_cos = np.cos(self.phase[i])
                
                # Add noise
                joint_pos = joint_pos + np.random.randn(self.NUM_JOINTS) * self.cfg.joint_pos_noise
                joint_vel = joint_vel + np.random.randn(self.NUM_JOINTS) * self.cfg.joint_vel_noise
                
                self.obs_buf[i] = np.concatenate([
                    joint_pos, joint_vel, base_ang_vel, gravity_body,
                    self.commands[i], self.last_actions[i],
                    [phase_sin, phase_cos]
                ])
    
    def _compute_rewards(self):
        """Compute reward for each environment"""
        import torch
        
        scales = self.cfg.reward_scales
        
        if self.backend == "genesis":
            # Get state
            base_pos = self.robot.get_pos()
            base_quat = self.robot.get_quat()
            base_lin_vel = self.robot.get_vel()
            base_ang_vel = self.robot.get_ang()
            joint_vel = self.robot.get_dofs_velocity()
            
            # Alive reward - encourage survival
            r_alive = scales["alive"] * torch.ones(self.num_envs, device=self.device)
            
            # Upright reward - penalize tilting
            gravity_body = self._quat_rotate_inverse(
                base_quat, 
                torch.tensor([0, 0, -1], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            )
            r_upright = scales["upright"] * (gravity_body[:, 2] + 1) / 2  # 0 to 1
            
            # Height reward - maintain target height
            target_height = 0.22
            height_error = torch.abs(base_pos[:, 2] - target_height)
            r_height = scales["height"] * torch.exp(-10 * height_error)
            
            # Velocity tracking reward
            vel_error_x = torch.square(base_lin_vel[:, 0] - self.commands[:, 0])
            vel_error_y = torch.square(base_lin_vel[:, 1] - self.commands[:, 1])
            r_vel = scales["velocity_tracking"] * torch.exp(-5 * (vel_error_x + vel_error_y))
            
            # Angular velocity tracking
            yaw_rate_error = torch.square(base_ang_vel[:, 2] - self.commands[:, 2])
            r_ang_vel = scales["angular_velocity_tracking"] * torch.exp(-5 * yaw_rate_error)
            
            # Action rate penalty - smooth actions
            action_diff = self.actions - self.last_actions
            r_action_rate = scales["action_rate"] * torch.sum(torch.square(action_diff), dim=1)
            
            # Joint torque penalty - energy efficiency
            r_torque = scales["joint_torque"] * torch.sum(torch.square(joint_vel), dim=1)
            
            # Total reward
            self.rewards = r_alive + r_upright + r_height + r_vel + r_ang_vel + r_action_rate + r_torque
        
        else:
            # MuJoCo rewards (CPU)
            for i in range(self.num_envs):
                d = self.mj_datas[i]
                
                base_pos = d.qpos[:3]
                base_quat = d.qpos[3:7]
                base_lin_vel = d.qvel[:3]
                base_ang_vel = d.qvel[3:6]
                joint_vel = d.qvel[6:6+self.NUM_JOINTS]
                
                # Alive
                r_alive = scales["alive"]
                
                # Upright
                gravity_body = self._quat_rotate_inverse_np(base_quat, np.array([0, 0, -1]))
                r_upright = scales["upright"] * (gravity_body[2] + 1) / 2
                
                # Height
                height_error = abs(base_pos[2] - 0.22)
                r_height = scales["height"] * np.exp(-10 * height_error)
                
                # Velocity tracking
                vel_error = (base_lin_vel[0] - self.commands[i, 0])**2 + (base_lin_vel[1] - self.commands[i, 1])**2
                r_vel = scales["velocity_tracking"] * np.exp(-5 * vel_error)
                
                # Angular velocity
                yaw_error = (base_ang_vel[2] - self.commands[i, 2])**2
                r_ang_vel = scales["angular_velocity_tracking"] * np.exp(-5 * yaw_error)
                
                # Action rate
                action_diff = self.actions[i] - self.last_actions[i]
                r_action_rate = scales["action_rate"] * np.sum(action_diff**2)
                
                # Torque
                r_torque = scales["joint_torque"] * np.sum(joint_vel**2)
                
                self.rewards[i] = r_alive + r_upright + r_height + r_vel + r_ang_vel + r_action_rate + r_torque
    
    def _check_termination(self):
        """Check termination conditions"""
        import torch
        
        if self.backend == "genesis":
            base_pos = self.robot.get_pos()
            base_quat = self.robot.get_quat()
            
            # Height check
            height_terminate = base_pos[:, 2] < self.cfg.min_height
            
            # Orientation check (roll/pitch)
            gravity_body = self._quat_rotate_inverse(
                base_quat,
                torch.tensor([0, 0, -1], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            )
            roll = torch.atan2(gravity_body[:, 1], gravity_body[:, 2])
            pitch = torch.atan2(gravity_body[:, 0], gravity_body[:, 2])
            
            orientation_terminate = (torch.abs(roll) > self.cfg.max_roll) | (torch.abs(pitch) > self.cfg.max_pitch)
            
            self.dones = height_terminate | orientation_terminate
        
        else:
            # MuJoCo termination
            for i in range(self.num_envs):
                d = self.mj_datas[i]
                base_pos = d.qpos[:3]
                base_quat = d.qpos[3:7]
                
                # Height
                if base_pos[2] < self.cfg.min_height:
                    self.dones[i] = True
                    continue
                
                # Orientation
                gravity_body = self._quat_rotate_inverse_np(base_quat, np.array([0, 0, -1]))
                roll = np.arctan2(gravity_body[1], gravity_body[2])
                pitch = np.arctan2(gravity_body[0], gravity_body[2])
                
                if abs(roll) > self.cfg.max_roll or abs(pitch) > self.cfg.max_pitch:
                    self.dones[i] = True
                else:
                    self.dones[i] = False
    
    def _quat_rotate_inverse(self, q, v):
        """Rotate vector by inverse of quaternion (wxyz format)"""
        import torch
        
        q_w = q[:, 0:1]
        q_vec = q[:, 1:4]
        
        a = v * (2.0 * q_w ** 2 - 1.0)
        b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
        c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
        
        return a - b + c
    
    def _quat_rotate_inverse_np(self, q, v):
        """Numpy version of quaternion rotation (wxyz format)"""
        q_w = q[0]
        q_vec = q[1:4]
        
        a = v * (2.0 * q_w ** 2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0
        
        return a - b + c
    
    def render(self):
        """Render the environment"""
        if self.backend == "genesis" and self.render_enabled:
            self.scene.viewer.update()
        elif self.backend == "mujoco" and hasattr(self, 'mj_viewer'):
            self.mj_viewer.sync()
    
    def close(self):
        """Clean up resources"""
        if self.backend == "mujoco" and hasattr(self, 'mj_viewer'):
            self.mj_viewer.close()
    
    @property
    def observation_space(self):
        """Gymnasium-compatible observation space"""
        import gymnasium as gym
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.num_obs,), 
            dtype=np.float32
        )
    
    @property
    def action_space(self):
        """Gymnasium-compatible action space"""
        import gymnasium as gym
        return gym.spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.num_actions,), 
            dtype=np.float32
        )


def make_env(
    stage: str = "stand",
    num_envs: int = 1024,
    backend: str = "genesis",
    device: str = "cuda",
    render: bool = False,
    headless: bool = True,
) -> MicroduckEnv:
    """
    Factory function to create Microduck environment.
    
    Args:
        stage: Curriculum stage - "stand", "walk", or "command"
        num_envs: Number of parallel environments
        backend: "genesis" (GPU) or "mujoco" (CPU)
        device: "cuda" or "cpu"
        render: Enable rendering
        headless: Run without display
    
    Returns:
        MicroduckEnv instance
    """
    stage_map = {
        "stand": CurriculumStage.STAND,
        "walk": CurriculumStage.WALK,
        "command": CurriculumStage.COMMAND,
    }
    
    cfg = EnvConfig(
        num_envs=num_envs,
        stage=stage_map.get(stage.lower(), CurriculumStage.STAND),
    )
    
    return MicroduckEnv(
        cfg=cfg,
        backend=backend,
        device=device,
        render=render,
        headless=headless,
    )
