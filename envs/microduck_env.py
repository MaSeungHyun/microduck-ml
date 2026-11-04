"""
Microduck Bipedal Robot Environment

A Gymnasium-style vectorized environment for training the Pollen Robotics Microduck robot.

Robot: Microduck by Pollen Robotics
  - 14 DOF bipedal robot (~25cm tall, ~800g)
  - Source: https://github.com/pollen-robotics/microduck_rl
  - License: STL meshes CC BY-NC-SA 4.0, MJCF Apache 2.0

Curriculum Stages:
1. STAND - Learn to balance and maintain upright posture
2. WALK - Walk forward at commanded velocity
3. COMMAND - Track velocity commands with full domain randomization
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List
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
    
    # Termination - more lenient for learning
    min_height: float = 0.05  # Terminate if trunk below this height (Microduck is smaller)
    max_pitch: float = 1.2    # ~70 degrees - allow significant lean
    max_roll: float = 1.2
    
    # Robot initial height (Microduck spawns at 0.12m per scene.xml)
    init_height: float = 0.12  # Initial spawn height
    target_height: float = 0.10  # Target standing height for reward
    
    # Assets
    robot_xml: str = "assets/scene.xml"


class MicroduckEnv:
    """
    Gymnasium-style vectorized environment for Pollen Robotics Microduck robot.
    
    The real Microduck has 14 actuated joints:
    - Left leg: hip_yaw, hip_roll, hip_pitch, knee, ankle (5)
    - Head: neck_pitch, head_pitch, head_yaw, head_roll (4)
    - Right leg: hip_yaw, hip_roll, hip_pitch, knee, ankle (5)
    
    Supports both Genesis (GPU-accelerated) and MuJoCo backends.
    """
    
    # Joint configuration for real Pollen Robotics Microduck (14 DOF)
    JOINT_NAMES = [
        # Left leg
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        # Head
        "neck_pitch", "head_pitch", "head_yaw", "head_roll",
        # Right leg
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    NUM_JOINTS = 14
    
    # Indices for leg joints only (for locomotion-focused training)
    LEG_JOINT_INDICES = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]  # Exclude head joints
    HEAD_JOINT_INDICES = [5, 6, 7, 8]
    NUM_LEG_JOINTS = 10
    NUM_HEAD_JOINTS = 4
    
    # Default standing pose from Pollen Robotics (STAND keyframe)
    # Joint order: L_hip_yaw, L_hip_roll, L_hip_pitch, L_knee, L_ankle, 
    #              neck_pitch, head_pitch, head_yaw, head_roll,
    #              R_hip_yaw, R_hip_roll, R_hip_pitch, R_knee, R_ankle
    DEFAULT_JOINT_POS = np.array([
        0.0, -0.0873, -0.4579, -0.0049, 0.4530,   # Left leg
        0.3491, 0.3491, 0.0, 0.0,                  # Head
        0.0, 0.0873, 0.4579, 0.0049, -0.4530,     # Right leg
    ], dtype=np.float32)
    
    # Action scaling per joint (radians)
    ACTION_SCALE = np.array([
        0.4, 0.3, 1.0, 1.0, 1.0,   # Left leg
        0.5, 0.5, 0.5, 0.3,        # Head
        0.4, 0.3, 1.0, 1.0, 1.0,   # Right leg
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
        # Obs: joint_pos(14) + joint_vel(14) + base_ang_vel(3) + gravity_vec(3) + 
        #      commands(3) + last_actions(14) + phase(2)
        self.num_obs = 14 + 14 + 3 + 3 + 3 + 14 + 2  # = 53
        self.num_actions = self.NUM_JOINTS
        
        # State buffers (will be torch tensors on GPU or numpy on CPU)
        self.num_envs = self.cfg.num_envs
        
        # Genesis-specific: DOF indices for leg joints (excluding free joint base DOFs)
        self.leg_dof_indices: Optional[List[int]] = None
        
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
            self.cfg.cmd_vel_x_range = (0.1, 0.3)
            self.cfg.cmd_vel_y_range = (0.0, 0.0)
            self.cfg.cmd_vel_yaw_range = (0.0, 0.0)
            self.cfg.enable_domain_rand = False
            self.cfg.enable_push = False
            
        elif stage == CurriculumStage.COMMAND:
            # Stage 3: Full velocity tracking with domain randomization
            self.cfg.cmd_vel_x_range = (-0.1, 0.3)
            self.cfg.cmd_vel_y_range = (-0.1, 0.1)
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
        
        # Create scene with viewer options
        viewer_opts = None
        vis_opts = None
        show_viewer = False
        
        if self.render_enabled and not self.headless:
            viewer_opts = gs.options.ViewerOptions(
                refresh_rate=60,
                camera_pos=(-1.2, -1.2, 1.0),
                camera_lookat=(0.0, 0.0, 0.05),
            )
            _side = int(np.ceil(np.sqrt(self.num_envs)))
            vis_opts = gs.options.VisOptions(rendered_envs_idx=[r * _side + c for r in range(3) for c in range(3)])
            show_viewer = True
        else:
            vis_opts = gs.options.VisOptions(
                show_world_frame=False,
                visualize_mpm_boundary=False,
            )
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.cfg.sim_dt,
                substeps=2,
            ),
            viewer_options=viewer_opts,
            vis_options=vis_opts,
            show_viewer=show_viewer,
        )
        
        # Add ground plane
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        
        # Load robot
        robot_path = os.path.join(os.path.dirname(__file__), "..", self.cfg.robot_xml)
        if not os.path.exists(robot_path):
            robot_path = self.cfg.robot_xml
        
        self.robot = self.scene.add_entity(
            # Genesis treats the base as fixed when loading scene.xml (via <include>); load the robot file directly
            gs.morphs.MJCF(file=robot_path.replace('scene.xml', 'microduck.xml')),
        )
        
        # Build scene with parallel environments
        self.scene.build(n_envs=self.num_envs, env_spacing=(0.4, 0.4) if self.render_enabled else (0.0, 0.0))
        
        # Get joint DOF indices
        self.leg_dof_indices = []
        for name in self.JOINT_NAMES:
            try:
                joint = self.robot.get_joint(name)
                idx = joint.dof_idx_local
                self.leg_dof_indices.append(idx)
            except Exception as e:
                print(f"Warning: Could not find joint {name}: {e}")
        
        self.total_dofs = self.robot.n_dofs
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
        
        self.mj_models = [self.mj_model]
        self.mj_datas = [mujoco.MjData(self.mj_model) for _ in range(self.num_envs)]
        
        if self.render_enabled and not self.headless:
            self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_datas[0])
        
        self._backend_initialized = True
    
    def _init_buffers(self):
        """Initialize state buffers"""
        import torch
        
        self.use_torch = self.backend == "genesis"
        
        if self.use_torch:
            device = torch.device(self.device)
            self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=device, dtype=torch.float32)
            self.actions = torch.zeros((self.num_envs, self.num_actions), device=device, dtype=torch.float32)
            self.last_actions = torch.zeros((self.num_envs, self.num_actions), device=device, dtype=torch.float32)
            self.rewards = torch.zeros(self.num_envs, device=device, dtype=torch.float32)
            self.dones = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self.episode_length = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
            self.commands = torch.zeros((self.num_envs, 3), device=device, dtype=torch.float32)
            self.command_time = torch.zeros(self.num_envs, device=device, dtype=torch.float32)
            self.phase = torch.zeros(self.num_envs, device=device, dtype=torch.float32)
            self.gait_freq = 2.0
            self.push_time = torch.zeros(self.num_envs, device=device, dtype=torch.float32)
            self.episode_rewards = torch.zeros(self.num_envs, device=device, dtype=torch.float32)
        else:
            self.obs_buf = np.zeros((self.num_envs, self.num_obs), dtype=np.float32)
            self.actions = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
            self.last_actions = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
            self.rewards = np.zeros(self.num_envs, dtype=np.float32)
            self.dones = np.zeros(self.num_envs, dtype=bool)
            self.episode_length = np.zeros(self.num_envs, dtype=np.float32)
            self.commands = np.zeros((self.num_envs, 3), dtype=np.float32)
            self.command_time = np.zeros(self.num_envs, dtype=np.float32)
            self.phase = np.zeros(self.num_envs, dtype=np.float32)
            self.gait_freq = 2.0
            self.push_time = np.zeros(self.num_envs, dtype=np.float32)
            self.episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
    
    def reset(self, env_ids=None):
        """Reset environments."""
        if env_ids is None:
            env_ids = None
        
        if self.backend == "genesis":
            self._reset_genesis(env_ids)
        else:
            self._reset_mujoco(env_ids)
        
        self._resample_commands(env_ids)
        
        if self.use_torch:
            import torch
            if env_ids is None:
                self.phase.zero_()
            else:
                env_ids_t = torch.tensor(env_ids, device=self.device, dtype=torch.long)
                self.phase[env_ids_t] = 0.0
        else:
            if env_ids is None:
                self.phase[:] = 0
            else:
                self.phase[env_ids] = 0
        
        self._compute_observations()
        return self.obs_buf, {}
    
    def _reset_genesis(self, env_ids):
        """Reset Genesis environments"""
        import torch
        
        if env_ids is None:
            n = self.num_envs
            
            leg_pos = torch.tensor(self.DEFAULT_JOINT_POS, device=self.device, dtype=torch.float32)
            leg_pos = leg_pos.unsqueeze(0).expand(n, -1)
            
            if self.leg_dof_indices:
                dof_indices = torch.tensor(self.leg_dof_indices, device=self.device, dtype=torch.long)
                self.robot.set_dofs_position(leg_pos, dofs_idx_local=dof_indices)
                self.robot.set_dofs_velocity(
                    torch.zeros((n, len(self.leg_dof_indices)), device=self.device, dtype=torch.float32),
                    dofs_idx_local=dof_indices
                )
            
            init_pos = torch.tensor([[0.0, 0.0, self.cfg.init_height]], device=self.device, dtype=torch.float32)
            init_pos = init_pos.expand(n, -1)
            init_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)
            init_quat = init_quat.expand(n, -1)
            self.robot.set_pos(init_pos)
            self.robot.set_quat(init_quat)
            self.robot.set_dofs_velocity(torch.zeros((n, 3), device=self.device, dtype=torch.float32), dofs_idx_local=[0, 1, 2])
            self.robot.set_dofs_velocity(torch.zeros((n, 3), device=self.device, dtype=torch.float32), dofs_idx_local=[3, 4, 5])
            
            self.episode_length.zero_()
            self.last_actions.zero_()
            self.episode_rewards.zero_()
        else:
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            n = len(env_ids)
            
            leg_pos = torch.tensor(self.DEFAULT_JOINT_POS, device=self.device, dtype=torch.float32)
            leg_pos = leg_pos.unsqueeze(0).expand(n, -1)
            
            if self.leg_dof_indices:
                dof_indices = torch.tensor(self.leg_dof_indices, device=self.device, dtype=torch.long)
                self.robot.set_dofs_position(leg_pos, dofs_idx_local=dof_indices, envs_idx=env_ids_tensor)
                self.robot.set_dofs_velocity(
                    torch.zeros((n, len(self.leg_dof_indices)), device=self.device, dtype=torch.float32),
                    dofs_idx_local=dof_indices,
                    envs_idx=env_ids_tensor
                )
            
            init_pos = torch.tensor([[0.0, 0.0, self.cfg.init_height]], device=self.device, dtype=torch.float32)
            init_pos = init_pos.expand(n, -1)
            init_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)
            init_quat = init_quat.expand(n, -1)
            
            self.robot.set_pos(init_pos, envs_idx=env_ids_tensor)
            self.robot.set_quat(init_quat, envs_idx=env_ids_tensor)
            self.robot.set_dofs_velocity(torch.zeros((n, 3), device=self.device, dtype=torch.float32), dofs_idx_local=[0, 1, 2], envs_idx=env_ids_tensor)
            self.robot.set_dofs_velocity(torch.zeros((n, 3), device=self.device, dtype=torch.float32), dofs_idx_local=[3, 4, 5], envs_idx=env_ids_tensor)
            
            self.episode_length[env_ids_tensor] = 0.0
            self.last_actions[env_ids_tensor] = 0.0
            self.episode_rewards[env_ids_tensor] = 0.0
    
    def _reset_mujoco(self, env_ids):
        """Reset MuJoCo environments"""
        if env_ids is None:
            env_ids = range(self.num_envs)
        
        for i in env_ids:
            mujoco.mj_resetData(self.mj_model, self.mj_datas[i])
            # Set initial position (7 DOF for free joint: x,y,z, qw,qx,qy,qz)
            self.mj_datas[i].qpos[:7] = [0, 0, self.cfg.init_height, 1, 0, 0, 0]
            # Set joint positions (14 actuated joints)
            self.mj_datas[i].qpos[7:7+self.NUM_JOINTS] = self.DEFAULT_JOINT_POS
            mujoco.mj_forward(self.mj_model, self.mj_datas[i])
            
            self.episode_length[i] = 0
            self.last_actions[i] = 0
            self.episode_rewards[i] = 0
    
    def _resample_commands(self, env_ids=None):
        """Resample velocity commands"""
        import torch
        
        if self.use_torch:
            if env_ids is None:
                n = self.num_envs
                self.commands[:, 0] = torch.empty(n, device=self.device, dtype=torch.float32).uniform_(
                    self.cfg.cmd_vel_x_range[0], self.cfg.cmd_vel_x_range[1])
                self.commands[:, 1] = torch.empty(n, device=self.device, dtype=torch.float32).uniform_(
                    self.cfg.cmd_vel_y_range[0], self.cfg.cmd_vel_y_range[1])
                self.commands[:, 2] = torch.empty(n, device=self.device, dtype=torch.float32).uniform_(
                    self.cfg.cmd_vel_yaw_range[0], self.cfg.cmd_vel_yaw_range[1])
                self.command_time.zero_()
            else:
                env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
                n = len(env_ids)
                self.commands[env_ids_tensor, 0] = torch.empty(n, device=self.device, dtype=torch.float32).uniform_(
                    self.cfg.cmd_vel_x_range[0], self.cfg.cmd_vel_x_range[1])
                self.commands[env_ids_tensor, 1] = torch.empty(n, device=self.device, dtype=torch.float32).uniform_(
                    self.cfg.cmd_vel_y_range[0], self.cfg.cmd_vel_y_range[1])
                self.commands[env_ids_tensor, 2] = torch.empty(n, device=self.device, dtype=torch.float32).uniform_(
                    self.cfg.cmd_vel_yaw_range[0], self.cfg.cmd_vel_yaw_range[1])
                self.command_time[env_ids_tensor] = 0.0
        else:
            if env_ids is None:
                env_ids = range(self.num_envs)
            for i in env_ids:
                self.commands[i, 0] = np.random.uniform(*self.cfg.cmd_vel_x_range)
                self.commands[i, 1] = np.random.uniform(*self.cfg.cmd_vel_y_range)
                self.commands[i, 2] = np.random.uniform(*self.cfg.cmd_vel_yaw_range)
                self.command_time[i] = 0
    
    def step(self, actions):
        """Step the environment."""
        import torch
        
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = actions
        
        if self.use_torch:
            if not isinstance(actions, torch.Tensor):
                actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
            self.actions = torch.clamp(actions, -1.0, 1.0)
            action_scale = torch.tensor(self.ACTION_SCALE, device=self.device, dtype=torch.float32)
            default_pos = torch.tensor(self.DEFAULT_JOINT_POS, device=self.device, dtype=torch.float32)
            target_positions = default_pos + self.actions * action_scale
        else:
            self.actions = np.clip(actions_np, -1.0, 1.0)
            target_positions = self.DEFAULT_JOINT_POS + self.actions * self.ACTION_SCALE
        
        if self.backend == "genesis":
            self._step_genesis(target_positions)
        else:
            self._step_mujoco(target_positions)
        
        if self.use_torch:
            self.phase = self.phase + 2.0 * np.pi * self.gait_freq * self.cfg.dt
            self.phase = torch.fmod(self.phase, 2.0 * np.pi)
        else:
            self.phase += 2 * np.pi * self.gait_freq * self.cfg.dt
            self.phase = np.fmod(self.phase, 2 * np.pi)
        
        if self.cfg.enable_push:
            self._apply_push()
        
        if self.use_torch:
            self.command_time = self.command_time + self.cfg.dt
            resample_mask = self.command_time > self.cfg.command_resample_time
            if resample_mask.any():
                resample_ids = torch.where(resample_mask)[0]
                self._resample_commands(resample_ids.tolist())
        else:
            self.command_time += self.cfg.dt
            resample_ids = np.where(self.command_time > self.cfg.command_resample_time)[0]
            if len(resample_ids) > 0:
                self._resample_commands(resample_ids)
        
        self._compute_observations()
        self._compute_rewards()
        self._check_termination()
        
        self.episode_length = self.episode_length + 1.0
        if self.use_torch:
            truncated = self.episode_length >= self.cfg.episode_length
            self.episode_rewards = self.episode_rewards + self.rewards
        else:
            truncated = self.episode_length >= self.cfg.episode_length
            self.episode_rewards += self.rewards
        
        if self.use_torch:
            self.last_actions = self.actions.clone()
        else:
            self.last_actions = self.actions.copy()
        
        if self.use_torch:
            reset_mask = self.dones | truncated
            reset_ids = torch.where(reset_mask)[0]
            if len(reset_ids) > 0:
                self.reset(reset_ids.tolist())
        else:
            reset_ids = np.where(self.dones | truncated)[0]
            if len(reset_ids) > 0:
                self.reset(reset_ids.tolist())
        
        return self.obs_buf, self.rewards, self.dones, truncated, {}
    
    def _step_genesis(self, target_positions):
        """Step Genesis simulation"""
        import torch
        
        if self.leg_dof_indices:
            dof_indices = torch.tensor(self.leg_dof_indices, device=self.device, dtype=torch.long)
            self.robot.control_dofs_position(target_positions, dofs_idx_local=dof_indices)
        
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
        """Apply random push perturbations"""
        import torch
        
        if self.use_torch:
            self.push_time = self.push_time + self.cfg.dt
            push_mask = self.push_time > self.cfg.push_interval
            
            if push_mask.any():
                push_ids = torch.where(push_mask)[0]
                n = len(push_ids)
                
                force_mag = torch.empty(n, device=self.device, dtype=torch.float32).uniform_(
                    self.cfg.push_force_range[0], self.cfg.push_force_range[1])
                angle = torch.empty(n, device=self.device, dtype=torch.float32).uniform_(0, 2 * np.pi)
                
                current_vel = self.robot.get_vel()
                push_vel = torch.zeros((n, 3), device=self.device, dtype=torch.float32)
                push_vel[:, 0] = force_mag * torch.cos(angle) * 0.1
                push_vel[:, 1] = force_mag * torch.sin(angle) * 0.1
                
                new_vel = current_vel.clone()
                new_vel[push_ids] = current_vel[push_ids] + push_vel
                self.robot.set_dofs_velocity(new_vel[push_ids], dofs_idx_local=[0, 1, 2], envs_idx=push_ids)
                
                self.push_time[push_ids] = 0.0
    
    def _compute_observations(self):
        """Compute observation vector"""
        import torch
        
        if self.backend == "genesis":
            if self.leg_dof_indices:
                dof_indices = torch.tensor(self.leg_dof_indices, device=self.device, dtype=torch.long)
                joint_pos = self.robot.get_dofs_position(dofs_idx_local=dof_indices)
                joint_vel = self.robot.get_dofs_velocity(dofs_idx_local=dof_indices)
            else:
                joint_pos = self.robot.get_dofs_position()[:, :self.NUM_JOINTS]
                joint_vel = self.robot.get_dofs_velocity()[:, :self.NUM_JOINTS]
            
            base_quat = self.robot.get_quat()
            base_ang_vel = self.robot.get_ang()
            
            gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
            gravity_body = self._quat_rotate_inverse(base_quat, gravity_world.unsqueeze(0).expand(self.num_envs, -1))
            
            phase_sin = torch.sin(self.phase).unsqueeze(1)
            phase_cos = torch.cos(self.phase).unsqueeze(1)
            
            if self.cfg.joint_pos_noise > 0:
                joint_pos = joint_pos + torch.randn_like(joint_pos) * self.cfg.joint_pos_noise
            if self.cfg.joint_vel_noise > 0:
                joint_vel = joint_vel + torch.randn_like(joint_vel) * self.cfg.joint_vel_noise
            
            self.obs_buf = torch.cat([
                joint_pos, joint_vel, base_ang_vel, gravity_body,
                self.commands, self.last_actions, phase_sin, phase_cos,
            ], dim=1)
        
        else:
            for i in range(self.num_envs):
                d = self.mj_datas[i]
                joint_pos = d.qpos[7:7+self.NUM_JOINTS]
                joint_vel = d.qvel[6:6+self.NUM_JOINTS]
                base_quat = d.qpos[3:7]
                base_ang_vel = d.qvel[3:6]
                
                gravity_body = self._quat_rotate_inverse_np(base_quat, np.array([0, 0, -1]))
                
                phase_sin = np.sin(self.phase[i])
                phase_cos = np.cos(self.phase[i])
                
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
            base_pos = self.robot.get_pos()
            base_quat = self.robot.get_quat()
            base_lin_vel = self.robot.get_vel()
            base_ang_vel = self.robot.get_ang()
            
            if self.leg_dof_indices:
                dof_indices = torch.tensor(self.leg_dof_indices, device=self.device, dtype=torch.long)
                joint_vel = self.robot.get_dofs_velocity(dofs_idx_local=dof_indices)
            else:
                joint_vel = self.robot.get_dofs_velocity()[:, :self.NUM_JOINTS]
            
            r_alive = scales["alive"] * torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
            
            gravity_body = self._quat_rotate_inverse(
                base_quat, 
                torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).unsqueeze(0).expand(self.num_envs, -1)
            )
            r_upright = scales["upright"] * (-gravity_body[:, 2] + 1.0) / 2.0
            
            target_height = self.cfg.target_height
            height_error = torch.abs(base_pos[:, 2] - target_height)
            r_height = scales["height"] * torch.exp(-10.0 * height_error)
            
            vel_error_x = torch.square(base_lin_vel[:, 0] - self.commands[:, 0])
            vel_error_y = torch.square(base_lin_vel[:, 1] - self.commands[:, 1])
            r_vel = scales["velocity_tracking"] * torch.exp(-5.0 * (vel_error_x + vel_error_y))
            
            yaw_rate_error = torch.square(base_ang_vel[:, 2] - self.commands[:, 2])
            r_ang_vel = scales["angular_velocity_tracking"] * torch.exp(-5.0 * yaw_rate_error)
            
            action_diff = self.actions - self.last_actions
            r_action_rate = scales["action_rate"] * torch.sum(torch.square(action_diff), dim=1)
            
            r_torque = scales["joint_torque"] * torch.sum(torch.square(joint_vel), dim=1)
            
            self.rewards = r_alive + r_upright + r_height + r_vel + r_ang_vel + r_action_rate + r_torque
        
        else:
            for i in range(self.num_envs):
                d = self.mj_datas[i]
                
                base_pos = d.qpos[:3]
                base_quat = d.qpos[3:7]
                base_lin_vel = d.qvel[:3]
                base_ang_vel = d.qvel[3:6]
                joint_vel = d.qvel[6:6+self.NUM_JOINTS]
                
                r_alive = scales["alive"]
                
                gravity_body = self._quat_rotate_inverse_np(base_quat, np.array([0, 0, -1]))
                r_upright = scales["upright"] * (-gravity_body[2] + 1.0) / 2.0
                
                height_error = abs(base_pos[2] - self.cfg.target_height)
                r_height = scales["height"] * np.exp(-10 * height_error)
                
                vel_error = (base_lin_vel[0] - self.commands[i, 0])**2 + (base_lin_vel[1] - self.commands[i, 1])**2
                r_vel = scales["velocity_tracking"] * np.exp(-5 * vel_error)
                
                yaw_error = (base_ang_vel[2] - self.commands[i, 2])**2
                r_ang_vel = scales["angular_velocity_tracking"] * np.exp(-5 * yaw_error)
                
                action_diff = self.actions[i] - self.last_actions[i]
                r_action_rate = scales["action_rate"] * np.sum(action_diff**2)
                
                r_torque = scales["joint_torque"] * np.sum(joint_vel**2)
                
                self.rewards[i] = r_alive + r_upright + r_height + r_vel + r_ang_vel + r_action_rate + r_torque
    
    def _check_termination(self):
        """Check termination conditions"""
        import torch
        
        if self.backend == "genesis":
            base_pos = self.robot.get_pos()
            base_quat = self.robot.get_quat()
            
            height_terminate = base_pos[:, 2] < self.cfg.min_height
            
            gravity_body = self._quat_rotate_inverse(
                base_quat,
                torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).unsqueeze(0).expand(self.num_envs, -1)
            )
            
            roll_angle = torch.asin(torch.clamp(gravity_body[:, 1], -1.0, 1.0))
            pitch_angle = torch.asin(torch.clamp(gravity_body[:, 0], -1.0, 1.0))
            
            orientation_terminate = (torch.abs(roll_angle) > self.cfg.max_roll) | (torch.abs(pitch_angle) > self.cfg.max_pitch)
            
            self.dones = height_terminate | orientation_terminate
        
        else:
            for i in range(self.num_envs):
                d = self.mj_datas[i]
                base_pos = d.qpos[:3]
                base_quat = d.qpos[3:7]
                
                if base_pos[2] < self.cfg.min_height:
                    self.dones[i] = True
                    continue
                
                gravity_body = self._quat_rotate_inverse_np(base_quat, np.array([0, 0, -1]))
                roll_angle = np.arcsin(np.clip(gravity_body[1], -1.0, 1.0))
                pitch_angle = np.arcsin(np.clip(gravity_body[0], -1.0, 1.0))
                
                if abs(roll_angle) > self.cfg.max_roll or abs(pitch_angle) > self.cfg.max_pitch:
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
        import gymnasium as gym
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)
    
    @property
    def action_space(self):
        import gymnasium as gym
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)


def make_env(
    stage: str = "stand",
    num_envs: int = 1024,
    backend: str = "genesis",
    device: str = "cuda",
    render: bool = False,
    headless: bool = True,
) -> MicroduckEnv:
    """Factory function to create Microduck environment."""
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
