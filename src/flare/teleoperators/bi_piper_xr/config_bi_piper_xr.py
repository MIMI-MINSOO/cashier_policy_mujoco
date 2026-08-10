#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from pathlib import Path

from lerobot.teleoperators.config import TeleoperatorConfig

from .neck import NeckConfig

# Follower's physical "ready" pose (deg). Used as the XR IK seed so the arm holds
# there when idle. MUST match PiperFollowerConfig.joints_init: stock
# lerobot-teleoperate only calls send_feedback for unitree_g1, so without this
# the IK seed stays at zeros and the arm snaps to a flat pose once the loop starts.
_ARM_READY_DEG = [-3.98, 13.53, -20.24, 4.9, 38.24, -6.47]
_ARM_READY_RAD = [float(np.radians(d)) for d in _ARM_READY_DEG]

@dataclass
class MotionTrackerConfig:
    """Configuration for a motion tracker to retarget a specific link."""
    serial: str = ""
    link_target: str = "link3"

@dataclass
class PiperArmConfig:
    """Base configuration for a single Piper arm in BiPiperXR teleoperator."""
    side: str = "left"  # "left" or "right"
    
    # Pose source string for xrobotoolkit_sdk, typically "left_controller" or "right_controller"
    pose_source: str = "left_controller"
    control_trigger: str = "left_grip"
    gripper_trigger: str = "left_trigger"
    
    # IK scaling parameters
    position_scale: float = 1.0
    rotation_scale: float = 1.0

    # 1-Euro Filter
    euro_min_cutoff: float = 1.0
    euro_beta: float = 0.01
    euro_d_cutoff: float = 1.0
    
    # Motion tracker configuration
    motion_tracker: Optional[MotionTrackerConfig] = None
    
    # Placo solver and URDF
    urdf_path: str = str(Path(__file__).parents[2] / "assets" / "urdf" / "piper_description.urdf")
    link_name: str = "link6"  # Target link name for IK (end effector)
    
    # Initial joint positions
    joints_init: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    
    # Gripper limits (in degrees corresponding to Piper master gripper range)
    gripper_open_pos: float = 101.4
    gripper_close_pos: float = 0.0
    # Trigger deadzone: a resting / barely-touched trigger maps to fully open.
    gripper_trigger_deadzone: float = 0.1

@TeleoperatorConfig.register_subclass("bi_piper_xr")
@dataclass
class BiPiperXRTeleopConfig(TeleoperatorConfig):
    # Base configuration for dual arm xr teleoperation
    dt: float = 0.02  # Control loop time step (50 Hz)
    
    left_arm: PiperArmConfig = field(
        default_factory=lambda: PiperArmConfig(
            side="left",
            pose_source="left_controller",
            control_trigger="left_grip",
            gripper_trigger="left_trigger",
            # Calibrated to this arm's measured range (raw 10360..90020).
            gripper_open_pos=90.0,
            gripper_close_pos=10.4,
            joints_init=list(_ARM_READY_RAD),
            motion_tracker=MotionTrackerConfig(
                serial="PC2310MLKB041941G",
                link_target="link3"
            )
        )
    )
    right_arm: PiperArmConfig = field(
        default_factory=lambda: PiperArmConfig(
            side="right",
            pose_source="right_controller",
            control_trigger="right_grip",
            gripper_trigger="right_trigger",
            joints_init=list(_ARM_READY_RAD),
            motion_tracker=MotionTrackerConfig(
                serial="PC2310MLKB041978G",
                link_target="link3"
            )
        )
    )

    # 2-DOF Dynamixel pan/tilt neck driven by the headset pose (None = disabled)
    enable_neck: bool = True
    neck: NeckConfig = field(default_factory=NeckConfig)
    # False면 Dynamixel 하드웨어에 쓰지 않고(dry_run) 각도 계산/기록만 한다.
    # 심(bi_piper_mujoco)에서 VR 텔레옵을 쓸 때 --teleop.neck_write_hardware=false
    neck_write_hardware: bool = True

    # Headset to world transformation matrix (flattened row-major)
    R_headset_world: list[float] = field(
        default_factory=lambda: [
            0.0, 0.0, -1.0,
           -1.0, 0.0,  0.0,
            0.0, 1.0,  0.0
        ]
    )


# --- MuJoCo sim variant: same XRoboToolkit (Quest/Pico 공용) + placo IK 경로,
# IK 시드만 심 home 키프레임(PiperRobot.EVAL_INIT_JOINT_DEG)으로 바꾼다. draccus는
# 부모 필드의 default_factory를 무시하고 중첩 dataclass는 그 클래스 자신의 필드
# 기본값을 쓰므로 (CLI에서 --teleop.left_arm.joints_init=... 로 오버라이드하면
# gripper_open_pos/motion_tracker 등 나머지 필드가 PiperArmConfig 기본값으로
# 리셋되는 부작용을 실제로 겪음), left/right 차이를 별도 클래스로 인코딩한다.
def _sim_joints_init_rad() -> list[float]:
    from flare.robots.piper_robot import PiperRobot

    return [float(np.radians(d)) for d in PiperRobot.EVAL_INIT_JOINT_DEG]


@dataclass
class SimLeftArmConfig(PiperArmConfig):
    side: str = "left"
    pose_source: str = "left_controller"
    control_trigger: str = "left_grip"
    gripper_trigger: str = "left_trigger"
    joints_init: list[float] = field(default_factory=_sim_joints_init_rad)
    # VIVE 모션 트래커 없이 씀 - 매칭 안 되는 시리얼을 남겨두면 tracker_pose가
    # 항상 None이라 실질적 영향은 없지만, 명시적으로 꺼서 헷갈리지 않게 한다.
    motion_tracker: Optional[MotionTrackerConfig] = None


@dataclass
class SimRightArmConfig(PiperArmConfig):
    side: str = "right"
    pose_source: str = "right_controller"
    control_trigger: str = "right_grip"
    gripper_trigger: str = "right_trigger"
    joints_init: list[float] = field(default_factory=_sim_joints_init_rad)
    motion_tracker: Optional[MotionTrackerConfig] = None


@TeleoperatorConfig.register_subclass("bi_piper_xr_mujoco")
@dataclass
class BiPiperXRMujocoConfig(BiPiperXRTeleopConfig):
    left_arm: SimLeftArmConfig = field(default_factory=SimLeftArmConfig)
    right_arm: SimRightArmConfig = field(default_factory=SimRightArmConfig)
    # 심에는 Dynamixel 목 하드웨어가 없음 - dry-run으로 각도만 계산해서 action에 싣는다.
    neck_write_hardware: bool = False

