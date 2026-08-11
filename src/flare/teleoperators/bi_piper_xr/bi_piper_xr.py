#!/usr/bin/env python

import logging
import time
from functools import cached_property
from pathlib import Path
import math

import numpy as np
import meshcat.transformations as tf

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.teleoperators.teleoperator import Teleoperator
from .config_bi_piper_xr import BiPiperXRTeleopConfig, BiPiperXRMujocoConfig
from .geometry import (
    quat_diff_as_angle_axis,
    is_valid_quaternion,
    quaternion_to_angle_axis,
    R_HEADSET_TO_WORLD,
    apply_delta_pose
)
from .neck import (
    NeckAngleMapper,
    DynamixelNeckController,
    CLUTCH_BUTTONS,
    headset_rotation,
)

try:
    import placo
    PLACO_AVAILABLE = True
except ImportError:
    PLACO_AVAILABLE = False

try:
    import xrobotoolkit_sdk as xrt
    XR_SDK_AVAILABLE = True
except ImportError:
    XR_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

class OneEuroFilter:
    def __init__(self, dim, rate, min_cutoff, beta, d_cutoff):
        self.rate = rate
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = np.zeros(dim)

    def _alpha(self, cutoff):
        te = 1.0 / self.rate
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        x = np.asarray(x, dtype=float)
        if self.x_prev is None:
            self.x_prev = x.copy()
            return x.copy()
        a_d = self._alpha(self.d_cutoff)
        dx = (x - self.x_prev) * self.rate
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = np.array([self._alpha(c) for c in cutoff])
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        return x_hat

class XrIKController:
    """IK solver wrapper for a single arm using placo."""

    def __init__(self, config, dt: float, R_headset_world: np.ndarray):
        self.config = config
        self.dt = dt
        self.R_headset_world = R_headset_world

        self.robot = placo.RobotWrapper(config.urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.dt = self.dt
        self.solver.mask_fbase(True)
        
        # Initialize joint states (first 7 are the floating base for placo [x,y,z,qx,qy,qz,qw])
        self.robot.state.q[:7] = np.array([0, 0, 0, 0, 0, 0, 1])
        if len(self.robot.state.q) >= 7 + len(config.joints_init):
            self.robot.state.q[7:7+len(config.joints_init)] = config.joints_init
            
        self.robot.update_kinematics()

        self.ee_link_name = config.link_name
        ee_transform = self.robot.get_T_world_frame(self.ee_link_name)
        self.ref_ee_xyz = ee_transform[:3, 3]
        # Store initial rotation as quaternion (w, x, y, z) for meshcat.transformations
        self.ref_ee_quat = tf.quaternion_from_matrix(ee_transform)
        
        self.effector_task = self.solver.add_frame_task(self.ee_link_name, ee_transform)
        self.effector_task.configure("ee", "soft", 1.0)
        
        manipulability = self.solver.add_manipulability_task(self.ee_link_name, "both", 1.0)
        manipulability.configure("manipulability", "soft", 1e-2)

        # Add joint regularization task to prevent sudden flips/jumping
        self.joints_task = self.solver.add_joints_task()
        default_joints = {
            self.robot.model.names[7 + i]: config.joints_init[i] 
            for i in range(len(config.joints_init)) 
            if 7 + i < len(self.robot.model.names)
        }
        self.joints_task.set_joints(default_joints)
        self.joints_task.configure("joints_regularization", "soft", 1e-3)

        # Set up motion tracker tasks if configured (position only)
        self.motion_tracker_task = None
        if config.motion_tracker:
            target_xyz = self.robot.get_T_world_frame(config.motion_tracker.link_target)[:3, 3]
            self.motion_tracker_task = self.solver.add_position_task(config.motion_tracker.link_target, target_xyz)
            self.motion_tracker_task.configure("tracker", "soft", 0.05)

        self.solver.add_kinetic_energy_regularization_task(1e-6)

        self.robot.update_kinematics()
        
        self.ref_controller_xyz = None
        self.ref_controller_quat = None
        self.ref_tracker_xyz = None
        self.ref_robot_xyz = None
        
        # 1-Euro Filters
        self._pos_filter = None
        self._rot_filter = None

    def _reset_filters(self):
        cfg = self.config
        rate = 1.0 / self.dt
        self._pos_filter = OneEuroFilter(3, rate, cfg.euro_min_cutoff, cfg.euro_beta, cfg.euro_d_cutoff)
        self._rot_filter = OneEuroFilter(3, rate, cfg.euro_min_cutoff, cfg.euro_beta, cfg.euro_d_cutoff)

    def update_robot_state(self, actual_joints: np.ndarray):
        """Updates Placo's internal joint states with actual physical angles (closed-loop)."""
        # First 7 are the floating base [x,y,z,qx,qy,qz,qw], joints start at index 7
        if len(actual_joints) == 6:
            self.robot.state.q[7:13] = actual_joints
            self.robot.update_kinematics()

    def process_xr_pose(self, xr_pose: np.ndarray, tracker_pose: np.ndarray = None):
        """Processes XR controller pose and solves IK. xr_pose is [x, y, z, qx, qy, qz, qw]"""
        # Get position and orientation (convert to w,x,y,z for tf)
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = [
            xr_pose[6],  # w
            xr_pose[3],  # x
            xr_pose[4],  # y
            xr_pose[5],  # z
        ]

        controller_xyz = self.R_headset_world @ controller_xyz
        
        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat),
            tf.quaternion_conjugate(R_quat),
        )
        
        # We use scipy Rotation here only to convert to/from rotvec for filtering
        from scipy.spatial.transform import Rotation
        
        # controller_quat is [w, x, y, z]. Scipy needs [x, y, z, w]
        rot_scipy = Rotation.from_quat([controller_quat[1], controller_quat[2], controller_quat[3], controller_quat[0]])
        rot_vec = rot_scipy.as_rotvec()

        if self.ref_controller_xyz is None:
            self._reset_filters()
            
            # Feed initial values to filters
            self._pos_filter.filter(controller_xyz)
            self._rot_filter.filter(rot_vec)

            self.ref_controller_xyz = controller_xyz.copy()
            self.ref_controller_quat = controller_quat.copy()
            
            # Re-sync base robot pose on activation (clutch)
            ee_transform = self.robot.get_T_world_frame(self.ee_link_name)
            self.ref_ee_xyz = ee_transform[:3, 3].copy()
            self.ref_ee_quat = tf.quaternion_from_matrix(ee_transform)
            
            # Re-sync motion tracker base pose on activation
            if self.motion_tracker_task is not None and tracker_pose is not None:
                tracker_xyz = self.R_headset_world @ np.array(tracker_pose[:3])
                self.ref_tracker_xyz = tracker_xyz.copy()
                self.ref_robot_xyz = self.robot.get_T_world_frame(self.config.motion_tracker.link_target)[:3, 3].copy()
                
            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])
        else:
            # 1-Euro Filter
            filt_pos = self._pos_filter.filter(controller_xyz)
            filt_rotvec = self._rot_filter.filter(rot_vec)
            
            # Convert filtered rotvec back to [w, x, y, z] quaternion
            filt_rot_scipy = Rotation.from_rotvec(filt_rotvec)
            filt_quat_xyzw = filt_rot_scipy.as_quat()
            filt_quat = np.array([filt_quat_xyzw[3], filt_quat_xyzw[0], filt_quat_xyzw[1], filt_quat_xyzw[2]])

            delta_xyz = (filt_pos - self.ref_controller_xyz) * self.config.position_scale
            delta_rot = quat_diff_as_angle_axis(self.ref_controller_quat, filt_quat) * self.config.rotation_scale
        
        # Apply to end effector
        target_xyz, target_quat = apply_delta_pose(
            self.ref_ee_xyz,
            self.ref_ee_quat,
            delta_xyz,
            delta_rot,
        )
        
        target_pose = tf.quaternion_matrix(target_quat)
        target_pose[:3, 3] = target_xyz
        
        self.effector_task.T_world_frame = target_pose
        
        # Apply motion tracker delta to tracker task
        if self.motion_tracker_task is not None and tracker_pose is not None:
            tracker_xyz = self.R_headset_world @ np.array(tracker_pose[:3])
            
            if self.ref_tracker_xyz is None:
                self.ref_tracker_xyz = tracker_xyz.copy()
                self.ref_robot_xyz = self.robot.get_T_world_frame(self.config.motion_tracker.link_target)[:3, 3].copy()
                
            tracker_delta = tracker_xyz - self.ref_tracker_xyz
            final_target_xyz = self.ref_robot_xyz + tracker_delta * self.config.position_scale
            self.motion_tracker_task.target_world = final_target_xyz

        self.solver.solve(True)
        self.robot.update_kinematics()

    def get_joints(self):
        """Returns joint values."""
        # Typically the arm joints start at index 7 for placo because of floating base [x,y,z,qx,qy,qz,qw]
        return self.robot.state.q[7:13].copy()

    def deactivate(self):
        self.ref_controller_xyz = None
        self.ref_controller_rot = None
        self.ref_tracker_xyz = None
        self.ref_robot_xyz = None


class BiPiperXRTeleop(Teleoperator):
    """
    Bimanual Agilex PiPER teleoperator using Pico XR controller and Placo IK.
    """

    config_class = BiPiperXRTeleopConfig
    name = "bi_piper_xr"

    def __init__(self, config: BiPiperXRTeleopConfig):
        super().__init__(config)
        self.config = config
        
        if not PLACO_AVAILABLE:
            raise ImportError("placo is not installed. Please install it to use BiPiperXRTeleop.")
        if not XR_SDK_AVAILABLE:
            raise ImportError("xrobotoolkit_sdk is not installed. Please install it to use BiPiperXRTeleop.")

        self.R_headset_world = np.array(config.R_headset_world).reshape(3, 3)
        self._is_connected = False
        self._last_time = time.time()
        
        self.left_ik = None
        self.right_ik = None

        # 2-DOF Dynamixel neck. Gated by the app head-tracking checkbox (pose
        # validity) AND a clutch button toggle; tracks head delta when both on.
        self.neck_ctrl = None
        self.neck_mapper = None
        self.neck_read_button = None
        self.neck_active = False
        self._prev_neck_btn = False

        # Expose states matching quest_controller for lerobot_record
        self.left = type('Obj', (object,), {'buttons': {}, 'thumbstick': [0.0, 0.0]})()
        self.right = type('Obj', (object,), {'buttons': {}, 'thumbstick': [0.0, 0.0]})()

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = {}
        for side in ["left", "right"]:
            for name in JOINT_NAMES:
                features[f"{side}_{name}.pos"] = float
            features[f"{side}_gripper.pos"] = float
        if self.config.enable_neck:
            features["neck_yaw.pos"] = float
            features["neck_pitch.pos"] = float
        return features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        # Declare that we want the current joint positions of both arms from the robot
        features = {}
        for side in ["left", "right"]:
            for name in JOINT_NAMES:
                features[f"{side}_{name}.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:

        if not self.config.left_arm.urdf_path or not self.config.right_arm.urdf_path:
            raise ValueError("You must provide URDF paths for both left and right Piper arms in the config.")
            
        xrt.init()
        
        self.left_ik = XrIKController(self.config.left_arm, self.config.dt, self.R_headset_world)
        self.right_ik = XrIKController(self.config.right_arm, self.config.dt, self.R_headset_world)

        if self.config.enable_neck:
            self.neck_mapper = NeckAngleMapper(self.config.neck)
            self.neck_ctrl = DynamixelNeckController(
                self.config.neck, dry_run=not self.config.neck_write_hardware
            )
            self.neck_read_button = getattr(xrt, CLUTCH_BUTTONS[self.config.neck.clutch_button])
            logger.info(
                f"Neck enabled (head checkbox + '{self.config.neck.clutch_button}' "
                f"button toggle){' [dry-run: no Dynamixel]' if not self.config.neck_write_hardware else ''}."
            )

        self._is_connected = True
        logger.info(f"{self.name} initialized with XR SDK and Placo IK.")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        
        action = {}
        
        tracker_data = {}
        try:
            if hasattr(xrt, "num_motion_data_available"):
                num = xrt.num_motion_data_available()
                if num > 0:
                    poses = xrt.get_motion_tracker_pose()
                    serials = xrt.get_motion_tracker_serial_numbers()
                    for i in range(num):
                        tracker_data[serials[i]] = poses[i]
        except Exception as e:
            logger.debug(f"Failed to fetch motion tracker data: {e}")
        
        for side, ik_ctrl, cfg in [("left", self.left_ik, self.config.left_arm), ("right", self.right_ik, self.config.right_arm)]:
            # Check grip for activation (Clutch behavior)
            grip_val = xrt.get_left_grip() if cfg.control_trigger == "left_grip" else xrt.get_right_grip()
            if grip_val > 0.9:
                pose = xrt.get_left_controller_pose() if cfg.pose_source == "left_controller" else xrt.get_right_controller_pose()
                
                tracker_pose = None
                if cfg.motion_tracker and cfg.motion_tracker.serial in tracker_data:
                    tracker_pose = tracker_data[cfg.motion_tracker.serial]
                    
                ik_ctrl.process_xr_pose(pose, tracker_pose)
            else:
                ik_ctrl.deactivate()
                
            joints = ik_ctrl.get_joints()
            for i, name in enumerate(JOINT_NAMES):
                action[f"{side}_{name}.pos"] = float(np.degrees(joints[i]))
                
            # Process gripper trigger
            trigger_val = xrt.get_left_trigger() if cfg.gripper_trigger == "left_trigger" else xrt.get_right_trigger()
            # Deadzone: a resting / barely-touched trigger -> fully open (avoids drift-closing at rest)
            trigger_val = 0.0 if trigger_val < cfg.gripper_trigger_deadzone else min(trigger_val, 1.0)
            # Simple linear interpolation for gripper position
            gripper_pos = cfg.gripper_open_pos + trigger_val * (cfg.gripper_close_pos - cfg.gripper_open_pos)
            action[f"{side}_gripper.pos"] = float(gripper_pos)

        # Neck: head checkbox (pose validity) = data gate, clutch button = toggle.
        #   head OFF -> home; head ON + clutch OFF -> hold; head ON + clutch ON -> track
        if self.neck_mapper is not None:
            rot = headset_rotation(xrt.get_headset_pose())
            btn = bool(self.neck_read_button())
            if btn and not self._prev_neck_btn:   # rising edge -> toggle clutch
                self.neck_active = not self.neck_active
            self._prev_neck_btn = btn
            neck_pitch, neck_yaw = self.neck_mapper.step(rot, self.neck_active)
            self.neck_ctrl.write(neck_pitch, neck_yaw)
            action["neck_yaw.pos"] = float(np.degrees(neck_yaw))
            action["neck_pitch.pos"] = float(np.degrees(neck_pitch))

        # Update controller states for lerobot_record
        try:
            self.right.thumbstick = xrt.get_right_axis()
            self.left.thumbstick = xrt.get_left_axis()

            # Map Y button to the left controller's 'Y' key to match Quest behavior
            self.left.buttons["Y"] = xrt.get_Y_button()
            # Right controller B button: scene reset trigger (심 전용, teleop_xr.py에서 사용).
            self.right.buttons["B"] = xrt.get_B_button()
        except Exception as e:
            logger.debug(f"Failed to read controller inputs for state: {e}")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self.name} read action: {dt_ms:.1f}ms")
        
        # Enforce rate
        elapsed = time.time() - self._last_time
        if elapsed < self.config.dt:
            time.sleep(self.config.dt - elapsed)
        self._last_time = time.time()
            
        return action

    def reset_neck(self) -> None:
        """목 트래킹 상태를 home으로 강제 리셋 (씬 리셋 시 호출용).

        씬 리셋(예: teleop_xr.py의 B버튼)은 robot.env만 리셋하고 이 텔레옵의
        neck_mapper 내부 상태(current_pitch/yaw)는 안 건드리기 때문에, 안 하면
        리셋 다음 프레임에 바로 리셋 전 각도로 되돌아가버린다.
        """
        if self.neck_mapper is not None:
            self.neck_mapper.reset()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """
        Receives the actual physical state of the robot from LeRobot's main loop
        and updates the internal Placo IK models (Closed-loop control).
        """
        if not feedback:
            return
            
        # Extract left and right joints (converting from degrees back to radians for Placo)
        left_actual = np.zeros(6)
        right_actual = np.zeros(6)
        
        for i, name in enumerate(JOINT_NAMES):
            if f"left_{name}.pos" in feedback:
                left_actual[i] = np.radians(feedback[f"left_{name}.pos"])
            if f"right_{name}.pos" in feedback:
                right_actual[i] = np.radians(feedback[f"right_{name}.pos"])
                
        # Update Placo models
        self.left_ik.update_robot_state(left_actual)
        self.right_ik.update_robot_state(right_actual)

    def _reset_filters(self):
        cfg = self.config
        self._pos_filter = OneEuroFilter(3, cfg.control_hz, cfg.euro_min_cutoff, cfg.euro_beta, cfg.euro_d_cutoff)
        self._rot_filter = OneEuroFilter(3, cfg.control_hz, cfg.euro_min_cutoff, cfg.euro_beta, cfg.euro_d_cutoff)
        self._prev_filt_pos = None
        self._prev_filt_rot = None    

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.neck_ctrl is not None:
            try:
                self.neck_ctrl.close()
            except Exception:
                pass
        try:
            xrt.close()
        except Exception:
            pass
        self._is_connected = False
        logger.info(f"{self.name} disconnected.")


class BiPiperXRMujoco(BiPiperXRTeleop):
    """XRoboToolkit(Quest/Pico 공용) -> MuJoCo 심 텔레옵: 동일한 placo IK 경로,
    IK 시드만 심 home 키프레임(PiperRobot.EVAL_INIT_JOINT_DEG)으로 바꾼다.
    VIVE 모션 트래커 없이 동작(양팔 motion_tracker=None). 목은 기본값 그대로
    enable_neck=True라 헤드셋 헤드 트래킹으로 계속 구동되고, neck_write_hardware만
    False라 실제 Dynamixel에는 안 쓴다."""

    config_class = BiPiperXRMujocoConfig
    name = "bi_piper_xr_mujoco"
