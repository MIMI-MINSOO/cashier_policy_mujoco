"""키보드 양팔+목 텔레옵 — VR(Pico) 없이 심 파이프라인을 굴리기 위한 폴백.

BiPiperXRTeleop과 동일한 action dict(12관절 + 2그리퍼 + neck 2축, deg)를
내보내므로 record/robot 쪽에서는 구분 없이 동작한다. EE 목표 pose를 키
입력으로 증분하고 placo IK(XrIKController와 동일한 태스크 구성)로 푼다.

- 키 입력: pynput 전역 리스너 (X11 데스크톱 세션 필요).
  record의 에피소드 제어(→/←/ESC)와 겹치지 않는 키만 사용한다.
- 내부 sleep 없음: 페이싱은 record/teleoperate 루프가 담당.
- send_feedback으로 실제(심) 관절각을 받아 placo 상태를 동기화한다.
"""

import logging
from functools import cached_property

import numpy as np
import meshcat.transformations as tf

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.teleoperators.teleoperator import Teleoperator

from flare.teleoperators.bi_piper_xr.config_bi_piper_xr import PiperArmConfig
from .config_bi_piper_keyboard import BiPiperKeyboardTeleopConfig

try:
    import placo

    PLACO_AVAILABLE = True
except ImportError:
    PLACO_AVAILABLE = False

logger = logging.getLogger(__name__)

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# 목표 pose가 실제 EE에서 이 이상 벗어나면 끌어당김 (IK 불가능 영역으로의 폭주 방지)
_TARGET_LEASH_M = 0.10


class KeyboardIKController:
    """단일 팔 절대 목표 pose 증분 + placo IK (XrIKController의 태스크 구성 재사용)."""

    def __init__(self, config: PiperArmConfig, dt: float):
        self.config = config
        self.robot = placo.RobotWrapper(config.urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.dt = dt
        self.solver.mask_fbase(True)

        # placo state.q 선두 7개는 floating base [x,y,z,qx,qy,qz,qw]
        self.robot.state.q[:7] = np.array([0, 0, 0, 0, 0, 0, 1])
        if len(self.robot.state.q) >= 7 + len(config.joints_init):
            self.robot.state.q[7 : 7 + len(config.joints_init)] = config.joints_init
        self.robot.update_kinematics()

        self.ee_link_name = config.link_name
        ee_transform = self.robot.get_T_world_frame(self.ee_link_name)
        self.target_xyz = ee_transform[:3, 3].copy()
        self.target_quat = tf.quaternion_from_matrix(ee_transform)  # (w, x, y, z)

        self.effector_task = self.solver.add_frame_task(self.ee_link_name, ee_transform)
        self.effector_task.configure("ee", "soft", 1.0)

        manipulability = self.solver.add_manipulability_task(self.ee_link_name, "both", 1.0)
        manipulability.configure("manipulability", "soft", 1e-2)

        self.joints_task = self.solver.add_joints_task()
        default_joints = {
            self.robot.model.names[7 + i]: config.joints_init[i]
            for i in range(len(config.joints_init))
            if 7 + i < len(self.robot.model.names)
        }
        self.joints_task.set_joints(default_joints)
        self.joints_task.configure("joints_regularization", "soft", 1e-3)

        self.solver.add_kinetic_energy_regularization_task(1e-6)
        self.robot.update_kinematics()

    def nudge(self, dxyz: np.ndarray, drpy_rad: np.ndarray):
        """목표 pose를 월드 좌표계에서 증분한다."""
        self.target_xyz = self.target_xyz + dxyz
        if np.any(drpy_rad):
            dq = tf.quaternion_from_euler(*drpy_rad)  # (w, x, y, z)
            self.target_quat = tf.quaternion_multiply(dq, self.target_quat)

    def solve(self) -> np.ndarray:
        """현재 목표 pose로 IK 1스텝을 풀고 관절각(rad 6개)을 반환한다."""
        # 목표가 도달 불가 영역으로 계속 멀어지지 않게 leash로 제한
        actual_xyz = self.robot.get_T_world_frame(self.ee_link_name)[:3, 3]
        err = np.linalg.norm(self.target_xyz - actual_xyz)
        if err > _TARGET_LEASH_M:
            self.target_xyz = actual_xyz + (self.target_xyz - actual_xyz) * (_TARGET_LEASH_M / err)

        target_pose = tf.quaternion_matrix(self.target_quat)
        target_pose[:3, 3] = self.target_xyz
        self.effector_task.T_world_frame = target_pose

        self.solver.solve(True)
        self.robot.update_kinematics()
        return self.robot.state.q[7:13].copy()

    def update_robot_state(self, actual_joints_rad: np.ndarray):
        """실제(심) 관절각으로 placo 내부 상태를 동기화한다 (closed-loop)."""
        if len(actual_joints_rad) == 6:
            self.robot.state.q[7:13] = actual_joints_rad
            self.robot.update_kinematics()


class BiPiperKeyboardTeleop(Teleoperator):
    """키보드로 양팔 EE + 그리퍼 + 목을 조작하는 텔레옵."""

    config_class = BiPiperKeyboardTeleopConfig
    name = "bi_piper_keyboard"

    def __init__(self, config: BiPiperKeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        if not PLACO_AVAILABLE:
            raise ImportError("placo is not installed. pip install placo (teleop extra).")

        self._is_connected = False
        self.left_ik: KeyboardIKController | None = None
        self.right_ik: KeyboardIKController | None = None
        self._listener = None

        self._held: set[str] = set()
        self.active_side = "left"
        # 그리퍼 목표값 (direct 단위) — f키로 열림/닫힘 토글
        self._gripper_pos = {
            "left": config.left_arm.gripper_open_pos,
            "right": config.right_arm.gripper_open_pos,
        }
        self._gripper_closed = {"left": False, "right": False}
        # 목 상태 (deg, 실물 커맨드 공간: +pitch=아래, +yaw=왼쪽)
        nc = config.neck
        self._neck_home = (float(np.degrees(nc.home_pitch_rad)), float(np.degrees(nc.home_yaw_rad)))
        self.neck_pitch, self.neck_yaw = self._neck_home

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = {}
        for side in ["left", "right"]:
            for name in JOINT_NAMES:
                features[f"{side}_{name}.pos"] = float
            features[f"{side}_gripper.pos"] = float
        # neck 키는 항상 포함 — 로봇(action_features 16차원)과 규약을 같이 간다
        features["neck_pitch.pos"] = float
        features["neck_yaw.pos"] = float
        return features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
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

    # ------------------------------------------------------------------
    # 키 입력 (pynput 전역 리스너)
    # ------------------------------------------------------------------

    def _on_press(self, key):
        ch = getattr(key, "char", None)
        if ch is None:
            return
        ch = ch.lower()
        if ch == "z":
            self.active_side = "right" if self.active_side == "left" else "left"
            logger.info(f"[keyboard] active arm -> {self.active_side}")
        elif ch == "f":
            side = self.active_side
            arm_cfg = self.config.left_arm if side == "left" else self.config.right_arm
            self._gripper_closed[side] = not self._gripper_closed[side]
            self._gripper_pos[side] = (
                arm_cfg.gripper_close_pos if self._gripper_closed[side] else arm_cfg.gripper_open_pos
            )
            logger.info(f"[keyboard] {side} gripper -> {'close' if self._gripper_closed[side] else 'open'}")
        elif ch == "h":
            self.neck_pitch, self.neck_yaw = self._neck_home
            logger.info("[keyboard] neck -> home")
        else:
            self._held.add(ch)

    def _on_release(self, key):
        ch = getattr(key, "char", None)
        if ch is not None:
            self._held.discard(ch.lower())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        from pynput import keyboard  # X 세션 필요 — 지연 import

        self.left_ik = KeyboardIKController(self.config.left_arm, self.config.dt)
        self.right_ik = KeyboardIKController(self.config.right_arm, self.config.dt)

        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        self._is_connected = True
        logger.info(
            f"{self.name} connected. z=팔전환 wasdqe=이동 uoikjl=회전 f=그리퍼 "
            f"[]=목yaw ;'=목pitch h=목홈 (에피소드: →종료 ←재녹화 ESC중단)"
        )

    # ------------------------------------------------------------------
    # 텔레옵 루프
    # ------------------------------------------------------------------

    def _jog_deltas(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.config.pos_step_m
        r = float(np.radians(self.config.rot_step_deg))
        held = self._held
        dxyz = np.array(
            [
                p * (("w" in held) - ("s" in held)),
                p * (("a" in held) - ("d" in held)),
                p * (("q" in held) - ("e" in held)),
            ]
        )
        drpy = np.array(
            [
                r * (("o" in held) - ("u" in held)),
                r * (("i" in held) - ("k" in held)),
                r * (("j" in held) - ("l" in held)),
            ]
        )
        return dxyz, drpy

    def _update_neck(self):
        n = self.config.neck_step_deg
        held = self._held
        self.neck_yaw += n * (("[" in held) - ("]" in held))      # +yaw = 왼쪽
        self.neck_pitch += n * (("'" in held) - (";" in held))    # +pitch = 아래
        nc = self.config.neck
        self.neck_pitch = float(
            np.clip(self.neck_pitch, np.degrees(nc.pitch_min_rad), np.degrees(nc.pitch_max_rad))
        )
        yaw_home = self._neck_home[1]
        lim = float(np.degrees(nc.yaw_limit_rad))
        self.neck_yaw = float(np.clip(self.neck_yaw, yaw_home - lim, yaw_home + lim))

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        dxyz, drpy = self._jog_deltas()
        action: dict[str, float] = {}
        for side, ik in [("left", self.left_ik), ("right", self.right_ik)]:
            if side == self.active_side:
                ik.nudge(dxyz, drpy)
            joints = ik.solve()
            for i, name in enumerate(JOINT_NAMES):
                action[f"{side}_{name}.pos"] = float(np.degrees(joints[i]))
            action[f"{side}_gripper.pos"] = float(self._gripper_pos[side])

        self._update_neck()
        action["neck_pitch.pos"] = float(self.neck_pitch)
        action["neck_yaw.pos"] = float(self.neck_yaw)
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """실제(심) 관절각으로 placo 상태를 동기화한다 (record 루프가 매 틱 호출)."""
        if not feedback:
            return
        for side, ik in [("left", self.left_ik), ("right", self.right_ik)]:
            actual = np.zeros(6)
            found = False
            for i, name in enumerate(JOINT_NAMES):
                key = f"{side}_{name}.pos"
                if key in feedback:
                    actual[i] = np.radians(feedback[key])
                    found = True
            if found:
                ik.update_robot_state(actual)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._is_connected = False
        logger.info(f"{self.name} disconnected.")
