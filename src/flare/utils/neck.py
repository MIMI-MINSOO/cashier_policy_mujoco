#!/usr/bin/env python
"""2-DOF Dynamixel pan/tilt neck driven by the headset pose.

Ported from the standalone head_pico.py (verified on hardware):
  - head orientation comes from xrt.get_headset_pose() -> [x,y,z,qx,qy,qz,qw]
  - forward-vector method gives decoupled yaw/pitch (no euler gimbal coupling)
  - delta/clutch mapping: while engaged the neck moves by the head delta from the
    engage reference, on top of the neck pose at engage time (no jump)
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

# --- Dynamixel (RX-28, protocol 1.0) ----------------------------------------
ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_PRESENT_POSITION = 36
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
DXL_CENTER_TICK = 512
DXL_MIN_TICK = 0
DXL_MAX_TICK = 1023

# Clutch button name -> xrt getter
CLUTCH_BUTTONS = {
    "A": "get_A_button",
    "B": "get_B_button",
    "X": "get_X_button",
    "Y": "get_Y_button",
}


@dataclass
class NeckConfig:
    pitch_gain: float = 1.0
    yaw_gain: float = 1.0
    smoothing_factor: float = 0.2

    # Neck angles below are in "command space" (fed to radian_to_tick), AFTER the
    # direction signs are applied. Verified on hardware:
    #   pitch: +down / -up      yaw: +left / -right     (center tick 512 = 0 rad)
    home_pitch_rad: float = math.radians(2.72)   # initial pose (captured from teleop)
    home_yaw_rad: float = math.radians(-1.58)
    pitch_min_rad: float = math.radians(-25.0)   # most "up" allowed
    pitch_max_rad: float = math.radians(55.0)    # most "down" allowed (asymmetric)
    yaw_limit_rad: float = math.radians(80.0)    # +/- around home_yaw

    # Direction signs, calibrated to physical motor mounting (both inverted).
    pitch_sign: float = -1.0
    yaw_sign: float = -1.0

    # Hardware + clutch
    device_name: str = "/dev/ttyACM0"
    baudrate: int = 1000000
    protocol_version: float = 1.0
    pitch_id: int = 1
    yaw_id: int = 2
    clutch_button: str = "A"
    # Tick that corresponds to command 0 (physical forward / level). Calibrated by
    # hand (torque off) because the motor's mechanical center (512) is offset from
    # the neck's true forward.  +deg -> higher tick.
    pitch_center_tick: int = 587
    yaw_center_tick: int = 818
    # consecutive all-zero (head-tracking-off) frames before returning home;
    # absorbs momentary streaming dropouts so the neck doesn't flicker to home
    home_grace_frames: int = 10


def headset_rotation(pose):
    """Convert an xrt headset pose [x,y,z,qx,qy,qz,qw] to a scipy Rotation.

    Returns None if the packet is all-zero (headset not streaming yet)."""
    quat_xyzw = [pose[3], pose[4], pose[5], pose[6]]
    if np.linalg.norm(quat_xyzw) < 1e-6:
        return None
    return Rotation.from_quat(quat_xyzw)


class NeckAngleMapper:
    """Delta (clutch) mapping. While engaged the neck moves by the amount the head
    moved relative to the pose captured at engage time, added on top of the neck
    position at that moment (so engaging never causes a jump)."""

    def __init__(self, config: NeckConfig):
        self.config = config
        self.neutral_rot = None              # head reference, set on engage
        self.base_pitch = config.home_pitch_rad   # neck position at engage time
        self.base_yaw = config.home_yaw_rad
        self.current_pitch = config.home_pitch_rad  # smoothed output
        self.current_yaw = config.home_yaw_rad
        self.last_yaw_deg = 0.0              # measured head delta, for debug
        self.last_pitch_deg = 0.0
        self._invalid_count = 0             # consecutive head-tracking-off frames

    def engage(self, rot):
        """Capture references so tracking starts from the current neck pose."""
        self.neutral_rot = rot
        self.base_pitch = self.current_pitch
        self.base_yaw = self.current_yaw

    def disengage(self):
        self.neutral_rot = None

    def home(self):
        """Head tracking is OFF: smoothly drive the neck back to its home pose."""
        self.neutral_rot = None
        alpha = self.config.smoothing_factor
        self.current_pitch += (self.config.home_pitch_rad - self.current_pitch) * alpha
        self.current_yaw += (self.config.home_yaw_rad - self.current_yaw) * alpha
        return self.current_pitch, self.current_yaw

    def step(self, rot, active):
        """Single control tick combining the head checkbox and the clutch button.

        rot:    headset Rotation, or None when head tracking is OFF (app checkbox
                unchecked -> all-zero pose).
        active: clutch state (button A toggle).

        Behavior:
          head OFF              -> return to home pose
          head ON + clutch OFF  -> hold current pose (frozen)
          head ON + clutch ON   -> track head delta (engages on activation)
        Returns (pitch, yaw) in rad."""
        if rot is None:
            self._invalid_count += 1
            if self._invalid_count >= self.config.home_grace_frames:
                return self.home()
            # within grace window: hold the current pose (absorb dropouts)
            return self.current_pitch, self.current_yaw
        self._invalid_count = 0
        if not active:
            # clutch off: freeze, and drop the reference so the next activation
            # re-engages from the current pose (no jump).
            self.neutral_rot = None
            return self.current_pitch, self.current_yaw
        if self.neutral_rot is None:
            self.engage(rot)   # clutch just turned on -> no-jump engage
        return self.map(rot)

    def map(self, rot):
        """Update neck target from the current head pose (must be engaged)."""
        rel = self.neutral_rot.inv() * rot
        # Forward-vector method: decoupled yaw/pitch, no euler gimbal coupling.
        # OpenXR local convention: forward = -Z, up = +Y. Verified on Pico.
        f = rel.apply([0.0, 0.0, -1.0])
        yaw_meas = math.atan2(f[0], -f[2])                     # head left -, right +
        pitch_meas = math.atan2(f[1], math.hypot(f[0], f[2]))  # head up +, down -
        self.last_yaw_deg = math.degrees(yaw_meas)
        self.last_pitch_deg = math.degrees(pitch_meas)

        delta_pitch = pitch_meas * self.config.pitch_sign * self.config.pitch_gain
        delta_yaw = yaw_meas * self.config.yaw_sign * self.config.yaw_gain

        target_pitch = np.clip(
            self.base_pitch + delta_pitch,
            self.config.pitch_min_rad,
            self.config.pitch_max_rad,
        )
        target_yaw = np.clip(
            self.base_yaw + delta_yaw,
            self.config.home_yaw_rad - self.config.yaw_limit_rad,
            self.config.home_yaw_rad + self.config.yaw_limit_rad,
        )

        alpha = self.config.smoothing_factor
        self.current_pitch += (target_pitch - self.current_pitch) * alpha
        self.current_yaw += (target_yaw - self.current_yaw) * alpha
        return self.current_pitch, self.current_yaw


class DynamixelNeckController:
    def __init__(self, config: NeckConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.port = None
        self.packet = None
        self.port_ready = False
        # 최근 명령값(rad) — dry-run이나 읽기 실패 시 read()의 폴백으로 사용.
        self.last_pitch = config.home_pitch_rad
        self.last_yaw = config.home_yaw_rad

        if not self.dry_run:
            from dynamixel_sdk import PacketHandler, PortHandler

            self.port = PortHandler(config.device_name)
            self.packet = PacketHandler(config.protocol_version)

            if not self.port.openPort():
                raise RuntimeError(f"failed to open Dynamixel port: {config.device_name}")
            if not self.port.setBaudRate(config.baudrate):
                self.port.closePort()
                raise RuntimeError(f"failed to set Dynamixel baudrate: {config.baudrate}")

            self.port_ready = True
            self.enable_torque(config.pitch_id)
            self.enable_torque(config.yaw_id)

    def write(self, pitch, yaw):
        self.last_pitch = float(pitch)
        self.last_yaw = float(yaw)
        if self.dry_run:
            return
        self.write_position(self.config.pitch_id,
                            self.radian_to_tick(pitch, self.config.pitch_center_tick))
        self.write_position(self.config.yaw_id,
                            self.radian_to_tick(yaw, self.config.yaw_center_tick))

    def read(self):
        """Present Position을 읽어 (pitch, yaw) rad로 반환.

        dry-run이거나 통신에 실패하면 최근 명령값을 반환한다 (레코딩 루프가
        일시적 통신 오류로 죽지 않도록)."""
        if self.dry_run:
            return self.last_pitch, self.last_yaw
        pitch = self.read_position_rad(
            self.config.pitch_id, self.config.pitch_center_tick, fallback=self.last_pitch
        )
        yaw = self.read_position_rad(
            self.config.yaw_id, self.config.yaw_center_tick, fallback=self.last_yaw
        )
        return pitch, yaw

    def read_position_rad(self, motor_id, center, fallback):
        tick, comm, err = self.packet.read2ByteTxRx(self.port, motor_id, ADDR_PRESENT_POSITION)
        if comm != 0:
            print(f"ID {motor_id} read failed: {self.packet.getTxRxResult(comm)}")
            return fallback
        if err != 0:
            print(f"ID {motor_id} read packet error: {self.packet.getRxPacketError(err)}")
            return fallback
        return self.tick_to_radian(tick, center)

    def close(self):
        if self.port_ready:
            self.disable_torque(self.config.pitch_id)
            self.disable_torque(self.config.yaw_id)
            self.port.closePort()
            self.port_ready = False

    @staticmethod
    def radian_to_tick(rad, center=DXL_CENTER_TICK):
        tick = int(center + math.degrees(rad) * (1023.0 / 300.0))
        return int(np.clip(tick, DXL_MIN_TICK, DXL_MAX_TICK))

    @staticmethod
    def tick_to_radian(tick, center=DXL_CENTER_TICK):
        # radian_to_tick의 역변환 (같은 center/스케일 규약).
        return math.radians((tick - center) * (300.0 / 1023.0))

    def write_position(self, motor_id, position):
        comm, err = self.packet.write2ByteTxRx(self.port, motor_id, ADDR_GOAL_POSITION, position)
        if comm != 0:
            print(f"ID {motor_id} write failed: {self.packet.getTxRxResult(comm)}")
        elif err != 0:
            print(f"ID {motor_id} packet error: {self.packet.getRxPacketError(err)}")

    def enable_torque(self, motor_id):
        self.set_torque(motor_id, TORQUE_ENABLE)

    def disable_torque(self, motor_id):
        self.set_torque(motor_id, TORQUE_DISABLE)

    def set_torque(self, motor_id, value):
        comm, err = self.packet.write1ByteTxRx(self.port, motor_id, ADDR_TORQUE_ENABLE, value)
        action = "enable" if value == TORQUE_ENABLE else "disable"
        if comm != 0:
            print(f"ID {motor_id} torque {action} failed: {self.packet.getTxRxResult(comm)}")
        elif err != 0:
            print(f"ID {motor_id} torque {action} packet error: {self.packet.getRxPacketError(err)}")
