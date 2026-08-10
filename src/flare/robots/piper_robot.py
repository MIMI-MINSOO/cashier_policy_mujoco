import time
import threading
import numpy as np
from flare.robots.base_robot import BaseRobot


class PiperRobot(BaseRobot):
    """PiPER robot control via piper_sdk CAN bus interface.

    State/Action format: 7D [joint1, ..., joint6, gripper] in degrees.
    Gripper action is in leader (master) space and scaled to follower (slave) before sending.

    Features:
    - Optional 200Hz interpolation thread for smooth motion
    - Smooth transition between consecutive actions
    """

    # Safe rest position
    DEFAULT_JOINTS_INIT = [-0.51, 0.49, 1.6, 3.96, 4.83, -8.33]
    # Task-ready position (mean of dataset initial poses — term-project_ljw, 50 episodes)
    EVAL_INIT_JOINT_DEG = [-0.20, 35.31, -57.85, -0.36, 60.46, 0.74]

    def __init__(
        self,
        can_name: str = "can_slave",
        init_move_speed_pct: int = 50,
        joints_init: list[float] | None = DEFAULT_JOINTS_INIT,
        eval_init: list[float] | None = EVAL_INIT_JOINT_DEG,
        enable_timeout: float = 5.0,
        return_to_init_on_disconnect: bool = True,
        disable_on_disconnect: bool = True,
        use_interpolation: bool = True,
        interpolation_freq: float = 180.0,
    ):
        self.can_name = can_name
        self.init_move_speed_pct = init_move_speed_pct
        self.joints_init = joints_init
        self.eval_init = eval_init
        self.enable_timeout = enable_timeout
        self.return_to_init_on_disconnect = return_to_init_on_disconnect
        self.disable_on_disconnect = disable_on_disconnect
        self.use_interpolation = use_interpolation
        self.interpolation_freq = interpolation_freq
        self._piper = None
        self._connected = False

        # Interpolation thread state
        self._interp_thread = None
        self._interp_stop = threading.Event()
        self._target_action = None
        self._prev_target = None
        self._next_target = None   # look-ahead for velocity estimation
        self._target_time = None
        self._action_period = 1.0 / 30.0
        self._target_lock = threading.Lock()

    def connect(self):
        from piper_sdk import C_PiperInterface_V2

        self._piper = C_PiperInterface_V2(self.can_name)
        self._piper.ConnectPort()
        self._piper.EnableArm(7)
        self._wait_for_enable()

        self._piper.SetSDKJointLimitParam("j4", -1.7977, 1.7977)
        self._piper.SetSDKJointLimitParam("j5", -1.4265, 1.4265)
        self._piper.SetSDKJointLimitParam("j6", -2.0071, 2.2166)

        if self.joints_init is not None:
            self._move_to_init()
            time.sleep(5.0)

        # Start interpolation thread
        if self.use_interpolation:
            self._interp_stop.clear()
            self._interp_thread = threading.Thread(target=self._interpolation_loop, daemon=True)
            self._interp_thread.start()

        self._connected = True
        mode = f"interpolation {self.interpolation_freq}Hz" if self.use_interpolation else "direct"
        print(f"PiperRobot connected on {self.can_name} ({mode})")

    def _wait_for_enable(self):
        start = time.time()
        while time.time() - start < self.enable_timeout:
            msgs = self._piper.GetArmLowSpdInfoMsgs()
            all_enabled = all(
                getattr(msgs, f"motor_{i}").foc_status.driver_enable_status
                for i in range(1, 7)
            )
            if all_enabled:
                print("All motors enabled.")
                return
            self._piper.EnableArm(7)
            self._piper.GripperCtrl(0, 5000, 0x01, 0)
            time.sleep(0.5)
        raise TimeoutError(f"Failed to enable PiPER arm within {self.enable_timeout}s")

    def _move_to_init(self):
        if self.joints_init is None:
            return
        self._piper.MotionCtrl_2(0x01, 0x01, self.init_move_speed_pct, 0x00)
        joints_raw = [int(j * 1000) for j in self.joints_init[:6]]
        self._piper.JointCtrl(*joints_raw)
        self._piper.GripperCtrl(0, 1000, 0x01, 0)

    @staticmethod
    def _quintic_coeffs(s0, sf, v0, vf, a0, af, T):
        """Compute quintic polynomial coefficients for given boundary conditions.

        s(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵

        Boundary conditions:
            s(0)=s0, v(0)=v0, a(0)=a0
            s(T)=sf, v(T)=vf, a(T)=af
        """
        c0 = s0
        c1 = v0
        c2 = a0 / 2.0
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T
        # Solve 3x3 system for c3, c4, c5
        ds = sf - s0 - v0 * T - (a0 / 2.0) * T2
        dv = vf - v0 - a0 * T
        da = af - a0
        c3 = (20 * ds - (8 * dv + da * T) * T) / (2 * T3)
        c4 = (-30 * ds + (14 * dv + 2 * da * T) * T) / (2 * T4)
        c5 = (12 * ds - (6 * dv + da * T) * T) / (2 * T5)
        return np.array([c0, c1, c2, c3, c4, c5])

    @staticmethod
    def _quintic_eval(coeffs, t):
        """Evaluate quintic polynomial at time t."""
        return coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5

    @staticmethod
    def _quintic_vel(coeffs, t):
        """Evaluate quintic polynomial velocity at time t."""
        return coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 + 4*coeffs[4]*t**3 + 5*coeffs[5]*t**4

    def _interpolation_loop(self):
        """High-frequency loop: quintic interpolation with velocity continuity."""
        dt = 1.0 / self.interpolation_freq
        action_dim = 7  # 6 joints + 1 gripper
        current_vel = np.zeros(action_dim)
        coeffs = None  # (action_dim, 6) quintic coefficients per joint

        self._piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)

        while not self._interp_stop.is_set():
            t_start = time.perf_counter()

            with self._target_lock:
                target = self._target_action
                prev = self._prev_target
                next_tgt = self._next_target
                target_time = self._target_time
                T = self._action_period

            if target is not None and prev is not None and target_time is not None:
                elapsed = time.perf_counter() - target_time
                t = min(elapsed, T)

                # Recompute coefficients when target changes
                if coeffs is None or elapsed < dt * 1.5:
                    v0 = current_vel.copy()
                    # vf: next target과의 변화량에 비례, 감쇠 적용
                    if next_tgt is not None:
                        vf = 0.4 * (next_tgt - target) / T
                    else:
                        vf = np.zeros(action_dim)
                    coeffs = np.stack([
                        self._quintic_coeffs(prev[j], target[j], v0[j], vf[j], 0.0, 0.0, T)
                        for j in range(action_dim)
                    ])

                # Evaluate position and velocity
                interp = np.array([self._quintic_eval(coeffs[j], t) for j in range(action_dim)])
                current_vel = np.array([self._quintic_vel(coeffs[j], t) for j in range(action_dim)])

                joints_raw = [int(interp[i] * 1000) for i in range(6)]
                self._piper.JointCtrl(*joints_raw)

                gripper_val = interp[6]
                gripper_scaled = -2.4 + (101.4 / 72.75) * (gripper_val + 1.75)
                self._piper.GripperCtrl(int(gripper_scaled * 1000), 1000, 0x01, 0)
            elif target is not None and prev is None:
                # 첫 action: 그대로 전송
                joints_raw = [int(target[i] * 1000) for i in range(6)]
                self._piper.JointCtrl(*joints_raw)
                gripper_val = target[6]
                gripper_scaled = -2.4 + (101.4 / 72.75) * (gripper_val + 1.75)
                self._piper.GripperCtrl(int(gripper_scaled * 1000), 1000, 0x01, 0)

            elapsed_loop = time.perf_counter() - t_start
            sleep_time = dt - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)

    def disconnect(self):
        if self._piper is None:
            return

        # Stop interpolation thread
        self._interp_stop.set()
        if self._interp_thread is not None:
            self._interp_thread.join(timeout=1.0)
            self._interp_thread = None

        try:
            if self.return_to_init_on_disconnect and self.joints_init is not None:
                print("Returning to initial position...")
                # Direct control for disconnect (no interpolation)
                self._piper.MotionCtrl_2(0x01, 0x01, self.init_move_speed_pct, 0x00)
                joints_raw = [int(j * 1000) for j in self.joints_init[:6]]
                self._piper.JointCtrl(*joints_raw)
                self._piper.GripperCtrl(0, 1000, 0x01, 0)
                time.sleep(5.0)
            if self.disable_on_disconnect:
                self._piper.DisableArm(7)
                print("Arm disabled (torque off).")
        except Exception as e:
            print(f"Error during disconnect: {e}")
        self._piper = None
        self._connected = False
        print("PiperRobot disconnected")

    def reset_filter(self):
        """Reset interpolation state (call when starting new episode)."""
        with self._target_lock:
            self._target_action = None
            self._prev_target = None
            self._next_target = None
            self._target_time = None
            self._action_period = 1.0 / 30.0

    def move_to_eval_pose(self, duration: float = 5.0):
        """Move robot to task-ready eval pose (uses eval_init or joints_init fallback).

        Bypasses the interpolation thread and sends a direct JointCtrl + open gripper.
        Blocks for `duration` seconds to allow the move to complete.
        """
        joints = self.eval_init if self.eval_init else self.joints_init
        if joints is None or self._piper is None:
            return
        self._piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)  # 30% speed
        self._piper.JointCtrl(*[int(j * 1000) for j in joints[:6]])
        self._piper.GripperCtrl(int(70.0 * 1000), 1000, 0x01, 0)  # open gripper
        time.sleep(duration)

    def get_state(self) -> np.ndarray:
        joint_msg = self._piper.GetArmJointMsgs()
        gripper_msg = self._piper.GetArmGripperMsgs()
        return np.array([
            joint_msg.joint_state.joint_1 * 0.001,
            joint_msg.joint_state.joint_2 * 0.001,
            joint_msg.joint_state.joint_3 * 0.001,
            joint_msg.joint_state.joint_4 * 0.001,
            joint_msg.joint_state.joint_5 * 0.001,
            joint_msg.joint_state.joint_6 * 0.001,
            gripper_msg.gripper_state.grippers_angle * 0.001,
        ], dtype=np.float32)

    def send_action(self, action: np.ndarray, next_action: np.ndarray = None):
        if self.use_interpolation and self._interp_thread is not None:
            with self._target_lock:
                now = time.perf_counter()
                if self._target_action is None:
                    self._prev_target = action.copy()
                else:
                    self._prev_target = self._target_action.copy()
                    if self._target_time is not None:
                        measured = now - self._target_time
                        self._action_period = max(measured, 1.0 / 60.0)
                self._target_action = action.copy()
                self._next_target = next_action.copy() if next_action is not None else None
                self._target_time = now
        else:
            # Direct control (no interpolation)
            self._piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
            joints_raw = [int(action[i] * 1000) for i in range(6)]
            self._piper.JointCtrl(*joints_raw)

            gripper_val = action[6]
            gripper_scaled = -2.4 + (101.4 / 72.75) * (gripper_val + 1.75)
            self._piper.GripperCtrl(int(gripper_scaled * 1000), 1000, 0x01, 0)

    def get_end_pose(self) -> np.ndarray:
        pose_msg = self._piper.GetArmEndPoseMsgs()
        return np.array([
            pose_msg.end_pose.X_axis * 0.001,
            pose_msg.end_pose.Y_axis * 0.001,
            pose_msg.end_pose.Z_axis * 0.001,
            pose_msg.end_pose.RX_axis * 0.001,
            pose_msg.end_pose.RY_axis * 0.001,
            pose_msg.end_pose.RZ_axis * 0.001,
        ], dtype=np.float32)

    def is_connected(self) -> bool:
        return self._connected
