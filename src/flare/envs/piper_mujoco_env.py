"""MuJoCo simulation environment mirroring PiperRealEnv's interface.

Loads piper.xml (single) / piper_dual.xml (dual) / piper_dual_capstone.xml
and exposes the same connect/disconnect/reset/step/get_obs API as
PiperRealEnv, with the same unit conventions as the real robot stack
(flare/robots/piper_robot.py):

    - observation.state / action: (7,) single / (14,) dual /
        (16,) dual+neck, float32 — 순서는 state_feature_names() 참조.
        [joint1..joint6 in degrees, gripper] per arm
        (gripper follows the piper_sdk feedback convention:
         grippers_angle * 0.001 -> here 2 * finger_qpos[m] * 1000;
         actions use the PiperRobot.send_action affine calibration:
         scaled = -2.4 + (101.4 / 72.75) * (val + 1.75))
        neck: [pitch, yaw] in degrees, 실물 커맨드 공간 규약
        (neck.py NeckConfig: pitch +아래/-위, yaw +왼쪽/-오른쪽).

camera_names를 주면 get_obs()가 각 카메라를 렌더링해
observation.images.<키>로 내보낸다 (raw uint8 병행 반환).

torch is optional: if unavailable (minimal mujoco-only venv), get_obs()
returns numpy arrays instead of torch tensors.
"""

import time
from pathlib import Path

import numpy as np
import mujoco

try:
    import torch
except ImportError:  # minimal sim venv without the full flare install
    torch = None

_ASSETS_DIR = Path(__file__).parents[1] / "assets"

DEFAULT_XML = {
    "single": _ASSETS_DIR / "piper.xml",
    "dual": _ASSETS_DIR / "piper_dual.xml",
}

# piper_dual_capstone.xml의 바구니/상품 body 이름 (랜덤 초기화용)
CAPSTONE_BASKET = "basket"
# 2026-08-10: spam 제외 나머지는 MJCF에서도 임시 비활성화(주석)됨 - 나중에 다시 켤 예정.
CAPSTONE_PRODUCTS = [
    # "acafela",
    # "hotsix",
    # "dalbam",
    "spam",
    # "flavono",
]

# piper_dual_capstone.xml의 카메라: obs 키 -> MJCF 카메라 이름.
# head = 목(h2)의 Femto Bolt (메인 학습용), top = 헤드 플레이트의 D435 (sub/대체 학습용),
# left/right_wrist = 그리퍼 손목의 D405 (장착 위치는 임시값)
CAPSTONE_CAMERAS = {
    "head": "head_cam",
    "top": "top_cam",
    "left_wrist": "left_wrist_cam",
    "right_wrist": "right_wrist_cam",
}

# 상품 body 이름 -> EAN-13 (스캔 이벤트 보고용)
CAPSTONE_EAN13 = {
    "acafela": "8801104667043",
    "spam": "8801007519456",
    "flavono": "8801062323159",
    "dalbam": "8809821801283",
    "hotsix": "8801056240998",
}

# Actuator name layout per arm; must match the <actuator> blocks in the MJCF.
_ARM_ACTUATORS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

# 목(2-DOF) 액추에이터/조인트 이름 (piper_dual_capstone.xml).
# h1 = yaw (axis z), h2 = pitch (axis -y).
NECK_ACTUATORS = {"pitch": "h2_joint_pos", "yaw": "h1_joint_pos"}
NECK_JOINTS = {"pitch": "h2_joint", "yaw": "h1_joint"}

# MJCF 조인트 rad <-> 실물 커맨드 공간 deg 부호.
# 실물 규약(neck.py NeckConfig, 하드웨어 검증): +pitch=아래, +yaw=왼쪽.
# 2026-07-10 head_cam 시선 벡터로 수치 검증: h1(+qpos)=왼쪽, h2(+qpos)=아래
# — MJCF 축이 실물 규약과 일치하므로 부호 둘 다 +1 확정.
NECK_PITCH_SIGN = 1.0
NECK_YAW_SIGN = 1.0


def state_feature_names(robot_mode: str = "dual", enable_neck: bool = True) -> list[str]:
    """state/action 벡터의 차원 순서 규약 (전 구간 단일 소스).

    수집(BiPiperMujocoFollower features), 텔레옵, 학습 데이터, 추론(eval_sim)이
    모두 이 순서를 따른다. 값은 전부 도(deg):
        [<arm>joint1..6, <arm>gripper] x (single: 1 / dual: left,right)
        + [neck_pitch, neck_yaw]  (enable_neck일 때)
    """
    prefixes = [""] if robot_mode == "single" else ["left_", "right_"]
    names = []
    for p in prefixes:
        names += [f"{p}joint{k}.pos" for k in range(1, 7)] + [f"{p}gripper.pos"]
    if enable_neck:
        names += ["neck_pitch.pos", "neck_yaw.pos"]
    return names

# Gripper calibration constants from PiperRobot.send_action.
_GRIP_OFFSET = -2.4
_GRIP_SCALE = 101.4 / 72.75
_GRIP_SHIFT = 1.75


class PiperMujocoEnv:
    """MuJoCo twin of PiperRealEnv (same reset/step/get_obs contract).

    Like PiperRealEnv, reset() and step() return the (obs, raw_images)
    tuple produced by get_obs().
    """

    def __init__(
        self,
        robot_mode: str = "dual",
        xml_path: str | Path | None = None,
        image_size: tuple[int, int] = (240, 320),
        control_freq: float = 30.0,
        camera_names: dict[str, str] | None = None,
        product_names: list[str] | None = None,
        basket_name: str | None = None,
        randomize_products: bool = True,
        seed: int | None = None,
        enable_neck: bool | None = None,
    ):
        """
        Args:
            robot_mode: 'single' or 'dual'.
            xml_path: MJCF to load. Defaults to piper.xml / piper_dual.xml
                from flare/assets depending on robot_mode.
            image_size: (H, W) for rendered camera images (unused until
                cameras are added to the scene).
            control_freq: Control loop frequency in Hz. Each step() advances
                the simulation by 1/control_freq seconds.
            camera_names: obs 키 -> MJCF 카메라 이름 매핑 (예: CAPSTONE_CAMERAS).
                지정하면 get_obs()가 각 카메라를 image_size로 렌더링해
                obs["observation.images.<키>"]로 내보낸다. None = 이미지 없음.
            product_names: free-joint product bodies to randomize inside the
                basket on reset (e.g. CAPSTONE_PRODUCTS). None = no products.
            basket_name: basket body name (e.g. CAPSTONE_BASKET). The basket
                itself stays at its keyframe pose.
            randomize_products: if False, products keep their keyframe poses.
            seed: RNG seed for reproducible product layouts.
            enable_neck: 목(h1/h2)을 state/action에 포함할지. None이면 모델에
                목 액추에이터가 있을 때 자동 활성. True인데 모델에 목이 없으면
                에러. 활성 시 state/action 끝에 [neck_pitch, neck_yaw](deg)가
                붙어 dual 기준 16차원이 된다 (state_feature_names() 참조).
        """
        if robot_mode not in DEFAULT_XML:
            raise ValueError(f"Unknown robot_mode='{robot_mode}'. Use 'single' or 'dual'.")
        self.robot_mode = robot_mode
        self.xml_path = str(xml_path if xml_path is not None else DEFAULT_XML[robot_mode])

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.image_size = image_size
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self._substeps = max(1, round(self.dt / self.model.opt.timestep))

        # Resolve actuator ids / joint qpos addresses arm by arm.
        prefixes = [""] if robot_mode == "single" else ["left_", "right_"]
        self._arm_act_ids = []       # per arm: 7 actuator ids
        self._arm_joint_qadr = []    # per arm: qpos addr of joint1..6
        self._arm_finger_qadr = []   # per arm: qpos addr of the actuated finger
        self._finger_ctrl_lo = []
        self._finger_ctrl_hi = []
        for prefix in prefixes:
            act_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + n)
                for n in _ARM_ACTUATORS
            ]
            if min(act_ids) < 0:
                missing = [n for n, i in zip(_ARM_ACTUATORS, act_ids) if i < 0]
                raise ValueError(f"Actuators not found in {self.xml_path}: {prefix}{missing}")
            self._arm_act_ids.append(act_ids)
            qadr = []
            for k in range(1, 7):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}joint{k}")
                qadr.append(self.model.jnt_qposadr[jid])
            self._arm_joint_qadr.append(qadr)
            finger_jid = self.model.actuator_trnid[act_ids[6], 0]
            self._arm_finger_qadr.append(self.model.jnt_qposadr[finger_jid])
            lo, hi = self.model.actuator_ctrlrange[act_ids[6]]
            self._finger_ctrl_lo.append(lo)
            self._finger_ctrl_hi.append(hi)

        # 목(2-DOF): [pitch, yaw] 순서로 액추에이터/조인트 주소 해석
        neck_act = {
            k: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for k, n in NECK_ACTUATORS.items()
        }
        has_neck = min(neck_act.values()) >= 0
        if enable_neck is None:
            enable_neck = has_neck
        elif enable_neck and not has_neck:
            missing = [n for k, n in NECK_ACTUATORS.items() if neck_act[k] < 0]
            raise ValueError(f"enable_neck=True but actuators missing in {self.xml_path}: {missing}")
        self.enable_neck = enable_neck
        self._neck_act_ids: list[int] = []
        self._neck_qadr: list[int] = []
        self._neck_signs = np.array([NECK_PITCH_SIGN, NECK_YAW_SIGN])
        if enable_neck:
            for k in ("pitch", "yaw"):
                self._neck_act_ids.append(neck_act[k])
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, NECK_JOINTS[k])
                self._neck_qadr.append(self.model.jnt_qposadr[jid])

        self.action_dim = 7 * len(prefixes) + (2 if enable_neck else 0)
        self.state_names = state_feature_names(robot_mode, enable_neck)
        self._home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._connected = False

        # EvalRunner 호환 로봇 표면 (reset_filter/move_to_eval_pose/send_action/idle_action)
        self.robot = SimRobotShim(self)
        # eval_sim 등이 passive 뷰어를 붙이면 apply_action마다 sync된다
        self.viewer = None

        # 카메라 렌더링 세팅 (렌더러는 첫 get_obs 때 생성 — GL 컨텍스트 지연 초기화)
        self._camera_map = dict(camera_names) if camera_names else {}
        for key, cam in self._camera_map.items():
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
                raise ValueError(f"camera '{cam}' (obs key '{key}') not found in {self.xml_path}")
        self._renderer = None

        # 바코드 스캐너 (모델에 'barcode_scanner' site가 있을 때만 활성).
        # 판정: 바코드 데칼 중심이 스캔 볼륨(반경/높이) 안 + 바코드 면이 아래를 향함(허용각)
        #       + dwell_time 연속 유지. 파라미터는 실물 스캐너 특성 확인 전 임시값.
        self._scanner_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "barcode_scanner")
        self._scanner_indicator = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "scanner_indicator")
        self.scan_radius = 0.07       # 구멍 중심 기준 수평 반경 [m]
        self.scan_height = 0.25       # 구멍 위 인식 높이 [m]
        self.scan_max_angle_deg = 60  # 바코드 법선과 '수직 아래' 사이 허용각
        self.scan_dwell_time = 0.2    # 연속 만족 시간 [s]
        self._barcode_geoms: dict[str, int] = {}
        if self._scanner_site >= 0 and product_names:
            for n in product_names:
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{n}_barcode")
                if gid >= 0:
                    self._barcode_geoms[n] = gid
        self._scan_streak: dict[str, int] = {n: 0 for n in self._barcode_geoms}
        self._indicator_steps = 0
        self.last_scan: dict | None = None   # 직전 step에서 발생한 스캔 이벤트
        self.scan_events: list[dict] = []    # 에피소드 내 누적 이벤트 (reset 시 초기화)

        # 상품 랜덤 초기화 세팅
        self._rng = np.random.default_rng(seed)
        self._randomize = randomize_products
        self._basket_id = -1
        self._product_ids: list[int] = []
        if product_names:
            if basket_name is None:
                raise ValueError("product_names를 쓰려면 basket_name도 지정해야 합니다.")
            self._basket_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, basket_name)
            if self._basket_id < 0:
                raise ValueError(f"basket body '{basket_name}' not found in {self.xml_path}")
            for n in product_names:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                if bid < 0:
                    raise ValueError(f"product body '{n}' not found in {self.xml_path}")
                self._product_ids.append(bid)
            # 바구니 내부 절반 크기: 'basket_bottom' geom에서 유도
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "basket_bottom")
            if gid < 0:
                raise ValueError("geom 'basket_bottom'이 없어 바구니 내부 크기를 알 수 없습니다.")
            self._basket_inner = self.model.geom_size[gid][:2].copy()

    # ------------------------------------------------------------------
    # PiperRealEnv-compatible lifecycle (no devices in simulation).
    # ------------------------------------------------------------------

    def connect(self):
        self._connected = True
        print("PiperMujocoEnv ready.")

    def disconnect(self):
        self._connected = False
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        print("PiperMujocoEnv disconnected.")

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Unit conversions (real-robot conventions <-> MuJoCo qpos/ctrl).
    # ------------------------------------------------------------------

    def _gripper_action_to_ctrl(self, val: float, arm: int) -> float:
        # Same calibration as PiperRobot.send_action, producing the commanded
        # stroke that GripperCtrl receives; MuJoCo drives one finger in meters,
        # so stroke[mm] -> per-finger slide = stroke / 1000 / 2.
        stroke_mm = _GRIP_OFFSET + _GRIP_SCALE * (val + _GRIP_SHIFT)
        ctrl = stroke_mm / 1000.0 / 2.0
        return float(np.clip(ctrl, self._finger_ctrl_lo[arm], self._finger_ctrl_hi[arm]))

    def _gripper_state_from_qpos(self, arm: int) -> float:
        # Inverse of the feedback path in PiperRobot.get_state
        # (grippers_angle * 0.001 -> commanded stroke value).
        return float(self.data.qpos[self._arm_finger_qadr[arm]] * 2.0 * 1000.0)

    # ------------------------------------------------------------------
    # Env API
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """현재 state 벡터 (7/14/16,) float32, state_feature_names() 순서 (deg)."""
        state = []
        for arm in range(len(self._arm_act_ids)):
            for adr in self._arm_joint_qadr[arm]:
                state.append(np.degrees(self.data.qpos[adr]))
            state.append(self._gripper_state_from_qpos(arm))
        if self.enable_neck:
            for i, adr in enumerate(self._neck_qadr):
                state.append(self._neck_signs[i] * np.degrees(self.data.qpos[adr]))
        return np.asarray(state, dtype=np.float32)

    def render_images(self) -> dict[str, np.ndarray]:
        """camera_names에 지정된 각 카메라를 렌더링해 {키: (H,W,3) uint8 RGB} 반환."""
        raw_images: dict[str, np.ndarray] = {}
        if self._camera_map:
            if self._renderer is None:
                h, w = self.image_size
                self._renderer = mujoco.Renderer(self.model, height=h, width=w)
            for key, cam in self._camera_map.items():
                self._renderer.update_scene(self.data, camera=cam)
                raw_images[key] = self._renderer.render()  # (H, W, 3) uint8 RGB
        return raw_images

    def get_obs(self):
        """Get current observation matching training format.

        Returns:
            obs: dict with
                - 'observation.state': (7,)/(14,)/(16,) float32 — torch tensor면 torch, 아니면 numpy
                - 'observation.images.<키>': (3, H, W) float32 [0,1] — camera_names에 지정된 각 카메라
            raw_images: dict {키: (H, W, 3) uint8 RGB} — 시각화/로깅용 원본.
        """
        state = self.get_state()
        obs = {
            "observation.state": torch.from_numpy(state) if torch is not None else state,
        }

        raw_images = self.render_images()
        for key, image in raw_images.items():
            if torch is not None:
                image_t = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            else:
                image_t = image.astype(np.float32).transpose(2, 0, 1) / 255.0
            obs[f"observation.images.{key}"] = image_t

        return obs, raw_images

    def apply_action(self, action: np.ndarray):
        """Apply action and advance the simulation by one control period.

        Args:
            action: (7,)/(14,)/(16,) — state_feature_names() 순서
                ([joint1..6 deg, gripper] per arm + [neck_pitch, neck_yaw] deg),
                same convention as PiperRealEnv/PiperRobot.
        """
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape ({self.action_dim},), got {action.shape}")

        # 실물 파이프라인은 30Hz 액션 사이를 고주파 보간 루프(piper_robot._interpolation_loop,
        # quintic)로 연결한다 — 시뮬도 이전 목표→새 목표를 substep에 걸쳐 선형 보간해서
        # 목표각 스텝 점프로 인한 비현실적 whip(파지 물체 이탈)을 방지한다.
        prev = self.data.ctrl.copy()
        target = prev.copy()
        for arm, act_ids in enumerate(self._arm_act_ids):
            base = arm * 7
            for k in range(6):
                target[act_ids[k]] = np.radians(action[base + k])
            target[act_ids[6]] = self._gripper_action_to_ctrl(action[base + 6], arm)
        if self.enable_neck:
            for i, aid in enumerate(self._neck_act_ids):
                lo, hi = self.model.actuator_ctrlrange[aid]
                target[aid] = np.clip(
                    self._neck_signs[i] * np.radians(action[-2 + i]), lo, hi
                )

        for i in range(self._substeps):
            frac = (i + 1) / self._substeps
            self.data.ctrl[:] = prev + (target - prev) * frac
            mujoco.mj_step(self.model, self.data)

        self._update_scanner()
        self._sync_viewer()

    def _sync_viewer(self):
        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()
            else:
                self.viewer = None

    def step(self, action: np.ndarray, next_action: np.ndarray = None):
        """apply_action + get_obs. PiperRealEnv와 동일 시그니처.

        next_action: accepted for signature compatibility; unused in sim
            (it only feeds the real robot's velocity interpolation).
        """
        self.apply_action(action)
        return self.get_obs()

    def _update_scanner(self):
        """기하 판정 바코드 스캔: 조건 만족이 dwell_time 지속되면 이벤트 발생.

        결과는 self.last_scan(이번 step 이벤트, 없으면 None)과
        self.scan_events(에피소드 누적)로 노출된다. 시각 피드백으로
        scanner_indicator geom이 잠시 초록색이 된다.
        """
        self.last_scan = None
        if self._scanner_site < 0 or not self._barcode_geoms:
            return
        m, d = self.model, self.data
        hole = d.site_xpos[self._scanner_site]
        need = max(1, round(self.scan_dwell_time * self.control_freq))
        cos_tol = np.cos(np.radians(self.scan_max_angle_deg))
        for name, gid in self._barcode_geoms.items():
            p = d.geom_xpos[gid]
            in_zone = (
                np.hypot(p[0] - hole[0], p[1] - hole[1]) <= self.scan_radius
                and hole[2] < p[2] <= hole[2] + self.scan_height
            )
            # 데칼 geom의 로컬 +z = 바코드 면 바깥 법선; 아래(-z)를 향해야 스캐너가 읽음
            normal = d.geom_xmat[gid].reshape(3, 3)[:, 2]
            facing_down = -normal[2] >= cos_tol
            if in_zone and facing_down:
                self._scan_streak[name] += 1
                if self._scan_streak[name] == need:
                    event = {
                        "product": name,
                        "ean13": CAPSTONE_EAN13.get(name),
                        "time": float(d.time),
                    }
                    self.last_scan = event
                    self.scan_events.append(event)
                    self._indicator_steps = round(0.5 * self.control_freq)
                    print(f"[barcode] scanned: {name} ({event['ean13']}) t={d.time:.2f}s")
            else:
                self._scan_streak[name] = 0
        # 표시등: 스캔 직후 0.5초 초록
        if self._scanner_indicator >= 0:
            if self._indicator_steps > 0:
                m.geom_rgba[self._scanner_indicator] = [0.1, 0.9, 0.2, 1.0]
                self._indicator_steps -= 1
            else:
                m.geom_rgba[self._scanner_indicator] = [0.35, 0.35, 0.35, 1.0]

    def reset(self):
        """Reset to the MJCF 'home' keyframe (+ 상품 랜덤 배치) and return the initial observation."""
        if self._home_key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        if self._product_ids and self._randomize:
            self._randomize_products()
        self._scan_streak = {n: 0 for n in self._barcode_geoms}
        self._indicator_steps = 0
        self.last_scan = None
        self.scan_events = []
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def _randomize_products(self):
        """바구니 내부에 상품들을 무작위 xy/yaw로 배치 후 물리로 정착시킨다.

        배치 규칙: 각 상품의 경계반경(rbound) 기준으로 겹치지 않게 xy를
        기각샘플링하고, 자리가 안 나면 이미 놓인 상품 위 높이로 쌓는다.
        """
        m, d = self.model, self.data
        # 바구니가 free body면 qpos에서, 월드 고정이면 body_pos에서 위치를 읽는다
        if m.body_jntnum[self._basket_id] > 0:
            basket_pos = d.qpos[m.jnt_qposadr[m.body_jntadr[self._basket_id]]:][:3]
        else:
            basket_pos = m.body_pos[self._basket_id]
        bottom_top_z = basket_pos[2] + 0.006
        inner_x, inner_y = self._basket_inner

        placed: list[tuple[np.ndarray, float]] = []  # (xy, 반경)
        order = self._rng.permutation(len(self._product_ids))
        for k, idx in enumerate(order):
            bid = self._product_ids[idx]
            r = float(m.body_geomnum[bid] and max(
                m.geom_rbound[g] for g in range(m.body_geomadr[bid], m.body_geomadr[bid] + m.body_geomnum[bid])
                if m.geom_contype[g] or m.geom_conaffinity[g]
            ))
            # 바구니 벽에서 반경만큼 안쪽 (내부보다 큰 물체는 중앙 근처로 제한)
            mx = max(inner_x - r, 0.01)
            my = max(inner_y - r, 0.01)
            stack = 0.0
            for _ in range(40):
                xy = self._rng.uniform([-mx, -my], [mx, my])
                if all(np.linalg.norm(xy - p) >= r + pr for p, pr in placed):
                    break
            else:
                stack = max((pr for _, pr in placed), default=0.0) * 2.0  # 자리 없으면 위로
            placed.append((xy, r))

            adr = m.jnt_qposadr[m.body_jntadr[bid]]
            yaw = self._rng.uniform(-np.pi, np.pi)
            d.qpos[adr:adr + 3] = [basket_pos[0] + xy[0], basket_pos[1] + xy[1],
                                   bottom_top_z + r + stack + 0.005]
            d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            d.qvel[m.jnt_dofadr[m.body_jntadr[bid]]:m.jnt_dofadr[m.body_jntadr[bid]] + 6] = 0

        # 낙하 정착 (home ctrl 유지) — 2초: 1초로는 바닥 접촉이 과도 상태(~4mm 관통)로 남음
        if self._home_key >= 0:
            d.ctrl[:] = m.key_ctrl[self._home_key]
        for _ in range(int(2.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)


class SimRobotShim:
    """EvalRunner가 기대하는 로봇 표면을 PiperMujocoEnv 위에 흉내 낸다.

    실물 경로에서는 env.robot이 PiperRobot/DualArmRobot이고 EvalRunner가
    reset_filter / move_to_eval_pose / send_action(idle 유지) / eval_init을
    호출한다. 심에서는 물리가 env 한 곳에 있으므로 전부 env로 위임한다.
    """

    def __init__(self, env: "PiperMujocoEnv"):
        self._env = env

    def reset_filter(self):
        pass

    @property
    def idle_action(self) -> np.ndarray:
        """home keyframe에서 유도한 유지용 액션 (state_feature_names 순서, deg).

        그리퍼는 state 규약(스트로크 mm) 값을 그대로 액션으로 쓴다 — affine
        캘리브레이션을 거치면 열림 위치에서 포화되어 열림 유지가 보장된다.
        """
        env = self._env
        if env._home_key < 0:
            return env.get_state()
        q = env.model.key_qpos[env._home_key]
        vals: list[float] = []
        for arm in range(len(env._arm_act_ids)):
            vals += [float(np.degrees(q[adr])) for adr in env._arm_joint_qadr[arm]]
            vals.append(float(q[env._arm_finger_qadr[arm]] * 2.0 * 1000.0))
        if env.enable_neck:
            vals += [
                float(env._neck_signs[i] * np.degrees(q[adr]))
                for i, adr in enumerate(env._neck_qadr)
            ]
        return np.asarray(vals, dtype=np.float32)

    def get_state(self) -> np.ndarray:
        return self._env.get_state()

    def move_to_eval_pose(self, duration: float = 5.0):
        """현재 자세에서 home keyframe ctrl로 duration 동안 선형 보간 이동."""
        env = self._env
        m, d = env.model, env.data
        if env._home_key < 0:
            return
        start = d.ctrl.copy()
        target = m.key_ctrl[env._home_key]
        n = max(1, int(duration / m.opt.timestep))
        sync_every = max(1, round((1.0 / 30.0) / m.opt.timestep))  # 뷰어 ~30fps
        for i in range(n):
            d.ctrl[:] = start + (target - start) * ((i + 1) / n)
            mujoco.mj_step(m, d)
            if i % sync_every == 0:
                env._sync_viewer()
        env._sync_viewer()

    def send_action(self, action):
        self._env.apply_action(np.asarray(action, dtype=np.float64))
