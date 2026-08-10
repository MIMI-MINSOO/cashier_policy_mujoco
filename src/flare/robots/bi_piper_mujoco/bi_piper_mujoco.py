"""MuJoCo 심 양팔 Piper(+목) lerobot Robot — BiPiperFollower의 시뮬레이션 쌍둥이.

PiperMujocoEnv를 감싸서 lerobot record/teleoperate 루프에 실물 로봇처럼
끼워 넣는다. 물리(그리퍼 캘리브레이션, substep 보간, 바코드 스캐너, 상품
랜덤화)는 env 한 곳에만 있으므로 수집과 심 추론(eval_sim)이 동일한 동역학을
공유한다.

심 시계는 send_action() 안에서 동기로 1/sim_fps초씩 진행된다 (step+4캠 렌더
~8.4ms < 33ms — 30Hz 실시간 여유, manipulation_pipeline에서 측정). 벽시계
페이싱은 record 루프의 precise_sleep이 담당한다.
"""

import logging
from functools import cached_property

import numpy as np

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.robots.robot import Robot

from flare.envs.piper_mujoco_env import (
    CAPSTONE_BASKET,
    CAPSTONE_CAMERAS,
    CAPSTONE_PRODUCTS,
    PiperMujocoEnv,
    _ASSETS_DIR,
    state_feature_names,
)
from .config_bi_piper_mujoco import BiPiperMujocoConfig

logger = logging.getLogger(__name__)

CAPSTONE_XML = _ASSETS_DIR / "piper_dual_capstone.xml"


class BiPiperMujoco(Robot):
    """Bimanual PiPER (+2-DOF neck) simulated in MuJoCo."""

    config_class = BiPiperMujocoConfig
    name = "bi_piper_mujoco"

    def __init__(self, config: BiPiperMujocoConfig):
        super().__init__(config)
        self.config = config
        self._state_names = state_feature_names("dual", config.enable_neck)
        if config.camera_keys is None:
            self._camera_map = dict(CAPSTONE_CAMERAS)
        else:
            unknown = [k for k in config.camera_keys if k not in CAPSTONE_CAMERAS]
            if unknown:
                raise ValueError(f"Unknown camera keys {unknown}; available: {list(CAPSTONE_CAMERAS)}")
            self._camera_map = {k: CAPSTONE_CAMERAS[k] for k in config.camera_keys}
        self.env: PiperMujocoEnv | None = None
        self._viewer = None
        self.cameras = {}  # 실물 Camera 객체 없음 — 렌더는 env가 담당

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {n: float for n in self._state_names}
        shape = (self.config.image_height, self.config.image_width, 3)
        features.update({cam: shape for cam in self._camera_map})
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {n: float for n in self._state_names}

    @property
    def is_connected(self) -> bool:
        return self.env is not None and self.env.is_connected()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        xml = self.config.xml_path if self.config.xml_path else CAPSTONE_XML
        self.env = PiperMujocoEnv(
            robot_mode="dual",
            xml_path=xml,
            image_size=(self.config.image_height, self.config.image_width),
            control_freq=self.config.sim_fps,
            camera_names=self._camera_map,
            product_names=CAPSTONE_PRODUCTS,
            basket_name=CAPSTONE_BASKET,
            randomize_products=self.config.randomize_products,
            seed=self.config.seed,
            enable_neck=self.config.enable_neck,
        )
        self.env.connect()
        self.env.reset()
        if self.config.show_viewer:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
            if self.config.show_scanner_indicator:
                self._viewer.opt.geomgroup[5] = 1
            self._viewer.sync()
        logger.info(
            f"{self} connected: action_dim={self.env.action_dim}, "
            f"cameras={list(self._camera_map)}, sim_fps={self.config.sim_fps} "
            f"(--dataset.fps와 일치해야 심 시간이 벽시계와 맞음)"
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: RobotObservation = {
            name: float(v) for name, v in zip(self._state_names, self.env.get_state())
        }
        obs.update(self.env.render_images())
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        missing = [n for n in self._state_names if n not in action]
        if missing:
            raise KeyError(f"action에 없는 키: {missing} (텔레옵이 neck 키를 내보내는지 확인)")
        vec = np.array([action[n] for n in self._state_names], dtype=np.float64)
        self.env.apply_action(vec)
        self._sync_viewer()
        return {n: float(v) for n, v in zip(self._state_names, vec)}

    def reset_scene(self) -> None:
        """에피소드 간 씬 리셋: home 자세 + 바구니 내 상품 랜덤 재배치."""
        self.env.reset()
        self._sync_viewer()
        logger.info("Scene reset (products re-randomized).")

    def _sync_viewer(self) -> None:
        if self._viewer is not None:
            if self._viewer.is_running():
                self._viewer.sync()
            else:
                self._viewer = None

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self.env.disconnect()
        self.env = None
