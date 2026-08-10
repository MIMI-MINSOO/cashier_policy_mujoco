from dataclasses import dataclass, field

import numpy as np

from lerobot.teleoperators.config import TeleoperatorConfig

from flare.robots.piper_robot import PiperRobot
from flare.teleoperators.bi_piper_xr.config_bi_piper_xr import PiperArmConfig
from flare.teleoperators.bi_piper_xr.neck import NeckConfig

# IK 시드/유휴 자세 = 심 home keyframe과 동일한 실물 task-ready 자세
_SIM_READY_RAD = [float(np.radians(d)) for d in PiperRobot.EVAL_INIT_JOINT_DEG]


def _arm(side: str) -> PiperArmConfig:
    # XR 텔레옵의 팔 설정을 재사용하되 VR 전용 필드(모션트래커)는 끄고,
    # IK 시드를 심 home 자세로 맞춘다. 그리퍼 범위는 실물 수집과 동일
    # (좌 10.4~90 / 우 0~101.4, direct 단위).
    kwargs = dict(side=side, joints_init=list(_SIM_READY_RAD), motion_tracker=None)
    if side == "left":
        kwargs.update(gripper_open_pos=90.0, gripper_close_pos=10.4)
    return PiperArmConfig(**kwargs)


@TeleoperatorConfig.register_subclass("bi_piper_keyboard")
@dataclass
class BiPiperKeyboardTeleopConfig(TeleoperatorConfig):
    """VR 없이 심 파이프라인을 검증하기 위한 키보드 양팔+목 텔레옵 설정.

    키맵 (record의 에피소드 제어 키인 화살표/ESC와 겹치지 않게 선정):
        z          활성 팔 전환 (left <-> right)
        w/s a/d q/e  활성 팔 EE 이동 +x/-x +y/-y +z/-z
        u/o i/k j/l  활성 팔 EE 회전 roll-/+ pitch+/- yaw+/-
        f          활성 팔 그리퍼 토글 (열림 <-> 닫힘)
        [ / ]      목 yaw 왼쪽(+) / 오른쪽(-)
        ; / '      목 pitch 위(-) / 아래(+)
        h          목 홈 복귀
    """

    # placo solver dt (record 루프 주기와 일치시킬 것)
    dt: float = 1.0 / 30.0

    left_arm: PiperArmConfig = field(default_factory=lambda: _arm("left"))
    right_arm: PiperArmConfig = field(default_factory=lambda: _arm("right"))

    # 틱당 조그 스텝 (키를 누르고 있는 동안 매 루프 적용, 30Hz 기준)
    pos_step_m: float = 0.003        # ~9 cm/s
    rot_step_deg: float = 0.8        # ~24 deg/s
    neck_step_deg: float = 1.0       # ~30 deg/s

    # 목 제한: 실물 NeckConfig 규약 재사용 (pitch -25~+55, yaw home±80)
    neck: NeckConfig = field(default_factory=NeckConfig)
