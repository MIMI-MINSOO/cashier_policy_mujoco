from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("bi_piper_mujoco")
@dataclass
class BiPiperMujocoConfig(RobotConfig):
    """MuJoCo 심 양팔 Piper(+목) — lerobot record/teleoperate용 로봇 설정.

    BiPiperFollower(실물)와 동일한 action/observation 규약에 목 2축을 더한
    16차원을 노출한다. 순서는 flare.envs.piper_mujoco_env.state_feature_names().

    주의: --dataset.fps는 sim_fps와 반드시 일치시켜야 한다. record 루프가
    프레임당 send_action 1회를 호출하고 그때마다 심이 1/sim_fps초 진행되므로,
    fps가 다르면 심 시간이 벽시계와 어긋난다.
    """

    # None이면 piper_dual_capstone.xml (flare/assets)
    xml_path: str | None = None
    # 렌더 이미지 크기 (관측 카메라 4개 공통)
    image_height: int = 240
    image_width: int = 320
    # 심 제어 주기 [Hz] — 프레임당 심이 1/sim_fps초 진행
    sim_fps: float = 30.0
    # 관측 카메라 obs 키 (CAPSTONE_CAMERAS의 부분집합). None = 4개 전부
    camera_keys: list[str] | None = None
    # 텔레옵용 passive 뷰어 표시
    show_viewer: bool = True
    # 뷰어에서 스캐너 표시등(geom group 5)을 보이게 할지 (관측 렌더에는 안 찍힘)
    show_scanner_indicator: bool = True
    # 에피소드 리셋 시 바구니 내 상품 랜덤 배치
    randomize_products: bool = True
    seed: int | None = None
    # 목(h1/h2)을 state/action에 포함 (16차원). 끄면 14차원 —
    # 텔레옵이 neck 키를 내보내는 조합과 섞어 쓰지 말 것.
    enable_neck: bool = True
