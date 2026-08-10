"""PiperMujocoEnv 캡스톤 씬 데모 뷰어.

실행:  python -m flare.scripts.demo_random_reset
키:    R / Space = 상품 랜덤 재배치 (env.reset())

스캐너 표시등(geom group 5)이 켜진 상태로 뷰어를 띄우므로, 상품 바코드를
스캐너 구멍 위에 아래로 향하게 가져가면 초록 불빛으로 스캔을 확인할 수 있다
(관측 카메라 렌더에는 표시등이 찍히지 않음).
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer

from flare.envs.piper_mujoco_env import (
    CAPSTONE_BASKET,
    CAPSTONE_CAMERAS,
    CAPSTONE_PRODUCTS,
    PiperMujocoEnv,
)

XML = Path(__file__).parents[1] / "assets" / "piper_dual_capstone.xml"


def main():
    env = PiperMujocoEnv(
        robot_mode="dual",
        xml_path=XML,
        product_names=CAPSTONE_PRODUCTS,
        basket_name=CAPSTONE_BASKET,
    )
    env.reset()

    reset_requested = False

    def key_cb(keycode):
        nonlocal reset_requested
        if keycode in (ord("R"), ord("r"), ord(" ")):
            reset_requested = True

    print("뷰어에서 R 또는 Space 키 = 상품 랜덤 재배치")
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_cb) as viewer:
        viewer.opt.geomgroup[5] = 1  # 스캐너 표시등은 사람용 뷰어에서만 표시
        env.data.ctrl[:] = env.model.key_ctrl[0]
        while viewer.is_running():
            if reset_requested:
                reset_requested = False
                env.reset()
                print("재배치 완료")
            mujoco.mj_step(env.model, env.data)
            env._update_scanner()  # 뷰어 조작 중에도 스캔 판정/표시등 동작
            viewer.sync()
            time.sleep(env.model.opt.timestep)


if __name__ == "__main__":
    main()
