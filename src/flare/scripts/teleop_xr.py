#!/usr/bin/env python3
"""Camera-capable teleoperation CLI for the in-repo flare robot/teleop (MuJoCo 심 전용).

이 저장소는 cashier_policy의 MuJoCo 심 파이프라인(듀얼암+목, Meta Quest/XRoboToolkit
텔레옵)만 추출한 것이라 실물 로봇 관련 등록(piper_follower 등)은 빠져 있다. 이 wrapper는
flare의 로봇/텔레옵을 먼저 import해서(``--robot.type`` / ``--teleop.type`` 선택지 등록)
lerobot의 ``teleoperate``와 동일한 루프를 돌리되, 씬 리셋 트리거(오른쪽 컨트롤러 B버튼,
teleop.right.buttons["B"] 지원하는 텔레옵에 한함, 예: bi_piper_xr*)가 눌리면
``robot.reset_scene()``을 호출하는 것 하나만 추가되어 있다.

Usage:

    python -m flare.scripts.teleop_xr \
        --robot.type=bi_piper_mujoco \
        --teleop.type=bi_piper_xr_mujoco \
        --display_data=true
"""

import logging
import time
from dataclasses import dataclass
from pprint import pformat

# Register flare robot/teleop via import side effects (must precede the draccus
# parse so their choices are available).
import flare.robots.bi_piper_mujoco  # noqa: F401
import flare.teleoperators.bi_piper_xr  # noqa: F401
import flare.teleoperators.bi_piper_keyboard  # noqa: F401

from lerobot.configs import parser
from lerobot.processor import RobotProcessorPipeline, make_default_processors
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)


@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    fps: int = 60
    teleop_time_s: float | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    """lerobot_teleoperate.teleop_loop과 동일 + 씬 리셋 버튼(오른쪽 B) 지원."""
    can_reset = hasattr(robot, "reset_scene")
    reset_btn_prev = False

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()

        obs = robot.get_observation()

        if robot.name == "unitree_g1" or hasattr(teleop, "send_feedback"):
            try:
                teleop.send_feedback(obs)
            except (NotImplementedError, Exception):
                pass

        raw_action = teleop.get_action()

        # 씬 리셋: 오른쪽 컨트롤러 B버튼 rising edge. bi_piper_xr(_mujoco)/bi_piper_quest_vive
        # 계열은 teleop.right.buttons["B"]를 노출함 - 없는 텔레옵(키보드 등)은 조용히 스킵.
        if can_reset:
            right = getattr(teleop, "right", None)
            reset_btn = bool(getattr(right, "buttons", {}).get("B", False)) if right is not None else False
            if reset_btn and not reset_btn_prev:
                logger.info("Reset button pressed - resetting scene.")
                robot.reset_scene()
                # 씬(로봇 qpos 포함)은 리셋됐는데 텔레옵의 목 트래킹 상태는 그대로라,
                # 안 하면 다음 프레임에 바로 리셋 전 각도로 되돌아간다.
                if hasattr(teleop, "reset_neck"):
                    teleop.reset_neck()
            reset_btn_prev = reset_btn

        teleop_action = teleop_action_processor((raw_action, obs))
        robot_action_to_send = robot_action_processor((teleop_action, obs))
        _ = robot.send_action(robot_action_to_send)

        if display_data:
            obs_transition = robot_observation_processor(obs)
            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )
            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(cfg))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            import rerun as rr

            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    teleoperate()
