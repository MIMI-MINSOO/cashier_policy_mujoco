# MuJoCo 심 평가 스크립트.
# 정책 로딩/키 조작/카메라 미리보기는 eval_utils.py(cashier_policy의 eval_real.py에서
# 실물 하드웨어 무관한 부분만 추출) 공용, 로봇은 PiperMujocoEnv(capstone 씬).
#
# Usage:
#   python -m flare.scripts.eval_sim --checkpoint path/to/training_state.pt --policy diffusion
#
# Controls (터미널):
#   c = Start policy execution
#   s = Stop and return to initial pose
#   q / ESC = Quit
#   Ctrl+C = Emergency stop

import argparse
from datetime import datetime
from pathlib import Path

from flare.envs.piper_mujoco_env import (
    CAPSTONE_BASKET,
    CAPSTONE_CAMERAS,
    CAPSTONE_PRODUCTS,
    PiperMujocoEnv,
    _ASSETS_DIR,
)
from flare.inference import EvalRunner
from flare.scripts.eval_utils import CameraDisplay, KeyListener, load_policy


def build_camera_map(cfg) -> dict[str, str]:
    """task.image_keys -> {obs 키: MJCF 카메라 이름} (capstone 카메라만 지원)."""
    cam_map = {}
    for key in cfg.task.image_keys:
        cam_name = key.replace("observation.images.", "")
        if cam_name in CAPSTONE_CAMERAS:
            cam_map[cam_name] = CAPSTONE_CAMERAS[cam_name]
        else:
            print(f"[WARN] '{cam_name}'은 capstone 카메라가 아님 (key={key}); skipping. "
                  f"available: {list(CAPSTONE_CAMERAS)}")
    if not cam_map:
        raise ValueError("task.image_keys에서 사용할 수 있는 capstone 카메라가 없습니다.")
    return cam_map


def main():
    parser = argparse.ArgumentParser(description="Run trained policy in the MuJoCo capstone sim.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--policy", type=str, default="vita")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=2700)
    parser.add_argument("--freq", type=float, default=30.0,
                        help="Control frequency [Hz]; sim advances 1/freq per step.")
    parser.add_argument("--xml", type=str, default=None,
                        help="MJCF path (default: piper_dual_capstone.xml)")
    parser.add_argument("--seed", type=int, default=None, help="Product layout RNG seed")
    parser.add_argument("--no-randomize", action="store_true",
                        help="Keep keyframe product poses (no random reset)")
    parser.add_argument("--no-viewer", action="store_true", help="Disable MuJoCo passive viewer")
    parser.add_argument("--display", action="store_true",
                        help="Show camera preview window (default: off).")
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--sync", action="store_true",
                        help="Inference in-line each step (default: async 3-thread).")
    parser.add_argument("--merger", type=str, default="temporal_ensemble",
                        choices=["overwrite", "temporal_ensemble"])
    parser.add_argument("--te-coeff", type=float, default=0.01)
    args = parser.parse_args()

    policy, cfg = load_policy(args.checkpoint, args.policy, args.device)

    cam_map = build_camera_map(cfg)
    print(f"Cameras: {cam_map} (from image_keys: {list(cfg.task.image_keys)})")

    xml = args.xml if args.xml else _ASSETS_DIR / "piper_dual_capstone.xml"
    env = PiperMujocoEnv(
        robot_mode=cfg.task.get("robot_mode", "dual"),
        xml_path=xml,
        image_size=tuple(cfg.resize_shape),
        control_freq=args.freq,
        camera_names=cam_map,
        product_names=CAPSTONE_PRODUCTS,
        basket_name=CAPSTONE_BASKET,
        randomize_products=not args.no_randomize,
        seed=args.seed,
    )
    if env.action_dim != cfg.task.action_dim:
        raise ValueError(
            f"env action_dim({env.action_dim}) != task action_dim({cfg.task.action_dim}) — "
            f"목 포함 여부/task config를 확인하세요."
        )

    save_video = not args.no_save_video
    video_dir = args.video_dir
    if save_video and video_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = Path(args.checkpoint).parent.parent.parent
        video_dir = str(ckpt_dir / "eval_videos_sim" / timestamp)

    try:
        env.connect()
        env.reset()

        if not args.no_viewer:
            import mujoco.viewer

            env.viewer = mujoco.viewer.launch_passive(env.model, env.data)
            env.viewer.opt.geomgroup[5] = 1  # 스캐너 표시등 (관측 렌더에는 안 찍힘)
            env.viewer.sync()

        state = env.get_state()
        print(f"\nPre-check (LEFT) :  joints={state[:6]}  gripper={state[6]:.1f}")
        print(f"Pre-check (RIGHT):  joints={state[7:13]}  gripper={state[13]:.1f}")
        if env.enable_neck:
            print(f"Pre-check (NECK) :  pitch={state[14]:.1f}  yaw={state[15]:.1f}")

        keys = KeyListener()
        keys.start()

        cam_display = None
        if args.display or save_video:
            if save_video:
                print(f"Video save: {video_dir}")
            cam_display = CameraDisplay(
                camera_names=list(cam_map.keys()),
                display=args.display,
                save_video=save_video,
                video_dir=video_dir,
                fps=int(args.freq),
            )

        try:
            runner = EvalRunner(
                policy=policy,
                cfg=cfg,
                env=env,
                merger_name=args.merger,
                te_coeff=args.te_coeff,
                max_steps=args.max_steps,
                control_freq=args.freq,
                device=args.device,
                sync=args.sync,
                key_listener=keys,
                cam_display=cam_display,
            )
            runner.run()
            if env.scan_events:
                print(f"\n[barcode] episode scan events: {env.scan_events}")
        finally:
            keys.stop()
            if cam_display is not None:
                cam_display.stop()
    finally:
        env.disconnect()


if __name__ == "__main__":
    main()
