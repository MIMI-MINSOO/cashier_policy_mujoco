# 심 평가(eval_sim.py)용 공용 유틸 — cashier_policy의 eval_real.py에서 실물 하드웨어와
# 무관한 부분(체크포인트 로딩, 키 입력 리스너, 카메라 미리보기)만 뽑아온 것.
# eval_real.py 원본은 최상단에서 flare.envs.piper_real_env(피어-SDK 등 실물 전용)를
# import하기 때문에 여기서는 재사용하지 않고 따로 둔다.

import threading
import queue
from pathlib import Path

import torch
import cv2
import numpy as np
from omegaconf import OmegaConf


def _build_dummy_stats(cfg):
    """Placeholder stats with the right keys / shapes. Actual normalization
    values are loaded from ``model.safetensors`` (or overwritten by EMA), so
    the exact numbers here do not matter — they just let the policy constructor
    allocate the correct buffer structure."""
    state_dim = cfg.task.state_dim
    action_dim = cfg.task.action_dim

    def _zeros(dim):
        return {
            "mean": [0.0] * dim, "std": [1.0] * dim,
            "min": [0.0] * dim, "max": [1.0] * dim,
        }

    stats = {
        cfg.task.state_key: _zeros(state_dim),
        cfg.task.action_key: _zeros(action_dim),
    }
    for key in cfg.task.image_keys:
        stats[key] = {
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0],
        }
    return stats


def load_policy(checkpoint_path: str, policy_name: str, device: str):
    from flare.factory import get_policy_class

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_dir = Path(checkpoint_path).parent.parent.parent / "logs"
    config_files = sorted(config_dir.glob("train_config_*.yaml"))
    if not config_files:
        raise FileNotFoundError(f"No training config found in {config_dir}")
    cfg = OmegaConf.load(config_files[-1])

    # Build policy with dummy stats — real normalization buffers come from
    # the checkpoint (model.safetensors / EMA shadow), so we no longer need
    # the dataset to exist at eval time.
    stats = _build_dummy_stats(cfg)
    policy_cls = get_policy_class(policy_name)
    policy = policy_cls(cfg, stats)

    model_path = Path(checkpoint_path).parent / "model.safetensors"

    if "ema" in checkpoint and checkpoint["ema"] is not None:
        # EMA shadow_params include all nn.Parameter (also the normalize
        # buffers registered as Parameter(requires_grad=False)), so copy_to
        # restores both model weights and normalization stats in one shot.
        from diffusers.training_utils import EMAModel
        ema = EMAModel(policy.parameters())
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(policy.parameters())
        print("Loaded EMA weights")
    elif model_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(model_path), device=str(device))
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        print(f"Loaded model from {model_path.name}")
    else:
        raise FileNotFoundError(
            f"No EMA in checkpoint and {model_path} not found. Cannot load weights."
        )

    policy.to(device)
    policy.eval()
    policy.reset()
    print(f"Policy '{policy_name}' loaded from {checkpoint_path}")
    return policy, cfg


class KeyListener:
    def __init__(self):
        self.latest_key = None
        self._stop = False
        self._old_settings = None

    def start(self):
        import sys, termios
        self._old_settings = termios.tcgetattr(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        import sys, tty
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not self._stop:
                self.latest_key = sys.stdin.read(1)
        except Exception:
            pass

    def get(self):
        k = self.latest_key
        self.latest_key = None
        return k

    def stop(self):
        self._stop = True
        if self._old_settings is not None:
            import sys, termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)


class CameraDisplay:
    """Non-blocking camera display and deferred video saving.

    Display runs in a background thread. Video frames are buffered in memory
    and written to disk on stop(). 카메라 이름 개수/이름을 가리지 않고 들어오는
    camera_names 전부를 가로로 이어붙여 한 창에 보여준다 (원본 eval_real.py의
    CameraDisplay는 "main"/"wrist" 두 이름만 하드코딩되어 있어서, capstone의
    4캠(head/top/left_wrist/right_wrist)에는 아무것도 안 뜨는 문제가 있었음).
    """

    def __init__(self, camera_names, display=True, save_video=False, video_dir=None, fps=30):
        self.camera_names = list(camera_names)
        self.display = display
        self.save_video = save_video
        self.fps = fps
        self.window_name = "Camera View"

        self._display_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._display_thread = None

        self._frame_buffers = {name: [] for name in self.camera_names}
        self.video_dir = None
        if save_video and video_dir:
            self.video_dir = Path(video_dir)
            self.video_dir.mkdir(parents=True, exist_ok=True)

        if display:
            self._display_thread = threading.Thread(target=self._display_worker, daemon=True)
            self._display_thread.start()

    def update(self, camera_images: dict[str, np.ndarray], step: int, running: bool):
        """Non-blocking update. Queues display frame, buffers save frame."""
        if self.save_video:
            for name in self.camera_names:
                if name in camera_images:
                    self._frame_buffers[name].append(camera_images[name].copy())

        if self.display:
            try:
                self._display_queue.put_nowait((camera_images, step, running))
            except queue.Full:
                pass

    def _display_worker(self):
        """Background thread: display only."""
        while not self._stop_event.is_set():
            try:
                camera_images, step, running = self._display_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            display_height = 360
            parts = []
            for name in self.camera_names:
                if name not in camera_images:
                    continue
                img = camera_images[name].copy()
                status = f"RUNNING step:{step}" if running else "READY (press c)"
                cv2.putText(img, f"{name} | {status}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                h, w = img.shape[:2]
                scale = display_height / h
                new_w = int(w * scale)
                parts.append(cv2.resize(img, (new_w, display_height)))

            if parts:
                combined = np.hstack(parts)
                cv2.imshow(self.window_name, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def stop(self):
        self._stop_event.set()
        if self._display_thread is not None:
            self._display_thread.join(timeout=2.0)
        if self.display:
            cv2.destroyAllWindows()

        if self.save_video and self.video_dir:
            for name, frames in self._frame_buffers.items():
                if not frames:
                    continue
                h, w = frames[0].shape[:2]
                path = str(self.video_dir / f"{name}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
                print(f"Saving {name}: {len(frames)} frames -> {path}")
                for frame in frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
            print(f"Videos saved to {self.video_dir}")
