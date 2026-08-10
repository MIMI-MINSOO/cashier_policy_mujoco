# Convert LeRobot datasets to Zarr format for fast training.
#
# Registration in DATASET_CONFIGS is OPTIONAL — unregistered repos use defaults
# (episodes=all, remove_keys=[], output_name auto-derived
# from repo_id). Register only if you need non-default settings.
#
# Images are stored at native resolution; resizing/cropping happens in the
# policy's vision encoder (resize_shape/crop_shape) right before the model.
#
# Usage:
#   List registered datasets:
#       python convert.py -l
#   Convert from HuggingFace cache (~/.cache/huggingface/lerobot/<repo_id>/):
#       python convert.py -r Leejungwook/cube_stack
#   Convert from local path (repo_id auto-inferred from last two parts of path):
#       python convert.py --local-dir /path/to/Leejungwook/new_task
#   Override output directory:
#       python convert.py -r Leejungwook/new_task -o data/piper_new_task
#   Merge multiple repos into one zarr (episodes appended, stats aggregated):
#       python convert.py --repos mimiminsoo/spam_1 mimiminsoo/spam_2 -o data/piper_spam
#   Convert all registered datasets:
#       python convert.py --all

import argparse
import json
import shutil
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as torch_mp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


# Use file_system sharing strategy to avoid /dev/shm exhaustion
# with large multi-camera datasets (especially bimanual).
torch_mp.set_sharing_strategy('file_system')

from flare.datasets.replay_buffer import ReplayBuffer
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Project root (src/flare/scripts/convert.py -> manipulation_pipeline)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Default output directory
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"

# Default image resolution (H, W) stored in the zarr — the policy's resize_shape,
# i.e. the model input size before the random/center crop. Keeps the in-RAM zarr
# small while preserving the crop augmentation. Use --native to store full res.
DEFAULT_RESIZE = (240, 320)

DATASET_CONFIGS = {
    "Leejungwook/cube_stack": {
        "episodes": None,
        "remove_keys": [],
        "output_name": "piper_cube_stack",
    },
    "Leejungwook/cube_stack_v2": {
        "episodes": None,
        "remove_keys": [],
        "output_name": "piper_cube_stack_v2",
    },
    "Leejungwook/peg_in_hole": {
        "episodes": None,
        "remove_keys": [],
        "output_name": "piper_peg_in_hole",
    },
    "Leejungwook/cube_stack_single-arm": {
        "episodes": None,
        "remove_keys": [],
        "output_name": "piper_cube_stack_single_arm",
    },
    "Leejungwook/term-project": {
        "episodes": None,
        "remove_keys": [],
        "output_name": "piper_term_project",
    },
    "Leejungwook/term-project_ljw": {
        "episodes": None,
        "remove_keys": [],
        "output_name": "piper_term_project_ljw",
    },
    "Leejungwook/vr_test": {
        "episodes": None,
        "remove_keys": [],
        "output_name": "piper_bimanual_cube_stack",
    },
}


def make_json_serializable(obj):
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def create_piper_dataset_from_lerobot(
    repo_id: str | list[str],
    root: Path,
    episodes: list[int] | None = None,
    remove_keys: list[str] | None = None,
    local_dir: str | Path | None = None,
    target_fps: int | None = None,
    resize: tuple[int, int] | None = None,
):
    # repo_id에 리스트를 주면 여러 LeRobot 데이터셋을 하나의 zarr로 병합한다:
    # 에피소드를 순서대로 이어 붙이고(episode_index 재부여), stats는
    # lerobot aggregate_stats(count 가중)로 결합. features/fps 불일치는 에러.
    repo_ids = [repo_id] if isinstance(repo_id, str) else list(repo_id)
    if remove_keys is None:
        remove_keys = []
    if local_dir is not None and len(repo_ids) > 1:
        raise ValueError("--local-dir is not supported with multiple repos.")

    # lerobot은 torchcodec이 import만 되면 기본 백엔드로 쓰는데, 시스템에
    # FFmpeg 공유 라이브러리(libavdevice 등)가 없으면 로드 단계에서 죽는다.
    # 실제 로드 가능 여부를 확인하고 안 되면 pyav로 폴백한다.
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
        video_backend = None
    except Exception:
        print("  Video backend: torchcodec unavailable -> pyav fallback")
        video_backend = "pyav"

    print(f"  Output: {root}")
    print(f"  Images: {'native resolution' if resize is None else f'resized to {resize[0]}x{resize[1]}'}")

    datasets = []
    for rid in repo_ids:
        lerobot_kwargs = {"repo_id": rid}
        if episodes is not None:
            lerobot_kwargs["episodes"] = episodes
        if local_dir is not None:
            lerobot_kwargs["root"] = Path(local_dir)
        if video_backend is not None:
            lerobot_kwargs["video_backend"] = video_backend
        print(f"Loading: {rid}")
        if local_dir:
            print(f"  Source: {local_dir}")
        datasets.append(LeRobotDataset(**lerobot_kwargs))

    # 병합 가능 검증: fps와 feature 시그니처(키/dtype/shape)가 전부 같아야 한다.
    def feature_signature(ds):
        return {
            k: (v.get("dtype"), tuple(v.get("shape") or ()))
            for k, v in ds.features.items() if k not in remove_keys
        }

    fps = datasets[0].meta.fps
    signature = feature_signature(datasets[0])
    for rid, ds in zip(repo_ids[1:], datasets[1:]):
        if ds.meta.fps != fps:
            raise ValueError(f"FPS mismatch: {repo_ids[0]}={fps}, {rid}={ds.meta.fps}")
        if feature_signature(ds) != signature:
            raise ValueError(
                f"Feature mismatch between {repo_ids[0]} and {rid}:\n"
                f"  {signature}\n  vs\n  {feature_signature(ds)}"
            )

    # Episode boundaries — read from each source (use dataset.root provided by lerobot)
    # (dataset, from_idx, to_idx, source_repo) 순서대로 이어 붙인다.
    episode_specs = []
    for rid, ds in zip(repo_ids, datasets):
        ep_parquet_dir = Path(ds.root) / "meta" / "episodes"
        if not ep_parquet_dir.exists():
            raise FileNotFoundError(
                f"Cannot find episode metadata at {ep_parquet_dir}. "
                "Please ensure the dataset has the expected LeRobot format."
            )
        # Read all episode parquet files (may be split across multiple files)
        ep_files = sorted(ep_parquet_dir.rglob("*.parquet"))
        episodes_meta = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
        for f_idx, t_idx in zip(
            episodes_meta["dataset_from_index"], episodes_meta["dataset_to_index"]
        ):
            episode_specs.append((ds, int(f_idx), int(t_idx), rid))

    num_episodes = len(episode_specs)

    if target_fps is not None:
        if fps % target_fps != 0:
            raise ValueError(f"Source FPS ({fps}) must be divisible by target FPS ({target_fps}).")
        downsample_ratio = fps // target_fps
        out_fps = target_fps
    else:
        downsample_ratio = 1
        out_fps = fps

    out_num_frames = sum(
        len(range(f_idx, t_idx, downsample_ratio)) for _, f_idx, t_idx, _ in episode_specs
    )

    dataset = datasets[0]
    features = {k: v for k, v in dataset.features.items() if k not in remove_keys}

    camera_keys = [k for k in dataset.meta.camera_keys if k in features]
    video_keys = [k for k in dataset.meta.video_keys if k in features]
    image_keys = [k for k in (dataset.meta.image_keys if hasattr(dataset.meta, 'image_keys') else []) if k in features]

    # When resizing, reflect the new resolution in the stored feature shapes (HWC).
    if resize is not None:
        for k in set(camera_keys + video_keys + image_keys):
            if "shape" in features.get(k, {}):
                features[k] = {**features[k], "shape": [resize[0], resize[1], 3]}

    stats_list = [
        {k: v for k, v in ds.meta.stats.items() if k in features} for ds in datasets
    ]
    stats = stats_list[0] if len(stats_list) == 1 else aggregate_stats(stats_list)

    # task 문자열 기준으로 중복 제거해 병합 인덱스를 새로 부여한다.
    tasks = {}
    tasks_reversed = {}
    for ds in datasets:
        for task_name, _row in ds.meta.tasks.iterrows():
            if task_name not in tasks_reversed:
                new_idx = len(tasks_reversed)
                tasks_reversed[task_name] = new_idx
                tasks[new_idx] = task_name

    # Remove old output
    if root.exists():
        print(f"Removing existing: {root}")
        shutil.rmtree(root)

    # Create replay buffer
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=root, mode="a")

    # Save metadata
    config = {
        "repo_id": "+".join(repo_ids),
        "source_repo_ids": repo_ids,
        "stats": stats,
        "num_frames": out_num_frames,
        "num_episodes": num_episodes,
        "features": features,
        "camera_keys": camera_keys,
        "video_keys": video_keys,
        "image_keys": image_keys,
        "fps": out_fps,
        "tasks": tasks,
    }
    with open(root / "config.json", "w") as f:
        json.dump(make_json_serializable(config), f, indent=4)

    def convert(k, v: torch.Tensor):
        dtype = features[k]["dtype"]
        if dtype in ["image", "video"]:
            # v: (N, C, H, W) float in [0, 1]. Optionally resize before storing so
            # the zarr (loaded fully into RAM by PiperDataset) stays small.
            if resize is not None:
                v = torch.nn.functional.interpolate(
                    v, size=resize, mode="bilinear", align_corners=False
                )
            v = v.permute(0, 2, 3, 1)
            v = (v * 255).to(torch.uint8).numpy()
        else:
            v = v.numpy()
        return v

    # Convert episodes
    for i, (ds, from_idx, to_idx, rid) in enumerate(episode_specs):
        indices = list(range(from_idx, to_idx, downsample_ratio))
        ep_len = len(indices)
        print(f"  Episode {i}/{num_episodes} ({ep_len} frames, {rid})...")
        subset = Subset(ds, indices)
        dataloader = DataLoader(
            subset, batch_size=16, shuffle=False, num_workers=8
        )
        data = []
        for batch in tqdm(dataloader, leave=False):
            if "task_index" in batch:
                batch["task_index"] = torch.tensor(
                    [tasks_reversed[k] for k in batch["task"]], dtype=int
                )
                del batch["task"]
            batch["episode_index"] = torch.full_like(batch["episode_index"], i)
            data.append(batch)

        batch = {k: torch.cat([d[k] for d in data], dim=0) for k in data[0].keys()}
        assert batch["action"].shape[0] == ep_len
        batch = {k: convert(k, v) for k, v in batch.items() if k in features}
        replay_buffer.add_episode(batch, compressors="disk")

    print(f"\nDone! {num_episodes} episodes, {out_num_frames} frames → {root}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot datasets to Zarr format.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-l", "--ls", action="store_true", help="List registered datasets.")
    group.add_argument("-r", "--repo", type=str, metavar="REPO_ID",
                       help="HuggingFace repo ID (e.g. Leejungwook/cube_stack).")
    group.add_argument("--repos", type=str, nargs="+", metavar="REPO_ID",
                       help="Multiple repo IDs to merge into a single zarr (requires -o).")
    group.add_argument("--local-dir", type=str, default=None,
                       help="Local path to LeRobot dataset. repo_id auto-inferred from path.")
    group.add_argument("--all", action="store_true", help="Convert all registered datasets.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory. Default: data/<dataset_name>")
    parser.add_argument("--target-fps", type=int, default=None,
                        help="Target FPS to downsample the dataset (e.g., 10).")
    parser.add_argument("--resize", type=int, nargs=2, default=list(DEFAULT_RESIZE), metavar=("H", "W"),
                        help=f"Resize images to H W before storing. Default: {DEFAULT_RESIZE[0]} {DEFAULT_RESIZE[1]} "
                             "(the policy resize_shape / model input size). e.g. --resize 360 640")
    parser.add_argument("--native", action="store_true",
                        help="Store images at native resolution (no resize). Large in-RAM zarr.")

    args = parser.parse_args()

    # Native overrides the resize default; otherwise use the (default or given) H W.
    resize = None if args.native else tuple(args.resize)

    if args.ls:
        print("--- Registered Datasets ---")
        for repo_id, cfg in DATASET_CONFIGS.items():
            ep_info = f"all" if cfg["episodes"] is None else f"{len(cfg['episodes'])}"
            print(f"  {repo_id}  →  {DEFAULT_OUTPUT_DIR / cfg['output_name']}  ({ep_info} episodes)")
        print("---------------------------")
        return

    if args.repos:
        if not args.output:
            parser.error("--repos requires -o/--output (merged dataset directory).")
        create_piper_dataset_from_lerobot(
            repo_id=args.repos,
            root=Path(args.output),
            target_fps=args.target_fps,
            resize=resize,
        )
        return

    if args.all:
        for repo_id, cfg in DATASET_CONFIGS.items():
            output_dir = DEFAULT_OUTPUT_DIR / cfg["output_name"]
            create_piper_dataset_from_lerobot(
                repo_id=repo_id,
                root=output_dir,
                episodes=cfg["episodes"],
                remove_keys=cfg["remove_keys"],
                local_dir=args.local_dir,
                target_fps=args.target_fps,
                resize=resize,
            )
        return

    # Single dataset
    local_dir = None
    if args.local_dir:
        local_dir = args.local_dir
        # Auto-infer repo_id from local path: .../Leejungwook/cube_stack → Leejungwook/cube_stack
        parts = Path(local_dir).parts
        repo_id = f"{parts[-2]}/{parts[-1]}"
        print(f"Auto-inferred repo_id: {repo_id}")
    else:
        repo_id = args.repo

    # Use DATASET_CONFIGS entry if registered, else fall back to defaults.
    # Auto-derive output_name from repo_id: "Leejungwook/foo-bar" → "piper_foo_bar"
    if repo_id in DATASET_CONFIGS:
        cfg = DATASET_CONFIGS[repo_id]
    else:
        auto_output_name = "piper_" + repo_id.split("/")[-1].replace("-", "_")
        cfg = {
            "episodes": None,
            "remove_keys": [],
            "output_name": auto_output_name,
        }
        print(f"'{repo_id}' not registered — using defaults (output_name='{auto_output_name}').")

    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / cfg["output_name"]
    create_piper_dataset_from_lerobot(
        repo_id=repo_id,
        root=output_dir,
        episodes=cfg["episodes"],
        remove_keys=cfg["remove_keys"],
        local_dir=local_dir,
        target_fps=args.target_fps,
        resize=resize,
    )


if __name__ == "__main__":
    main()
