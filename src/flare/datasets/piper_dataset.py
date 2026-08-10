import torch
from pathlib import Path
import numpy as np
from typing import Callable
from itertools import accumulate

from flare.datasets.replay_buffer import ReplayBuffer
try:
    from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices
except ImportError:
    from lerobot.datasets.feature_utils import check_delta_timestamps, get_delta_indices


import json


def get_episode_data_index(
    episode_dicts: dict, episodes: list[int] | None = None
) -> dict[str, torch.Tensor]:
    episode_lengths = {ep_idx: ep_dict["length"] for ep_idx, ep_dict in episode_dicts.items()}
    if episodes is not None:
        episode_lengths = {ep_idx: episode_lengths[ep_idx] for ep_idx in episodes}
    cumulative_lengths = list(accumulate(episode_lengths.values()))
    return {
        "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
        "to": torch.LongTensor(cumulative_lengths),
    }


def check_timestamps_sync(
    timestamps: np.ndarray,
    episode_indices: np.ndarray,
    episode_data_index: dict[str, np.ndarray],
    fps: int,
    tolerance_s: float,
    raise_value_error: bool = True,
) -> bool:
    if timestamps.shape != episode_indices.shape:
        raise ValueError(
            f"timestamps and episode_indices should have the same shape. "
            f"Found {timestamps.shape=} and {episode_indices.shape=}."
        )
    diffs = np.diff(timestamps)
    within_tolerance = np.abs(diffs - (1.0 / fps)) <= tolerance_s
    mask = np.ones(len(diffs), dtype=bool)
    ignored_diffs = episode_data_index["to"][:-1] - 1
    mask[ignored_diffs] = False
    filtered_within_tolerance = within_tolerance[mask]
    if not np.all(filtered_within_tolerance):
        if raise_value_error:
            original_indices = np.arange(len(diffs))
            filtered_indices = original_indices[mask]
            bad_indices = filtered_indices[np.nonzero(~filtered_within_tolerance)[0]]
            raise ValueError(
                f"Timestamps violate tolerance at indices: {bad_indices[:10]}... "
                f"(expected diff={1.0/fps:.6f}, tolerance={tolerance_s})"
            )
        return False
    return True


def get_dataset_config(
    repo_id: str | None = None,
    root: str | Path | None = None,
) -> dict:
    root = Path(root)
    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}. Please create the dataset first.")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


class PiperDatasetMeta:
    def __init__(self, repo_id: str | None = None, root: str | Path | None = None):
        self.repo_id = repo_id
        self.root = Path(root)
        self.config = get_dataset_config(repo_id=self.repo_id, root=self.root)

        if 'tasks' in self.config:
            self.config['tasks'] = {int(k): v for k, v in self.config['tasks'].items()}

    @property
    def stats(self) -> dict:
        return self.config['stats']

    @property
    def num_frames(self) -> int:
        return self.config['num_frames']

    @property
    def num_episodes(self) -> int:
        return self.config['num_episodes']

    @property
    def features(self):
        return self.config['features']

    @property
    def camera_keys(self):
        return self.config['camera_keys']

    @property
    def video_keys(self):
        return self.config['video_keys']

    @property
    def image_keys(self):
        return self.config['image_keys']

    @property
    def fps(self) -> float:
        return self.config['fps']

    @property
    def tasks(self):
        return self.config['tasks']


class PiperDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str | None = None,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict | None = None,
        tolerance_s: float = 1e-4,
    ):
        super().__init__()

        self.repo_id = repo_id
        self.root = Path(root)
        self.episodes = episodes
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s

        # Load zarr dataset + metadata
        self.replay_buffer = ReplayBuffer.copy_from_path(self.root)
        self.meta = PiperDatasetMeta(repo_id=self.repo_id, root=self.root)

        if not self.episodes:
            self.episodes = list(range(self.meta.num_episodes))

        self.length = sum([self.replay_buffer.episode_lengths[i] for i in self.episodes])

        if 'task_index' in self.features:
            self.delta_timestamps['task_index'] = [0]

        # Episode boundaries (for ALL episodes, so original ep_idx works in __getitem__)
        self.episode_data_index = get_episode_data_index({
            i: {'length': length}
            for i, length in enumerate(self.replay_buffer.episode_lengths)
        })

        # Timestamp validation (use filtered episode boundaries)
        timestamps = np.array(self.replay_buffer['timestamp'])
        episode_indices = np.array(self.replay_buffer['episode_index'])
        if self.episodes is not None:
            mask = np.isin(episode_indices, self.episodes)
            timestamps = timestamps[mask]
            episode_indices = episode_indices[mask]
        filtered_ep_data_index = get_episode_data_index({
            i: {'length': length}
            for i, length in enumerate(self.replay_buffer.episode_lengths)
        }, self.episodes)
        filtered_ep_np = {k: t.numpy() for k, t in filtered_ep_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, filtered_ep_np, self.fps, self.tolerance_s)

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple:
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_replay_buffer(self, query_indices: dict) -> dict:
        return {
            key: self.replay_buffer[key][q_idx]
            for key, q_idx in query_indices.items()
        }

    @property
    def stats(self):
        return self.meta.stats

    @property
    def features(self):
        return self.meta.features

    @property
    def fps(self) -> float:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        return self.meta.num_frames

    @property
    def num_episodes(self) -> int:
        return self.meta.num_episodes

    @property
    def video_keys(self):
        return self.meta.video_keys

    @property
    def image_keys(self):
        return self.meta.image_keys

    @property
    def camera_keys(self):
        return self.meta.camera_keys

    @property
    def tasks(self):
        return self.meta.tasks

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx = self.replay_buffer["episode_index"][idx]
        item = {"episode_index": torch.tensor(ep_idx)}

        query_indices, padding = self._get_query_indices(idx, ep_idx)
        query_result = self._query_replay_buffer(query_indices)
        item = {**item, **padding}
        for key, val in query_result.items():
            if key in self.image_keys or key in self.video_keys:
                item[key] = torch.from_numpy(val).type(torch.float32).permute(0, 3, 1, 2) / 255.0
            else:
                item[key] = torch.from_numpy(val)

        if self.image_transforms is not None:
            for cam in self.camera_keys:
                item[cam] = self.image_transforms(item[cam])

        if "task_index" in item:
            task_idx = item["task_index"].item()
            item["task"] = self.tasks[task_idx]

        return item
