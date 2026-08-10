import torch
from torch import Tensor, nn


def create_stats_buffers(
    keys: list[str],
    stats: dict[str, dict[str, Tensor]],
    mode: str = "meanstd",
) -> dict[str, dict[str, nn.ParameterDict]]:
    stats_buffers = {}
    for key, stat in stats.items():
        if key not in keys:
            continue
        if mode == "meanstd":
            mean = torch.tensor(stat["mean"], dtype=torch.float32)
            std = torch.tensor(stat["std"], dtype=torch.float32)
            buffer = nn.ParameterDict(
                {
                    "mean": nn.Parameter(mean, requires_grad=False),
                    "std": nn.Parameter(std, requires_grad=False),
                }
            )
        elif mode == "minmax":
            vmin = torch.tensor(stat["min"], dtype=torch.float32)
            vmax = torch.tensor(stat["max"], dtype=torch.float32)
            buffer = nn.ParameterDict(
                {
                    "min": nn.Parameter(vmin, requires_grad=False),
                    "max": nn.Parameter(vmax, requires_grad=False),
                }
            )
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
        stats_buffers[key] = buffer
    return stats_buffers


class Normalize(nn.Module):
    def __init__(
        self,
        keys: list[str],
        stats: dict[str, dict[str, Tensor]],
    ):
        super().__init__()
        self.keys = keys
        self.stats = stats
        stats_buffers = create_stats_buffers(keys, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key in self.keys:
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))
            mean = buffer["mean"]
            std = buffer["std"]
            batch[key] = (batch[key] - mean) / (std + 1e-8)
        return batch


class Unnormalize(nn.Module):
    def __init__(
        self,
        keys: list[str],
        stats: dict[str, dict[str, Tensor]],
    ):
        super().__init__()
        self.keys = keys
        self.stats = stats
        stats_buffers = create_stats_buffers(keys, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key in self.keys:
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))
            mean = buffer["mean"]
            std = buffer["std"]
            batch[key] = batch[key] * std + mean
        return batch


# ============ MinMax [-1, 1] ============

class NormalizeMinMax(nn.Module):
    """MinMax normalization: (x - min) / (max - min) * 2 - 1  →  [-1, 1]"""
    def __init__(
        self,
        keys: list[str],
        stats: dict[str, dict[str, Tensor]],
    ):
        super().__init__()
        self.keys = keys
        self.stats = stats
        stats_buffers = create_stats_buffers(keys, stats, mode="minmax")
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        for key in self.keys:
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))
            vmin = buffer["min"]
            vmax = buffer["max"]
            batch[key] = (batch[key] - vmin) / (vmax - vmin + 1e-8) * 2 - 1
        return batch


class UnnormalizeMinMax(nn.Module):
    """MinMax unnormalization: (x + 1) / 2 * (max - min) + min"""
    def __init__(
        self,
        keys: list[str],
        stats: dict[str, dict[str, Tensor]],
    ):
        super().__init__()
        self.keys = keys
        self.stats = stats
        stats_buffers = create_stats_buffers(keys, stats, mode="minmax")
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        for key in self.keys:
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))
            vmin = buffer["min"]
            vmax = buffer["max"]
            batch[key] = (batch[key] + 1) / 2 * (vmax - vmin) + vmin
        return batch
