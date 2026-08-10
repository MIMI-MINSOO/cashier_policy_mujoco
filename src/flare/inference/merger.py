"""Action chunk mergers.

A Merger receives action chunks via `submit(t_obs, chunk)` and returns the
action for a requested global timestep via `get_action(t)`. Timestep-aware:
chunk[k] of a submission with t_obs=N is the prediction for time (N+k).

Two strategies:
    Overwrite           — keep only the latest chunk (no ensembling)
    TemporalEnsembler   — exponential-weighted average over overlapping chunks
                          (ACT-style; coeff controls recency bias)
"""

from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from .timed_chunk import TimedChunk


class Merger(ABC):
    @abstractmethod
    def submit(self, t_obs: int, chunk: np.ndarray) -> None: ...

    @abstractmethod
    def get_action(self, t: int) -> np.ndarray | None: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def remaining_after(self, t: int) -> int:
        """Number of future timesteps (>= t) for which a prediction exists.
        Used for debug logging and request-scheduling heuristics."""


class Overwrite(Merger):
    """Latest-chunk-only. Old chunks are discarded on each submit."""

    def __init__(self):
        self._latest: TimedChunk | None = None

    def submit(self, t_obs: int, chunk: np.ndarray) -> None:
        self._latest = TimedChunk(t_obs=t_obs, chunk=np.asarray(chunk))

    def get_action(self, t: int) -> np.ndarray | None:
        if self._latest is None:
            return None
        return self._latest.get(t)

    def clear(self) -> None:
        self._latest = None

    def remaining_after(self, t: int) -> int:
        if self._latest is None:
            return 0
        return max(self._latest.last_t() - t + 1, 0)


class TemporalEnsembler(Merger):
    """ACT-style exponential-weighted average across overlapping chunks.

    For each timestep t we collect every stored chunk that predicts t and take
    a weighted average. Weights follow weight[age] = exp(-coeff * age), where
    age=0 is the most-recently-submitted chunk.

        coeff > 0   → recent-dominant
        coeff = 0   → uniform average
        coeff < 0   → old-dominant
    """

    def __init__(self, coeff: float = 0.01, max_chunks: int = 16):
        self.coeff = float(coeff)
        self._chunks: deque[TimedChunk] = deque(maxlen=max_chunks)

    def submit(self, t_obs: int, chunk: np.ndarray) -> None:
        self._chunks.append(TimedChunk(t_obs=t_obs, chunk=np.asarray(chunk)))

    def get_action(self, t: int) -> np.ndarray | None:
        preds: list[np.ndarray] = []
        ages: list[int] = []
        # age = 0 for the newest submission (rightmost in deque)
        for age, tc in enumerate(reversed(self._chunks)):
            a = tc.get(t)
            if a is not None:
                preds.append(a)
                ages.append(age)

        if not preds:
            return None

        weights = np.exp(-self.coeff * np.asarray(ages, dtype=np.float64))
        weights /= weights.sum()
        stacked = np.stack(preds, axis=0)
        return (stacked * weights[:, None]).sum(axis=0).astype(stacked.dtype, copy=False)

    def clear(self) -> None:
        self._chunks.clear()

    def remaining_after(self, t: int) -> int:
        if not self._chunks:
            return 0
        last = max(tc.last_t() for tc in self._chunks)
        return max(last - t + 1, 0)


def make_merger(name: str, te_coeff: float = 0.01) -> Merger:
    """Factory used by eval CLI.

    `name`:
        "overwrite"          — Overwrite
        "temporal_ensemble"  — TemporalEnsembler(coeff=te_coeff)
    """
    if name == "overwrite":
        return Overwrite()
    if name == "temporal_ensemble":
        return TemporalEnsembler(coeff=te_coeff)
    raise ValueError(
        f"Unknown merger: {name!r}. Available: 'overwrite', 'temporal_ensemble'."
    )
