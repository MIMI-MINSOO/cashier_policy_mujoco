from dataclasses import dataclass
import numpy as np


@dataclass
class TimedChunk:
    """An action chunk anchored to the timestep at which its observation was captured.

    chunk[k] is the predicted action for global timestep (t_obs + k).
    """

    t_obs: int
    chunk: np.ndarray  # shape (chunk_size, action_dim)

    def get(self, t: int) -> np.ndarray | None:
        idx = t - self.t_obs
        if 0 <= idx < len(self.chunk):
            return self.chunk[idx]
        return None

    def last_t(self) -> int:
        """Last timestep for which this chunk has a prediction (inclusive)."""
        return self.t_obs + len(self.chunk) - 1
