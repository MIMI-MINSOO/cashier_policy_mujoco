"""Middle-layer between ControlLoop (real-time) and PolicyServer (GPU).

For every iteration:
    1. Block on ObsProvider.get() — wait for the next fresh observation
    2. Stack obs_horizon frames (duplicating on cold start)
    3. Ship (t_obs, stacked_obs) to PolicyServer
    4. Block on chunk_queue.get() — wait for the resulting chunk
    5. merger.submit(true_anchor, chunk) under merger_lock, where
       `true_anchor = t_obs - (obs_horizon - 1)` is the timestep predicted by
       chunk[0]. (See "Anchor correction" below.)
    6. Loop immediately

The immediate re-loop is what gives us "continuous inference": as soon
as a chunk lands we capture the latest obs and start the next inference,
without any executed-since-chunk gating.  This is the in-process
analogue of lerobot's RobotClient.receive_actions thread.

Anchor correction
-----------------
A chunk-based policy (Diffusion, ACT, ...) trained with obs_horizon h returns
a full pred_horizon prediction where the first (h-1) entries cover past
timesteps inside the obs window:

    chunk[0]        = action at time (t_obs - (h - 1))   ← past
    chunk[h - 1]    = action at time t_obs                ← obs capture
    chunk[k]        = action at time (t_obs - (h - 1) + k)

The policy's own select_action() slices [h-1 : h-1 + action_horizon] before
use.  In async mode we keep the whole chunk and just submit with the
correct anchor — the merger naturally ignores anything <= current_step
(the executed-during-latency portion is never queried), and overlapping
chunks line up on the same timestep grid for temporal ensemble.
"""

from __future__ import annotations

import collections
import queue
import threading
from typing import Callable

import torch


class ClientManager(threading.Thread):
    def __init__(
        self,
        obs_provider,
        policy_obs_queue: "queue.Queue",
        policy_chunk_queue: "queue.Queue",
        merger,
        merger_lock: threading.Lock,
        cfg,
        device: str,
        stop_event: threading.Event,
        on_chunk_submitted: Callable[[int, float], None] | None = None,
    ) -> None:
        super().__init__(daemon=True, name="ClientManager")
        self.obs_provider = obs_provider
        self.policy_obs_queue = policy_obs_queue
        self.policy_chunk_queue = policy_chunk_queue
        self.merger = merger
        self.merger_lock = merger_lock
        self.cfg = cfg
        self.device = device
        self.stop_event = stop_event
        self.on_chunk_submitted = on_chunk_submitted

        self.obs_horizon = cfg.policy.obs_horizon
        self._anchor_offset = self.obs_horizon - 1  # chunk[0] sits this far before obs capture
        self._obs_buffer: collections.deque = collections.deque(maxlen=self.obs_horizon)
        self._buffer_lock = threading.Lock()
        self._input_keys = list(cfg.task.image_keys) + [cfg.task.state_key]

    def reset_buffer(self) -> None:
        with self._buffer_lock:
            self._obs_buffer.clear()

    def _stack(self, obs) -> dict:
        """Push obs into buffer (cold-start by duplicating first frame) and stack."""
        obs_device = {
            k: v.unsqueeze(0).to(self.device, non_blocking=True) for k, v in obs.items()
        }
        with self._buffer_lock:
            self._obs_buffer.append(obs_device)
            while len(self._obs_buffer) < self.obs_horizon:
                self._obs_buffer.appendleft(self._obs_buffer[0])
            frames = list(self._obs_buffer)
        return {k: torch.stack([f[k] for f in frames], dim=1) for k in self._input_keys}

    def run(self) -> None:
        while not self.stop_event.is_set():
            item = self.obs_provider.get(timeout=0.5)
            if item is None:
                continue
            t_obs, obs = item

            stacked = self._stack(obs)

            try:
                self.policy_obs_queue.put((t_obs, stacked), timeout=0.1)
            except queue.Full:
                continue

            try:
                t_obs_recv, chunk, latency_ms = self.policy_chunk_queue.get(timeout=5.0)
            except queue.Empty:
                continue

            # Anchor correction: chunk[0] is the prediction for
            # (t_obs_recv - obs_horizon + 1), not for t_obs_recv. Submit with
            # the corrected anchor so merger.get_action(step) returns the
            # action whose timestep is `step`. Past entries (timestep < step)
            # are simply never queried — the lerobot-style "drop executed,
            # combine from current" behavior falls out for free.
            true_anchor = t_obs_recv - self._anchor_offset
            with self.merger_lock:
                self.merger.submit(true_anchor, chunk)

            if self.on_chunk_submitted is not None:
                # Pass the obs-capture step (not the corrected anchor) so the
                # debug log shows the intuitive "when was this obs taken?" value.
                self.on_chunk_submitted(t_obs_recv, latency_ms)
