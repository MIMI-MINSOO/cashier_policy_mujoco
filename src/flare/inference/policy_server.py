"""Inference worker thread.

Pure inference loop:
    pull (t_obs, stacked_obs) from input queue
    -> policy.generate_actions(...)
    -> push (t_obs, chunk_np, latency_ms) to output queue

This is the in-process analogue of lerobot's gRPC `PolicyServer`
(SendObservations + GetActions).  Same role, no network.  Keeping the
class small and dependency-free makes it easy to swap for a remote
gRPC server later — only the queue boundary needs to change.
"""

from __future__ import annotations

import queue
import threading
import time

import torch


class PolicyServer(threading.Thread):
    def __init__(
        self,
        policy,
        cfg,
        obs_queue: "queue.Queue",
        chunk_queue: "queue.Queue",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name="PolicyServer")
        self.policy = policy
        self.cfg = cfg
        self.obs_queue = obs_queue
        self.chunk_queue = chunk_queue
        self.stop_event = stop_event
        self._input_keys = list(cfg.task.image_keys) + [cfg.task.state_key]

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                t_obs, stacked_obs = self.obs_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            t_start = time.time()
            with torch.inference_mode():
                batch = {k: v for k, v in stacked_obs.items() if k in self._input_keys}
                batch = self.policy.normalize_inputs(batch)
                actions = self.policy.generate_actions(batch)
                actions = self.policy.unnormalize_outputs({"action": actions})["action"]
                chunk_np = actions.squeeze(0).cpu().numpy()
            latency_ms = (time.time() - t_start) * 1000.0

            try:
                self.chunk_queue.put((t_obs, chunk_np, latency_ms), timeout=0.1)
            except queue.Full:
                # Receiver lagged — drop the chunk. In continuous mode this
                # only happens if ClientManager is stuck; the next chunk will
                # supersede this one anyway.
                pass
