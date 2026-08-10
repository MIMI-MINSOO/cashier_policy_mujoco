"""Eval orchestrator. Async (3-thread) is default; sync (in-line) is optional.

Async roles
-----------
Thread 1 (main / control loop)
    Real-time loop at `control_freq` Hz.  Pops an action from the merger for
    the current global step, sends it to the robot, calls env.step() to
    capture the next observation, then publishes (step+1, obs) to the
    ObsProvider for the inference path.  Also handles c/s/q key input.

Thread 2 (ClientManager, daemon)
    Continuously pulls the freshest obs from ObsProvider, ships it to
    PolicyServer, waits for the resulting chunk, and submits it to the merger
    under the merger lock.  Loops back immediately — this is what gives us
    "continuous inference": the next request is issued the instant the
    previous chunk lands, with no executed-since-chunk gating.

Thread 3 (PolicyServer, daemon)
    Pure inference worker.  Pulls (t_obs, obs) from one queue, runs
    `policy.generate_actions`, pushes (t_obs, chunk, latency_ms) to another.

Sync mode (`sync=True`)
-----------------------
ClientManager + PolicyServer threads are not started.  The control loop runs
inference in-line at the top of each iteration, slices [obs_horizon-1 : +action_horizon]
so chunk[0] aligns with the current step (matching the legacy convention),
submits to the merger, then pops the action for the current step.  Mergers
work the same way — all four pairs (sync, async) x (overwrite, temporal_ensemble)
are valid, although sync mode produces non-overlapping chunks and the temporal
ensembler degenerates to overwrite in practice (single chunk covers each step).

Mirrors the role split of lerobot's RobotClient(control_loop +
receive_actions) + PolicyServer(gRPC), collapsed to a single process —
queue.Queue replaces gRPC, threading.Lock replaces RPC framing.
"""

from __future__ import annotations

import collections
import queue
import threading
import time

import numpy as np
import torch

from .client_manager import ClientManager
from .merger import make_merger
from .obs_provider import ObsProvider
from .policy_server import PolicyServer


def _build_idle_action(robot, action_dim: int) -> np.ndarray:
    # 심 로봇(SimRobotShim 등)은 자기 idle 액션을 직접 제공한다 (목 포함 16차원 등)
    if hasattr(robot, "idle_action"):
        return np.asarray(robot.idle_action, dtype=np.float32)
    if hasattr(robot, "left") and hasattr(robot, "right"):
        l = robot.left.eval_init if robot.left.eval_init else robot.left.joints_init
        r = robot.right.eval_init if robot.right.eval_init else robot.right.joints_init
        return np.array(list(l) + [70.0] + list(r) + [70.0], dtype=np.float32)
    joints = robot.eval_init if robot.eval_init else robot.joints_init
    return np.array(list(joints) + [70.0], dtype=np.float32)


class EvalRunner:
    def __init__(
        self,
        policy,
        cfg,
        env,
        *,
        merger_name: str = "temporal_ensemble",
        te_coeff: float = 0.01,
        max_steps: int = 2700,
        control_freq: float = 30.0,
        device: str = "cuda:0",
        sync: bool = False,
        key_listener=None,
        cam_display=None,
    ) -> None:
        self.policy = policy
        self.cfg = cfg
        self.env = env
        self.max_steps = max_steps
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.device = device
        self.sync = sync
        self.key_listener = key_listener
        self.cam_display = cam_display

        # Shared state — merger/lock used in both modes; obs_provider/queues
        # only in async (kept around in sync for code uniformity but unused).
        self.merger = make_merger(merger_name, te_coeff=te_coeff)
        self.merger_lock = threading.Lock()
        self.obs_provider = ObsProvider()
        self.policy_obs_queue: queue.Queue = queue.Queue(maxsize=1)
        self.policy_chunk_queue: queue.Queue = queue.Queue(maxsize=4)
        self.stop_event = threading.Event()

        # Threads — created now, started in run() if async.
        self.policy_server = PolicyServer(
            policy=policy,
            cfg=cfg,
            obs_queue=self.policy_obs_queue,
            chunk_queue=self.policy_chunk_queue,
            stop_event=self.stop_event,
        )
        self.client_manager = ClientManager(
            obs_provider=self.obs_provider,
            policy_obs_queue=self.policy_obs_queue,
            policy_chunk_queue=self.policy_chunk_queue,
            merger=self.merger,
            merger_lock=self.merger_lock,
            cfg=cfg,
            device=device,
            stop_event=self.stop_event,
            on_chunk_submitted=self._on_chunk_submitted,
        )

        # Banner state
        self._merger_name = merger_name
        self._te_coeff = te_coeff

        # Chunk-arrival logging (set by ClientManager callback in async;
        # not used in sync — sync produces a chunk every step, too noisy).
        self._chunk_event = threading.Event()
        self._last_chunk_t_obs: int | None = None
        self._last_chunk_latency: float | None = None

        # Sync-mode local state — small obs buffer owned by the control thread
        # since there's no ClientManager. Always allocated; ignored in async.
        self.obs_horizon = cfg.policy.obs_horizon
        self._sync_obs_buffer: collections.deque = collections.deque(maxlen=self.obs_horizon)
        self._policy_input_keys = list(cfg.task.image_keys) + [cfg.task.state_key]

    # -------------------- chunk-arrival callback --------------------
    def _on_chunk_submitted(self, t_obs: int, latency_ms: float) -> None:
        self._last_chunk_t_obs = t_obs
        self._last_chunk_latency = latency_ms
        self._chunk_event.set()

    # -------------------- episode lifecycle --------------------
    def _drain_queues(self) -> None:
        for q in (self.policy_obs_queue, self.policy_chunk_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _start_episode(self):
        self.policy.reset()
        self.env.robot.reset_filter()
        obs, raw_images = self.env.reset()

        with self.merger_lock:
            self.merger.clear()
        self._drain_queues()
        self.obs_provider.reset()
        self.client_manager.reset_buffer()
        self._sync_obs_buffer.clear()
        self._chunk_event.clear()
        self._last_chunk_t_obs = None
        self._last_chunk_latency = None

        # Async warm-up: publish initial obs so the inference path can start
        # before the first action is needed. In sync, _sync_step pulls obs
        # directly each tick, so no publish is needed.
        if not self.sync:
            self.obs_provider.put(0, obs)
        return obs, raw_images

    # -------------------- sync inference --------------------
    def _sync_step(self, obs, step: int) -> None:
        """Run inference in-line and submit chunk to the merger.

        Convention: slice [obs_horizon-1 : +action_horizon] so chunk[0] is the
        prediction for the current step (matching legacy sync behavior).
        """
        obs_device = {
            k: v.unsqueeze(0).to(self.device, non_blocking=True) for k, v in obs.items()
        }
        self._sync_obs_buffer.append(obs_device)
        while len(self._sync_obs_buffer) < self.obs_horizon:
            self._sync_obs_buffer.appendleft(self._sync_obs_buffer[0])

        stacked = {
            k: torch.stack([f[k] for f in self._sync_obs_buffer], dim=1)
            for k in self._policy_input_keys
        }

        with torch.inference_mode():
            batch = self.policy.normalize_inputs(stacked)
            actions = self.policy.generate_actions(batch)
            start = self.obs_horizon - 1
            end = start + self.cfg.policy.action_horizon
            actions = actions[:, start:end]
            actions = self.policy.unnormalize_outputs({"action": actions})["action"]
            chunk_np = actions.squeeze(0).cpu().numpy()

        with self.merger_lock:
            self.merger.submit(step, chunk_np)

    # -------------------- main loop --------------------
    def run(self) -> None:
        merger_desc = self._merger_name + (
            f"(coeff={self._te_coeff})" if self._merger_name == "temporal_ensemble" else ""
        )
        mode_desc = "sync (in-line)" if self.sync else "async (3-thread)"
        print("\nControls: c=start  s=stop  q/ESC=quit  Ctrl+C=emergency")
        print(
            f"max_steps={self.max_steps}, freq={self.control_freq}Hz, "
            f"mode={mode_desc}, merger={merger_desc}"
        )

        if not self.sync:
            self.policy_server.start()
            self.client_manager.start()

        print("Moving to initial pose...")
        self.env.robot.move_to_eval_pose(duration=5.0)
        print("Ready. Press 'c' to start.")

        action_dim = self.cfg.task.action_dim
        eval_action = _build_idle_action(self.env.robot, action_dim)

        running = False
        step = 0
        last_action_np: np.ndarray | None = None
        prev_action_np: np.ndarray | None = None
        raw_images = None
        obs = None  # latest observation, owned by control thread

        try:
            while step < self.max_steps:
                pressed = self.key_listener.get() if self.key_listener else None

                if pressed == "c" and not running:
                    running = True
                    obs, raw_images = self._start_episode()
                    step = 0
                    last_action_np = None
                    prev_action_np = None
                    print("\n>>> Started")

                elif pressed == "s" and running:
                    running = False
                    print("\n>>> Stopped")
                    self.env.robot.move_to_eval_pose(duration=5.0)
                    print("Ready. Press 'c' to restart.")

                elif pressed in ("q", "\x1b"):
                    print("\n>>> Quit")
                    break

                if self.cam_display is not None and raw_images is not None:
                    self.cam_display.update(raw_images, step, running)

                t_loop_start = time.time()

                if not running:
                    self.env.robot.send_action(eval_action)
                    self._pace(t_loop_start)
                    continue

                # Sync: run inference in-line so a fresh chunk is submitted
                # before pop. Async: chunks are submitted by ClientManager.
                if self.sync and obs is not None:
                    self._sync_step(obs, step)

                # Pop predictions for now & next step (look-ahead for quintic interp)
                with self.merger_lock:
                    action_np = self.merger.get_action(step)
                    next_action_np = self.merger.get_action(step + 1)

                if action_np is not None:
                    last_action_np = action_np
                elif last_action_np is not None:
                    action_np = last_action_np
                    print(f"  !! STALL step:{step} (no prediction, reusing last)")
                else:
                    # Cold start — first chunk not yet submitted. Hold pose.
                    self.env.robot.send_action(eval_action)
                    self._pace(t_loop_start)
                    continue

                obs, raw_images = self.env.step(action_np, next_action=next_action_np)

                # Async only: hand the freshly captured obs to ClientManager.
                if not self.sync:
                    self.obs_provider.put(step + 1, obs)

                self._pace(t_loop_start)
                freq = 1.0 / max(time.time() - t_loop_start, 1e-6)

                # Periodic debug — delta only (per-arm for bimanual; 16 = bimanual+neck).
                if action_dim in (14, 16) and prev_action_np is not None:
                    dl_l = np.linalg.norm(action_np[:6] - prev_action_np[:6])
                    dl_r = np.linalg.norm(action_np[7:13] - prev_action_np[7:13])
                    delta_str = f"dl_L:{dl_l:.2f} dl_R:{dl_r:.2f}"
                elif prev_action_np is not None:
                    delta_str = f"dl:{np.linalg.norm(action_np[:6] - prev_action_np[:6]):.2f}"
                else:
                    delta_str = "dl:--"
                prev_action_np = action_np.copy()

                if step % 30 == 0:
                    print(f"  step:{step:4d}  {delta_str}  freq:{freq:.0f}Hz")

                if self._chunk_event.is_set():
                    self._chunk_event.clear()
                    print(
                        f"  [chunk] t_obs:{self._last_chunk_t_obs} "
                        f"infer:{self._last_chunk_latency:.0f}ms"
                    )

                step += 1

        except KeyboardInterrupt:
            print("\n>>> Emergency stop")
        finally:
            self.stop_event.set()
            self.obs_provider.stop()
            # Daemon threads will exit on their own when stop_event is observed.
            time.sleep(0.1)

        if step > 0:
            print(f"Finished after {step} steps.")

    def _pace(self, t_loop_start: float) -> None:
        elapsed = time.time() - t_loop_start
        if elapsed < self.dt:
            time.sleep(self.dt - elapsed)
