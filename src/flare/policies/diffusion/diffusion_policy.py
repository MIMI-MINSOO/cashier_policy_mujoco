"""
Diffusion Policy for visuomotor manipulation.
Adapted from Chi et al., 2023 and LeRobot implementation.

Performs DDPM/DDIM diffusion directly in action space,
conditioned on visual observations via FiLM modulation.
Uses SpatialSoftmax vision encoder and MinMax normalization.
"""

import torch
import logging
import einops
import torch.nn.functional as F
from itertools import chain
from tqdm import tqdm
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

from flare.factory import registry
from flare.policies import BasePolicy
from flare.policies.observers.resnet_observer import DiffusionRgbEncoder
from flare.networks.diffusion import ConditionalUnet1d
from flare.utils.normalize import NormalizeMinMax, UnnormalizeMinMax

logger = logging.getLogger(__name__)

NOISE_SCHEDULER_REGISTRY = {
    "DDPM": DDPMScheduler,
    "DDIM": DDIMScheduler,
}


@registry.register_policy("diffusion")
class DiffusionPolicy(BasePolicy):
    def __init__(self, config, stats):
        super().__init__(config, stats)

        # Override normalization: MinMax for state and action (same as LeRobot)
        self.normalize_inputs = NormalizeMinMax(
            [config.task.state_key], stats
        )
        self.normalize_targets = NormalizeMinMax(
            [config.task.action_key], stats
        )
        self.unnormalize_outputs = UnnormalizeMinMax(
            [config.task.action_key], stats
        )

        # Vision encoder — DiffusionRgbEncoder (ResNet18 + GroupNorm + SpatialSoftmax)
        diffusion_cfg = config.policy
        self.use_separate_encoder = diffusion_cfg.get("use_separate_rgb_encoder_per_camera", False)
        num_cameras = len(config.task.image_keys)

        encoder_kwargs = dict(
            resize_shape=tuple(config.resize_shape),
            crop_shape=tuple(config.crop_shape),
            crop_is_random=diffusion_cfg.get("crop_is_random", True),
            spatial_softmax_num_keypoints=diffusion_cfg.get("spatial_softmax_num_keypoints", 32),
        )

        if self.use_separate_encoder:
            self.rgb_encoder = torch.nn.ModuleList([
                DiffusionRgbEncoder(**encoder_kwargs) for _ in range(num_cameras)
            ])
        else:
            self.rgb_encoder = DiffusionRgbEncoder(**encoder_kwargs)

        # Compute global_cond_dim
        single_encoder = self.rgb_encoder[0] if self.use_separate_encoder else self.rgb_encoder
        img_feature_dim = single_encoder.feature_dim * num_cameras
        state_dim = config.task.state_dim
        self.global_cond_dim = state_dim + img_feature_dim

        # Noise scheduler
        scheduler_cfg = config.policy.noise_scheduler
        scheduler_cls = NOISE_SCHEDULER_REGISTRY[scheduler_cfg.type]
        self.noise_scheduler = scheduler_cls(
            num_train_timesteps=scheduler_cfg.num_train_timesteps,
            beta_start=scheduler_cfg.get("beta_start", 0.0001),
            beta_end=scheduler_cfg.get("beta_end", 0.02),
            beta_schedule=scheduler_cfg.beta_schedule,
            prediction_type=scheduler_cfg.prediction_type,
            clip_sample=scheduler_cfg.get("clip_sample", True),
            clip_sample_range=scheduler_cfg.get("clip_sample_range", 1.0),
        )
        self.num_inference_steps = scheduler_cfg.num_inference_steps

        # 1D UNet
        unet_cfg = config.policy.unet
        self.unet = ConditionalUnet1d(
            input_dim=config.task.action_dim,
            global_cond_dim=self.global_cond_dim,
            down_dims=list(unet_cfg.down_dims),
            kernel_size=unet_cfg.kernel_size,
            n_groups=unet_cfg.n_groups,
            diffusion_step_embed_dim=unet_cfg.diffusion_step_embed_dim,
            use_film_scale_modulation=unet_cfg.use_film_scale_modulation,
        )

        logger.info(f"UNet parameters: {sum(p.numel() for p in self.unet.parameters()) / 1e6:.2f}M")
        logger.info(f"Global cond dim: {self.global_cond_dim} (state={state_dim} + img={img_feature_dim})")
        logger.info(f"Scheduler: {scheduler_cfg.type}, train_steps={scheduler_cfg.num_train_timesteps}, "
                     f"inference_steps={self.num_inference_steps}")

        self.reset()

    def _encode_observations(self, batch):
        """Encode images + state into global conditioning vector.

        Returns:
            (B, global_cond_dim) conditioning vector.
        """
        # State: already normalized by normalize_inputs
        state = batch[self.config.task.state_key]  # (B, obs_horizon, state_dim)
        B, S = state.shape[:2]

        # Images: stack all cameras → (B, S, N, C, H, W)
        images = torch.stack([batch[key] for key in self.config.task.image_keys], dim=2)
        N = images.shape[2]

        if self.use_separate_encoder:
            # (B, S, N, C, H, W) → (N, B*S, C, H, W)
            images_per_cam = einops.rearrange(images, "b s n c h w -> n (b s) c h w")
            img_features = torch.cat([
                encoder(cam_imgs)
                for encoder, cam_imgs in zip(self.rgb_encoder, images_per_cam)
            ])
            # (N*B*S, feat) → (B, S, N*feat)
            img_features = einops.rearrange(
                img_features, "(n b s) d -> b s (n d)", b=B, s=S, n=N
            )
        else:
            # (B, S, N, C, H, W) → (B*S*N, C, H, W)
            flat_images = einops.rearrange(images, "b s n c h w -> (b s n) c h w")
            img_features = self.rgb_encoder(flat_images)
            # (B*S*N, feat) → (B, S, N*feat)
            img_features = einops.rearrange(
                img_features, "(b s n) d -> b s (n d)", b=B, s=S, n=N
            )

        # Concatenate state + image features, then flatten
        global_cond = torch.cat([state, img_features], dim=-1)
        global_cond = global_cond.flatten(start_dim=1)

        return global_cond

    def compute_loss(self, batch):
        global_cond = self._encode_observations(batch)
        actions = batch[self.config.task.action_key]  # (B, pred_horizon, action_dim)

        noise = torch.randn_like(actions)
        B = actions.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,),
            device=actions.device, dtype=torch.long,
        )

        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        pred = self.unet(noisy_actions, timesteps, global_cond)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = actions
        else:
            raise ValueError(f"Unknown prediction_type: {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(pred, target)
        return loss, {"diffusion_loss": loss.item()}

    def generate_actions(self, batch, global_cond=None):
        """Generate a full action chunk via diffusion denoising.

        Args:
            batch: Normalized observation dict (output of normalize_inputs).
            global_cond: Optional pre-encoded conditioning vector (B, global_cond_dim).
                         If None, _encode_observations(batch) is called internally.
                         Pass a pre-computed value to avoid double-encoding (ADR-004).

        Returns:
            Tensor(B, pred_horizon, action_dim) — normalized predicted actions.
        """
        if global_cond is None:
            global_cond = self._encode_observations(batch)
        B = global_cond.shape[0]
        sample = torch.randn(
            (B, self.pred_horizon, self.action_dim),
            device=global_cond.device,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            t_batch = t.unsqueeze(0).expand(B).to(global_cond.device)
            pred = self.unet(sample, t_batch, global_cond)
            sample = self.noise_scheduler.step(pred, t, sample).prev_sample
        return sample

    @torch.no_grad()
    def select_action(self, batch):
        self.eval()
        batch = {
            k: v.unsqueeze(1)
            for k, v in batch.items()
            if k in self.config.task.image_keys + [self.config.task.state_key]
        }
        batch = self.normalize_inputs(batch)

        if len(self._action_queue) == 0:
            actions = self.generate_actions(batch)
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            actions = actions[:, start:end]
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def select_action_ood(
        self,
        batch: dict,
        ood_ensemble=None,
    ) -> tuple:
        """OOD-aware action selection. Encodes observations exactly once per chunk.

        Follows ADR-004: _encode_observations is called at most once per new chunk,
        and the resulting global_cond is shared with the OOD ensemble — no double-encoding.

        Args:
            batch: Raw observation dict (same format as select_action's input —
                   keys must include image_keys + state_key, NO batch/time dims yet).
            ood_ensemble: OODDetectorEnsemble or None. If None, behaves identically
                          to select_action() but returns (action, None).

        Returns:
            action: Tensor(B, action_dim) — unnormalized action, same as select_action().
            ood_score: Tensor(B,) CDF percentile scores [0,1] when the action queue
                       was empty and a new chunk was generated; None otherwise.
                       High value (→1.0) = out-of-distribution.
        """
        self.eval()
        batch = {
            k: v.unsqueeze(1)
            for k, v in batch.items()
            if k in self.config.task.image_keys + [self.config.task.state_key]
        }
        batch = self.normalize_inputs(batch)

        ood_score = None
        if len(self._action_queue) == 0:
            # Encode observations exactly once (ADR-004: double-encoding 금지)
            global_cond = self._encode_observations(batch)  # (B, global_cond_dim) — 2D

            # Generate full action chunk using pre-encoded global_cond
            norm_actions = self.generate_actions(batch, global_cond=global_cond)  # (B, pred_horizon, action_dim)

            # Compute OOD score before slicing or unnormalizing
            if ood_ensemble is not None:
                # global_cond: (B, global_cond_dim) 2D tensor — NOT 3D
                # norm_actions: (B, pred_horizon, action_dim) full chunk — NOT sliced
                ood_score = ood_ensemble.detect(global_cond, norm_actions)  # (B,)

            # Slice to action_horizon and unnormalize (same as select_action)
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            actions = norm_actions[:, start:end]
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft(), ood_score

    def reset(self):
        super().reset()

    def get_optimizer(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config.optimizer_lr,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
            weight_decay=self.config.optimizer_weight_decay,
        )

    def get_scheduler(self, optimizer, num_training_steps):
        return get_scheduler(
            name=self.config.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.config.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def encode_observations(self, batch):
        """Public API for encoding observations into the global conditioning vector.

        Returns:
            (B, global_cond_dim) tensor — same as _encode_observations output.
        """
        return self._encode_observations(batch)

    @torch.no_grad()
    def compute_ood_loss(self, global_cond, nactions, num_repeats=4):
        """Compute per-sample diffusion loss for OOD detection.

        Averages MSE over num_repeats random timesteps to produce a stable
        loss estimate. Corresponds to DiffDAgger's get_avg_diffusion_loss_ndata
        but returns per-sample Tensor(B,) instead of a scalar.

        Args:
            global_cond: (B, global_cond_dim) — output of encode_observations().
            nactions:    (B, pred_horizon, action_dim) — normalized actions.
            num_repeats: Number of random-timestep samples per item (default 4).

        Returns:
            Tensor(B,) of mean per-sample diffusion losses.

        # From: references/DiffDAgger/diffdagger/agents/diffusion_policy.py
        """
        T = self.noise_scheduler.config.num_train_timesteps
        B = global_cond.shape[0]
        device = global_cond.device
        total_loss = torch.zeros(B, device=device)

        for _ in range(num_repeats):
            timesteps = torch.randint(0, T, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(nactions)
            noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)
            pred = self.unet(noisy_actions, timesteps, global_cond)

            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "sample":
                target = nactions
            else:
                raise ValueError(
                    f"Unknown prediction_type: {self.noise_scheduler.config.prediction_type}"
                )

            # MSE averaged over (pred_horizon, action_dim), keeping batch dim
            per_sample = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2))
            total_loss += per_sample

        return total_loss / num_repeats

    @torch.no_grad()
    def calibrate_ood(self, dataset, alpha=0.95, num_iter=16):
        """Build empirical loss CDF from training data for OOD threshold calibration.

        Iterates over dataset num_iter times, computing compute_ood_loss per sample.
        Stores diffusion_loss_cdf and diffusion_loss_threshold on self so they are
        persisted via state_dict().

        Args:
            dataset:  Iterable of single-sample dicts (no batch dim).
            alpha:    CDF quantile that becomes the OOD threshold (default 0.95).
            num_iter: Number of full dataset passes (default 16).

        # From: references/DiffDAgger/diffdagger/agents/diffusion_policy.py
        """
        from flare.ood_detectors.utils.cdf import CDF

        self.eval()
        all_losses = []

        calib_device = next(self.parameters()).device
        with tqdm(chain.from_iterable([dataset] * num_iter), desc="Calibrating OOD") as pbar:
            for datapoint in pbar:
                batch = {k: v.unsqueeze(0).to(calib_device) for k, v in datapoint.items() if isinstance(v, torch.Tensor)}
                batch = self.normalize_inputs(batch)
                batch = self.normalize_targets(batch)

                global_cond = self._encode_observations(batch)   # (1, global_cond_dim)
                nactions = batch[self.config.task.action_key]    # (1, pred_horizon, action_dim)

                loss = self.compute_ood_loss(global_cond, nactions)  # (1,)
                all_losses.append(loss.item())

        self.diffusion_loss_cdf = CDF(all_losses)
        self.diffusion_loss_threshold = self.diffusion_loss_cdf.get_quantile(alpha)
        logger.info(
            f"OOD calibration complete: {len(all_losses)} samples, "
            f"threshold(alpha={alpha}) = {self.diffusion_loss_threshold:.6f}"
        )

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if hasattr(self, 'diffusion_loss_cdf'):
            # Store raw losses as float32 tensor — safetensors-compatible
            sd['_ood_losses'] = torch.tensor(
                list(self.diffusion_loss_cdf.data), dtype=torch.float32
            )
        if hasattr(self, 'diffusion_loss_threshold'):
            sd['_ood_threshold'] = torch.tensor(
                self.diffusion_loss_threshold, dtype=torch.float32
            )
        return sd

    def load_state_dict(self, state_dict, strict=True):
        # Shallow copy so we don't mutate the caller's dict
        state_dict = dict(state_dict)
        ood_losses = state_dict.pop('_ood_losses', None)
        ood_threshold = state_dict.pop('_ood_threshold', None)
        state_dict.pop('_ood_cdf', None)  # backwards compat: drop old non-tensor key
        result = super().load_state_dict(state_dict, strict=strict)
        if ood_losses is not None:
            self.diffusion_loss_cdf = CDF(ood_losses.tolist())
        if ood_threshold is not None:
            self.diffusion_loss_threshold = float(ood_threshold.item())
        return result

    @torch.no_grad()
    def score_dataset_ood(
        self,
        dataset,
        ood_ensemble,
        device=None,
    ) -> list:
        """Compute OOD score for each sample in a dataset.

        Unlike calibrate_ood() which builds a CDF, this returns the CDF
        percentile score [0, 1] for each sample. Requires calibrate_ood()
        to have been run first (diffusion_loss_cdf must exist).

        Args:
            dataset:      Iterable of single-sample dicts (no batch dim).
                          Must contain image_keys + state_key + action_key.
            ood_ensemble: OODDetectorEnsemble (or any detector with detect()).
                          Must be initialized with a calibrated CDF.
            device:       Device override. Defaults to policy's current device.

        Returns:
            List of float OOD scores in [0, 1], one per dataset sample.
            Higher = more out-of-distribution.

        Raises:
            RuntimeError: if diffusion_loss_cdf is not set (not calibrated).
        """
        if not hasattr(self, 'diffusion_loss_cdf'):
            raise RuntimeError(
                "OOD calibration has not been run. "
                "Call calibrate_ood(dataset) before score_dataset_ood()."
            )

        if device is None:
            device = next(self.parameters()).device

        self.eval()
        scores = []

        for sample in dataset:
            batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if isinstance(v, torch.Tensor)}
            batch = self.normalize_inputs(batch)
            batch = self.normalize_targets(batch)

            global_cond = self._encode_observations(batch)       # (1, global_cond_dim)
            nactions = batch[self.config.task.action_key]        # (1, pred_horizon, action_dim)

            score = ood_ensemble.detect(global_cond, nactions)   # Tensor(1,)
            scores.append(score.item())

        return scores

    def get_ood_detector(self):
        """Return DiffusionLossCDFDetector if calibration CDF is available."""
        if not hasattr(self, 'diffusion_loss_cdf'):
            return None

        from flare.ood_detectors import DiffusionLossCDFDetector
        return DiffusionLossCDFDetector(
            policy=self,
            config={},
            device=next(self.parameters()).device
        )
