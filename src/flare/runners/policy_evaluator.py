import torch
import logging
from pathlib import Path
from collections import defaultdict

from flare.factory import registry
from flare.utils.logger import FlareLogger
from flare.utils.eval import eval_policy
from flare.utils.checkpoints import load_model_weights

logger = logging.getLogger(__name__)


@registry.register_runner("policy_evaluator")
class PolicyEvaluator:
    def __init__(
        self,
        config,
        network,
        device,
        val_dataloader=None,
        eval_env=None,
        ood_ensemble=None,
    ):
        self.config = config
        self.network = network.to(device)
        self.device = device
        self.val_dataloader = val_dataloader
        self.eval_env = eval_env
        self.ood_ensemble = ood_ensemble
        self.ema = network.get_ema()

        self.flare_logger = FlareLogger(config)

    def validate_offline(self):
        self.network.eval()
        val_metrics = defaultdict(float)
        num_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)

                output_dict = self.network.validate(batch)

                batch_size = next(iter(batch.values())).shape[0]
                for k, v in output_dict.items():
                    val_metrics[k] += v * batch_size
                num_samples += batch_size

                if batch_idx == 0 and hasattr(self.network, 'visualize'):
                    viz_results = self.network.visualize(
                        batch,
                        num_samples=self.config.val.num_viz_samples
                    )

                    if viz_results:
                        self.flare_logger.log_figures(
                            viz_results,
                            prefix='plot',
                        )

        val_metrics = {k: v / num_samples for k, v in val_metrics.items()}
        return val_metrics

    def validate_online(self):
        model = self.network

        if self.config.use_ema and self.ema is not None:
            logger.info("Using EMA weights for evaluation")
            self.ema.store(model.parameters())
            self.ema.copy_to(model.parameters())

        model.eval()
        with torch.no_grad():
            eval_info = eval_policy(
                self.eval_env,
                model,
                self.config.val.eval_n_episodes,
                videos_dir=Path(self.config.eval_dir) / f"videos",
                max_episodes_rendered=self.config.val.num_viz_videos,
                start_seed=self.config.seed,
                ood_ensemble=self.ood_ensemble,
            )

        if self.config.use_ema and self.ema is not None:
            self.ema.restore(model.parameters())

        # OOD 메트릭 계산 및 로깅 (ood_ensemble이 활성화된 경우에만)
        if "ood_chunk_scores" in eval_info:
            ood_chunk_scores = eval_info["ood_chunk_scores"]
            if ood_chunk_scores.numel() > 0:
                ood_scores_flat = ood_chunk_scores.flatten()  # (num_chunks * B)
                mean_ood_score = ood_scores_flat.mean().item()
                max_ood_score = ood_scores_flat.max().item()
                threshold = self.config.ood_detection.get(
                    "threshold", self.config.ood_detection.calibration.alpha
                )
                ood_trigger_rate = (ood_scores_flat > threshold).float().mean().item()

                eval_info["aggregated"]["ood_mean_score"] = mean_ood_score
                eval_info["aggregated"]["ood_max_score"] = max_ood_score
                eval_info["aggregated"]["ood_trigger_rate"] = ood_trigger_rate

        return eval_info

    def eval(self):
        offline_metrics = self.validate_offline()
        online_metrics = self.validate_online()
        logger.info(f"Online metrics: {online_metrics}")
        logger.info(f"Offline metrics: {offline_metrics}")
        aggregated = online_metrics.get("aggregated", {})
        per_episode = online_metrics.get("per_episode", [])
        num_episodes = len(per_episode)

        summary = {
            "episodes": num_episodes,
            "pc_success": aggregated.get("pc_success"),
            "avg_max_reward": aggregated.get("avg_max_reward"),
            "loss": offline_metrics.get("loss"),
            "action_mse": offline_metrics.get("action_mse"),
            "eval_s": aggregated.get("eval_s"),
            "eval_ep_s": aggregated.get("eval_ep_s"),
            "ood_mean_score": aggregated.get("ood_mean_score"),
            "ood_max_score": aggregated.get("ood_max_score"),
            "ood_trigger_rate": aggregated.get("ood_trigger_rate"),
        }
        logger.info(f"Eval summary: {summary}")

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)

        training_state_file = checkpoint_dir / "training_state.pt"

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        if not training_state_file.exists():
            raise FileNotFoundError(f"Training state file not found: {training_state_file}")

        load_model_weights(self.network, checkpoint_dir, self.device)
        self.network.to(self.device)

        training_state = torch.load(
            training_state_file,
            map_location=self.device,
            weights_only=False
        )

        if self.ema and training_state.get("ema"):
            self.ema.load_state_dict(training_state["ema"])

        step = training_state.get("step")

        ood_calibration_file = checkpoint_dir / "ood_calibration.pt"
        if ood_calibration_file.exists():
            ood_data = torch.load(
                ood_calibration_file, map_location="cpu", weights_only=False
            )
            if "diffusion_loss_cdf" in ood_data:
                self.network.diffusion_loss_cdf = ood_data["diffusion_loss_cdf"]
            if "diffusion_loss_threshold" in ood_data:
                self.network.diffusion_loss_threshold = ood_data["diffusion_loss_threshold"]
            logger.info(
                f"Loaded OOD calibration: threshold={ood_data.get('diffusion_loss_threshold', 'N/A')}"
            )

        logger.info(f"Loaded checkpoint from {checkpoint_dir} at step {step}")
        return step
