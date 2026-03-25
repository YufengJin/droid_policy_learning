"""
Drift Policy: same UNet architecture as Diffusion Policy, but training uses
mean-shift drift loss (from drift_model/run_experiment.py) instead of noise prediction.
Inference: 1 NFE via single DDIM step from pure noise.

Improvements over base drift:
  - Two-phase training: warmup with diffusion loss, then switch to drift loss
  - Multi-scale kernel: sum of RBF kernels at multiple bandwidths
  - Curriculum max-drift: anneal max_drift from large to small over training
  - Scheduled EMA decay: anneal EMA decay from fast to slow over training
  - Drift alignment monitoring: cosine similarity between V and (nearest_pos - gen)
"""

import warnings
from collections import OrderedDict

import torch
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet
from robomimic.algo.drift_utils import compute_drift, compute_adaptive_temp, clip_drift


@register_algo_factory_func("drift_policy")
def drift_algo_config_to_class(algo_config):
    if algo_config.unet.enabled:
        return DriftPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class DriftPolicyUNet(DiffusionPolicyUNet):
    """
    Drift Policy: reuses Diffusion Policy's obs_encoder + ConditionalUnet1D + EMA.
    Training: z ~ N(0,I) -> pred_x0 via epsilon-to-x0 -> drift(gen, pos) -> MSE(student_x0, target).
    Inference: 1 NFE (config sets ddim.num_inference_timesteps=1).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_step_counter = 0

    def _pred_x0_from_noise(self, sample, timesteps, noise_pred):
        """
        Convert epsilon prediction to pred_original_sample (x0) using scheduler alphas.
        sample: [B, Tp, Da], timesteps: [B], noise_pred: [B, Tp, Da]
        """
        scheduler = self.noise_scheduler
        alpha_prod_t = scheduler.alphas_cumprod.to(sample.device)[timesteps]
        alpha_prod_t = alpha_prod_t.view(-1, 1, 1)
        beta_prod_t = 1.0 - alpha_prod_t
        pred_x0 = (sample - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt().clamp(min=1e-8)
        if getattr(scheduler.config, "clip_sample", True):
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
        return pred_x0

    def _get_drift_cfg_value(self, drift_cfg, key, default):
        """Helper to safely read drift config values."""
        if drift_cfg is None:
            return default
        return getattr(drift_cfg, key, default)

    def _compute_scheduled_max_drift(self, drift_cfg, epoch, base_max_drift):
        """Compute max_drift with optional curriculum annealing."""
        anneal_epochs = self._get_drift_cfg_value(drift_cfg, "max_drift_anneal_epochs", 0)
        if anneal_epochs <= 0:
            return base_max_drift
        start = self._get_drift_cfg_value(drift_cfg, "max_drift_start", base_max_drift)
        end = self._get_drift_cfg_value(drift_cfg, "max_drift_end", base_max_drift)
        progress = min(1.0, epoch / anneal_epochs)
        return start + (end - start) * progress

    def _update_ema_decay(self, drift_cfg, epoch):
        """Update EMA max_value for scheduled decay annealing."""
        if self.ema is None:
            return
        anneal_epochs = self._get_drift_cfg_value(drift_cfg, "ema_decay_anneal_epochs", 0)
        if anneal_epochs <= 0:
            return
        start = self._get_drift_cfg_value(drift_cfg, "ema_decay_start", None)
        end = self._get_drift_cfg_value(drift_cfg, "ema_decay_end", None)
        if start is None or end is None:
            return
        progress = min(1.0, epoch / anneal_epochs)
        new_max_value = start + (end - start) * progress
        self.ema.max_value = new_max_value

    def train_on_batch(self, batch, epoch, validate=False):
        drift_cfg = getattr(self.algo_config, "drift", None)

        # ── Two-phase training: warmup with diffusion loss ──
        warmup_epochs = self._get_drift_cfg_value(drift_cfg, "warmup_epochs", 0)
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            info = DiffusionPolicyUNet.train_on_batch(self, batch, epoch, validate=validate)
            info["_training_phase"] = "diffusion_warmup"
            return info

        # ── Scheduled EMA decay ──
        if not validate:
            self._update_ema_decay(drift_cfg, epoch)

        B = batch["actions"].shape[0]

        temp = self._get_drift_cfg_value(drift_cfg, "temp", 0.05)
        max_drift = self._get_drift_cfg_value(drift_cfg, "max_drift", 0.1)
        use_adaptive_temp = self._get_drift_cfg_value(drift_cfg, "use_adaptive_temp", True)
        kernel_type = self._get_drift_cfg_value(drift_cfg, "kernel_type", "laplace")
        dist_scale_mode = self._get_drift_cfg_value(drift_cfg, "dist_scale_mode", "none")
        rbf_sigma = self._get_drift_cfg_value(drift_cfg, "rbf_sigma", 1.0)
        multiscale_sigmas = self._get_drift_cfg_value(drift_cfg, "multiscale_sigmas", [0.25, 0.5, 1.0, 2.0])
        adaptive_median_scale = self._get_drift_cfg_value(drift_cfg, "adaptive_median_scale", 0.3)
        adaptive_min_temp = self._get_drift_cfg_value(drift_cfg, "adaptive_min_temp", 0.01)
        log_kernel_stats = self._get_drift_cfg_value(drift_cfg, "log_kernel_stats", True)
        log_every_n = self._get_drift_cfg_value(drift_cfg, "log_kernel_every_n_steps", 100)
        assert_alive = self._get_drift_cfg_value(drift_cfg, "assert_alive_kernel", False)
        alive_threshold = self._get_drift_cfg_value(drift_cfg, "alive_kernel_threshold", 1e-6)

        # ── Curriculum max-drift ──
        max_drift = self._compute_scheduled_max_drift(drift_cfg, epoch, max_drift)

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]

            inputs = {"obs": batch["obs"]}
            obs_features = TensorUtils.time_distributed(
                {"obs": inputs["obs"]}, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True
            )
            assert obs_features.ndim == 3
            obs_cond = obs_features.flatten(start_dim=1)

            t_fixed = self.noise_scheduler.config.num_train_timesteps - 1
            timesteps = torch.full((B,), t_fixed, device=self.device, dtype=torch.long)

            z = torch.randn_like(actions, device=self.device)

            with torch.no_grad():
                if self.ema is not None:
                    noise_pred_ema = self.ema.averaged_model["policy"]["noise_pred_net"](
                        z, timesteps, global_cond=obs_cond
                    )
                else:
                    self.nets.eval()
                    noise_pred_ema = self.nets["policy"]["noise_pred_net"](
                        z, timesteps, global_cond=obs_cond
                    )
                    self.nets.train()
                pred_x0_ema = self._pred_x0_from_noise(z, timesteps, noise_pred_ema)

            gen_flat = pred_x0_ema.reshape(B, -1)
            pos_flat = actions.reshape(B, -1)

            adaptive_temp = (
                compute_adaptive_temp(
                    gen_flat, pos_flat,
                    base_temp=temp,
                    median_scale=adaptive_median_scale,
                    min_temp=adaptive_min_temp,
                    dist_scale_mode=dist_scale_mode,
                )
                if use_adaptive_temp
                else temp
            )

            V, kernel_stats = compute_drift(
                gen_flat, pos_flat,
                temp=adaptive_temp,
                kernel_type=kernel_type,
                dist_scale_mode=dist_scale_mode,
                rbf_sigma=rbf_sigma,
                multiscale_sigmas=list(multiscale_sigmas) if kernel_type == "multiscale_rbf" else None,
            )
            V = clip_drift(V, max_drift)
            target = (gen_flat + V).detach()

            noise_pred_student = self.nets["policy"]["noise_pred_net"](
                z, timesteps, global_cond=obs_cond
            )
            pred_x0_student = self._pred_x0_from_noise(z, timesteps, noise_pred_student)
            pred_x0_student_flat = pred_x0_student.reshape(B, -1)

            loss = F.mse_loss(pred_x0_student_flat, target)

            info["losses"] = TensorUtils.detach({"l2_loss": loss, "drift_loss": loss})
            info["_drift_norm"] = V.norm(dim=-1).mean().item()
            info["_adaptive_temp"] = adaptive_temp
            info["_max_drift"] = max_drift
            info["_training_phase"] = "drift"

            # Kernel observability
            self._train_step_counter += 1
            should_log_kernel = log_kernel_stats and (self._train_step_counter % log_every_n == 0)
            if should_log_kernel:
                info["_kernel_stats"] = kernel_stats

                # ── Drift alignment monitoring ──
                with torch.no_grad():
                    dists_to_pos = torch.cdist(gen_flat, pos_flat)
                    nearest_idx = dists_to_pos.argmin(dim=1)
                    direction_to_nearest = pos_flat[nearest_idx] - gen_flat
                    dir_norm = direction_to_nearest.norm(dim=-1)
                    valid_mask = dir_norm > 1e-8
                    if valid_mask.any():
                        cos_sim = F.cosine_similarity(
                            V[valid_mask], direction_to_nearest[valid_mask], dim=-1
                        )
                        info["_drift_alignment_mean"] = cos_sim.mean().item()
                        info["_drift_alignment_std"] = cos_sim.std().item()

            # Kernel alive check
            if assert_alive and kernel_stats["kernel_max"] < alive_threshold:
                warnings.warn(
                    f"Drift kernel appears dead: kernel_max={kernel_stats['kernel_max']:.2e} "
                    f"< threshold={alive_threshold:.2e}. Consider enabling dist_scale_mode=sqrt_dim "
                    f"or switching to kernel_type=rbf."
                )

            if not validate:
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                    max_grad_norm=1.0,
                )
                if self.ema is not None:
                    self.ema.step(self.nets)
                info["policy_grad_norms"] = policy_grad_norms

        return info

    def log_info(self, info):
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "_training_phase" in info:
            log["Training_Phase"] = 0.0 if info["_training_phase"] == "diffusion_warmup" else 1.0
        if "_drift_norm" in info:
            log["Drift_Norm"] = info["_drift_norm"]
        if "_adaptive_temp" in info:
            log["Drift_Temp"] = info["_adaptive_temp"]
        if "_max_drift" in info:
            log["Drift_Max_Drift"] = info["_max_drift"]
        if "_kernel_stats" in info:
            stats = info["_kernel_stats"]
            log["Drift_Kernel_Max"] = stats["kernel_max"]
            log["Drift_Kernel_Mean"] = stats["kernel_mean"]
            log["Drift_Dist_Median"] = stats["dist_median"]
        if "_drift_alignment_mean" in info:
            log["Drift_Alignment_Mean"] = info["_drift_alignment_mean"]
            log["Drift_Alignment_Std"] = info["_drift_alignment_std"]
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
