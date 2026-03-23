"""
Drift Policy: same UNet architecture as Diffusion Policy, but training uses
mean-shift drift loss (from drift_model/run_experiment.py) instead of noise prediction.
Inference: 1 NFE via single DDIM step from pure noise.
"""

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

    def train_on_batch(self, batch, epoch, validate=False):
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        B = batch["actions"].shape[0]

        drift_cfg = getattr(self.algo_config, "drift", None)
        temp = getattr(drift_cfg, "temp", 0.05) if drift_cfg else 0.05
        max_drift = getattr(drift_cfg, "max_drift", 0.1) if drift_cfg else 0.1
        use_adaptive_temp = getattr(drift_cfg, "use_adaptive_temp", True) if drift_cfg else True

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

            adaptive_temp = compute_adaptive_temp(gen_flat, pos_flat, base_temp=temp) if use_adaptive_temp else temp
            V = compute_drift(gen_flat, pos_flat, temp=adaptive_temp)
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
        if "_drift_norm" in info:
            log["Drift_Norm"] = info["_drift_norm"]
        if "_adaptive_temp" in info:
            log["Drift_Temp"] = info["_adaptive_temp"]
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
