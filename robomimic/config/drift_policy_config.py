"""
Config for Drift Policy algorithm.
Same UNet as Diffusion Policy; training uses mean-shift drift loss; inference 1 NFE.
"""

from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class DriftPolicyConfig(DiffusionPolicyConfig):
    ALGO_NAME = "drift_policy"

    def algo_config(self):
        super().algo_config()

        # Drift-specific: noise_samples=1 to avoid O(B²) explosion
        self.algo.noise_samples = 1

        # 1-NFE inference
        self.algo.ddpm.num_inference_timesteps = 1
        self.algo.ddim.num_inference_timesteps = 1

        # Drift loss parameters (from drift_model/run_experiment.py)
        self.algo.drift.temp = 0.05
        self.algo.drift.max_drift = 0.1
        self.algo.drift.use_adaptive_temp = True
