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

        # Drift loss parameters
        self.algo.drift.temp = 0.05
        self.algo.drift.max_drift = 0.1
        self.algo.drift.use_adaptive_temp = True

        # Kernel type and distance scaling
        self.algo.drift.kernel_type = "laplace"         # "laplace", "rbf", or "multiscale_rbf"
        self.algo.drift.dist_scale_mode = "none"         # "none" or "sqrt_dim"
        self.algo.drift.rbf_sigma = 1.0                  # σ for RBF kernel
        self.algo.drift.multiscale_sigmas = [0.25, 0.5, 1.0, 2.0]  # σ list for multiscale_rbf

        # Adaptive temperature parameters
        self.algo.drift.adaptive_median_scale = 0.3
        self.algo.drift.adaptive_min_temp = 0.01

        # Kernel observability
        self.algo.drift.log_kernel_stats = True
        self.algo.drift.log_kernel_every_n_steps = 100

        # Kernel alive assertion
        self.algo.drift.assert_alive_kernel = False
        self.algo.drift.alive_kernel_threshold = 1e-6

        # Two-phase training: warmup with diffusion loss before switching to drift
        self.algo.drift.warmup_epochs = 0               # 0 = disabled (pure drift from epoch 1)

        # Curriculum max-drift: anneal max_drift over training
        self.algo.drift.max_drift_start = 0.1            # initial max_drift (only used if anneal_epochs > 0)
        self.algo.drift.max_drift_end = 0.1              # final max_drift
        self.algo.drift.max_drift_anneal_epochs = 0      # 0 = disabled (use fixed max_drift)

        # Scheduled EMA decay: anneal EMA decay over training
        self.algo.drift.ema_decay_start = None           # None = disabled
        self.algo.drift.ema_decay_end = None
        self.algo.drift.ema_decay_anneal_epochs = 0
