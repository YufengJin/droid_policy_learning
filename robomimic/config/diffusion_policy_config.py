"""
Config for Diffusion Policy algorithm.
"""

from robomimic.config.base_config import BaseConfig

class DiffusionPolicyConfig(BaseConfig):
    ALGO_NAME = "diffusion_policy"
    
    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        
        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adamw"          # optimizer type: "adam" or "adamw" (recommended)
        self.algo.optim_params.policy.betas = (0.9, 0.999)              # Adam/AdamW betas (momentum coefficients)
        self.algo.optim_params.policy.eps = 1e-8                        # Adam/AdamW epsilon for numerical stability
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        
        # Learning rate scheduler type: "multistep", "linear", or "cosine_warmup" (recommended for stable training)
        self.algo.optim_params.policy.learning_rate.scheduler_type = "cosine_warmup"
        self.algo.optim_params.policy.learning_rate.warmup_epochs = 10  # number of warmup epochs (0 = no warmup)
        self.algo.optim_params.policy.learning_rate.warmup_type = "linear"  # "linear" or "constant"
        self.algo.optim_params.policy.learning_rate.total_epochs = 1000 # total training epochs (for cosine scheduler)
        self.algo.optim_params.policy.learning_rate.min_lr_ratio = 0.01 # minimum lr = initial_lr * min_lr_ratio
        
        self.algo.optim_params.policy.regularization.L2 = 0.01          # L2 regularization strength (weight decay for AdamW)

        # horizon parameters
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.action_horizon = 8
        self.algo.horizon.prediction_horizon = 16
        
        # UNet parameters
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256,512,1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8
        
        # EMA parameters
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75
        
        # Noise Scheduler
        ## DDPM
        self.algo.ddpm.enabled = True
        self.algo.ddpm.num_train_timesteps = 100
        self.algo.ddpm.num_inference_timesteps = 100
        self.algo.ddpm.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddpm.clip_sample = True
        self.algo.ddpm.prediction_type = 'epsilon'
        self.algo.noise_samples = 1

        ## DDIM
        self.algo.ddim.enabled = False
        self.algo.ddim.num_train_timesteps = 50 #100
        self.algo.ddim.num_inference_timesteps = 10
        self.algo.ddim.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddim.clip_sample = True
        self.algo.ddim.set_alpha_to_one = True
        self.algo.ddim.steps_offset = 0
        self.algo.ddim.prediction_type = 'epsilon'
        
        # CleanDIFT Alignment Loss parameters
        self.algo.cleandift_alignment_weight = 0.0  # Alignment loss weight (0.0 = disabled)
        self.algo.cleandift_alignment_freq = 1.0    # Frequency of computing alignment loss (1.0=every batch, 0.2=20% of batches)
        self.algo.cleandift_alignment_freq_min = 0.2             # Lower bound on sampling frequency after stabilization
        self.algo.cleandift_alignment_freq_drop_ratio = 0.01     # Switch to low frequency when weight <= ratio * initial
        # Alignment schedule (to mitigate representation drift during long runs)
        # If warmdown_steps is set, it overrides warmdown_frac and makes the schedule independent of total training epochs.
        self.algo.cleandift_alignment_warmdown_frac = 0.5        # Portion of training used for warmdown (quadratic decay)
        self.algo.cleandift_alignment_warmdown_steps = None      # Fixed warmdown steps (recommended for long runs)
        self.algo.cleandift_alignment_min_decay_factor = 0.001   # Final weight multiplier after warmdown
        self.algo.cleandift_alignment_decay_power = 2.0          # Decay exponent (2.0 = quadratic)

        # Language dropout for robustness (per-sample)
        self.algo.language_dropout_prob = 0.0

        # Two-stage training configuration
        # Stage 1: freeze encoder, train decoder (noise_pred_net) with policy loss
        # Stage 2: freeze decoder, train encoder with policy loss + alignment loss
        self.algo.freeze_decoder = False      # Stage 2: freeze noise_pred_net completely
        self.algo.rgb_only = False            # Stage 2: disable low_dim inputs (pure RGB mode)
