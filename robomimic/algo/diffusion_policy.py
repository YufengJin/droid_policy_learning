from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import os


from transformers import AutoTokenizer, AutoModel

try:
    from transformers.utils import hub as _transformers_hub
    from huggingface_hub import RemoteEntryNotFoundError

    _orig_list_repo_templates = _transformers_hub.list_repo_templates

    def _safe_list_repo_templates(*args, **kwargs):
        try:
            return _orig_list_repo_templates(*args, **kwargs)
        except RemoteEntryNotFoundError:
            return []

    _transformers_hub.list_repo_templates = _safe_list_repo_templates
except Exception:
    pass

_tokenizer = None
_lang_model = None
_lang_model_device = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased",
                local_files_only=True,
            )
    return _tokenizer


def get_lang_model(device):
    global _lang_model, _lang_model_device
    
    if _lang_model is not None and _lang_model_device == device:
        return _lang_model

    def _load_lang_model(local_only):
        kwargs = {"local_files_only": True} if local_only else {}
        try:
            return AutoModel.from_pretrained(
                "distilbert-base-uncased",
                torch_dtype=torch.float16,
                **kwargs,
            )
        except TypeError:
            return AutoModel.from_pretrained(
                "distilbert-base-uncased",
                **kwargs,
            )

    try:
        _lang_model = _load_lang_model(local_only=False)
    except Exception:
        _lang_model = _load_lang_model(local_only=True)
    _lang_model.to(device)
    _lang_model_device = device
    
    return _lang_model


# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

import cv2
import copy


@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return DiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()

class DiffusionPolicyUNet(PolicyAlgo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_alignment_schedule()

    def _init_alignment_schedule(self):
        cfg = self.algo_config
        self.alignment_freq_high = float(getattr(cfg, 'cleandift_alignment_freq', 1.0))
        freq_min_cfg = getattr(cfg, 'cleandift_alignment_freq_min', self.alignment_freq_high)
        self.alignment_freq_low = float(max(0.0, min(self.alignment_freq_high, freq_min_cfg)))
        self.alignment_freq_drop_ratio = float(getattr(cfg, 'cleandift_alignment_freq_drop_ratio', 0.01))
        self.current_alignment_freq = self.alignment_freq_high
        # Track when alignment is (re-)enabled so schedules are measured in alignment steps,
        # not absolute training epochs (important for two-stage finetune / resume).
        self._alignment_start_epoch = None

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )


        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
        )

        # Two-stage training configuration:
        # Stage 1: freeze encoder (backbone), train decoder (noise_pred_net)
        # Stage 2: freeze decoder (noise_pred_net), train encoder
        self._freeze_decoder = bool(getattr(self.algo_config, 'freeze_decoder', False))
        if self._freeze_decoder:
            quiet = os.environ.get("ROBOMIMIC_QUIET", "0") == "1"
            if not quiet:
                print(f"\n{'='*60}")
                print(f"[DiffusionPolicy] Stage 2 Mode: FREEZE decoder (noise_pred_net)")
                print(f"  -> noise_pred_net: FROZEN (no training)")
                print(f"  -> obs_encoder: trained by policy loss + alignment loss")
                print(f"  -> Goal: Train a foundation image encoder")
                print(f"{'='*60}\n")
            for param in noise_pred_net.parameters():
                param.requires_grad = False

        # the final arch has 2 parts
        # Check if we're using DDP - if so, don't use DataParallel (DDP will handle multi-GPU)
        use_ddp = getattr(self.global_config.train, 'use_ddp', False)
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        
        # Only use DataParallel in single-process multi-GPU mode (not DDP)
        if (not use_ddp) and (not is_distributed) and torch.cuda.device_count() > 1:
            # Traditional DataParallel for single-process multi-GPU
            nets = nn.ModuleDict({
                'policy': nn.ModuleDict({
                    'obs_encoder': torch.nn.parallel.DataParallel(obs_encoder, device_ids=list(range(0,torch.cuda.device_count()))),
                    'noise_pred_net': torch.nn.parallel.DataParallel(noise_pred_net, device_ids=list(range(0,torch.cuda.device_count())))
                })
            })
        else:
            # DDP mode or single GPU: use plain networks (DDP will wrap them at higher level)
            nets = nn.ModuleDict({
                'policy': nn.ModuleDict({
                    'obs_encoder': obs_encoder,
                    'noise_pred_net': noise_pred_net
                })
            })

        nets = nets.float().to(self.device)

        # AMP configuration (defaults controlled via training config)
        requested_amp = bool(self.global_config.train.get("use_amp", False))
        self.use_amp = requested_amp and torch.cuda.is_available()
        amp_dtype_cfg = str(self.global_config.train.get("amp_dtype", "bfloat16")).lower()
        self.autocast_dtype = None
        if self.use_amp:
            if amp_dtype_cfg == "float16":
                self.autocast_dtype = torch.float16
            elif amp_dtype_cfg == "bfloat16":
                bf16_supported = False
                is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
                if callable(is_bf16_supported):
                    try:
                        bf16_supported = bool(is_bf16_supported())
                    except RuntimeError:
                        bf16_supported = False
                if bf16_supported:
                    self.autocast_dtype = torch.bfloat16
                else:
                    # Fall back to fp16 if bf16 is not available
                    self.autocast_dtype = torch.float16
                    if str(getattr(self.global_config.train, "amp_dtype", "")).lower() == "bfloat16":
                        quiet = os.environ.get("ROBOMIMIC_QUIET", "0") == "1"
                        if not quiet:
                            if torch.distributed.is_available() and torch.distributed.is_initialized():
                                if torch.distributed.get_rank() == 0:
                                    print("WARNING:  BF16 autocast requested but not supported on this device. Falling back to FP16.")
                            else:
                                print("WARNING:  BF16 autocast requested but not supported on this device. Falling back to FP16.")
            else:
                # default to float16
                self.autocast_dtype = torch.float16
        self.grad_scaler = None
        if self.use_amp:
            scaler_enabled = self.autocast_dtype == torch.float16
            # Use torch.amp.GradScaler instead of deprecated torch.cuda.amp.GradScaler
            self.grad_scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema_kwargs = {"power": self.algo_config.ema.power}
            ema_parameters = list(nets.parameters())
            try:
                ema = EMAModel(parameters=ema_parameters, **ema_kwargs)
            except TypeError:
                # Backwards compatibility with older diffusers releases that expected ``model=...``
                ema = EMAModel(model=nets, **ema_kwargs)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None

    def _decode_language_entry(self, entry):
        if entry is None:
            return ""
        if isinstance(entry, str):
            return entry
        if isinstance(entry, (bytes, bytearray)):
            try:
                return entry.decode("utf-8")
            except Exception:
                return entry.decode("latin1", errors="ignore")
        if isinstance(entry, torch.Tensor):
            entry = entry.detach().cpu().tolist()
        elif hasattr(entry, "tolist") and not isinstance(entry, (list, tuple, dict)):
            entry = entry.tolist()
        if isinstance(entry, (list, tuple)):
            parts = [self._decode_language_entry(e) for e in entry]
            return " ".join([p for p in parts if p])
        return str(entry)

    def _clean_prompt(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace("\n", " ")
        cleaned = re.sub(r"!+", " ", cleaned)
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return ""
        tokens = cleaned.split()
        clip_word_limit = 70
        if len(tokens) > clip_word_limit:
            tokens = tokens[:clip_word_limit]
        cleaned = " ".join(tokens)
        if len(cleaned) > 300:
            cleaned = cleaned[:300]
        return cleaned

    def _decode_language_prompts(self, raw_entries):
        if raw_entries is None:
            return None
        prompts = []
        for entry in raw_entries:
            decoded = self._decode_language_entry(entry)
            decoded = self._clean_prompt(decoded)
            if decoded:
                prompts.append(decoded)
        return prompts if len(prompts) > 0 else None

    def _resolve_lang_prompts(self, lang_prompts, obs_source=None):
        if lang_prompts is not None:
            return list(lang_prompts)
        if isinstance(obs_source, dict):
            if "lang_prompts" in obs_source:
                value = obs_source["lang_prompts"]
                if isinstance(value, (list, tuple)):
                    return list(value)
            if "raw_language" in obs_source:
                decoded = self._decode_language_prompts(obs_source["raw_language"])
                if decoded:
                    return decoded
        return None

    def _encode_obs_sequence(self, obs_encoder, obs_sequence, lang_prompts=None):
        To = self.algo_config.horizon.observation_horizon
        lang_cond = list(lang_prompts) if lang_prompts is not None else None
        features = []
        for t in range(To):
            obs_t = TensorUtils.index_at_time(obs_sequence, t)
            if lang_cond is not None:
                feats_t = obs_encoder(obs=obs_t, lang_cond=lang_cond)
            else:
                feats_t = obs_encoder(obs=obs_t)
            features.append(feats_t)
        return torch.stack(features, dim=1)
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()

        input_batch["obs"] = {}
        for k, v in batch["obs"].items():
            if "raw" in k:
                continue
            input_batch["obs"][k] = v[:, :To, ...]

        lang_prompts = self._decode_language_prompts(batch["obs"].get("raw_language"))
        if not lang_prompts:
            lang_prompts = None

        lang_keep_mask = None
        lang_dropout_prob = float(getattr(self.algo_config, "language_dropout_prob", 0.0) or 0.0)
        if lang_prompts is not None and lang_dropout_prob > 0:
            drop_mask = torch.rand(len(lang_prompts)) < lang_dropout_prob
            if drop_mask.all():
                lang_prompts = None
            else:
                lang_prompts = [
                    "" if drop else prompt
                    for prompt, drop in zip(lang_prompts, drop_mask.tolist())
                ]
                lang_keep_mask = ~drop_mask

        with torch.no_grad():
            if lang_prompts is not None:
                nets = self._get_nets()
                device = next(nets["policy"].parameters()).device

                encoded_lang = None
                cache_allowed = lang_dropout_prob == 0.0
                if cache_allowed:
                    try:
                        unique_prompts = set(lang_prompts)
                    except Exception:
                        unique_prompts = None
                    if unique_prompts is not None and len(unique_prompts) == 1:
                        cache_key = (next(iter(unique_prompts)), len(lang_prompts), To, str(device))
                        if not hasattr(self, "_lang_cache"):
                            self._lang_cache = {}
                        cached = self._lang_cache.get(cache_key)
                        if cached is not None:
                            encoded_lang = cached

                if encoded_lang is None:
                    lang_model = get_lang_model(device)
                    tokenizer = get_tokenizer()
                    encoded_input = tokenizer(
                        lang_prompts,
                        padding=True,
                        truncation=True,
                        max_length=tokenizer.model_max_length,
                        return_tensors='pt'
                    ).to(device)
                    outputs = lang_model(**encoded_input)
                    encoded_lang = outputs.last_hidden_state.sum(1).unsqueeze(1).repeat(1, To, 1)
                    if cache_allowed and unique_prompts is not None and len(unique_prompts) == 1:
                        self._lang_cache[cache_key] = encoded_lang

                if lang_keep_mask is not None:
                    keep = lang_keep_mask.to(device=device, dtype=encoded_lang.dtype).view(-1, 1, 1)
                    encoded_lang = encoded_lang * keep
                input_batch["obs"]["lang_fixed/language_distilbert"] = encoded_lang.float()

        if lang_prompts is not None:
            input_batch["lang_prompts"] = lang_prompts

        input_batch["actions"] = batch["actions"][:, :Tp, :]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError('"actions" must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.')
            self.action_check_done = True

        ## LOGGING HOW MANY NANs there are
        # bz = input_batch["actions"].shape[0]
        # nanamt = torch.BoolTensor([False] * bz)
        # for key in input_batch["obs"]:
        #     if key == "pad_mask":
        #         continue
        #     nanamt = torch.logical_or(nanamt, torch.isnan(input_batch["obs"][key].reshape(bz, -1).mean(1)))
        # print(nanamt.float().mean())

        for key in input_batch["obs"]:
            input_batch["obs"][key] = torch.nan_to_num(input_batch["obs"][key])
        input_batch["actions"] = torch.nan_to_num(input_batch["actions"])
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch['actions'].shape[0]

        
        lang_prompts = batch.get("lang_prompts", None)

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch['actions']
            
            # encode obs
            inputs = {
                'obs': batch["obs"],
            }
            for k in self.obs_shapes:
                ## Shape assertion does not apply to list of strings for raw language
                if "raw" in k:
                    continue
                # first two dimensions should be [B, T] for inputs
                assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])
            
            nets = self._get_nets()  # Unwrap DDP if necessary
            resolved_prompts = self._resolve_lang_prompts(lang_prompts, batch)
            alignment_weight_base = float(getattr(self.algo_config, 'cleandift_alignment_weight', 0.0) or 0.0)
            alignment_weight = 0.0
            if alignment_weight_base > 0.0:
                if getattr(self, "_alignment_start_epoch", None) is None:
                    self._alignment_start_epoch = int(epoch)
                alignment_epoch = int(epoch) - int(self._alignment_start_epoch)

                # Alignment schedule: by default uses a warmdown fraction of total epochs (backwards compatible),
                # but can be switched to a fixed warmdown step budget to avoid longer runs drifting more.
                total_epochs = int(getattr(self.global_config.train, 'num_epochs', 1000) or 1000)
                warmdown_steps_cfg = getattr(self.algo_config, 'cleandift_alignment_warmdown_steps', None)
                warmdown_frac = float(getattr(self.algo_config, 'cleandift_alignment_warmdown_frac', 0.5) or 0.0)
                min_decay_factor = float(getattr(self.algo_config, 'cleandift_alignment_min_decay_factor', 0.001) or 0.0)
                decay_power = float(getattr(self.algo_config, 'cleandift_alignment_decay_power', 2.0) or 2.0)

                if warmdown_steps_cfg is not None:
                    warmdown_steps = max(1, int(warmdown_steps_cfg))
                else:
                    warmdown_steps = max(1, int(total_epochs * warmdown_frac)) if warmdown_frac > 0.0 else 1

                min_decay_factor = max(0.0, min(1.0, min_decay_factor))
                decay_power = max(1.0, decay_power)

                if alignment_epoch < warmdown_steps:
                    normalized_epoch = float(alignment_epoch) / float(warmdown_steps)
                    decay_factor = min_decay_factor + (1.0 - min_decay_factor) * ((1.0 - normalized_epoch) ** decay_power)
                else:
                    decay_factor = min_decay_factor
                alignment_weight = alignment_weight_base * decay_factor

            # adjust alignment sampling frequency based on current weight (drop when sufficiently small)
            if alignment_weight_base > 0.0:
                drop_threshold = alignment_weight_base * self.alignment_freq_drop_ratio
                if alignment_weight <= drop_threshold:
                    self.current_alignment_freq = self.alignment_freq_low
                else:
                    self.current_alignment_freq = self.alignment_freq_high
            else:
                self.current_alignment_freq = 0.0

            effective_freq = self.current_alignment_freq if alignment_weight > 0.0 else 0.0
            sample_fraction = max(0.0, min(1.0, float(effective_freq))) if effective_freq > 0.0 else 0.0
            alignment_freq = sample_fraction
            compute_alignment = (
                alignment_weight > 0.0
                and not validate
                and sample_fraction > 0.0
            )

            autocast_enabled = (
                self.use_amp
                and (self.autocast_dtype is not None)
                and not validate
                and torch.cuda.is_available()
            )

            autocast_kwargs = {'enabled': autocast_enabled}
            if autocast_enabled and self.autocast_dtype is not None:
                autocast_kwargs['dtype'] = self.autocast_dtype
            with torch.amp.autocast(device_type='cuda', **autocast_kwargs):
                obs_features = self._encode_obs_sequence(
                    nets['policy']['obs_encoder'],
                    inputs["obs"],
                    lang_prompts=resolved_prompts,
                )
                assert obs_features.ndim == 3  # [B, T, D]
                obs_cond = obs_features.flatten(start_dim=1)

                num_noise_samples = self.algo_config.noise_samples

                noise = torch.randn([num_noise_samples] + list(actions.shape), device=self.device)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (B,), device=self.device
                ).long()

                noisy_actions = torch.cat([
                    self.noise_scheduler.add_noise(actions, noise[i], timesteps)
                    for i in range(len(noise))
                ], dim=0)

                obs_cond = obs_cond.repeat(num_noise_samples, 1)
                timesteps = timesteps.repeat(num_noise_samples)

                noise_pred = nets['policy']['noise_pred_net'](
                    noisy_actions, timesteps, global_cond=obs_cond)

                noise = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])
                policy_loss = F.mse_loss(noise_pred, noise)

                alignment_loss = torch.tensor(0.0, device=self.device)

                sample_indices = None
                if compute_alignment:
                    if sample_fraction < 1.0:
                        sample_count = max(1, int(math.ceil(sample_fraction * B)))
                        sample_indices = torch.randperm(B, device=self.device)[:sample_count]
                    else:
                        sample_count = B
                        sample_indices = None

                if compute_alignment:
                    obs_encoder = nets['policy']['obs_encoder']
                    if hasattr(obs_encoder, 'module'):
                        obs_encoder = obs_encoder.module

                    if hasattr(obs_encoder, 'nets'):
                        for group_name, group_encoder in obs_encoder.nets.items():
                            if hasattr(group_encoder, 'obs_nets'):
                                for modality_name, modality_encoder in group_encoder.obs_nets.items():
                                    if 'image' in modality_name and hasattr(modality_encoder, 'backbone'):
                                        backbone = modality_encoder.backbone

                                        if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'compute_alignment_loss'):
                                            total_alignment = 0.0
                                            num_views = 0

                                            for obs_key in inputs['obs']:
                                                if 'image' in obs_key:
                                                    obs_images = inputs['obs'][obs_key]
                                                    obs_current = obs_images[:, -1]

                                                    if sample_indices is not None:
                                                        obs_current = obs_current.index_select(0, sample_indices)

                                                    if resolved_prompts is not None:
                                                        if sample_indices is not None:
                                                            idx_list = sample_indices.tolist()
                                                            captions = [resolved_prompts[i] for i in idx_list]
                                                        else:
                                                            captions = resolved_prompts
                                                    else:
                                                        captions = [""] * obs_current.shape[0]

                                                    view_alignment = backbone.encoder.compute_alignment_loss(
                                                        obs_current,
                                                        caption=captions
                                                    )
                                                    total_alignment += view_alignment
                                                    num_views += 1

                                            if num_views > 0:
                                                alignment_loss = total_alignment / num_views

                                            break
                                else:
                                    continue
                                break

            total_loss = policy_loss + alignment_weight * alignment_loss
            
            losses = {
                'l2_loss': policy_loss,
                'alignment_loss': alignment_loss,
                'total_loss': total_loss
            }
            info["losses"] = TensorUtils.detach(losses)
            info["alignment_weight"] = alignment_weight
            info["alignment_freq"] = float(self.current_alignment_freq if hasattr(self, "current_alignment_freq") else getattr(self.algo_config, 'cleandift_alignment_freq', 1.0))

            if not validate:
                optimizer = self.optimizers["policy"]
                optimizer.zero_grad(set_to_none=True)

                policy_grad_norms = 0.0
                grad_norm_clip = getattr(self.global_config.train, 'max_grad_norm', None)

                if self.use_amp and self.grad_scaler is not None and self.grad_scaler.is_enabled():
                    scaled_loss = self.grad_scaler.scale(total_loss)
                    scaled_loss.backward()
                    self.grad_scaler.unscale_(optimizer)
                    if grad_norm_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            nets['policy'].parameters(),
                            grad_norm_clip
                        )
                    for p in nets['policy'].parameters():
                        if p.grad is not None:
                            policy_grad_norms += p.grad.data.norm(2).pow(2).item()
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()
                else:
                    total_loss.backward()
                    if grad_norm_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            nets['policy'].parameters(),
                            grad_norm_clip
                        )
                    for p in nets['policy'].parameters():
                        if p.grad is not None:
                            policy_grad_norms += p.grad.data.norm(2).pow(2).item()
                    optimizer.step()

                if self.ema is not None:
                    self.ema.step(self.nets)

                info['policy_grad_norms'] = policy_grad_norms

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        
        log["Loss"] = info["losses"]["l2_loss"].item()
        
        log["Total_Loss"] = info["losses"]["total_loss"].item()
        
        if "alignment_loss" in info["losses"]:
            align_loss_val = info["losses"]["alignment_loss"]
            if isinstance(align_loss_val, torch.Tensor):
                align_loss_val = align_loss_val.item()
            log["Alignment_Loss"] = align_loss_val
            
            if log["Total_Loss"] > 0:
                log["Alignment_Loss_Ratio"] = abs(align_loss_val) / log["Total_Loss"]
            
            if "alignment_weight" in info:
                log["Alignment_Weight"] = info["alignment_weight"]
        
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]

        if "alignment_freq" in info:
            log["Alignment_Freq"] = info["alignment_freq"]
        
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
        
    def get_action(self, obs_dict, goal_mode=None, eval_mode=False):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """

        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon

        lang_prompts = None

        if eval_mode:
            from droid.misc.parameters import hand_camera_id, varied_camera_1_id, varied_camera_2_id
            root_path = os.path.join(os. getcwd(), "eval_params")

            if goal_mode is not None:
                # Read in goal images
                goal_hand_camera_left_image = torch.FloatTensor((cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{hand_camera_id}_left.png")), cv2.COLOR_BGR2RGB) / 255.0)).cuda().permute(2, 0, 1).unsqueeze(0).repeat([1, 1, 1, 1]).unsqueeze(0)
                goal_hand_camera_right_image = torch.FloatTensor((cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{hand_camera_id}_right.png")), cv2.COLOR_BGR2RGB) / 255.0)).cuda().permute(2, 0, 1).unsqueeze(0).repeat([1, 1, 1, 1]).unsqueeze(0)
                goal_varied_camera_1_left_image = torch.FloatTensor((cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{varied_camera_1_id}_left.png")), cv2.COLOR_BGR2RGB) / 255.0)).cuda().permute(2, 0, 1).unsqueeze(0).repeat([1, 1, 1, 1]).unsqueeze(0)
                goal_varied_camera_1_right_image = torch.FloatTensor((cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{varied_camera_1_id}_right.png")), cv2.COLOR_BGR2RGB) / 255.0)).cuda().permute(2, 0, 1).unsqueeze(0).repeat([1, 1, 1, 1]).unsqueeze(0)
                goal_varied_camera_2_left_image = torch.FloatTensor((cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{varied_camera_2_id}_left.png")), cv2.COLOR_BGR2RGB) / 255.0)).cuda().permute(2, 0, 1).unsqueeze(0).repeat([1, 1, 1, 1]).unsqueeze(0)
                goal_varied_camera_2_right_image = torch.FloatTensor((cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{varied_camera_2_id}_right.png")), cv2.COLOR_BGR2RGB) / 255.0)).cuda().permute(2, 0, 1).unsqueeze(0).repeat([1, 1, 1, 1]).unsqueeze(0)

                obs_dict['camera/image/hand_camera_left_image'] = torch.cat([obs_dict['camera/image/hand_camera_left_image'], goal_hand_camera_left_image.repeat(1, To, 1, 1, 1)], dim=2) 
                obs_dict['camera/image/hand_camera_right_image'] = torch.cat([obs_dict['camera/image/hand_camera_right_image'], goal_hand_camera_right_image.repeat(1, To, 1, 1, 1)], dim=2) 
                obs_dict['camera/image/varied_camera_1_left_image'] = torch.cat([obs_dict['camera/image/varied_camera_1_left_image'], goal_varied_camera_1_left_image.repeat(1, To, 1, 1, 1)], dim=2) 
                obs_dict['camera/image/varied_camera_1_right_image'] = torch.cat([obs_dict['camera/image/varied_camera_1_right_image'] , goal_varied_camera_1_right_image.repeat(1, To, 1, 1, 1)], dim=2) 
                obs_dict['camera/image/varied_camera_2_left_image'] = torch.cat([obs_dict['camera/image/varied_camera_2_left_image'] , goal_varied_camera_2_left_image.repeat(1, To, 1, 1, 1)], dim=2) 
                obs_dict['camera/image/varied_camera_2_right_image'] = torch.cat([obs_dict['camera/image/varied_camera_2_right_image'], goal_varied_camera_2_right_image.repeat(1, To, 1, 1, 1)], dim=2) 
            # Note: currently assumes that you are never doing both goal and language conditioning
            else:
                # Reads in current language instruction from file and fills the appropriate obs key, only will
                # actually use it if the policy uses language instructions
                with open(os.path.join(root_path, "lang_command.txt"), 'r') as file:
                    raw_lang = file.read()

                nets = self._get_nets()  # Unwrap DDP if necessary
                device = next(nets["policy"].parameters()).device
                lang_model = get_lang_model(device)
                
                tokenizer = get_tokenizer()
                encoded_input = tokenizer(raw_lang, return_tensors='pt').to(device)
                outputs = lang_model(**encoded_input)
                encoded_lang = outputs.last_hidden_state.sum(1).squeeze().unsqueeze(0).repeat(To, 1).unsqueeze(0)
                obs_dict["lang_fixed/language_distilbert"] = encoded_lang.type(torch.float32)
                lang_prompts = [raw_lang]

        ###############################

        lang_prompts = self._resolve_lang_prompts(lang_prompts, obs_dict)

        # TODO: obs_queue already handled by frame_stack
        # make sure we have at least To observations in obs_queue
        # if not enough, repeat
        # if already full, append one to the obs_queue
        # n_repeats = max(To - len(self.obs_queue), 1)
        # self.obs_queue.extend([obs_dict] * n_repeats)
        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # turn obs_queue into dict of tensors (concat at T dim)
            # import pdb; pdb.set_trace()
            # obs_dict_list = TensorUtils.list_of_flat_dict_to_dict_of_list(list(self.obs_queue))
            # obs_dict_tensor = dict((k, torch.cat(v, dim=0).unsqueeze(0)) for k,v in obs_dict_list.items())
            
            # run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict, lang_prompts=lang_prompts)
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        # [1,Da]
        action = action.unsqueeze(0)
        return action
        
    def _get_action_trajectory(self, obs_dict, lang_prompts=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self._get_nets()
        ema_restore_mode = None
        if self.ema is not None:
            if hasattr(self.ema, "store") and hasattr(self.ema, "copy_to"):
                # Newer diffusers EMA API: temporarily swap in EMA weights.
                self.ema.store(self.nets.parameters())
                self.ema.copy_to(self.nets.parameters())
                ema_restore_mode = "params"
                nets = self._get_nets()
            elif hasattr(self.ema, "averaged_model"):
                # Older diffusers EMA API: use the averaged model directly.
                nets = self.ema.averaged_model
                ema_restore_mode = "avg_model"
        
        # encode obs
        inputs = {
            'obs': obs_dict,
        }
        for k in self.obs_shapes:
            ## Shape assertion does not apply to list of strings for raw language
            if "raw" in k:
                continue
            # first two dimensions should be [B, T] for inputs
            assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])
        resolved_prompts = self._resolve_lang_prompts(lang_prompts, obs_dict)
        obs_encoder_eval = nets['policy']['obs_encoder']
        if hasattr(obs_encoder_eval, 'module'):
            obs_encoder_eval = obs_encoder_eval.module
        obs_features = self._encode_obs_sequence(
            obs_encoder_eval,
            inputs["obs"],
            lang_prompts=resolved_prompts,
        )
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)


        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred_module = nets['policy']['noise_pred_net']
            if hasattr(noise_pred_module, "module"):
                noise_pred_module = noise_pred_module.module
            noise_pred = noise_pred_module(
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        
        # Restore original parameters if EMA was used
        if self.ema is not None and ema_restore_mode == "params":
            if hasattr(self.ema, "restore"):
                self.ema.restore(self.nets.parameters())
        
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        def _ema_state(ema_obj):
            if ema_obj is None:
                return None
            if hasattr(ema_obj, "state_dict"):
                try:
                    return {"_type": "state_dict", "data": ema_obj.state_dict()}
                except Exception:
                    pass
            data = {}
            if hasattr(ema_obj, "shadow_params"):
                data["shadow_params"] = [p.detach().cpu() for p in ema_obj.shadow_params]
            for attr in ("decay", "power", "inv_gamma", "min_decay", "optimization_step"):
                if hasattr(ema_obj, attr):
                    data[attr] = getattr(ema_obj, attr)
            return {"_type": "raw", "data": data} if data else None

        return {
            "nets": self.nets.state_dict(),
            "ema": _ema_state(self.ema),
            "alignment_start_epoch": getattr(self, "_alignment_start_epoch", None),
        }

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        # Handle DDP wrapper mismatch: checkpoint may have 'module.' prefix but current model may not
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        nets_state = model_dict["nets"]
        current_is_ddp = isinstance(self.nets, DDP)
        
        # Check if saved state has 'module.' prefix
        saved_has_prefix = any(k.startswith('module.') for k in nets_state.keys())
        
        # If there's a mismatch, we need to adjust the keys
        if saved_has_prefix and not current_is_ddp:
            # Saved with DDP, loading to non-DDP: remove 'module.' prefix
            new_state = {}
            for k, v in nets_state.items():
                new_key = k[7:] if k.startswith('module.') else k  # remove 'module.'
                new_state[new_key] = v
            nets_state = new_state
        elif not saved_has_prefix and current_is_ddp:
            # Saved without DDP, loading to DDP: add 'module.' prefix
            new_state = {}
            for k, v in nets_state.items():
                new_key = f'module.{k}' if not k.startswith('module.') else k
                new_state[new_key] = v
            nets_state = new_state
        
        # If input dimensions changed (e.g., adding low_dim), pad matching linear weights.
        current_state = self.nets.state_dict()
        quiet = os.environ.get("ROBOMIMIC_QUIET", "0") == "1"
        for k, v in list(nets_state.items()):
            if k not in current_state:
                continue
            cur = current_state[k]
            if v.shape == cur.shape:
                continue
            # Handle Linear weights where in_features changed (pad or truncate).
            if v.ndim == 2 and cur.ndim == 2 and v.shape[0] == cur.shape[0]:
                if v.shape[1] < cur.shape[1]:
                    padded = torch.zeros_like(cur)
                    padded[:, :v.shape[1]] = v
                    nets_state[k] = padded
                    if not quiet:
                        print(f"WARNING: padded weight {k} from {tuple(v.shape)} -> {tuple(cur.shape)}")
                elif v.shape[1] > cur.shape[1]:
                    nets_state[k] = v[:, :cur.shape[1]]
                    if not quiet:
                        print(f"WARNING: truncated weight {k} from {tuple(v.shape)} -> {tuple(cur.shape)}")
        self.nets.load_state_dict(nets_state)
        ema_state = model_dict.get("ema", None)
        if ema_state is not None and self.ema is not None:
            loaded = False
            if isinstance(ema_state, dict) and ema_state.get("_type") == "state_dict":
                if hasattr(self.ema, "load_state_dict"):
                    self.ema.load_state_dict(ema_state["data"])
                    loaded = True
            else:
                if hasattr(self.ema, "load_state_dict"):
                    try:
                        self.ema.load_state_dict(ema_state)
                        loaded = True
                    except Exception:
                        loaded = False
            if not loaded:
                data = ema_state.get("data", ema_state) if isinstance(ema_state, dict) else None
                if isinstance(data, dict):
                    if "shadow_params" in data and hasattr(self.ema, "shadow_params"):
                        model_device = next(self.nets.parameters()).device
                        self.ema.shadow_params = [p.to(model_device) for p in data["shadow_params"]]
                    for attr in ("decay", "power", "inv_gamma", "min_decay", "optimization_step"):
                        if attr in data and hasattr(self.ema, attr):
                            setattr(self.ema, attr, data[attr])
            # CRITICAL: Move EMA shadow parameters to the same device as model parameters
            if hasattr(self.ema, 'shadow_params') and len(self.ema.shadow_params) > 0:
                model_device = next(self.nets.parameters()).device
                self.ema.shadow_params = [p.to(model_device) for p in self.ema.shadow_params]

        self._alignment_start_epoch = model_dict.get("alignment_start_epoch", None)

    
            
            

# =================== Vision Encoder Utils =====================
def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version('1.9.0'):
        raise ImportError('This function requires pytorch >= 1.9.0')

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module

# =================== UNet for Diffusion ==============

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM 
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level. 
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
