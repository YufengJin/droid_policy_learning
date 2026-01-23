import os
import warnings

try:
    from huggingface_hub import LocalEntryNotFoundError, snapshot_download
except ImportError:
    from huggingface_hub import snapshot_download

    class LocalEntryNotFoundError(FileNotFoundError):
        pass

import torch
from torch import nn
import torch.nn.functional as F
import einops
from diffusers import DiffusionPipeline
from jaxtyping import Float, Int
from pydoc import locate
from typing import Literal, Sequence
from .layers import FeedForwardBlock, FourierFeatures, Linear, MappingNetwork
from .min_sd15 import SD15UNetModel
from .min_sd21 import SD21UNetModel


class SD15UNetFeatureExtractor(SD15UNetModel):
    def __init__(self):
        super().__init__()

    def forward(self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, **kwargs):
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        # 3. down
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s4, s5, s6] = self.down_blocks[1](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s7, s8, s9] = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s10, s11] = self.down_blocks[3](
            sample,
            temb=emb,
        )

        # 4. mid
        sample_mid = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        _, [us1, us2, us3] = self.up_blocks[0](
            hidden_states=sample_mid,
            temb=emb,
            res_hidden_states_tuple=[s9, s10, s11],
        )

        _, [us4, us5, us6] = self.up_blocks[1](
            hidden_states=us3,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
        )

        _, [us7, us8, us9] = self.up_blocks[2](
            hidden_states=us6,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
        )

        _, [us10, us11, _] = self.up_blocks[3](
            hidden_states=us9,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
            encoder_hidden_states=encoder_hidden_states,
        )

        return {
            "mid": sample_mid,
            "us1": us1,
            "us2": us2,
            "us3": us3,
            "us4": us4,
            "us5": us5,
            "us6": us6,
            "us7": us7,
            "us8": us8,
            "us9": us9,
            "us10": us10,
        }


class SD21UNetFeatureExtractor(SD21UNetModel):
    def __init__(self):
        super().__init__()

    def forward(self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, **kwargs):
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        # 3. down
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s4, s5, s6] = self.down_blocks[1](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s7, s8, s9] = self.down_blocks[2](
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s10, s11] = self.down_blocks[3](
            sample,
            temb=emb,
        )

        # 4. mid
        sample_mid = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        _, [us1, us2, us3] = self.up_blocks[0](
            hidden_states=sample_mid,
            temb=emb,
            res_hidden_states_tuple=[s9, s10, s11],
        )

        _, [us4, us5, us6] = self.up_blocks[1](
            hidden_states=us3,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
        )

        _, [us7, us8, us9] = self.up_blocks[2](
            hidden_states=us6,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
        )

        _, [us10, us11, _] = self.up_blocks[3](
            hidden_states=us9,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
            encoder_hidden_states=encoder_hidden_states,
        )

        return {
            "mid": sample_mid,
            "us1": us1,
            "us2": us2,
            "us3": us3,
            "us4": us4,
            "us5": us5,
            "us6": us6,
            "us7": us7,
            "us8": us8,
            "us9": us9,
            "us10": us10,
        }

class FeedForwardBlockCustom(FeedForwardBlock):
    def __init__(self, d_model: int, d_ff: int, d_cond_norm: int = None, norm_type: Literal['AdaRMS', 'FiLM'] = 'AdaRMS', use_gating: bool = True):
        super().__init__(d_model=d_model, d_ff=d_ff, d_cond_norm=d_cond_norm)
        if not use_gating:
            self.up_proj = LinearSwish(d_model, d_ff, bias=False)
        if norm_type == 'FiLM':
            self.norm = FiLMNorm(d_model, d_cond_norm)

class FFNStack(nn.Module):
    def __init__(self, dim: int, depth: int, ffn_expansion: float, dim_cond: int, 
                 norm_type: Literal['AdaRMS', 'FiLM'] = 'AdaRMS', use_gating: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FeedForwardBlockCustom(d_model=dim, d_ff=int(dim * ffn_expansion), d_cond_norm=dim_cond, norm_type=norm_type, use_gating=use_gating) 
             for _ in range(depth)])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cond_norm=cond)
        return x

class FiLMNorm(nn.Module):
    def __init__(self, features, cond_features):
        super().__init__()
        self.linear = Linear(cond_features, features * 2, bias=False)
        self.feature_dim = features

    def forward(self, x, cond):
        B, _, D = x.shape
        scale, shift = self.linear(cond).chunk(2, dim=-1)
        # broadcast scale and shift across all features
        scale = scale.view(B, 1, D)
        shift = scale.view(B, 1, D) 
        return scale * x + shift

class LinearSwish(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        return F.silu(super().forward(x))
    

class ArgSequential(nn.Module):  # Utility class to enable instantiating nn.Sequential instances with Hydra
    def __init__(self, *layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x

class StableFeatureAligner(nn.Module):
    def __init__(
        self,
        ae: nn.Module,
        mapping,
        adapter_layer_class: str,
        feature_dims: dict[str, int],
        feature_extractor_cls: str,
        sd_version: Literal["sd15", "sd21"],
        adapter_layer_params: dict = {},
        use_text_condition: bool = False,
        t_min: int = 1,
        t_max: int = 999,
        t_max_model: int = 999,
        num_t_stratification_bins: int = 3,
        alignment_loss: Literal["cossim", "mse", "l1"] = "cossim",
        train_unet: bool = True,
        train_adapter: bool = True,
        t_init: int = 261,
        learn_timestep: bool = False,
        val_dataset: torch.utils.data.Dataset | None = None,
        val_t: int = 261,
        val_feature_key: str = "us6",
        val_chunk_size: int = 10,
        use_adapters: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        self.ae = ae
        self.sd_version = sd_version
        self.val_t = val_t
        self.val_feature_key = val_feature_key
        self.val_dataset = val_dataset
        self.val_chunk_size = val_chunk_size
        self.use_adapters = use_adapters
        self.device = device

        if device.startswith("cuda") and torch.cuda.is_available():
            if ":" in device:
                cuda_device = int(device.split(":")[1])
                try:
                    import torch.distributed as dist
                    if not dist.is_initialized():
                        torch.cuda.set_device(cuda_device)
                except ImportError:
                    torch.cuda.set_device(cuda_device)
            target_device = device
        else:
            target_device = device

        if sd_version == "sd15":
            self.repo = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        elif sd_version == "sd21":
            self.repo = "stabilityai/stable-diffusion-2-1"
        else:
            raise ValueError(f"Invalid SD version: {sd_version}")

        self._compile_enabled = os.environ.get("TORCHDYNAMO_DISABLE", "").lower() not in {"1", "true"}

        def _configure_static_compile():
            try:
                import torch._dynamo as dynamo
                if hasattr(dynamo.config, "dynamic_shapes"):
                    dynamo.config.dynamic_shapes = False
                if hasattr(dynamo.config, "assume_static_by_default"):
                    dynamo.config.assume_static_by_default = True
                if hasattr(dynamo.config, "specialize_int"):
                    dynamo.config.specialize_int = True
            except Exception:
                pass

        def _compile_module(module):
            try:
                module.compile(dynamic=False)
            except TypeError:
                module.compile()

        self._configure_static_compile = _configure_static_compile
        self._compile_module = _compile_module

        self.mapping = None
        if use_adapters:
            self.time_emb = FourierFeatures(1, mapping.width)
            self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
            self.mapping = MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout)
            if self._compile_enabled:
                try:
                    self._configure_static_compile()
                    self._compile_module(self.mapping)
                except Exception:
                    warnings.warn("Failed to compile MappingNetwork, falling back to eager.")

        if use_adapters:
            self.adapters = nn.ModuleDict()
            for k, dim in feature_dims.items():
                self.adapters[k] = locate(adapter_layer_class)(dim=dim, **adapter_layer_params)
                self.adapters[k].requires_grad_(train_adapter)

        self.unet_feature_extractor_base = locate(feature_extractor_cls)().to(target_device)
        repo_source = None
        try:
            repo_source = snapshot_download(self.repo, local_files_only=True)
        except LocalEntryNotFoundError:
            repo_source = None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            if repo_source is not None:
                self.pipe = DiffusionPipeline.from_pretrained(
                    repo_source,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    local_files_only=True,
                ).to(target_device)
            else:
                try:
                    self.pipe = DiffusionPipeline.from_pretrained(
                        self.repo,
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                    ).to(target_device)
                except Exception:
                    self.pipe = DiffusionPipeline.from_pretrained(
                        self.repo,
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                        local_files_only=True,
                    ).to(target_device)
        self.unet_feature_extractor_base.load_state_dict(self.pipe.unet.state_dict())
        self.unet_feature_extractor_base.eval()
        self.unet_feature_extractor_base.requires_grad_(False)
        if self._compile_enabled:
            try:
                self._configure_static_compile()
                self._compile_module(self.unet_feature_extractor_base)
            except Exception:
                warnings.warn("Failed to compile base UNet, falling back to eager.")

        self.unet_feature_extractor_cleandift = locate(feature_extractor_cls)().to(target_device)
        self.unet_feature_extractor_cleandift.load_state_dict(
            {k: v.detach().clone() for k, v in self.unet_feature_extractor_base.state_dict().items()}
        )

        if train_unet or learn_timestep:
            self.unet_feature_extractor_cleandift.train()
        else:
            self.unet_feature_extractor_cleandift.eval()
        self.unet_feature_extractor_cleandift.requires_grad_(train_unet)
        if self._compile_enabled:
            try:
                self.unet_feature_extractor_cleandift.compile()
            except Exception:
                warnings.warn("Failed to compile finetune UNet, falling back to eager.")

        self.use_text_condition = use_text_condition
        if self.use_text_condition:
            if self._compile_enabled:
                try:
                    self.pipe.text_encoder.compile()
                except Exception:
                    warnings.warn("Failed to compile text encoder, falling back to eager.")
        else:
            with torch.no_grad():
                prompt_embeds_dict = self.get_prompt_embeds([""])
                self._empty_prompt_embeds = prompt_embeds_dict["prompt_embeds"]
                del self.pipe.text_encoder

        del self.pipe.unet, self.pipe.vae

        self.t_min = t_min
        self.t_max = t_max
        self.t_max_model = t_max_model
        self.num_t_stratification_bins = num_t_stratification_bins
        self.alignment_loss = alignment_loss
        self.timestep = nn.Parameter(
            torch.tensor(float(t_init), requires_grad=learn_timestep), requires_grad=learn_timestep
        )

    def get_prompt_embeds(self, prompt: list[str]) -> dict[str, torch.Tensor | None]:
        self.prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            device=torch.device(self.device),
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return {"prompt_embeds": self.prompt_embeds}

    def _get_unet_conds(self, prompts: list[str], device, dtype, N_T) -> dict[str, torch.Tensor]:
        B = len(prompts)
        if self.use_text_condition:
            prompt_embeds_dict = self.get_prompt_embeds(prompts)
        else:
            prompt_embeds_dict = {"prompt_embeds": einops.repeat(self._empty_prompt_embeds, "b ... -> (B b) ...", B=B)}

        unet_conds = {
            "encoder_hidden_states": einops.repeat(
                prompt_embeds_dict["prompt_embeds"], "B ... -> (B N_T) ...", N_T=N_T
            ).to(dtype=dtype, device=device),
            "added_cond_kwargs": {},
        }

        return unet_conds

    def forward(
        self, x: Float[torch.Tensor, "b c h w"], caption: list[str], **kwargs
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        B, *_ = x.shape
        device = x.device
        t_range = self.t_max - self.t_min
        t_range_per_bin = t_range / self.num_t_stratification_bins
        t: Int[torch.Tensor, "B N_T"] = (
            self.t_min
            + torch.rand((B, self.num_t_stratification_bins), device=device) * t_range_per_bin
            + torch.arange(0, self.num_t_stratification_bins, device=device)[None, :] * t_range_per_bin
        ).long()
        B, N_T = t.shape

        with torch.no_grad():
            unet_conds = self._get_unet_conds(caption, device, x.dtype, N_T)
            x_0: Float[torch.Tensor, "(B N_T) ..."] = self.ae.encode(x)
            x_0 = einops.repeat(x_0, "B ... -> (B N_T) ...", N_T=N_T)
            _, *latent_shape = x_0.shape
            noise_sample = torch.randn((B * N_T, *latent_shape), device=device, dtype=x.dtype)
            
            x_t: Float[torch.Tensor, "(B N_T) ..."] = self.pipe.scheduler.add_noise(
                x_0,
                noise_sample,
                einops.rearrange(t, "B N_T -> (B N_T)"),
            )

            feats_base: dict[str, Float[torch.Tensor, "B N_T ..."]] = {
                k: einops.rearrange(v, "(B N_T) D H W -> B N_T (H W) D", B=B, N_T=N_T)
                for k, v in self.unet_feature_extractor_base(
                    x_t,
                    einops.rearrange(t, "B N_T -> (B N_T)"),
                    **unet_conds,
                ).items()
            }

        feats_cleandift: dict[str, Float[torch.Tensor, "B N_T ..."]] = {
            k: einops.rearrange(v, "(B N_T) D H W -> B N_T (H W) D", N_T=N_T)
            for k, v in self.unet_feature_extractor_cleandift(
                x_0,
                einops.rearrange(torch.ones_like(t) * self.timestep, "B N_T -> (B N_T)"),
                **unet_conds,
            ).items()
        }

        if self.use_adapters:
            # time conditioning for adapters
            if not self.mapping is None:
                map_cond: Float[torch.Tensor, "(B N_T) ..."] = self.mapping(
                    self.time_in_proj(
                        self.time_emb(
                            einops.rearrange(t, "B N_T -> (B N_T) 1").to(dtype=x.dtype, device=device) / self.t_max_model
                        )
                    )
                )
   
            feats_cleandift: dict[str, Float[torch.Tensor, "B N_T ..."]] = {
                k: einops.rearrange(
                    self.adapters[k](einops.rearrange(v, "B N_T ... -> (B N_T) ..."), cond=map_cond),
                    "(B N_T) ... -> B N_T ...",
                    B=B,
                    N_T=N_T,
                )
                for k, v in feats_cleandift.items()
            }

        if self.alignment_loss == "mse":
            return {f"mse_{k}": F.mse_loss(feats_cleandift[k], v.detach()) for k, v in feats_base.items()}
        elif self.alignment_loss == "l1":
            return {f"l1_{k}": F.l1_loss(feats_cleandift[k], v.detach()) for k, v in feats_base.items()}
        elif self.alignment_loss == "cossim":
            return {
                f"neg_cossim_{k}": -F.cosine_similarity(feats_cleandift[k], v.detach(), dim=-1).mean()
                for k, v in feats_base.items()
            }
        else:
            raise ValueError(f"Invalid alignment loss type: {self.alignment_loss}")

    def _normalize_feat_key(self, feat_key):
        if feat_key is None:
            return None, None, True
        if isinstance(feat_key, str):
            return [feat_key], feat_key, False
        if isinstance(feat_key, Sequence):
            keys = [str(k) for k in feat_key]
            if len(keys) == 0:
                raise ValueError("feat_key sequence must be non-empty.")
            if len(keys) == 1:
                key = keys[0]
                return keys, key, False
            return keys, None, True
        raise TypeError(f"Unsupported feat_key type: {type(feat_key)}")

    def _prepare_timestep_vector(self, t, batch_size: int, device: torch.device) -> torch.Tensor:
        if t is None:
            base = self.timestep.detach().to(device=device)
            return base.repeat(batch_size)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device, dtype=self.timestep.dtype)
        if t.ndim == 0:
            t = t.repeat(batch_size)
        elif t.shape[0] != batch_size:
            raise ValueError(
                f"Expected timestep tensor with first dimension {batch_size}, got {tuple(t.shape)}"
            )
        return t.to(device=device, dtype=self.timestep.dtype)

    @staticmethod
    def _finalize_feature_output(feature_dict: dict[str, torch.Tensor], single_key: str | None, return_dict: bool):
        if not return_dict:
            key = single_key if single_key is not None else next(iter(feature_dict))
            return feature_dict[key]
        return feature_dict

    @torch.no_grad()
    def get_features(
        self,
        x: Float[torch.Tensor, "b c h w"],
        caption: list[str] | None,
        t: Int[torch.Tensor, "b"] | None,
        feat_key,
        use_base_model: bool = False,
        input_pure_noise: bool = False,
        eps: torch.Tensor = None,
    ):
        keys, single_key, return_dict = self._normalize_feat_key(feat_key)

        if use_base_model:
            if t is None:
                raise ValueError("Base model feature extraction requires timestep tensor `t`.")
            B, *_ = x.shape

            if caption is None:
                caption = [""] * B

            unet_conds = self._get_unet_conds(caption, x.device, x.dtype, 1)
            x_0 = self.ae.encode(x)
            eps = torch.randn_like(x_0) if eps is None else eps
            if input_pure_noise:
                if not torch.allclose(t, torch.full_like(t, 999)):
                    raise ValueError("Pure noise input expects all timesteps to be 999.")
                x_t = eps
            else:
                x_t = self.pipe.scheduler.add_noise(x_0, eps, t)

            feats = self.unet_feature_extractor_base(x_t, t, **unet_conds)
            if keys is None:
                feature_dict = dict(feats)
            else:
                feature_dict = {}
                for key in keys:
                    if key not in feats:
                        raise KeyError(f"Feature key '{key}' not available in base model outputs.")
                    feature_dict[key] = feats[key]
            return self._finalize_feature_output(feature_dict, single_key, return_dict)

        (B, *_), device = x.shape, x.device

        if caption is None:
            caption = [""] * B

        unet_conds = self._get_unet_conds(caption, device, x.dtype, 1)
        x_0 = self.ae.encode(x)
        cleandift_t = torch.ones((B,), device=device, dtype=self.timestep.dtype) * self.timestep
        raw_feats = self.unet_feature_extractor_cleandift(
            x_0,
            cleandift_t,
            **unet_conds,
        )

        if keys is None:
            key_iterable = list(raw_feats.keys())
        else:
            key_iterable = list(keys)

        feature_dict = {}
        apply_adapter = self.use_adapters and hasattr(self, "adapters") and (t is not None)
        cond = None
        if apply_adapter:
            timestep_vec = self._prepare_timestep_vector(t, B, device)
            cond = self.mapping(
                self.time_in_proj(
                    self.time_emb(
                        timestep_vec[:, None].to(dtype=x.dtype, device=device) / self.t_max_model
                    )
                )
            )

        for key in key_iterable:
            if key not in raw_feats:
                raise KeyError(f"Feature key '{key}' not found in CleanDIFT features.")
            tensor = raw_feats[key]
            tensor = einops.rearrange(tensor, "B D H W -> B H W D")
            if apply_adapter:
                tensor = self.adapters[key](tensor, cond=cond)
            tensor = einops.rearrange(tensor, "B H W D -> B D H W")
            feature_dict[key] = tensor

        return self._finalize_feature_output(feature_dict, single_key, return_dict)
