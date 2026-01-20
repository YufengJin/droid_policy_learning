"""
CleanDIFT Image Encoder with S2-FPN Bidirectional Fusion + Queries Attention

Architecture:
    1. CleanDIFT Backbone: Extract multi-scale features from SD2.1 UNet
    2. S2-FPN Bidirectional Fusion: Top-down (semantic) + Bottom-up (geometric)
    3. Queries Attention: Learnable queries cross-attend to fused features
    4. Global Representation: Output latent vector for policy learning

Paper Reference:
    - S2-FPN: Scale-Selection Feature Pyramid Network
    - Layer Scale: DeiT-III, EVA, BEiT-3
"""

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
import logging
from collections import OrderedDict
from typing import Optional, List, Dict, Any, Tuple
from omegaconf import OmegaConf, ListConfig

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from pydoc import locate

try:
    from agents.encoders.cleandift.src.sd_feature_extraction import StableFeatureAligner
    from agents.encoders.cleandift.src.ae import AutoencoderKL
    from agents.encoders.cleandift.src.utils import MappingSpec
except ImportError:
    from cleandift.src.sd_feature_extraction import StableFeatureAligner
    from cleandift.src.ae import AutoencoderKL
    from cleandift.src.utils import MappingSpec


def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    """2D sin-cos absolute position embedding, returns [1, h*w, dim]."""
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    dim_half = dim // 2
    dim_y, dim_x = dim_half, dim - dim_half
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )

    def _pe(vec, d):
        L = max(1, d // 2)
        omega = 1.0 / (10000 ** (torch.arange(L, device=device, dtype=dtype) / max(1, L - 1)))
        out = torch.einsum("n,d->nd", vec.flatten(), omega)
        pe = torch.cat([out.sin(), out.cos()], dim=-1)
        if pe.shape[-1] < d:
            pe = F.pad(pe, (0, d - pe.shape[-1]))
        return pe

    pey, pex = _pe(yy, dim_y), _pe(xx, dim_x)
    pe = torch.cat([pey, pex], dim=-1).unsqueeze(0)
    return pe


def _make_group_norm(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """Build GroupNorm with automatic group size selection."""
    for g in [num_groups, 16, 8, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


# =============================================================================
# S2-FPN Components
# =============================================================================

class S2FPNFuseBlock(nn.Module):
    """Single fusion block for S2-FPN: Conv + GroupNorm + GELU."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm = _make_group_norm(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class S2FPNBidirectionalFusion(nn.Module):
    """
    S2-FPN Bidirectional Feature Fusion Module.
    
    Top-Down Path: High semantic -> Low semantic (spatial refinement)
    Bottom-Up Path: High resolution -> Low resolution (semantic enhancement)
    
    Args:
        feature_dims: Dict mapping feature keys to channel dimensions
        feature_keys: List of feature keys in order (low-res to high-res)
        fpn_dim: Unified channel dimension after projection
    """
    
    def __init__(
        self, 
        feature_dims: Dict[str, int],
        feature_keys: List[str],
        fpn_dim: int = 256,
        device: str = "cuda",
    ):
        super().__init__()
        self.feature_keys = feature_keys
        self.fpn_dim = fpn_dim
        
        # 1. Per-scale projection layers (keep original resolution)
        self.proj_layers = nn.ModuleDict()
        for key in feature_keys:
            c_in = feature_dims[key]
            self.proj_layers[key] = nn.Sequential(
                nn.Conv2d(c_in, fpn_dim, kernel_size=1, bias=True),
                _make_group_norm(fpn_dim),
                nn.GELU(),
            ).to(device)
        
        # 2. Top-Down fusion blocks (semantic enhancement to geometric layers)
        # Number of fusions = len(feature_keys) - 1
        self.td_fuse_blocks = nn.ModuleList([
            S2FPNFuseBlock(fpn_dim * 2, fpn_dim).to(device)
            for _ in range(len(feature_keys) - 1)
        ])
        
        # 3. Bottom-Up fusion blocks (geometric detail back to semantic layers)
        self.bu_fuse_blocks = nn.ModuleList([
            S2FPNFuseBlock(fpn_dim * 2, fpn_dim).to(device)
            for _ in range(len(feature_keys) - 1)
        ])
        
        # 4. Final fusion: combine top-down and bottom-up
        self.final_fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * 2, fpn_dim, kernel_size=1, bias=True),
            _make_group_norm(fpn_dim),
            nn.GELU(),
        ).to(device)
    
    def forward(self, feature_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feature_maps: Dict of {key: [B, C, H, W]} tensors
            
        Returns:
            Fused feature map [B, fpn_dim, H_max, W_max]
        """
        # Record original resolutions before any processing
        original_resolutions = {key: feature_maps[key].shape[-1] for key in self.feature_keys}
        
        # Sort keys by resolution: low-res (high semantic) first
        sorted_keys = sorted(self.feature_keys, key=lambda k: original_resolutions[k])
        
        # Step 1: Project all features to fpn_dim (keep original resolutions)
        projected = {}
        for key in sorted_keys:
            fmap = feature_maps[key]
            if fmap.dtype != torch.float32:
                fmap = fmap.float()
            projected[key] = self.proj_layers[key](fmap)
        
        # Get target resolution (highest resolution for final output)
        target_h = max(f.shape[-2] for f in projected.values())
        target_w = max(f.shape[-1] for f in projected.values())
        
        # Step 2: Top-Down Path (low-res -> high-res, semantic enhancement)
        # P_low -> upsample -> concat with P_high -> fuse
        td_features = {}
        td_features[sorted_keys[0]] = projected[sorted_keys[0]]
        
        for i, key in enumerate(sorted_keys[1:]):
            prev_key = sorted_keys[i]
            prev_feat = td_features[prev_key]
            curr_feat = projected[key]
            
            # Upsample previous (lower-res) to current resolution
            if prev_feat.shape[-2:] != curr_feat.shape[-2:]:
                prev_feat = F.interpolate(
                    prev_feat, size=curr_feat.shape[-2:], 
                    mode="bilinear", align_corners=False
                )
            
            # Concat and fuse
            fused = torch.cat([prev_feat, curr_feat], dim=1)
            td_features[key] = self.td_fuse_blocks[i](fused)
        
        # Step 3: Bottom-Up Path (high-res -> low-res, geometric detail)
        # P_high -> downsample -> concat with P_low -> fuse
        bu_features = {}
        bu_features[sorted_keys[-1]] = td_features[sorted_keys[-1]]
        
        for i, key in enumerate(reversed(sorted_keys[:-1])):
            next_key = sorted_keys[-(i+1)]
            next_feat = bu_features[next_key]
            curr_feat = td_features[key]
            
            # Downsample next (higher-res) to current resolution
            if next_feat.shape[-2:] != curr_feat.shape[-2:]:
                next_feat = F.interpolate(
                    next_feat, size=curr_feat.shape[-2:],
                    mode="bilinear", align_corners=False
                )
            
            # Concat and fuse
            fused = torch.cat([next_feat, curr_feat], dim=1)
            bu_features[key] = self.bu_fuse_blocks[i](fused)
        
        # Step 4: Final output - take highest resolution feature
        # Combine TD and BU paths at highest resolution
        final_key = sorted_keys[-1]
        td_final = td_features[final_key]
        bu_final = bu_features[final_key]
        
        # Ensure same resolution
        if td_final.shape[-2:] != (target_h, target_w):
            td_final = F.interpolate(td_final, size=(target_h, target_w), mode="bilinear", align_corners=False)
        if bu_final.shape[-2:] != (target_h, target_w):
            bu_final = F.interpolate(bu_final, size=(target_h, target_w), mode="bilinear", align_corners=False)
        
        # Final fusion
        output = self.final_fuse(torch.cat([td_final, bu_final], dim=1))
        
        return output


class QueriesAttentionPooling(nn.Module):
    """
    Learnable Queries Attention Pooling with Layer Scale.
    
    Cross-attention from learnable queries to spatial tokens,
    producing fixed-size global representation.
    
    Args:
        dim: Feature dimension
        num_queries: Number of learnable query vectors
        num_heads: Number of attention heads
        dropout: Dropout rate
        layer_scale_init: Initial value for layer scale (default 0.1)
    """
    
    def __init__(
        self,
        dim: int,
        num_queries: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        layer_scale_init: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        
        # Learnable queries
        self.queries = nn.Parameter(torch.empty(num_queries, dim, device=device))
        nn.init.trunc_normal_(self.queries, std=0.02)
        
        # Normalization
        self.token_norm = nn.LayerNorm(dim).to(device)
        self.query_norm = nn.LayerNorm(dim).to(device)
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        ).to(device)
        
        # FFN
        self.ffn_norm = nn.LayerNorm(dim).to(device)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        ).to(device)
        
        # Layer Scale (DeiT-III style)
        self.attn_layer_scale = nn.Parameter(torch.ones(dim, device=device) * layer_scale_init)
        self.ffn_layer_scale = nn.Parameter(torch.ones(dim, device=device) * layer_scale_init)
        
        # Position encoding dropout
        self.pos_dropout = nn.Dropout(dropout)
    
    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W] fused feature map
            
        Returns:
            queries: [B, num_queries, dim] attended query vectors
        """
        B, C, H, W = feature_map.shape
        
        # Flatten to tokens
        tokens = feature_map.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        # Add positional encoding
        pe = _build_2d_sincos_pos_embed(H, W, C, device=tokens.device, dtype=tokens.dtype)
        tokens = self.pos_dropout(tokens + pe)
        
        # Normalize tokens
        tokens_norm = self.token_norm(tokens)
        
        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        
        # Pre-Norm + Layer Scale Attention
        queries_norm = self.query_norm(queries)
        attn_out, _ = self.attn(
            query=queries_norm, 
            key=tokens_norm, 
            value=tokens_norm, 
            need_weights=False
        )
        queries = queries + self.attn_layer_scale * attn_out
        
        # Pre-Norm + Layer Scale FFN
        ffn_out = self.ffn(self.ffn_norm(queries))
        queries = queries + self.ffn_layer_scale * ffn_out
        
        return queries


# =============================================================================
# Main Encoder Class
# =============================================================================

class CleanDIFTImgEncoder(nn.Module):
    """
    CleanDIFT Image Encoder with S2-FPN Bidirectional Fusion.
    
    Architecture:
        Input Image [B, 3, 256, 256]
            ->
        CleanDIFT Backbone (SD2.1 UNet)
            ->
        Multi-scale Features: us3(32^2), us5(64^2), us6(64^2), us8(128^2)
            ->
        S2-FPN Bidirectional Fusion
            ->
        Queries Attention Pooling
            ->
        Global Vector [B, latent_dim]
    
    Args:
        sd_version: "sd15" or "sd21"
        feature_key: Single key or list of keys for multi-scale extraction
        freeze_backbone: Whether to freeze CleanDIFT backbone
        map_out_dim: Output latent dimension
        fpn_dim: FPN intermediate dimension (default 256)
        fpn_num_queries: Number of learnable queries (default 4)
        fpn_dropout: Dropout rate (default 0.1)
    """
    
    def __init__(
        self, 
        sd_version: str = "sd21",  
        feature_key: str = "us6", 
        freeze_backbone: bool = True,
        device: str = "cuda",
        camera_names: Optional[List[str]] = None,
        use_text_condition: bool = False,  
        yaml_file: Optional[str] = None,   
        map_out_dim: Optional[int] = None,
        use_fp32: Optional[bool] = None,
        custom_checkpoint: Optional[str] = None,
        alignment_cfg: Optional[Any] = None,
        # S2-FPN parameters
        fpn_dim: int = 256,
        fpn_num_queries: int = 4,
        fpn_dropout: float = 0.1,
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        
        # =================================================================
        # 1. Configuration
        # =================================================================
        self.sd_version = sd_version
        self.freeze_backbone = freeze_backbone
        self.device = device
        self.camera_names = camera_names
        self.use_text_condition = use_text_condition
        self.map_out_dim = map_out_dim
        self._fpn_dim = fpn_dim
        self._fpn_num_queries = fpn_num_queries
        self._fpn_dropout = fpn_dropout
        
        # Parse feature keys
        if isinstance(feature_key, ListConfig):
            feature_key = list(feature_key)
        if isinstance(feature_key, (list, tuple)):
            if len(feature_key) == 0:
                raise ValueError("feature_key must contain at least one entry.")
            self.feature_keys = [str(k) for k in feature_key]
        else:
            self.feature_keys = [str(feature_key)]
        
        self.multi_feature_mode = len(self.feature_keys) > 1
        self.primary_feature_key = self.feature_keys[0]
        
        # Precision settings
        self._force_backbone_fp32 = bool(use_fp32) if use_fp32 is not None else False
        
        # Alignment config
        if alignment_cfg is not None and not isinstance(alignment_cfg, dict):
            alignment_cfg = OmegaConf.to_container(alignment_cfg, resolve=True)
        self.alignment_cfg = alignment_cfg or {}
        
        # =================================================================
        # 2. Load CleanDIFT Backbone
        # =================================================================
        self._init_backbone(sd_version, yaml_file, device, custom_checkpoint)
        
        # =================================================================
        # 3. Build S2-FPN Fusion Head (Multi-Feature Mode)
        # =================================================================
        if self.multi_feature_mode:
            target_dim = map_out_dim if map_out_dim and map_out_dim > 0 else fpn_dim
            self._build_s2fpn_head(target_dim, fpn_dim, fpn_num_queries, fpn_dropout, layer_scale_init, device)
        else:
            # Single feature mode: simple pooling
            target_dim = map_out_dim if map_out_dim and map_out_dim > 0 else self.feature_dims[self.primary_feature_key]
            self._build_single_feature_head(target_dim, device)
        
        self._multi_feature_out_dim = target_dim
        
        # =================================================================
        # 4. Freeze Backbone if needed
        # =================================================================
        if freeze_backbone:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        
        self._logger = logging.getLogger(__name__)
    
    def _init_backbone(self, sd_version: str, yaml_file: Optional[str], device: str, custom_checkpoint: Optional[str]):
        """Initialize CleanDIFT backbone."""
        # Resolve config file
        if sd_version == "sd15":
            default_yaml = "sd15_feature_extractor.yaml"
            try:
                ckpt_pth = hf_hub_download(
                    repo_id="CompVis/cleandift",
                    filename="cleandift_sd15_full.safetensors",
                )
            except Exception:
                ckpt_pth = hf_hub_download(
                    repo_id="CompVis/cleandift",
                    filename="cleandift_sd15_full.safetensors",
                    local_files_only=True,
                )
        else:
            default_yaml = "sd21_feature_extractor.yaml"
            try:
                ckpt_pth = hf_hub_download(
                    repo_id="CompVis/cleandift",
                    filename="cleandift_sd21_full.safetensors",
                )
            except Exception:
                ckpt_pth = hf_hub_download(
                    repo_id="CompVis/cleandift",
                    filename="cleandift_sd21_full.safetensors",
                    local_files_only=True,
                )
        
        here = os.path.dirname(os.path.abspath(__file__))
        yaml_name = yaml_file or default_yaml
        candidates = [
            os.path.join(here, "cleandift", "configs", yaml_name),
            os.path.join(here, yaml_name),
        ]
        yaml_path = None
        for c in candidates:
            if os.path.exists(c):
                yaml_path = c
                break
        if yaml_path is None:
            raise FileNotFoundError(f"Could not find YAML: {yaml_name}. Tried: {candidates}")
        
        cfg = OmegaConf.load(yaml_path)
        mconf = cfg["model"]
        
        # Feature dimensions
        self.feature_dims = dict(mconf.get("feature_dims", {}))
        for key in self.feature_keys:
            if key not in self.feature_dims:
                raise KeyError(f"Feature key '{key}' not in config. Available: {list(self.feature_dims.keys())}")
        
        # Build model
        ae = AutoencoderKL(repo=mconf["ae"]["repo"]).to(device)
        mapping = MappingSpec(
            depth=mconf["mapping"]["depth"],
            width=mconf["mapping"]["width"],
            d_ff=mconf["mapping"]["d_ff"],
            dropout=mconf["mapping"]["dropout"],
        )
        
        def fix_class_path(path: str) -> str:
            if path.startswith("src."):
                try_path = "agents.encoders.cleandift." + path
                return try_path if locate(try_path) is not None else ("cleandift." + path)
            return path
        
        self.model = StableFeatureAligner(
            sd_version=sd_version,
            t_max=mconf.get("t_max", 999),
            num_t_stratification_bins=mconf.get("num_t_stratification_bins", 3),
            train_unet=mconf.get("train_unet", True),
            learn_timestep=mconf.get("learn_timestep", True),
            use_text_condition=self.use_text_condition,
            ae=ae,
            mapping=mapping,
            adapter_layer_class=fix_class_path(mconf["adapter_layer_class"]),
            adapter_layer_params=mconf.get("adapter_layer_params", {}),
            feature_extractor_cls=fix_class_path(mconf["feature_extractor_cls"]),
            feature_dims=self.feature_dims,
            device=device,
        )
        
        # Move to device and set precision
        self.model = self.model.to(device)
        if self._force_backbone_fp32:
            self.model = self.model.to(dtype=torch.float32)
        elif device.startswith("cuda") and torch.cuda.is_available():
            self.model = self.model.to(dtype=torch.bfloat16)
        else:
            self.model = self.model.to(dtype=torch.float32)
        
        self.model_dtype = next(self.model.parameters()).dtype
        self._amp_enabled = (
            not self._force_backbone_fp32 
            and device.startswith("cuda") 
            and torch.cuda.is_available()
        )
        self._amp_autocast_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16} if self._amp_enabled else {}
        
        # Load weights
        if custom_checkpoint:
            self._load_custom_checkpoint(custom_checkpoint)
        else:
            state_dict = load_file(ckpt_pth)
            self.model.load_state_dict(state_dict, strict=True)
    
    def _build_s2fpn_head(self, target_dim: int, fpn_dim: int, num_queries: int, dropout: float, layer_scale_init: float, device: str):
        """Build S2-FPN fusion and attention pooling modules."""
        # S2-FPN Bidirectional Fusion
        self.s2fpn_fusion = S2FPNBidirectionalFusion(
            feature_dims=self.feature_dims,
            feature_keys=self.feature_keys,
            fpn_dim=fpn_dim,
            device=device,
        )
        
        # Queries Attention Pooling
        num_heads = max(1, min(8, fpn_dim // 64))
        while num_heads > 1 and fpn_dim % num_heads != 0:
            num_heads -= 1
        
        self.queries_pooling = QueriesAttentionPooling(
            dim=fpn_dim,
            num_queries=num_queries,
            num_heads=num_heads,
            dropout=dropout,
            layer_scale_init=layer_scale_init,
            device=device,
        )
        
        # Final projection: [num_queries * fpn_dim] -> [target_dim]
        flat_dim = fpn_dim * num_queries
        self.final_proj = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, max(target_dim, flat_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(target_dim, flat_dim // 2), target_dim),
        ).to(device)
    
    def _build_single_feature_head(self, target_dim: int, device: str):
        """Build simple head for single feature mode."""
        c_in = self.feature_dims[self.primary_feature_key]
        
        self.single_proj = nn.Conv2d(c_in, target_dim, kernel_size=1, bias=True).to(device)
        self.single_norm = nn.LayerNorm(target_dim).to(device)
        self.single_pool = nn.AdaptiveAvgPool2d(1)
        
        nn.init.normal_(self.single_proj.weight, std=0.02)
        nn.init.zeros_(self.single_proj.bias)
    
    # =========================================================================
    # Forward Methods
    # =========================================================================
    
    def forward(self, x: torch.Tensor, lang_cond=None, alignment_context=None) -> tuple:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            lang_cond: Optional language condition
            alignment_context: Optional alignment loss context
            
        Returns:
            features: [B, 1, latent_dim] global features
            alignment_loss: Optional alignment loss tensor
        """
        x = x.to(self.device, dtype=self.model_dtype)
        caption = lang_cond if self.use_text_condition else None
        
        features = self._extract_features(x, caption)
        
        # Compute alignment loss if needed
        alignment_loss = None
        if alignment_context is not None and not self.freeze_backbone:
            alignment_loss = self._compute_alignment_loss(alignment_context, caption)
        
        return features, alignment_loss
    
    def _extract_features(self, x: torch.Tensor, caption=None) -> torch.Tensor:
        """Extract and process features through the full pipeline."""
        # Get multi-scale features from backbone
        feature_maps = self._encode_backbone(x, caption)
        
        if self.multi_feature_mode:
            # S2-FPN fusion
            fused = self.s2fpn_fusion(feature_maps)
            
            # Queries attention pooling
            queries = self.queries_pooling(fused)  # [B, num_queries, fpn_dim]
            
            # Final projection
            pooled = queries.flatten(1)  # [B, num_queries * fpn_dim]
            output = self.final_proj(pooled)  # [B, target_dim]
            
            return output.unsqueeze(1)  # [B, 1, target_dim]
        else:
            # Single feature mode
            fmap = feature_maps[self.primary_feature_key]
            if fmap.dtype != torch.float32:
                fmap = fmap.float()
            
            proj = self.single_proj(fmap)  # [B, target_dim, H, W]
            pooled = self.single_pool(proj).flatten(1)  # [B, target_dim]
            output = self.single_norm(pooled)
            
            return output.unsqueeze(1)  # [B, 1, target_dim]
    
    def _encode_backbone(self, x: torch.Tensor, caption=None) -> Dict[str, torch.Tensor]:
        """Extract features from CleanDIFT backbone."""
        if x.dtype != self.model_dtype:
            x = x.to(self.model_dtype)
        
        amp_ctx = torch.autocast(**self._amp_autocast_kwargs) if self._amp_enabled else contextlib.nullcontext()
        grad_ctx = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        
        with grad_ctx:
            with amp_ctx:
                if self.multi_feature_mode:
                    features = self.model.get_features(
                        x, caption=caption, t=None,
                        feat_key=self.feature_keys,
                        use_base_model=False,
                    )
                else:
                    features = self.model.get_features(
                        x, caption=caption, t=None,
                        feat_key=self.primary_feature_key,
                        use_base_model=False,
                    )
                    if not isinstance(features, dict):
                        features = {self.primary_feature_key: features}
        
        # Convert to float32
        return {k: v.float() if v.dtype != torch.float32 else v for k, v in features.items()}
    
    def _compute_alignment_loss(self, alignment_context: dict, caption) -> Optional[torch.Tensor]:
        """Compute optional alignment loss."""
        align_images = alignment_context.get("images")
        if align_images is None or align_images.shape[0] == 0:
            return None
        
        align_images = align_images.to(self.device, dtype=self.model_dtype)
        align_captions = alignment_context.get("captions")
        if align_captions is None:
            align_captions = caption if isinstance(caption, list) else [""] * align_images.shape[0]
        
        amp_ctx = torch.autocast(**self._amp_autocast_kwargs) if self._amp_enabled else contextlib.nullcontext()
        with amp_ctx:
            loss_dict = self.model.forward(align_images, align_captions)
        
        loss = sum(loss_dict.values()) / len(loss_dict)
        return loss.float() if loss.dtype != torch.float32 else loss
    
    def compute_alignment_loss(
        self,
        x: torch.Tensor,
        caption: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Public interface for computing alignment loss.
        Compatible with diffusion_policy.py's alignment loss computation.
        
        Args:
            x: Input images [B, 3, H, W]
            caption: Optional list of captions
            
        Returns:
            Alignment loss tensor (scalar)
        """
        if x.dtype != self.model_dtype:
            x = x.to(self.device, dtype=self.model_dtype)
        
        # If backbone is frozen, no alignment loss
        if self.freeze_backbone:
            return torch.tensor(0.0, device=x.device)
        
        B = x.shape[0]
        if caption is None:
            caption = [""] * B
        
        amp_ctx = torch.autocast(**self._amp_autocast_kwargs) if self._amp_enabled else contextlib.nullcontext()
        with amp_ctx:
            loss_dict = self.model.forward(x, caption)
        
        loss = sum(loss_dict.values()) / len(loss_dict)
        return loss.float() if loss.dtype != torch.float32 else loss
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def extract_feature_map(self, x: torch.Tensor, lang_cond=None) -> torch.Tensor:
        """Extract dense feature map (for visualization/analysis)."""
        caption = lang_cond if self.use_text_condition else None
        features = self._encode_backbone(x.to(self.device, dtype=self.model_dtype), caption)
        
        if self.multi_feature_mode:
            # Return fused feature map
            return self.s2fpn_fusion(features)
        else:
            return features[self.primary_feature_key]
    
    def get_parameter_groups(
        self,
        base_lr: float,
        backbone_lr_multiplier: float = 0.1,
        head_lr_multiplier: float = 1.0,
    ) -> List[dict]:
        """
        Get parameter groups with different learning rates.
        
        Recommended for transfer learning:
            - backbone_lr_multiplier: 0.1 (slow adaptation)
            - head_lr_multiplier: 1.0 (full learning)
        """
        groups = []
        
        # Backbone parameters
        if not self.freeze_backbone and self.model is not None:
            groups.append({
                "params": list(self.model.parameters()),
                "lr": base_lr * backbone_lr_multiplier,
                "name": "backbone",
            })
        
        # Head parameters
        head_params = []
        
        if self.multi_feature_mode:
            head_params.extend(self.s2fpn_fusion.parameters())
            head_params.extend(self.queries_pooling.parameters())
            head_params.extend(self.final_proj.parameters())
        else:
            head_params.extend(self.single_proj.parameters())
            head_params.extend(self.single_norm.parameters())
        
        if head_params:
            groups.append({
                "params": head_params,
                "lr": base_lr * head_lr_multiplier,
                "name": "head",
            })
        
        return groups
    
    def _load_custom_checkpoint(self, checkpoint_dir: str):
        """Load custom checkpoint."""
        import json
        
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
        
        # Try different checkpoint formats
        safetensor_path = os.path.join(checkpoint_dir, "cleandift_full_state.safetensors")
        pt_path = os.path.join(checkpoint_dir, "cleandift_full_state.pt")
        
        state_dict = None
        if os.path.exists(safetensor_path):
            state_dict = load_file(safetensor_path)
        elif os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location='cpu', weights_only=False)
        
        if state_dict is not None:
            # Handle legacy prefix
            if any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = OrderedDict(
                    (k.replace("model.", "", 1) if k.startswith("model.") else k, v)
                    for k, v in state_dict.items()
                )
            
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                warnings.warn(f"Missing keys: {missing[:5]}...")
            if unexpected:
                warnings.warn(f"Unexpected keys: {unexpected[:5]}...")
        else:
            warnings.warn(f"No checkpoint found in {checkpoint_dir}")
