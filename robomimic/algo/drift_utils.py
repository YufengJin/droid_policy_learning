"""
Drift policy utilities: mean-shift drift field for generative training.

Adapted from drift_model/run_experiment.py. Used by DriftPolicyUNet for
training loss computation (attraction toward real data, repulsion from other
generated samples).
"""

import torch


def compute_drift(
    gen: torch.Tensor,
    pos: torch.Tensor,
    temp: float = 0.05,
    kernel_type: str = "laplace",
    dist_scale_mode: str = "none",
    rbf_sigma: float = 1.0,
    multiscale_sigmas: list = None,
) -> tuple:
    """
    Compute mean-shift drift field V(gen) using batch-normalized kernel.

    The drift has two components:
      V⁺(x): weighted mean-shift toward real data points (attraction)
      V⁻(x): weighted mean-shift toward other generated points (repulsion)
      V(x) = V⁺(x) - V⁻(x)

    Args:
        gen: Generated samples [G, D]
        pos: Real data samples [P, D]
        temp: Temperature τ controlling kernel bandwidth (laplace)
        kernel_type: "laplace" or "rbf"
        dist_scale_mode: "none" or "sqrt_dim" (divide distances by sqrt(D))
        rbf_sigma: σ for RBF kernel
        multiscale_sigmas: list of σ values for multiscale_rbf kernel

    Returns:
        V: Drift vectors [G, D]
        stats: dict with kernel_max, kernel_mean, dist_median
    """
    targets = torch.cat([gen, pos], dim=0)  # [G+P, D]
    G = gen.shape[0]

    # Pairwise distances: gen → all (gen + pos)
    dist = torch.cdist(gen, targets)  # [G, G+P]
    dist[:, :G].fill_diagonal_(1e6)  # mask self-distances

    # Optional distance scaling for high-dimensional spaces
    if dist_scale_mode == "sqrt_dim":
        D = gen.shape[1]
        dist = dist / (D ** 0.5)

    # Compute kernel
    if kernel_type == "multiscale_rbf":
        sigmas = multiscale_sigmas or [0.25, 0.5, 1.0, 2.0]
        kernel = sum((-dist.pow(2) / (2.0 * s ** 2)).exp() for s in sigmas) / len(sigmas)
    elif kernel_type == "rbf":
        kernel = (-dist.pow(2) / (2.0 * rbf_sigma ** 2)).exp()
    else:  # laplace (default)
        kernel = (-dist / temp).exp()

    # Collect stats before normalization
    stats = {
        "kernel_max": kernel.max().item(),
        "kernel_mean": kernel.mean().item(),
        "dist_median": dist.median().item(),
    }

    # Batch-normalized kernel: normalize along both dimensions
    row_sum = kernel.sum(dim=-1, keepdim=True)  # [G, 1]
    col_sum = kernel.sum(dim=-2, keepdim=True)  # [1, G+P]
    normalizer = (row_sum * col_sum).clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer

    # Positive drift: attraction toward real data
    pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    pos_V = pos_coeff @ targets[G:]

    # Negative drift: repulsion from other generated samples
    neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
    neg_V = neg_coeff @ targets[:G]

    return pos_V - neg_V, stats  # attraction - repulsion


def compute_adaptive_temp(
    gen: torch.Tensor,
    pos: torch.Tensor,
    base_temp: float = 0.05,
    median_scale: float = 0.3,
    min_temp: float = 0.01,
    dist_scale_mode: str = "none",
) -> float:
    """
    Adaptive temperature based on median distance between gen and pos.
    temp = max(min_temp, max(base_temp, median_dist * median_scale))

    Args:
        gen: Generated samples [G, D]
        pos: Real data samples [P, D]
        base_temp: Base temperature floor
        median_scale: Multiplier for median distance
        min_temp: Absolute minimum temperature
        dist_scale_mode: "none" or "sqrt_dim"
    """
    dist = torch.cdist(gen, pos)
    if dist_scale_mode == "sqrt_dim":
        D = gen.shape[1]
        dist = dist / (D ** 0.5)
    median_dist = dist.median().item()
    return max(min_temp, max(base_temp, median_dist * median_scale))


def clip_drift(V: torch.Tensor, max_drift: float) -> torch.Tensor:
    """
    Clip drift vectors by max norm per sample.
    scale = min(||V||, max_drift) / ||V||
    """
    V_norm = V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = torch.minimum(V_norm, torch.tensor(max_drift, device=V.device, dtype=V.dtype)) / V_norm
    return V * scale
