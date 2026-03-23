"""
Drift policy utilities: mean-shift drift field for generative training.

Adapted from drift_model/run_experiment.py. Used by DriftPolicyUNet for
training loss computation (attraction toward real data, repulsion from other
generated samples).
"""

import torch


def compute_drift(gen: torch.Tensor, pos: torch.Tensor, temp: float = 0.05) -> torch.Tensor:
    """
    Compute mean-shift drift field V(gen) using batch-normalized kernel.

    The drift has two components:
      V⁺(x): weighted mean-shift toward real data points (attraction)
      V⁻(x): weighted mean-shift toward other generated points (repulsion)
      V(x) = V⁺(x) - V⁻(x)

    Kernel: k(x,y) = exp(-||x-y|| / τ)
    Batch normalization: normalize along both row and column dimensions.

    Args:
        gen: Generated samples [G, D]
        pos: Real data samples [P, D]
        temp: Temperature τ controlling kernel bandwidth

    Returns:
        V: Drift vectors [G, D]
    """
    targets = torch.cat([gen, pos], dim=0)  # [G+P, D]
    G = gen.shape[0]

    # Pairwise distances: gen → all (gen + pos)
    dist = torch.cdist(gen, targets)  # [G, G+P]
    dist[:, :G].fill_diagonal_(1e6)  # mask self-distances

    # Unnormalized kernel
    kernel = (-dist / temp).exp()  # [G, G+P]

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

    return pos_V - neg_V  # attraction - repulsion


def compute_adaptive_temp(gen: torch.Tensor, pos: torch.Tensor, base_temp: float = 0.05) -> float:
    """
    Adaptive temperature based on median distance between gen and pos.
    temp = max(base_temp, median_dist * 0.3)
    """
    median_dist = torch.cdist(gen, pos).median().item()
    return max(base_temp, median_dist * 0.3)


def clip_drift(V: torch.Tensor, max_drift: float) -> torch.Tensor:
    """
    Clip drift vectors by max norm per sample.
    scale = min(||V||, max_drift) / ||V||
    """
    V_norm = V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = torch.minimum(V_norm, torch.tensor(max_drift, device=V.device, dtype=V.dtype)) / V_norm
    return V * scale
