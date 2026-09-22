"""Reconstruction and similarity metrics.

All metrics take tensors whose LAST dimension is the vector dimension and
reduce over the leading dimensions, so they work for [N, D], [B, H, N, D], etc.
"""

from __future__ import annotations

import torch


def relative_l2(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-12) -> float:
    """||est - ref||_F / ||ref||_F  -- the standard CS reconstruction error."""
    num = torch.linalg.norm((estimate - reference).flatten())
    den = torch.linalg.norm(reference.flatten()) + eps
    return (num / den).item()


def per_row_relative_l2(estimate: torch.Tensor, reference: torch.Tensor,
                        eps: float = 1e-12) -> torch.Tensor:
    """Relative L2 computed independently for every row vector."""
    num = torch.linalg.norm(estimate - reference, dim=-1)
    den = torch.linalg.norm(reference, dim=-1) + eps
    return num / den


def cosine_similarity(estimate: torch.Tensor, reference: torch.Tensor,
                      eps: float = 1e-12) -> float:
    """Mean cosine similarity between corresponding row vectors."""
    cos = torch.nn.functional.cosine_similarity(
        estimate.flatten(0, -2), reference.flatten(0, -2), dim=-1, eps=eps
    )
    return cos.mean().item()


def nmse_db(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-12) -> float:
    """Normalised MSE in decibels: 10*log10(||e||^2 / ||ref||^2)."""
    num = torch.sum((estimate - reference) ** 2)
    den = torch.sum(reference ** 2) + eps
    return (10.0 * torch.log10(num / den + eps)).item()


def support_f1(alpha_hat: torch.Tensor, alpha_true: torch.Tensor,
               thresh: float = 1e-3) -> float:
    """F1 between the estimated and true supports (exact-recovery diagnostic).

    ``thresh`` is a relative magnitude cut-off: coefficients smaller than
    ``thresh * max|alpha|`` (per row) count as zero.
    """
    def support(a: torch.Tensor) -> torch.Tensor:
        scale = a.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        return (a.abs() > thresh * scale)

    s_hat, s_true = support(alpha_hat), support(alpha_true)
    tp = (s_hat & s_true).sum(dim=-1).float()
    fp = (s_hat & ~s_true).sum(dim=-1).float()
    fn = (~s_hat & s_true).sum(dim=-1).float()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1e-12)
    return f1.mean().item()


def sparsity_profile(x: torch.Tensor, energy: float = 0.95) -> dict[str, float]:
    """How many coefficients per row carry ``energy`` of the row's L2 energy?

    This is the direct empirical test of the paper's central assumption that
    context vectors are 'sparse or approximately compressible in some basis'.
    A value close to the ambient dimension means *not* compressible.
    """
    mag2 = x.flatten(0, -2) ** 2
    sorted_desc, _ = torch.sort(mag2, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_desc, dim=-1)
    total = cumulative[:, -1:].clamp_min(1e-12)
    frac = cumulative / total
    # first index where cumulative fraction >= energy (1-based count)
    k = (frac < energy).sum(dim=-1) + 1
    dim = x.shape[-1]
    return {
        "dim": float(dim),
        f"k_for_{int(energy * 100)}pct_mean": k.float().mean().item(),
        f"k_for_{int(energy * 100)}pct_median": k.float().median().item(),
        "compressibility_ratio": k.float().mean().item() / dim,
    }


def all_metrics(estimate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Bundle used by every experiment that compares two tensors."""
    return {
        "relative_l2": relative_l2(estimate, reference),
        "cosine_similarity": cosine_similarity(estimate, reference),
        "nmse_db": nmse_db(estimate, reference),
        "norm_ratio": (
            torch.linalg.norm(estimate.flatten()) /
            torch.linalg.norm(reference.flatten()).clamp_min(1e-12)
        ).item(),
    }
