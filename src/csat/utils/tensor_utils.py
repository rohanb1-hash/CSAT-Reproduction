"""Synthetic data generators and small tensor helpers.

Nothing here comes from the paper: the paper's experiments use WikiText-103,
LRA Pathfinder-X, Flickr30k and MS-COCO, none of which we train on in this
notebook. These generators create *controlled* inputs so that every claim we
test has a known ground truth.

Randomness is controlled by the global torch seed (utils.seed.set_seed) rather
than by explicit Generator objects, because a CPU Generator cannot be used for
CUDA tensors and we want every function here to work on both devices.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------- #
# Attention inputs
# --------------------------------------------------------------------------- #
def make_qkv(batch: int, heads: int, seq_len: int, d_k: int,
             device: torch.device | str = "cpu",
             dtype: torch.dtype = torch.float32,
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """i.i.d. Gaussian Q, K, V in the [B, H, N, D] layout."""
    shape = (batch, heads, seq_len, d_k)
    kw = {"device": device, "dtype": dtype}
    return torch.randn(shape, **kw), torch.randn(shape, **kw), torch.randn(shape, **kw)


def make_redundant_tokens(batch: int, heads: int, seq_len: int, d_k: int,
                          n_clusters: int = 16, noise: float = 0.1,
                          device: torch.device | str = "cpu",
                          ) -> torch.Tensor:
    """Token matrix with strong redundancy: ``n_clusters`` prototypes + noise.

    The paper motivates CSAT with 'spatially and perceptually redundant' visual
    tokens. i.i.d. Gaussian tokens have *no* redundancy, so a method that
    exploits redundancy cannot possibly look good on them. This generator lets
    us sweep redundancy explicitly instead of assuming it.
    """
    prototypes = torch.randn(batch, heads, n_clusters, d_k, device=device)
    idx = torch.randint(0, n_clusters, (batch, heads, seq_len), device=device)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, d_k)
    base = torch.gather(prototypes, 2, idx_exp)
    return base + noise * torch.randn(batch, heads, seq_len, d_k, device=device)


# --------------------------------------------------------------------------- #
# Sparse signals for the compressed-sensing experiments
# --------------------------------------------------------------------------- #
def make_sparse_signals(n_signals: int, dim: int, sparsity: int,
                        amplitude: tuple[float, float] = (0.5, 1.5),
                        device: torch.device | str = "cpu",
                        ) -> torch.Tensor:
    """Exactly ``sparsity``-sparse vectors, random support, random signs.

    Returns [n_signals, dim]. Random signs avoid the degenerate all-positive
    case in which even a non-negative least-squares solver succeeds.
    """
    assert 0 < sparsity <= dim, "sparsity must satisfy 0 < s <= dim"
    # Random support per row: rank the columns by a uniform key, keep the top s.
    keys = torch.rand(n_signals, dim, device=device)
    support = keys.argsort(dim=-1)[:, :sparsity]                     # [n, s]
    lo, hi = amplitude
    mag = torch.rand(n_signals, sparsity, device=device) * (hi - lo) + lo
    sign = torch.where(torch.rand(n_signals, sparsity, device=device) < 0.5, -1.0, 1.0)
    alpha = torch.zeros(n_signals, dim, device=device)
    alpha.scatter_(1, support, mag * sign)
    return alpha


def add_noise(y: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0:
        return y
    return y + noise_std * torch.randn_like(y)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def spectral_norm(A: torch.Tensor, n_iters: int = 100) -> float:
    """Largest singular value of A via power iteration on A^T A.

    Used to pick the ISTA step size eta = 1/L with L = sigma_max(A)^2, the
    classical condition for monotone descent. torch.linalg.matrix_norm(A, 2)
    would also work; power iteration keeps the Lipschitz logic explicit and is
    what a large-scale GPU implementation would actually use.
    """
    v = torch.randn(A.shape[1], device=A.device, dtype=A.dtype)
    v = v / v.norm().clamp_min(1e-20)
    nrm = torch.tensor(0.0, device=A.device, dtype=A.dtype)
    for _ in range(n_iters):
        v = A.T @ (A @ v)
        nrm = v.norm()
        if nrm < 1e-20:
            return 0.0
        v = v / nrm
    return torch.sqrt(nrm).item()


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
