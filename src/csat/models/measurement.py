"""Measurement matrices Phi in R^{m x n} (paper Section 3).

WHAT THE PAPER SAYS
    "Let Phi_K, Phi_V in R^{m x n} be measurement matrices satisfying the RIP,
     where m << n."
    "Here, Phi_K and Phi_V are typically drawn from sub-Gaussian ensembles
     (e.g., random Gaussian, Rademacher, or structured Hadamard matrices) that
     exhibit low coherence with sparse bases, ensuring stable signal recovery."

WHAT THE PAPER DOES NOT SAY
    * the normalisation constant (we use 1/sqrt(m), the standard choice under
      which E[Phi^T Phi] = I_n and the RIP constants of Candes-Tao apply);
    * whether Phi is shared across heads, layers, or modalities;
    * the value of m in any experiment, including Table 5;
    * whether Phi is re-drawn per batch or fixed at initialisation
      (we fix it at initialisation, which is what "fixed post-training" in
      Section 7 implies).

RIP REMINDER (why these ensembles are named)
    Phi satisfies the RIP of order s with constant delta_s if for every
    s-sparse x:   (1-delta_s)||x||^2 <= ||Phi x||^2 <= (1+delta_s)||x||^2.
    For i.i.d. sub-Gaussian entries this holds with high probability once
    m = O(s log(n/s)). Note the direction of this requirement: m must grow with
    the *sparsity of the signal being measured*, and the paper never identifies
    what is sparse along the token axis that Phi_K and Phi_V act on.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

EnsembleName = Literal["gaussian", "rademacher", "orthogonal", "hadamard"]


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
def gaussian_matrix(m: int, n: int, normalize: bool = True,
                    device: torch.device | str = "cpu",
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """i.i.d. N(0, 1/m) entries. E[Phi^T Phi] = I_n."""
    phi = torch.randn(m, n, device=device, dtype=dtype)
    return phi / math.sqrt(m) if normalize else phi


def rademacher_matrix(m: int, n: int, normalize: bool = True,
                      device: torch.device | str = "cpu",
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """i.i.d. +-1/sqrt(m) entries; sub-Gaussian with the same RIP guarantees."""
    signs = torch.randint(0, 2, (m, n), device=device, dtype=dtype) * 2 - 1
    return signs / math.sqrt(m) if normalize else signs


def orthogonal_matrix(m: int, n: int, normalize: bool = True,
                      device: torch.device | str = "cpu",
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """m orthonormal rows (from the QR of a Gaussian), optionally rescaled.

    With ``normalize=True`` we scale by sqrt(n/m) so that ||Phi x|| ~ ||x|| for a
    generic x, matching the energy convention of the Gaussian case. Row-orthogonal
    measurement matrices are a standard structured alternative; the paper mentions
    "structured orthogonal matrices" in Section 2 but does not specify a construction.
    """
    a = torch.randn(n, m, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(a)              # [n, m], orthonormal columns
    phi = q.T.contiguous()                 # [m, n], orthonormal rows
    return phi * math.sqrt(n / m) if normalize else phi


def _hadamard(n: int, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """Sylvester-construction Hadamard matrix, n a power of two."""
    assert n > 0 and (n & (n - 1)) == 0, "Hadamard construction needs n = 2^k"
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


def hadamard_matrix(m: int, n: int, normalize: bool = True,
                    device: torch.device | str = "cpu",
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Randomly subsampled, randomly sign-flipped Hadamard rows.

    This is the classical 'structured' measurement operator: it needs no dense
    storage in a production implementation (a fast Walsh-Hadamard transform
    costs O(n log n)), which is exactly why the paper lists it. Requires n to be
    a power of two; the caller must handle other n.
    """
    if (n & (n - 1)) != 0:
        raise ValueError(f"hadamard ensemble requires n to be a power of 2, got n={n}")
    h = _hadamard(n, device, dtype)
    rows = torch.randperm(n, device=device)[:m]
    col_signs = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0).to(dtype)
    phi = h[rows] * col_signs.unsqueeze(0)
    return phi / math.sqrt(m) if normalize else phi


_GENERATORS = {
    "gaussian": gaussian_matrix,
    "rademacher": rademacher_matrix,
    "orthogonal": orthogonal_matrix,
    "hadamard": hadamard_matrix,
}


def make_measurement_matrix(m: int, n: int, ensemble: EnsembleName = "gaussian",
                            normalize: bool = True,
                            device: torch.device | str = "cpu",
                            dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Dispatch to one of the sub-Gaussian / structured ensembles named in the paper."""
    if ensemble not in _GENERATORS:
        raise ValueError(f"unknown ensemble '{ensemble}', expected one of {list(_GENERATORS)}")
    if m > n:
        raise ValueError(f"compression requires m <= n, got m={m}, n={n}")
    return _GENERATORS[ensemble](m, n, normalize=normalize, device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def mutual_coherence(A: torch.Tensor) -> float:
    """max_{i != j} |<a_i, a_j>| / (||a_i|| ||a_j||) over the COLUMNS of A.

    Low coherence is the practical proxy for the RIP: exact RIP verification is
    NP-hard, so coherence (and the empirical estimate below) is what can actually
    be computed. The paper asserts RIP but never verifies it for its operators.
    """
    cols = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)
    gram = (cols.T @ cols).abs()
    gram.fill_diagonal_(0.0)
    return gram.max().item()


def empirical_rip_constant(A: torch.Tensor, sparsity: int, n_trials: int = 2000
                           ) -> dict:
    """Monte-Carlo LOWER bound on the RIP constant delta_s of A.

    For random s-sparse unit vectors x we measure ||Ax||^2 / ||x||^2 and report
    the worst observed deviation from 1. This is a *lower* bound on delta_s: the
    true constant is a worst case over all s-sparse x, which random sampling can
    only under-estimate. Reporting it as if it were delta_s would be wrong, so
    the return value is named accordingly.
    """
    n = A.shape[1]
    device = A.device
    keys = torch.rand(n_trials, n, device=device)
    support = keys.argsort(dim=-1)[:, :sparsity]
    vals = torch.randn(n_trials, sparsity, device=device)
    x = torch.zeros(n_trials, n, device=device, dtype=A.dtype)
    x.scatter_(1, support, vals.to(A.dtype))
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ratios = (x @ A.T).pow(2).sum(dim=1)          # ||A x||^2 with ||x|| = 1
    return {
        "sparsity": sparsity,
        "min_ratio": ratios.min().item(),
        "max_ratio": ratios.max().item(),
        "mean_ratio": ratios.mean().item(),
        "delta_s_lower_bound": max(abs(1 - ratios.min().item()),
                                   abs(ratios.max().item() - 1)),
        "n_trials": n_trials,
    }


# --------------------------------------------------------------------------- #
# Module wrapper
# --------------------------------------------------------------------------- #
class MeasurementMatrix(nn.Module):
    """Holds Phi as a buffer (fixed) or a Parameter (learnable).

    The paper supports both: Section 3 frames Phi as a fixed CS-style random
    operator, while Section 7 says measurement matrices "can be fixed
    post-training or made learnable".
    """

    def __init__(self, m: int, n: int, ensemble: EnsembleName = "gaussian",
                 normalize: bool = True, learnable: bool = False,
                 n_heads: int | None = None,
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.m, self.n, self.ensemble, self.learnable = m, n, ensemble, learnable
        self.n_heads = n_heads

        if n_heads is None:
            phi = make_measurement_matrix(m, n, ensemble, normalize, device, dtype)
        else:
            phi = torch.stack([
                make_measurement_matrix(m, n, ensemble, normalize, device, dtype)
                for _ in range(n_heads)
            ])                                   # [H, m, n]

        if learnable:
            self.phi = nn.Parameter(phi)
        else:
            self.register_buffer("phi", phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Phi along the TOKEN axis of x: [B, H, N, D] -> [B, H, M, D]."""
        if self.n_heads is None:
            return torch.einsum("mn,bhnd->bhmd", self.phi, x)
        return torch.einsum("hmn,bhnd->bhmd", self.phi, x)

    def extra_repr(self) -> str:
        return (f"m={self.m}, n={self.n}, ensemble={self.ensemble}, "
                f"learnable={self.learnable}, per_head={self.n_heads is not None}")
