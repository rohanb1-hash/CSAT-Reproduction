"""The sparsifying dictionary Psi (paper Section 3).

WHAT THE PAPER SAYS
    "Suppose there exists a dictionary Psi in R^{d_k x d_k}, such that the true
     context vector C_i admits a sparse representation: C_i = Psi alpha_i,
     where alpha_i in R^{d_k} is sparse."
    Elsewhere it says the context vectors are sparse "in some unknown or
    learnable basis".

WHAT THE PAPER DOES NOT SAY  -- all of this is [MISSING]
    * how Psi is obtained (fixed analytic basis? learned? learned from what?);
    * whether Psi is shared across heads / layers / modalities;
    * the sparsity level s = ||alpha_i||_0 actually observed;
    * any evidence that real context vectors are sparse in any Psi.

Note that a SQUARE Psi in R^{d_k x d_k} is a complete basis, so C_i = Psi alpha_i
has a unique exact solution alpha_i = Psi^{-1} C_i for every C_i whatsoever.
Sparsity of that solution is therefore an empirical property of the data, not a
consequence of the model -- it is an assumption that has to be measured. The
function ``fit_dictionary`` and ``utils.metrics.sparsity_profile`` exist so the
notebook measures it instead of assuming it.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

DictionaryName = Literal["identity", "dct", "random_orthogonal", "overcomplete", "learnable"]


# --------------------------------------------------------------------------- #
# Analytic bases
# --------------------------------------------------------------------------- #
def dct_matrix(d: int, device: torch.device | str = "cpu",
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II synthesis matrix, [d, d].

    Columns are cosine atoms. Chosen because the paper motivates sparsity by
    analogy with JPEG ("natural images are known to be sparse in wavelet, DCT,
    or learned convolutional bases"), so the DCT is the basis the paper's own
    argument points at -- though the paper never states which basis it used.
    """
    n = torch.arange(d, device=device, dtype=dtype).unsqueeze(1)   # rows
    k = torch.arange(d, device=device, dtype=dtype).unsqueeze(0)   # cols
    psi = torch.cos(math.pi / d * (n + 0.5) * k)
    psi[:, 0] *= 1.0 / math.sqrt(2.0)
    return psi * math.sqrt(2.0 / d)


def random_orthogonal_dictionary(d: int, device: torch.device | str = "cpu",
                                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(d, d, device=device, dtype=dtype))
    return q


def overcomplete_dictionary(d: int, k_atoms: int, device: torch.device | str = "cpu",
                            dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Random [d, k] dictionary with unit-norm columns (k > d = overcomplete).

    [CHOICE] The paper's Psi is square. Overcompleteness is the usual setting in
    sparse coding and is offered as an ablation, clearly outside the paper.
    """
    psi = torch.randn(d, k_atoms, device=device, dtype=dtype)
    return psi / psi.norm(dim=0, keepdim=True).clamp_min(1e-12)


def make_dictionary(kind: DictionaryName, d: int, k_atoms: int | None = None,
                    device: torch.device | str = "cpu",
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    k_atoms = k_atoms or d
    if kind == "identity":
        return torch.eye(d, device=device, dtype=dtype)
    if kind == "dct":
        if k_atoms != d:
            raise ValueError("the DCT basis is square; set dict_atoms=None or d_k")
        return dct_matrix(d, device, dtype)
    if kind == "random_orthogonal":
        if k_atoms != d:
            raise ValueError("a random orthogonal basis is square")
        return random_orthogonal_dictionary(d, device, dtype)
    if kind in ("overcomplete", "learnable"):
        return overcomplete_dictionary(d, k_atoms, device, dtype)
    raise ValueError(f"unknown dictionary kind '{kind}'")


# --------------------------------------------------------------------------- #
# Dictionary learning (for testing the paper's sparsity assumption)
# --------------------------------------------------------------------------- #
def fit_dictionary(X: torch.Tensor, k_atoms: int, sparsity_lambda: float = 0.1,
                   n_outer: int = 30, n_inner: int = 30, lr: float = 1e-2,
                   verbose: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Learn Psi and codes A from data X [N, d] by alternating minimisation of
        1/2 ||X - A Psi^T||_F^2 + lambda ||A||_1,
    with unit-norm dictionary columns re-imposed after every update.

    This is a compact stand-in for K-SVD / online dictionary learning. Its only
    purpose in this project is to answer the empirical question the paper leaves
    open: *can* context vectors be represented sparsely in a learned basis, and
    how sparse are they really? It is NOT a component of the paper's method.

    Returns (Psi [d, k], A [N, k]).
    """
    from .ista import ista  # local import, avoids cycle

    N, d = X.shape
    psi = overcomplete_dictionary(d, k_atoms, X.device, X.dtype)
    codes = torch.zeros(N, k_atoms, device=X.device, dtype=X.dtype)

    for outer in range(n_outer):
        # --- sparse coding step: fix Psi, solve for A with ISTA --------------
        out = ista(psi, X, lam=sparsity_lambda, n_iters=n_inner, alpha_init=codes)
        codes = out["alpha"]
        # --- dictionary step: fix A, gradient descent on Psi ----------------
        psi = psi.detach().requires_grad_(True)
        for _ in range(10):
            loss = 0.5 * ((codes @ psi.T) - X).pow(2).sum()
            grad, = torch.autograd.grad(loss, psi)
            with torch.no_grad():
                psi = psi - lr * grad / max(1.0, grad.norm().item())
                psi = psi / psi.norm(dim=0, keepdim=True).clamp_min(1e-12)
            psi = psi.detach().requires_grad_(True)
        psi = psi.detach()
        if verbose and outer % 10 == 0:
            rec = codes @ psi.T
            print(f"  [fit_dictionary] outer {outer:3d}  "
                  f"rel_err={(rec - X).norm() / X.norm():.4f}  "
                  f"nnz/row={(codes.abs() > 1e-3).float().sum(1).mean():.1f}")
    return psi.detach(), codes.detach()


# --------------------------------------------------------------------------- #
# Module wrapper
# --------------------------------------------------------------------------- #
class Dictionary(nn.Module):
    """Holds Psi as a buffer (fixed basis) or Parameter (learned end-to-end)."""

    def __init__(self, kind: DictionaryName, d: int, k_atoms: int | None = None,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.kind, self.d = kind, d
        self.k_atoms = k_atoms or d
        psi = make_dictionary(kind, d, self.k_atoms, device, dtype)
        if kind == "learnable":
            self.psi = nn.Parameter(psi)
        else:
            self.register_buffer("psi", psi)

    def synthesize(self, alpha: torch.Tensor) -> torch.Tensor:
        """C_hat = Psi alpha, applied to the last dimension: [..., k] -> [..., d]."""
        return alpha @ self.psi.T

    def analyze(self, x: torch.Tensor) -> torch.Tensor:
        """Psi^T x -- the analysis (adjoint) coefficients, exact only if Psi is orthonormal."""
        return x @ self.psi

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        return self.synthesize(alpha)

    def extra_repr(self) -> str:
        return f"kind={self.kind}, d={self.d}, atoms={self.k_atoms}"
