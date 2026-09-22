"""Orthogonal Matching Pursuit -- the other decoder the paper names.

The paper: "CSAT instead leverages fast approximate solvers, such as Iterative
Shrinkage-Thresholding Algorithm (ISTA) or its learned variant LISTA" and
earlier "using convex optimization solvers such as ISTA or OMP". OMP is a greedy
alternative that takes the sparsity level s as its input instead of a
regularisation weight lambda -- which is useful here precisely because it makes
the assumed sparsity explicit rather than implicit in a lambda the paper never
reports.

ALGORITHM (batched over N measurement vectors)
    r <- y ; S <- {}
    repeat s times:
        j*  <- argmax_j |<a_j, r>| / ||a_j||        (most correlated atom)
        S   <- S union {j*}
        x_S <- argmin ||y - A_S x||^2                (least squares on the support)
        r   <- y - A_S x_S
"""

from __future__ import annotations

import torch


@torch.no_grad()
def omp(A: torch.Tensor, y: torch.Tensor, sparsity: int,
        tol: float = 1e-8) -> dict[str, torch.Tensor]:
    """Batched OMP.

    Args:
        A: [p, k] dictionary/measurement operator (columns = atoms).
        y: [N, p] measurements.
        sparsity: number of atoms to select (s).
        tol: stop a row early once its residual norm falls below tol (applied as
            a mask, so the batch still runs for s steps -- GPU-friendly).

    Returns {'alpha': [N, k], 'reconstruction': [N, p], 'support': [N, s]}.
    """
    p, k = A.shape
    N = y.shape[0]
    assert y.shape[1] == p, f"A is [{p},{k}] so y must be [N,{p}], got {tuple(y.shape)}"
    sparsity = min(sparsity, p, k)

    A_n = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)
    residual = y.clone()
    support = torch.zeros(N, sparsity, dtype=torch.long, device=y.device)
    alpha = torch.zeros(N, k, device=y.device, dtype=y.dtype)

    chosen_mask = torch.zeros(N, k, dtype=torch.bool, device=y.device)
    for step in range(sparsity):
        corr = (residual @ A_n).abs()                     # [N, k]
        corr = corr.masked_fill(chosen_mask, -1.0)        # never re-select an atom
        j = corr.argmax(dim=1)                            # [N]
        support[:, step] = j
        chosen_mask.scatter_(1, j.unsqueeze(1), True)

        # Least squares on the current support, one small solve per row.
        idx = support[:, :step + 1]                       # [N, step+1]
        A_sub = A[:, idx].permute(1, 0, 2)                # [N, p, step+1]
        sol = torch.linalg.lstsq(A_sub, y.unsqueeze(-1)).solution  # [N, step+1, 1]
        alpha.zero_()
        alpha.scatter_(1, idx, sol.squeeze(-1))
        residual = y - alpha @ A.T
        if residual.norm(dim=1).max() < tol:
            break

    return {"alpha": alpha, "reconstruction": alpha @ A.T, "support": support}
