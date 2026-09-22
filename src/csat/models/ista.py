"""ISTA / FISTA -- the analytic sparse decoder the paper names.

OBJECTIVE (the standard LASSO form; the paper writes the constrained basis-pursuit
problem  min ||alpha||_1  s.t.  Z_i = Phi Psi alpha  and then says it uses ISTA,
which solves the unconstrained relaxation):

    F(alpha) = 1/2 ||y - A alpha||_2^2  +  lambda ||alpha||_1

UPDATE
    alpha_{t+1} = S_theta( alpha_t - eta A^T (A alpha_t - y) ),    theta = eta * lambda

    S_theta(x) = sign(x) * max(|x| - theta, 0)        (soft thresholding)

CONVERGENCE
    The gradient of the smooth part has Lipschitz constant L = sigma_max(A)^2.
    Any eta <= 1/L gives monotone non-increasing F; eta = 1/L is the standard
    choice and is what ``step_size=None`` computes.

WHAT COMES FROM THE PAPER
    * the name ISTA and the role of the decoder;
    * the update as quoted above (the paper writes the LISTA form of it).
WHAT DOES NOT  -- all [MISSING]
    * lambda, eta, the number of iterations, the stopping rule, the sparsity
      level s, and what A actually is in the paper's pipeline (see bridges.py).

BATCHING
    A is shared, [p, k]. Y is [N, p], one measurement vector per row. Everything
    below is a batched matmul, so n tokens are decoded in parallel -- this is
    what makes decoding tractable on a GPU, and it is what the FLOP counter in
    utils/flops.py assumes.
"""

from __future__ import annotations

import torch

from csat.utils.tensor_utils import spectral_norm


def soft_threshold(x: torch.Tensor, theta: float | torch.Tensor) -> torch.Tensor:
    """S_theta(x) = sign(x) * max(|x| - theta, 0), elementwise.

    This is the proximal operator of theta*||.||_1 and the only non-linearity in
    ISTA. ``theta`` may be a scalar or broadcastable tensor (LISTA learns a
    per-coordinate threshold, so the tensor case matters).
    """
    return torch.sign(x) * torch.clamp(x.abs() - theta, min=0.0)


def lasso_objective(A: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor,
                    lam: float) -> torch.Tensor:
    """F(alpha) per row: [N] tensor. Used to verify monotone descent in tests."""
    residual = alpha @ A.T - y                      # [N, p]
    return 0.5 * residual.pow(2).sum(dim=-1) + lam * alpha.abs().sum(dim=-1)


def estimate_step_size(A: torch.Tensor, n_power_iters: int = 100) -> float:
    """eta = 1 / L with L = sigma_max(A)^2."""
    sigma = spectral_norm(A, n_power_iters)
    return 1.0 / max(sigma ** 2, 1e-12)


@torch.no_grad()
def ista(A: torch.Tensor, y: torch.Tensor, lam: float = 0.1, n_iters: int = 100,
         step_size: float | None = None, alpha_init: torch.Tensor | None = None,
         track_objective: bool = False, tol: float = 0.0,
         use_fista: bool = False) -> dict[str, object]:
    """Solve min_alpha 1/2||y - A alpha||^2 + lam||alpha||_1 for every row of y.

    Args:
        A: [p, k] measurement-times-dictionary operator.
        y: [N, p] measurements (one per row).
        lam: l1 weight.
        n_iters: fixed iteration budget (no early stop when tol = 0, which keeps
            runtime comparisons against LISTA honest).
        step_size: eta. None -> 1/L via power iteration.
        alpha_init: [N, k] warm start, default zeros.
        track_objective: record F(alpha) each iteration (mean over rows).
        tol: stop when the mean relative change in alpha falls below tol.
        use_fista: Nesterov momentum (FISTA). [CHOICE] Not in the paper; included
            because it converges in O(1/t^2) vs ISTA's O(1/t) and makes the
            "iterations needed" discussion concrete.

    Returns dict with 'alpha' [N,k], 'reconstruction' [N,p] (= A alpha),
    'objective' (list), 'n_iters_run', 'step_size'.

    This function is intentionally decorated with ``@torch.no_grad()``: classical
    ISTA is a gradient-free fixed-point iteration with respect to the *network*,
    and running it under autograd would build an n_iters-deep graph. LISTA
    (models/lista.py) is the differentiable counterpart.
    """
    assert A.dim() == 2 and y.dim() == 2, "A must be [p,k] and y must be [N,p]"
    assert A.shape[0] == y.shape[1], \
        f"shape mismatch: A is {tuple(A.shape)} (p,k) but y is {tuple(y.shape)} (N,p)"

    p, k = A.shape
    eta = step_size if step_size is not None else estimate_step_size(A)
    theta = eta * lam

    alpha = torch.zeros(y.shape[0], k, device=y.device, dtype=y.dtype) \
        if alpha_init is None else alpha_init.clone()
    z, t_k = alpha.clone(), 1.0
    objective = []

    n_run = 0
    for it in range(n_iters):
        prev = alpha
        point = z if use_fista else alpha
        grad = (point @ A.T - y) @ A                 # A^T (A alpha - y), as [N,k]
        alpha = soft_threshold(point - eta * grad, theta)

        if use_fista:
            t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_k ** 2) ** 0.5)
            z = alpha + ((t_k - 1.0) / t_next) * (alpha - prev)
            t_k = t_next

        n_run = it + 1
        if track_objective:
            objective.append(lasso_objective(A, y, alpha, lam).mean().item())
        if tol > 0:
            rel = (alpha - prev).norm() / prev.norm().clamp_min(1e-12)
            if rel.item() < tol:
                break

    return {
        "alpha": alpha,
        "reconstruction": alpha @ A.T,
        "objective": objective,
        "n_iters_run": n_run,
        "step_size": eta,
        "threshold": theta,
    }


def fista(A: torch.Tensor, y: torch.Tensor, **kwargs) -> dict[str, object]:
    """Convenience alias for ``ista(..., use_fista=True)``."""
    kwargs["use_fista"] = True
    return ista(A, y, **kwargs)


@torch.no_grad()
def debias(A: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor,
           thresh: float = 1e-3) -> torch.Tensor:
    """Least-squares refit on the support detected by ISTA.

    Soft thresholding shrinks every surviving coefficient by theta, so the l1
    solution is biased towards zero even when the support is exactly right.
    Refitting on the support removes that bias. This is standard practice
    (LASSO debiasing) and is NOT part of the paper; it is reported separately in
    the experiments so the raw ISTA number is never quietly improved.
    """
    keep = alpha.abs() > thresh * alpha.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    out = torch.zeros_like(alpha)
    for i in range(alpha.shape[0]):
        idx = keep[i].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        sol = torch.linalg.lstsq(A[:, idx], y[i].unsqueeze(-1)).solution.squeeze(-1)
        out[i, idx] = sol
    return out


class ISTADecoder(torch.nn.Module):
    """nn.Module wrapper so ISTA and LISTA are interchangeable in a CSAT block.

    Holds A = Phi_eff @ Psi. Its forward returns the reconstructed signal
    C_hat = Psi alpha_hat, matching the paper's C_hat_i = Psi alpha_hat_i.
    """

    def __init__(self, A: torch.Tensor, psi: torch.Tensor, lam: float = 0.1,
                 n_iters: int = 50, step_size: float | None = None,
                 use_fista: bool = False):
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("psi", psi)
        self.lam, self.n_iters, self.use_fista = lam, n_iters, use_fista
        self.step_size = step_size if step_size is not None else estimate_step_size(A)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: [..., p] -> C_hat: [..., d]. Leading dims are flattened and restored."""
        lead, p = y.shape[:-1], y.shape[-1]
        out = ista(self.A, y.reshape(-1, p), lam=self.lam, n_iters=self.n_iters,
                   step_size=self.step_size, use_fista=self.use_fista)
        c_hat = out["alpha"] @ self.psi.T
        return c_hat.reshape(*lead, self.psi.shape[0])

    def extra_repr(self) -> str:
        return (f"A={tuple(self.A.shape)}, psi={tuple(self.psi.shape)}, "
                f"lam={self.lam}, n_iters={self.n_iters}, fista={self.use_fista}")
