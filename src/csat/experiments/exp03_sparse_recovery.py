"""EXPERIMENT 03 -- Sparse recovery with ISTA / FISTA / OMP (a genuine CS problem).

QUESTION
    Does our solver stack actually recover sparse signals, and in which regime?
    The paper states that "under standard RIP conditions, this formulation
    guarantees exact recovery when alpha_i is sufficiently sparse" but reports
    no recovery experiment, no sparsity level, no lambda and no iteration count.

SETUP (self-contained, fully specified -- this is OUR experiment, not the paper's)
    alpha in R^k, exactly s-sparse, random support and signs.
    A = Phi Psi with Phi in R^{p x d} Gaussian and Psi in R^{d x k}.
    y = A alpha (optionally + noise).
    Recover alpha_hat and compare.

SWEEPS
    * measurement ratio p/k, at fixed sparsity -> the phase-transition curve;
    * sparsity s, at fixed p -> where recovery breaks down;
    * iteration budget -> the accuracy/compute trade-off that the paper's
      efficiency argument depends on but never quantifies;
    * noise level -> stability.

STATUS: preliminary (Phase 4). Establishes that the SOLVERS work; it does NOT
establish anything about the paper's attention pipeline, because the paper's
Z_i is not a measurement of C_i (see models/bridges.py).
"""

from __future__ import annotations

import json
import os

import torch

from csat.models.ista import debias, fista, ista
from csat.models.omp import omp
from csat.utils.metrics import relative_l2, support_f1
from csat.utils.tensor_utils import add_noise, make_sparse_signals


def _make_problem(k: int, p: int, s: int, n_signals: int, noise_std: float,
                  device: torch.device):
    A = torch.randn(p, k, device=device) / (p ** 0.5)
    A = A / A.norm(dim=0, keepdim=True)
    alpha = make_sparse_signals(n_signals, k, s, device=device)
    y = add_noise(alpha @ A.T, noise_std)
    return A, alpha, y


def sweep_measurements(k: int = 128, s: int = 8, n_signals: int = 256,
                       p_values: list[int] = (16, 24, 32, 48, 64, 96, 128),
                       lam: float = 0.005, n_iters: int = 1000,
                       device: torch.device | str = "cpu") -> list[dict]:
    """Recovery error vs number of measurements -- the CS phase transition."""
    device = torch.device(device)
    rows = []
    for p in p_values:
        A, alpha, y = _make_problem(k, p, s, n_signals, 0.0, device)
        r_ista = ista(A, y, lam=lam, n_iters=n_iters)
        r_fista = fista(A, y, lam=lam, n_iters=n_iters)
        r_omp = omp(A, y, sparsity=s)
        rows.append({
            "k_atoms": k, "sparsity_s": s, "p_measurements": p,
            "measurement_ratio_p_over_k": p / k,
            "oversampling_p_over_s": p / s, "n_iters": n_iters, "lam": lam,
            "ista_rel_l2": relative_l2(r_ista["alpha"], alpha),
            "ista_support_f1": support_f1(r_ista["alpha"], alpha),
            "ista_debiased_rel_l2": relative_l2(debias(A, y, r_ista["alpha"]), alpha),
            "fista_rel_l2": relative_l2(r_fista["alpha"], alpha),
            "fista_support_f1": support_f1(r_fista["alpha"], alpha),
            "omp_rel_l2": relative_l2(r_omp["alpha"], alpha),
            "omp_support_f1": support_f1(r_omp["alpha"], alpha),
        })
    return rows


def sweep_sparsity(k: int = 128, p: int = 48, n_signals: int = 256,
                   s_values: list[int] = (2, 4, 8, 12, 16, 24, 32),
                   lam: float = 0.005, n_iters: int = 1000,
                   device: torch.device | str = "cpu") -> list[dict]:
    """Recovery error vs sparsity level at a fixed measurement budget."""
    device = torch.device(device)
    rows = []
    for s in s_values:
        A, alpha, y = _make_problem(k, p, s, n_signals, 0.0, device)
        r_ista = ista(A, y, lam=lam, n_iters=n_iters)
        r_fista = fista(A, y, lam=lam, n_iters=n_iters)
        r_omp = omp(A, y, sparsity=s)
        rows.append({
            "k_atoms": k, "p_measurements": p, "sparsity_s": s,
            "sparsity_fraction": s / k, "n_iters": n_iters,
            "ista_rel_l2": relative_l2(r_ista["alpha"], alpha),
            "fista_rel_l2": relative_l2(r_fista["alpha"], alpha),
            "omp_rel_l2": relative_l2(r_omp["alpha"], alpha),
            "fista_support_f1": support_f1(r_fista["alpha"], alpha),
            "omp_support_f1": support_f1(r_omp["alpha"], alpha),
        })
    return rows


def sweep_iterations(k: int = 128, p: int = 64, s: int = 8, n_signals: int = 256,
                     iter_values: list[int] = (10, 25, 50, 100, 250, 500, 1000, 2000),
                     lam: float = 0.005,
                     device: torch.device | str = "cpu") -> list[dict]:
    """Accuracy vs iteration budget -- the decoder's compute/accuracy trade-off.

    This is the sweep that bears directly on the paper's efficiency claim: the
    decoding cost is linear in the iteration count, and the paper reports
    neither the count nor the resulting error.
    """
    device = torch.device(device)
    A, alpha, y = _make_problem(k, p, s, n_signals, 0.0, device)
    rows = []
    for it in iter_values:
        r_ista = ista(A, y, lam=lam, n_iters=it)
        r_fista = fista(A, y, lam=lam, n_iters=it)
        rows.append({
            "n_iters": it, "k_atoms": k, "p_measurements": p, "sparsity_s": s,
            "ista_rel_l2": relative_l2(r_ista["alpha"], alpha),
            "fista_rel_l2": relative_l2(r_fista["alpha"], alpha),
            "ista_support_f1": support_f1(r_ista["alpha"], alpha),
            "fista_support_f1": support_f1(r_fista["alpha"], alpha),
        })
    return rows


def sweep_noise(k: int = 128, p: int = 64, s: int = 8, n_signals: int = 256,
                noise_values: list[float] = (0.0, 0.01, 0.05, 0.1, 0.2),
                lam: float = 0.01, n_iters: int = 1000,
                device: torch.device | str = "cpu") -> list[dict]:
    """Stability under measurement noise (the 'stable recovery' regime of CS)."""
    device = torch.device(device)
    rows = []
    for noise in noise_values:
        A, alpha, y = _make_problem(k, p, s, n_signals, noise, device)
        r = fista(A, y, lam=lam, n_iters=n_iters)
        rows.append({
            "noise_std": noise, "k_atoms": k, "p_measurements": p, "sparsity_s": s,
            "fista_rel_l2": relative_l2(r["alpha"], alpha),
            "fista_support_f1": support_f1(r["alpha"], alpha),
            "signal_rel_l2": relative_l2(r["reconstruction"], y),
        })
    return rows


def run(device: torch.device | str = "cpu", quick: bool = True,
        save_dir: str | None = None) -> dict[str, list[dict]]:
    n_sig = 128 if quick else 512
    iters = 600 if quick else 2000
    out = {
        "measurements": sweep_measurements(n_signals=n_sig, n_iters=iters, device=device),
        "sparsity": sweep_sparsity(n_signals=n_sig, n_iters=iters, device=device),
        "iterations": sweep_iterations(n_signals=n_sig, device=device),
        "noise": sweep_noise(n_signals=n_sig, n_iters=iters, device=device),
    }
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "exp03_sparse_recovery.json"), "w") as f:
            json.dump(out, f, indent=2)
    return out
