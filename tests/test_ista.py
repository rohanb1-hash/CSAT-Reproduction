"""Unit tests for the sparse solvers: soft thresholding, ISTA, FISTA, OMP."""

from __future__ import annotations

import torch

from csat.models.ista import (
    debias,
    estimate_step_size,
    fista,
    ista,
    lasso_objective,
    soft_threshold,
)
from csat.models.omp import omp
from csat.utils.metrics import relative_l2, support_f1
from csat.utils.tensor_utils import make_sparse_signals


def _cs_problem(p=64, k=128, s=8, n=128, seed=0):
    """A well-posed CS instance: A Gaussian with unit-norm columns, alpha s-sparse."""
    torch.manual_seed(seed)
    A = torch.randn(p, k) / (p ** 0.5)
    A = A / A.norm(dim=0, keepdim=True)
    alpha = make_sparse_signals(n, k, s)
    return A, alpha, alpha @ A.T


# --------------------------------------------------------------------------- #
def test_soft_threshold_values():
    """Establishes: S_theta is the exact proximal operator of theta*|.|_1."""
    x = torch.tensor([-3.0, -1.0, -0.4, 0.0, 0.4, 1.0, 3.0])
    out = soft_threshold(x, 1.0)
    expected = torch.tensor([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    assert torch.allclose(out, expected, atol=1e-6)


def test_soft_threshold_is_shrinkage_not_clipping():
    """Establishes: surviving coefficients are SHRUNK by theta (the l1 bias that
    motivates the debias() refit), not merely zeroed below a cut-off."""
    x = torch.tensor([5.0])
    assert torch.allclose(soft_threshold(x, 1.0), torch.tensor([4.0]))


def test_soft_threshold_supports_vector_threshold():
    """Establishes: per-coordinate thresholds work -- required by LISTA."""
    x = torch.tensor([[2.0, 2.0]])
    theta = torch.tensor([0.5, 1.5])
    assert torch.allclose(soft_threshold(x, theta), torch.tensor([[1.5, 0.5]]))


def test_ista_shapes():
    A, alpha, y = _cs_problem()
    out = ista(A, y, lam=0.01, n_iters=20)
    assert out["alpha"].shape == alpha.shape
    assert out["reconstruction"].shape == y.shape
    assert out["n_iters_run"] == 20


def test_ista_objective_decreases_monotonically():
    """Establishes: with eta = 1/L the objective is non-increasing, which is the
    theoretical guarantee. A wrong step size shows up here first."""
    A, _, y = _cs_problem()
    out = ista(A, y, lam=0.01, n_iters=100, track_objective=True)
    obj = out["objective"]
    assert len(obj) == 100
    assert all(obj[i + 1] <= obj[i] + 1e-6 for i in range(len(obj) - 1)), \
        "ISTA objective must not increase with eta <= 1/L"
    assert obj[-1] < obj[0]


def test_ista_step_size_matches_one_over_lipschitz():
    """Establishes: estimate_step_size returns 1/sigma_max(A)^2."""
    A, _, _ = _cs_problem()
    eta = estimate_step_size(A)
    sigma = torch.linalg.matrix_norm(A, 2).item()
    assert abs(eta - 1.0 / sigma ** 2) / (1.0 / sigma ** 2) < 1e-2


def test_ista_recovers_a_sparse_signal_given_enough_iterations():
    """Establishes recovery EMPIRICALLY rather than assuming it. We assert a
    modest threshold, and the accompanying experiment sweeps the regime where
    recovery fails -- the paper claims exact recovery under RIP but reports no
    recovery experiment at all."""
    A, alpha, y = _cs_problem(p=64, k=128, s=8)
    out = ista(A, y, lam=0.005, n_iters=3000)
    err = relative_l2(out["alpha"], alpha)
    assert err < 0.05, f"relative error {err:.4f} too high for p=64,k=128,s=8"


def test_fista_is_faster_than_ista_at_equal_iterations():
    """Establishes: the accelerated variant reaches a lower error for the same
    budget -- relevant because decoding cost is proportional to iterations."""
    A, alpha, y = _cs_problem(p=64, k=128, s=8)
    e_ista = relative_l2(ista(A, y, lam=0.005, n_iters=200)["alpha"], alpha)
    e_fista = relative_l2(fista(A, y, lam=0.005, n_iters=200)["alpha"], alpha)
    assert e_fista < e_ista, (e_ista, e_fista)


def test_zero_lambda_reduces_to_gradient_descent_on_least_squares():
    """Establishes: with lam = 0 the threshold vanishes and ISTA becomes plain
    gradient descent, so the residual must fall."""
    A, _, y = _cs_problem()
    out = ista(A, y, lam=0.0, n_iters=200)
    assert relative_l2(out["reconstruction"], y) < 0.1


def test_larger_lambda_gives_sparser_solutions():
    """Establishes: lambda controls sparsity in the expected direction."""
    A, _, y = _cs_problem()
    nnz = []
    for lam in (0.001, 0.05, 0.3):
        a = ista(A, y, lam=lam, n_iters=300)["alpha"]
        nnz.append((a.abs() > 1e-6).float().sum(dim=1).mean().item())
    assert nnz[0] > nnz[1] > nnz[2], nnz


def test_debias_improves_coefficient_error():
    """Establishes: soft thresholding biases coefficients toward zero; refitting
    on the support removes that bias."""
    A, alpha, y = _cs_problem(p=64, k=128, s=8)
    out = ista(A, y, lam=0.02, n_iters=1000)
    raw = relative_l2(out["alpha"], alpha)
    fixed = relative_l2(debias(A, y, out["alpha"]), alpha)
    assert fixed < raw, (raw, fixed)


def test_batch_independence():
    """Establishes: rows are decoded independently -- row i of a batched solve
    equals the solve of row i alone."""
    A, _, y = _cs_problem(n=16)
    batched = ista(A, y, lam=0.01, n_iters=50)["alpha"]
    single = ista(A, y[3:4], lam=0.01, n_iters=50)["alpha"]
    assert torch.allclose(batched[3:4], single, atol=1e-5)


def test_ista_is_gradient_free():
    """Establishes: the classical solver builds no autograd graph, so it cannot
    be trained end to end -- the reason LISTA exists."""
    A, _, y = _cs_problem()
    y = y.clone().requires_grad_(True)
    out = ista(A, y, lam=0.01, n_iters=10)
    assert not out["alpha"].requires_grad


def test_omp_recovers_support():
    """Establishes: given the true sparsity level, OMP finds the support."""
    A, alpha, y = _cs_problem(p=64, k=128, s=8)
    out = omp(A, y, sparsity=8)
    assert support_f1(out["alpha"], alpha) > 0.9


def test_shape_mismatch_raises_informative_error():
    A, _, y = _cs_problem()
    try:
        ista(A, y[:, :10], lam=0.01, n_iters=5)
    except AssertionError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("expected an informative shape error")


def test_lasso_objective_is_per_row():
    A, alpha, y = _cs_problem(n=7)
    obj = lasso_objective(A, y, alpha, lam=0.1)
    assert obj.shape == (7,)
    assert (obj >= 0).all()
