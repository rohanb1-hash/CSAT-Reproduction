"""Unit tests for measurement matrices, dictionaries and the bridge dimension check."""

from __future__ import annotations

import pytest
import torch

from csat.models.bridges import build_bridge, check_paper_equation, decode_context
from csat.models.dictionary import dct_matrix, make_dictionary
from csat.models.measurement import (
    empirical_rip_constant,
    make_measurement_matrix,
    mutual_coherence,
)


# --------------------------------------------------------------------------- #
# Measurement matrices
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ensemble", ["gaussian", "rademacher", "orthogonal", "hadamard"])
def test_ensembles_have_the_right_shape(ensemble):
    phi = make_measurement_matrix(16, 64, ensemble)
    assert phi.shape == (16, 64)
    assert torch.isfinite(phi).all()


def test_gaussian_normalisation_gives_identity_in_expectation():
    """Establishes: with entries N(0,1/m), E[Phi^T Phi] = I_n. This is the
    property the 1/sqrt(m) convention is chosen for, and it is what makes the
    softmax-free control in test_compressed_attention unbiased."""
    torch.manual_seed(0)
    n, m, trials = 32, 16, 200
    acc = torch.zeros(n, n)
    for _ in range(trials):
        phi = make_measurement_matrix(m, n, "gaussian")
        acc += phi.T @ phi
    acc /= trials
    off_diag = acc - torch.diag(torch.diag(acc))
    assert abs(torch.diag(acc).mean().item() - 1.0) < 0.05
    assert off_diag.abs().max().item() < 0.15


def test_rademacher_entries_are_two_valued():
    phi = make_measurement_matrix(8, 32, "rademacher")
    assert torch.unique(phi.abs()).numel() == 1


def test_orthogonal_rows_are_orthogonal():
    phi = make_measurement_matrix(16, 64, "orthogonal", normalize=False)
    gram = phi @ phi.T
    assert torch.allclose(gram, torch.eye(16), atol=1e-4)


def test_hadamard_requires_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        make_measurement_matrix(8, 48, "hadamard")


def test_m_greater_than_n_rejected():
    with pytest.raises(ValueError, match="m <= n"):
        make_measurement_matrix(64, 16, "gaussian")


def test_coherence_decreases_as_measurements_grow():
    """Establishes: more measurements -> lower coherence -> better recovery
    conditions. The paper asserts 'low coherence with sparse bases' without
    measuring it; this shows the quantity is computable."""
    torch.manual_seed(0)
    c_small = mutual_coherence(make_measurement_matrix(16, 128, "gaussian"))
    c_large = mutual_coherence(make_measurement_matrix(96, 128, "gaussian"))
    assert c_large < c_small, (c_small, c_large)


def test_empirical_rip_is_reported_as_a_lower_bound():
    """Establishes: the RIP estimate is Monte-Carlo and therefore a LOWER bound;
    it must not be presented as delta_s itself."""
    phi = make_measurement_matrix(64, 128, "gaussian")
    out = empirical_rip_constant(phi, sparsity=4, n_trials=500)
    assert "delta_s_lower_bound" in out
    assert out["min_ratio"] <= out["mean_ratio"] <= out["max_ratio"]


# --------------------------------------------------------------------------- #
# Dictionaries
# --------------------------------------------------------------------------- #
def test_dct_is_orthonormal():
    psi = dct_matrix(32)
    assert torch.allclose(psi.T @ psi, torch.eye(32), atol=1e-5)


def test_overcomplete_columns_are_unit_norm():
    psi = make_dictionary("overcomplete", 32, 96)
    assert psi.shape == (32, 96)
    assert torch.allclose(psi.norm(dim=0), torch.ones(96), atol=1e-5)


def test_square_dictionary_admits_an_exact_representation_for_any_vector():
    """Establishes the point made in models/dictionary.py: with a square,
    invertible Psi, C = Psi alpha ALWAYS has an exact solution. So sparsity is
    an empirical property of the data, never a consequence of the model."""
    psi = dct_matrix(32)
    c = torch.randn(4, 32)
    alpha = c @ torch.linalg.inv(psi).T
    assert torch.allclose(alpha @ psi.T, c, atol=1e-4)
    nnz = (alpha.abs() > 1e-3).float().sum(dim=1).mean().item()
    assert nnz > 20, "a generic vector is NOT sparse in the DCT basis"


# --------------------------------------------------------------------------- #
# The paper's equation
# --------------------------------------------------------------------------- #
def test_paper_equation_does_not_type_check():
    """Establishes, by arithmetic on the paper's own declared shapes, that
    Z_i = Phi_V Psi alpha_i is ill-formed whenever m << n."""
    report = check_paper_equation(m=64, n=4096, d_k=64)
    assert report["equation well-formed?"] is False
    assert report["inner dims agree (n == d_k)?"] is False


def test_paper_equation_only_closes_when_m_equals_n_equals_dk():
    report = check_paper_equation(m=64, n=64, d_k=64)
    assert report["equation well-formed?"] is True, \
        "the equation closes only in the degenerate case m = n = d_k (no compression)"


@pytest.mark.parametrize("name,kwargs", [
    ("denoise", {}),
    ("feature_cs", {"p_features": 32}),
    ("token_cs", {"n": 128, "m": 32}),
])
def test_every_bridge_is_dimensionally_consistent(name, kwargs):
    """Establishes: each of our three readings produces a well-formed y = A alpha."""
    spec = build_bridge(name, d_k=64, **kwargs)
    p, k = spec.A.shape
    d = spec.psi.shape[0]
    assert spec.psi.shape[1] == k, "Psi must map codes to signal space"
    y = torch.randn(10, p)
    out = decode_context(y, spec, lam=0.05, n_iters=5)
    assert out["alpha"].shape == (10, k)
    assert out["c_hat"].shape == (10, d)


def test_only_the_denoise_bridge_is_not_compressed_sensing():
    """Establishes the honest labelling: 'denoise' is square (no undersampling),
    the other two are genuinely underdetermined."""
    assert build_bridge("denoise", d_k=64).is_underdetermined is False
    assert build_bridge("feature_cs", d_k=64, p_features=32).is_underdetermined is True
    assert build_bridge("token_cs", d_k=64, n=128, m=32).is_underdetermined is True
