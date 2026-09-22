"""Unit tests for CSAT compressed attention.

Beyond shape checking, these tests pin down the two structural facts the paper
does not state: the compressed attention output is NOT a convex combination of
value vectors, and causal masking is undefined after token-axis compression.
"""

from __future__ import annotations

import pytest
import torch

from csat.models.compressed_attention import CompressedAttention, LinearCompressedAttention
from csat.models.measurement import make_measurement_matrix
from csat.utils.tensor_utils import make_qkv


def test_compressed_shapes():
    """Establishes: A~ is [B,H,N,M] and Z is [B,H,N,D]; the n x n matrix is never built."""
    q, k, v = make_qkv(2, 4, 64, 16)
    attn = CompressedAttention(seq_len=64, m=16, d_k=16, n_heads=4)
    z, a = attn(q, k, v, return_weights=True)
    assert a.shape == (2, 4, 64, 16), "A~ must be n x m, not n x n"
    assert z.shape == (2, 4, 64, 16)
    assert attn.compression_ratio() == 4.0


def test_compressed_rows_sum_to_one_but_output_is_not_convex():
    """Establishes BOTH halves of the key structural fact:
    (a) A~ is row-stochastic over the m compressed slots;
    (b) the effective map M_eff = A~ Phi_V is NOT row-stochastic and has
        negative entries, so Z is not a weighted average of value vectors.
    """
    q, k, v = make_qkv(1, 1, 64, 16)
    attn = CompressedAttention(seq_len=64, m=16, d_k=16)
    _, a = attn(q, k, v, return_weights=True)
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
    assert (a >= 0).all()

    m_eff = attn.effective_attention_matrix(q, k)
    assert m_eff.shape == (1, 1, 64, 64)
    assert (m_eff < 0).float().mean() > 0.1, \
        "with a Gaussian Phi_V, roughly half of M_eff should be negative"
    row_sums = m_eff.sum(-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2), \
        "M_eff rows are not expected to sum to 1"


def test_z_equals_effective_matrix_times_v():
    """Establishes: Z = (A~ Phi_V) V exactly, which is what licenses reading
    M_eff as 'the attention matrix CSAT actually applies'."""
    q, k, v = make_qkv(1, 2, 32, 8)
    attn = CompressedAttention(seq_len=32, m=8, d_k=8, n_heads=2)
    z, _ = attn(q, k, v)
    m_eff = attn.effective_attention_matrix(q, k)
    z_via_matrix = torch.einsum("bhnj,bhjd->bhnd", m_eff, v)
    assert torch.allclose(z, z_via_matrix, atol=1e-4), \
        (z - z_via_matrix).abs().max().item()


def test_causal_mask_is_rejected():
    """Establishes: the module refuses a per-token mask instead of silently
    applying a meaningless one. Each compressed slot mixes all n keys, so
    'token i may not see token j' cannot be expressed over m slots."""
    q, k, v = make_qkv(1, 1, 32, 8)
    attn = CompressedAttention(seq_len=32, m=8, d_k=8)
    causal = torch.tril(torch.ones(32, 32, dtype=torch.bool)).view(1, 1, 32, 32)
    with pytest.raises(ValueError, match="mathematically undefined"):
        attn(q, k, v, mask=causal)


def test_wrong_sequence_length_is_rejected():
    """Establishes: Phi is tied to a fixed n, exactly like Linformer's projection."""
    attn = CompressedAttention(seq_len=32, m=8, d_k=8)
    q, k, v = make_qkv(1, 1, 64, 8)
    with pytest.raises(AssertionError, match="built for n="):
        attn(q, k, v)


def test_m_greater_than_n_is_rejected():
    with pytest.raises(ValueError, match="requires m <= n"):
        CompressedAttention(seq_len=16, m=32, d_k=8)


def test_gradient_flows_and_learnable_phi_receives_gradient():
    """Establishes: the block trains, and a learnable Phi is actually updated."""
    q, k, v = make_qkv(2, 2, 32, 8)
    for t in (q, k, v):
        t.requires_grad_(True)
    attn = CompressedAttention(seq_len=32, m=8, d_k=8, n_heads=2, learnable_phi=True)
    z, _ = attn(q, k, v)
    z.sum().backward()
    assert attn.phi_k.phi.grad is not None and attn.phi_k.phi.grad.abs().sum() > 0
    assert attn.phi_v.phi.grad is not None and attn.phi_v.phi.grad.abs().sum() > 0
    for name, t in (("q", q), ("k", k), ("v", v)):
        assert t.grad is not None and t.grad.abs().sum() > 0, name


def test_shared_phi_is_a_single_tensor():
    """Establishes: share_phi=True really shares storage (Phi_K is Phi_V)."""
    attn = CompressedAttention(seq_len=32, m=8, d_k=8, share_phi=True)
    assert attn.phi_k is attn.phi_v


def test_linear_control_is_unbiased_for_shared_gaussian_phi():
    """Establishes the claim in LinearCompressedAttention's docstring:
    with Phi_K = Phi_V Gaussian and NO softmax, Q K~^T V~ estimates Q K^T V,
    because E[Phi^T Phi] = I. The error should shrink as m grows.

    This is the control that attributes CSAT's approximation error to the
    softmax rather than to the random projection.
    """
    torch.manual_seed(0)
    q, k, v = make_qkv(1, 1, 256, 8)
    target = torch.matmul(torch.matmul(q, k.transpose(-2, -1)) / (8 ** 0.5), v)

    errors = []
    for m in (32, 128):
        lin = LinearCompressedAttention(seq_len=256, m=m, d_k=8, share_phi=True)
        out, _ = lin(q, k, v)
        errors.append(((out - target).norm() / target.norm()).item())
    assert errors[1] < errors[0], f"error should fall as m grows, got {errors}"


def test_phi_normalization_preserves_energy_in_expectation():
    """Establishes: with the 1/sqrt(m) convention, E[Phi^T Phi] = I_n, so a
    generic vector keeps its norm under measurement (a JL-style property)."""
    torch.manual_seed(0)
    phi = make_measurement_matrix(256, 512, "gaussian", normalize=True)
    x = torch.randn(512)
    ratios = torch.stack([
        (make_measurement_matrix(256, 512, "gaussian") @ x).norm() / x.norm()
        for _ in range(20)
    ])
    assert 0.9 < ratios.mean().item() < 1.1, ratios.mean().item()
    assert phi.shape == (256, 512)
