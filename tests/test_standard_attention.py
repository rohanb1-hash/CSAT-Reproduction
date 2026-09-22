"""Unit tests for standard scaled dot-product attention.

Each test states what property it establishes; together they cover the checklist
of shape, normalisation, stability, gradient flow and agreement with PyTorch's
own kernel.
"""

from __future__ import annotations

import math

import torch

from csat.models.standard_attention import (
    MultiHeadSelfAttention,
    causal_mask,
    scaled_dot_product_attention,
)
from csat.utils.tensor_utils import make_qkv


def test_output_and_weight_shapes():
    """Establishes: [B,H,N,D] in -> [B,H,N,D] context and [B,H,N,N] weights."""
    q, k, v = make_qkv(3, 4, 16, 8)
    ctx, w = scaled_dot_product_attention(q, k, v, return_weights=True)
    assert ctx.shape == (3, 4, 16, 8)
    assert w.shape == (3, 4, 16, 16)


def test_rows_of_attention_sum_to_one():
    """Establishes: softmax is taken over the KEY axis, so every query row is a
    probability distribution. A bug that softmaxes the wrong axis passes the
    shape test but fails here."""
    q, k, v = make_qkv(2, 2, 12, 8)
    _, w = scaled_dot_product_attention(q, k, v, return_weights=True)
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    assert (w >= 0).all(), "attention weights must be non-negative"


def test_matches_pytorch_reference():
    """Establishes: our explicit implementation equals torch's fused kernel."""
    q, k, v = make_qkv(2, 3, 32, 16)
    ours, _ = scaled_dot_product_attention(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max().item()


def test_scaling_by_sqrt_dk():
    """Establishes: the 1/sqrt(d_k) factor is actually applied, by comparing
    against a hand-computed single-head case."""
    q = torch.tensor([[[[1.0, 0.0]]]])          # [1,1,1,2]
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    v = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    ctx, w = scaled_dot_product_attention(q, k, v, return_weights=True)
    logits = torch.tensor([1.0, 0.0]) / math.sqrt(2)
    expected = torch.softmax(logits, dim=-1)
    assert torch.allclose(w.flatten(), expected, atol=1e-6)
    assert torch.allclose(ctx.flatten(), expected, atol=1e-6)


def test_numerical_stability_large_logits():
    """Establishes: no NaN/Inf even with logits of magnitude ~1e4, because
    softmax subtracts the row maximum internally."""
    q, k, v = make_qkv(1, 1, 8, 4)
    q, k = q * 1e2, k * 1e2                     # logits ~ 1e4 / sqrt(4)
    ctx, w = scaled_dot_product_attention(q, k, v, return_weights=True)
    assert torch.isfinite(ctx).all() and torch.isfinite(w).all()
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)


def test_causal_mask_is_respected():
    """Establishes: masked positions receive exactly zero weight -- the property
    that compressed attention cannot provide (see test_compressed_attention)."""
    q, k, v = make_qkv(1, 1, 6, 4)
    mask = causal_mask(6, 6).view(1, 1, 6, 6)
    _, w = scaled_dot_product_attention(q, k, v, mask=mask, return_weights=True)
    upper = w[0, 0].triu(diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-7)


def test_gradients_flow_to_all_inputs():
    """Establishes: q, k and v all receive non-zero gradient."""
    q, k, v = make_qkv(2, 2, 10, 8)
    for t in (q, k, v):
        t.requires_grad_(True)
    ctx, _ = scaled_dot_product_attention(q, k, v)
    ctx.sum().backward()
    for name, t in (("q", q), ("k", k), ("v", v)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
        assert t.grad.abs().sum() > 0, f"{name} received zero gradient"


def test_multihead_module_roundtrip():
    """Establishes: the multi-head wrapper preserves [B,N,d_model] and trains."""
    mha = MultiHeadSelfAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 12, 32, requires_grad=True)
    out, _ = mha(x)
    assert out.shape == (2, 12, 32)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
