"""Standard scaled dot-product attention -- the paper's baseline.

PAPER EQUATION (Section 3)
    Attn(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
with Q, K, V in R^{n x d_k} obtained as Q = X W^Q, K = X W^K, V = X W^V,
X in R^{n x d}, W^* in R^{d x d_k}.

TENSOR LAYOUT USED HERE
    Q, K, V : [B, H, N, D]   (batch, heads, tokens, head-dim)
    scores  : [B, H, N, N]
    output  : [B, H, N, D]

The paper writes the single-head case; the [B, H, ...] layout is the standard
multi-head generalisation (Vaswani et al. 2017), which the paper implicitly uses
because it reports "8 attention heads".

COMPLEXITY: the Q K^T product is O(n^2 d_k) per head, which is exactly the cost
the paper sets out to remove.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    return_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """softmax(QK^T / sqrt(d_k)) V, written out explicitly.

    Args:
        q, k, v: [B, H, N, D] (k and v may have a different N; we call it N_kv).
        mask: broadcastable boolean/float mask over [B, H, N_q, N_kv].
            Boolean ``True`` means "keep". Float masks are added to the logits.
        dropout_p: attention dropout probability (applied to the weights).
        training: whether dropout is active.
        return_weights: also return the [B, H, N_q, N_kv] attention matrix.

    Returns:
        (context [B, H, N_q, D], weights or None)

    Numerical stability: ``torch.softmax`` internally subtracts the row max, so
    exp() never overflows. We rely on that rather than hand-rolling it, and the
    unit tests verify rows sum to 1 and that no NaNs appear for large logits.
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, \
        f"expected [B,H,N,D] tensors, got {q.shape}, {k.shape}, {v.shape}"
    assert q.shape[-1] == k.shape[-1], "q and k must share the head dimension d_k"
    assert k.shape[-2] == v.shape[-2], "k and v must share the key/value length"

    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)   # [B,H,N,N_kv]

    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask

    weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p, training=True)

    context = torch.matmul(weights, v)                               # [B,H,N,D]
    return (context, weights) if return_weights else (context, None)


def causal_mask(n_q: int, n_kv: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Lower-triangular boolean mask (True = attend). Used to show, in the
    notebook, that causal masking is well defined for standard attention and is
    NOT well defined once keys have been mixed across the token axis."""
    return torch.tril(torch.ones(n_q, n_kv, dtype=torch.bool, device=device))


class StandardAttention(nn.Module):
    """Module wrapper around :func:`scaled_dot_product_attention`."""

    def __init__(self, d_k: int, dropout: float = 0.0):
        super().__init__()
        self.d_k = d_k
        self.dropout = dropout

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor | None = None,
                return_weights: bool = False):
        return scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout,
            training=self.training, return_weights=return_weights,
        )

    def extra_repr(self) -> str:
        return f"d_k={self.d_k}, dropout={self.dropout}"


class MultiHeadSelfAttention(nn.Module):
    """Full multi-head self-attention block: X -> Q,K,V -> attention -> W^O.

    This is the ``Transformer (Full)`` row of the paper's tables, at the level of
    a single attention module. It exists so that the CSAT block (models/csat_block.py)
    can be compared against an identically-structured baseline.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        self.attn = StandardAttention(self.d_k, dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, self.d_k).transpose(1, 2)   # [B,H,N,D]

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * D)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                return_weights: bool = False):
        q, k, v = self._split(self.w_q(x)), self._split(self.w_k(x)), self._split(self.w_v(x))
        ctx, w = self.attn(q, k, v, mask=mask, return_weights=return_weights)
        return self.w_o(self._merge(ctx)), w
