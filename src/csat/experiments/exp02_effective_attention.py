"""EXPERIMENT 02 -- What does CSAT's effective attention matrix look like?

QUESTION
    Z = A~ V~ = (A~ Phi_V) V, so CSAT applies an effective n x n mixing matrix
        M_eff := A~ Phi_V
    in place of the true attention matrix A = softmax(QK^T/sqrt(d_k)).
    Standard attention's A is non-negative and row-stochastic: each context
    vector is a convex combination of value vectors. Is M_eff?

MEASURED
    * fraction of negative entries in M_eff;
    * distribution of row sums (should be 1.0 for a convex combination);
    * relative error ||M_eff - A||_F / ||A||_F;
    * the same statistics for the shared-Phi variant (Phi_K = Phi_V), where
      theory predicts the softmax-free version is unbiased.

WHY IT MATTERS FOR THE PAPER
    The paper's decoding story asks us to view Z_i as a measurement of C_i.
    If M_eff is not even a weighted average, then Z_i is not a "compressed
    version" of C_i in any averaging sense, and the interpretation of Z as a
    corrupted C has to be argued empirically rather than assumed.

STATUS: preliminary (Phase 4). This materialises an n x n matrix and is a
DIAGNOSTIC only -- it is never used in any timing measurement.
"""

from __future__ import annotations

import json
import os

import torch

from csat.models.compressed_attention import CompressedAttention
from csat.models.standard_attention import scaled_dot_product_attention
from csat.utils.metrics import relative_l2
from csat.utils.tensor_utils import make_qkv


def run(seq_len: int = 256, d_k: int = 64, n_heads: int = 4, batch: int = 1,
        m_values: list[int] = (16, 32, 64, 128),
        share_phi_options: list[bool] = (False, True),
        device: torch.device | str = "cpu",
        save_path: str | None = None) -> list[dict]:
    device = torch.device(device)
    q, k, v = make_qkv(batch, n_heads, seq_len, d_k, device=device)
    _, a_true = scaled_dot_product_attention(q, k, v, return_weights=True)

    rows: list[dict] = []
    for share in share_phi_options:
        for m in m_values:
            if m > seq_len:
                continue
            attn = CompressedAttention(seq_len=seq_len, m=m, d_k=d_k,
                                       n_heads=n_heads, share_phi=share, device=device)
            m_eff = attn.effective_attention_matrix(q, k)          # [B,H,N,N]
            row_sums = m_eff.sum(-1)
            rows.append({
                "n": seq_len, "m": m, "share_phi": share,
                "compression_ratio": seq_len / m,
                "negative_entry_fraction": (m_eff < 0).float().mean().item(),
                "row_sum_mean": row_sums.mean().item(),
                "row_sum_std": row_sums.std().item(),
                "row_sum_abs_mean": row_sums.abs().mean().item(),
                "max_abs_entry": m_eff.abs().max().item(),
                "rel_error_vs_true_A": relative_l2(m_eff, a_true),
                "true_A_min_entry": a_true.min().item(),
                "true_A_row_sum_mean": a_true.sum(-1).mean().item(),
            })

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(rows, f, indent=2)
    return rows
