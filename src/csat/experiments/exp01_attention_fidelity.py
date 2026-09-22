"""EXPERIMENT 01 -- How close is compressed attention Z to true attention C?

QUESTION
    The paper claims CSAT "significantly reduces attention complexity while
    maintaining semantic fidelity" and that Z_i is "a compressed version of the
    true context vector C_i". Fidelity is never measured in the paper. Here we
    measure it directly:
        C = softmax(Q K^T / sqrt(d_k)) V            (ground truth)
        Z = softmax(Q (Phi_K K)^T / sqrt(d_k)) Phi_V V
    and report relative L2, cosine similarity, NMSE and the norm ratio
    ||Z||/||C|| as functions of the compression ratio n/m.

WHY THE TOKEN DISTRIBUTION MATTERS
    The paper's motivation is that visual tokens are redundant. i.i.d. Gaussian
    tokens have no redundancy at all, so we sweep both regimes:
      'iid'       -- Q, K, V i.i.d. Gaussian (no structure to exploit);
      'redundant' -- tokens drawn from a small set of prototypes plus noise.

STATUS: preliminary (Phase 4). Final verification belongs to Phase 5.
"""

from __future__ import annotations

import json
import os

import torch

from csat.models.compressed_attention import CompressedAttention, LinearCompressedAttention
from csat.models.standard_attention import scaled_dot_product_attention
from csat.utils.metrics import all_metrics
from csat.utils.tensor_utils import make_qkv, make_redundant_tokens


def run(seq_len: int = 512, d_k: int = 64, n_heads: int = 8, batch: int = 2,
        m_values: list[int] = (16, 32, 64, 128, 256),
        ensembles: list[str] = ("gaussian", "rademacher", "orthogonal"),
        token_modes: list[str] = ("iid", "redundant"),
        scalings: list[str] = ("paper_sqrt_dk", "variance_calibrated"),
        n_clusters: int = 16, device: torch.device | str = "cpu",
        save_path: str | None = None) -> list[dict]:
    device = torch.device(device)
    rows: list[dict] = []

    for mode in token_modes:
        if mode == "iid":
            q, k, v = make_qkv(batch, n_heads, seq_len, d_k, device=device)
        else:
            q = make_redundant_tokens(batch, n_heads, seq_len, d_k, n_clusters, 0.1, device)
            k = make_redundant_tokens(batch, n_heads, seq_len, d_k, n_clusters, 0.1, device)
            v = make_redundant_tokens(batch, n_heads, seq_len, d_k, n_clusters, 0.1, device)

        # Ground truth: full attention.
        c_true, a_true = scaled_dot_product_attention(q, k, v, return_weights=True)
        # How concentrated is the true attention? (the paper's sparsity premise)
        entropy = -(a_true.clamp_min(1e-12) * a_true.clamp_min(1e-12).log()).sum(-1).mean().item()
        max_entropy = torch.log(torch.tensor(float(seq_len))).item()

        for ensemble in ensembles:
            for m in m_values:
                if m > seq_len:
                    continue
                for scaling in scalings:
                    attn = CompressedAttention(
                        seq_len=seq_len, m=m, d_k=d_k, n_heads=n_heads,
                        ensemble=ensemble, scaling=scaling, device=device)
                    with torch.no_grad():
                        z, _ = attn(q, k, v)
                    row = {
                        "token_mode": mode, "ensemble": ensemble, "scaling": scaling,
                        "n": seq_len, "m": m, "compression_ratio": seq_len / m,
                        "d_k": d_k, "heads": n_heads,
                        "true_attention_entropy_nats": entropy,
                        "max_entropy_nats": max_entropy,
                        "entropy_fraction": entropy / max_entropy,
                    }
                    row.update(all_metrics(z, c_true))
                    rows.append(row)

        # Softmax-free control with a single shared Phi: isolates the softmax as
        # the source of error (see LinearCompressedAttention's docstring).
        lin_target = torch.matmul(
            torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5), v)
        for m in m_values:
            if m > seq_len:
                continue
            lin = LinearCompressedAttention(seq_len=seq_len, m=m, d_k=d_k,
                                            share_phi=True, device=device)
            with torch.no_grad():
                z_lin, _ = lin(q, k, v)
            row = {
                "token_mode": mode, "ensemble": "gaussian",
                "scaling": "NO_SOFTMAX_control", "n": seq_len, "m": m,
                "compression_ratio": seq_len / m, "d_k": d_k, "heads": n_heads,
                "true_attention_entropy_nats": entropy,
                "max_entropy_nats": max_entropy, "entropy_fraction": entropy / max_entropy,
            }
            row.update(all_metrics(z_lin, lin_target))
            rows.append(row)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(rows, f, indent=2)
    return rows
