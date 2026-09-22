"""EXPERIMENT 05 -- Does the sparse decoder recover the true context vector?

THE CENTRAL QUESTION OF THE REPRODUCTION
    The paper's pipeline is: Z = A~ V~, then decode alpha_hat from Z, then
    C_hat = Psi alpha_hat, and use C_hat "for downstream tasks". The claim is
    that C_hat approximates the true context C = softmax(QK^T/sqrt(d_k)) V.
    Because the decoding equation is ill-posed as written (models/bridges.py),
    we test the only reading that plugs directly into Z -- BRIDGE 1 'denoise' --
    and ask the empirical question the paper never asks:

        Is  ||C_hat - C|| < ||Z - C|| ?
        i.e. does the decoding stage HELP at all?

    A decoder that increases the error is not a decoder; it is a distortion.

WE ALSO TEST THE PREMISE
    Before decoding can help, the true context vectors C must be sparse in some
    Psi. We measure that directly for (a) the DCT basis, (b) a random orthogonal
    basis, (c) a dictionary LEARNED from the context vectors themselves -- the
    most favourable case possible, since it is fit on the test data.

PART TWO -- the genuinely-CS bridges
    For completeness we also run BRIDGE 2 (feature_cs) and BRIDGE 3 (token_cs),
    which ARE well-posed CS problems, and report their recovery quality. Those
    results speak to the solvers and to the compressibility of the data; they do
    NOT reproduce the paper's pipeline, and are labelled accordingly.

STATUS: preliminary (Phase 4).
"""

from __future__ import annotations

import json
import os

import torch

from csat.models.bridges import build_bridge, decode_context
from csat.models.compressed_attention import CompressedAttention
from csat.models.dictionary import fit_dictionary, make_dictionary
from csat.models.ista import fista
from csat.models.standard_attention import scaled_dot_product_attention
from csat.utils.metrics import all_metrics, relative_l2, sparsity_profile
from csat.utils.tensor_utils import make_qkv, make_redundant_tokens


# --------------------------------------------------------------------------- #
def measure_context_compressibility(c: torch.Tensor, learn_dict: bool = True,
                                    k_atoms_mult: int = 2,
                                    device: torch.device | str = "cpu") -> list[dict]:
    """Is C sparse in any basis? Tests DCT, random orthogonal and a learned dictionary."""
    d_k = c.shape[-1]
    flat = c.reshape(-1, d_k)
    rows = []

    for name in ("dct", "random_orthogonal"):
        psi = make_dictionary(name, d_k, d_k, device=flat.device)
        alpha = flat @ torch.linalg.inv(psi).T           # exact code (square basis)
        prof = sparsity_profile(alpha, energy=0.95)
        rows.append({
            "basis": name, "exact_representation": True,
            "reconstruction_rel_l2": relative_l2(alpha @ psi.T, flat),
            **prof,
        })

    if learn_dict:
        k_atoms = d_k * k_atoms_mult
        subset = flat[torch.randperm(flat.shape[0])[:min(2048, flat.shape[0])]]
        psi_l, codes = fit_dictionary(subset, k_atoms=k_atoms, sparsity_lambda=0.05,
                                      n_outer=15, n_inner=30)
        rec = codes @ psi_l.T
        prof = sparsity_profile(codes, energy=0.95)
        rows.append({
            "basis": f"learned_overcomplete_{k_atoms}", "exact_representation": False,
            "reconstruction_rel_l2": relative_l2(rec, subset),
            "mean_nnz_at_1e-3": (codes.abs() > 1e-3).float().sum(1).mean().item(),
            **prof,
        })
    return rows


# --------------------------------------------------------------------------- #
def run(seq_len: int = 256, d_k: int = 64, n_heads: int = 4, batch: int = 1,
        m_values: list[int] = (32, 64, 128),
        lam_values: list[float] = (0.001, 0.01, 0.05, 0.2),
        token_mode: str = "redundant", n_iters: int = 300,
        device: torch.device | str = "cpu",
        save_dir: str | None = None) -> dict[str, object]:
    device = torch.device(device)

    if token_mode == "iid":
        q, k, v = make_qkv(batch, n_heads, seq_len, d_k, device=device)
    else:
        q = make_redundant_tokens(batch, n_heads, seq_len, d_k, 16, 0.1, device)
        k = make_redundant_tokens(batch, n_heads, seq_len, d_k, 16, 0.1, device)
        v = make_redundant_tokens(batch, n_heads, seq_len, d_k, 16, 0.1, device)

    c_true, _ = scaled_dot_product_attention(q, k, v)          # [B,H,N,D]

    # ---- premise check: is C compressible at all? ------------------------- #
    compressibility = measure_context_compressibility(c_true, learn_dict=True, device=device)

    # ---- BRIDGE 1: does decoding Z reduce the error to C? ----------------- #
    bridge1 = build_bridge("denoise", d_k=d_k, dictionary="dct", device=device)
    denoise_rows = []
    for m in m_values:
        if m > seq_len:
            continue
        attn = CompressedAttention(seq_len=seq_len, m=m, d_k=d_k, n_heads=n_heads,
                                   device=device)
        with torch.no_grad():
            z, _ = attn(q, k, v)
        base = all_metrics(z, c_true)                          # error BEFORE decoding
        for lam in lam_values:
            dec = decode_context(z, bridge1, lam=lam, n_iters=n_iters)
            after = all_metrics(dec["c_hat"], c_true)
            nnz = (dec["alpha"].abs() > 1e-3).float().sum(-1).mean().item()
            denoise_rows.append({
                "bridge": "denoise", "m": m, "n": seq_len, "lam": lam,
                "n_iters": n_iters,
                "rel_l2_Z_vs_C": base["relative_l2"],
                "rel_l2_Chat_vs_C": after["relative_l2"],
                "decoder_helps": after["relative_l2"] < base["relative_l2"],
                "improvement": base["relative_l2"] - after["relative_l2"],
                "cosine_Z_vs_C": base["cosine_similarity"],
                "cosine_Chat_vs_C": after["cosine_similarity"],
                "mean_nnz_alpha": nnz, "d_k": d_k,
            })

    # ---- BRIDGE 2: feature-space CS (our construction, well-posed) -------- #
    feature_rows = []
    c_flat = c_true.reshape(-1, d_k)
    for p_feat in (d_k // 4, d_k // 2, (3 * d_k) // 4):
        spec = build_bridge("feature_cs", d_k=d_k, p_features=p_feat,
                            dictionary="dct", device=device)
        y = c_flat @ spec.phi.T                                # y_i = Phi_f C_i
        out = fista(spec.A, y, lam=0.01, n_iters=n_iters)
        c_hat = out["alpha"] @ spec.psi.T
        feature_rows.append({
            "bridge": "feature_cs", "p_measurements": p_feat, "d_k": d_k,
            "undersampling_ratio": p_feat / d_k,
            "rel_l2_Chat_vs_C": relative_l2(c_hat, c_flat),
            "measurement_rel_l2": relative_l2(out["reconstruction"], y),
            "note": "well-posed CS, but Phi_f is OUR construction and y is not the paper's Z",
        })

    # ---- BRIDGE 3: token-axis CS on the value matrix ---------------------- #
    token_rows = []
    v_col = v[0, 0].T.contiguous()                             # [d_k, n] -> rows are columns of V
    for m in m_values:
        if m > seq_len:
            continue
        spec = build_bridge("token_cs", d_k=d_k, n=seq_len, m=m,
                            dictionary="dct", device=device)
        y = v_col @ spec.phi.T                                 # [d_k, m]
        out = fista(spec.A, y, lam=0.01, n_iters=n_iters)
        v_hat = out["alpha"] @ spec.psi.T                      # [d_k, n]
        token_rows.append({
            "bridge": "token_cs", "m": m, "n": seq_len,
            "undersampling_ratio": m / seq_len,
            "rel_l2_Vhat_vs_V": relative_l2(v_hat, v_col),
            "note": "recovers V along the token axis, NOT the context vector C",
        })

    out = {"compressibility_of_C": compressibility,
           "bridge1_denoise": denoise_rows,
           "bridge2_feature_cs": feature_rows,
           "bridge3_token_cs": token_rows,
           "config": {"n": seq_len, "d_k": d_k, "heads": n_heads,
                      "token_mode": token_mode, "n_iters": n_iters}}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "exp05_decoder_bridge.json"), "w") as f:
            json.dump(out, f, indent=2, default=str)
    return out
