"""Analytic FLOP counts for the complexity claims in the paper.

The paper claims a reduction from O(n^2 d) to O(nmd + decoding). This module
counts multiply-accumulate FLOPs (2 per MAC) for each stage so that the
*theoretical* claim can be separated from *measured* runtime -- these are
different things and the notebook never conflates them.

Convention: a matmul [a,b] @ [b,c] costs 2*a*b*c FLOPs.
Softmax and element-wise ops are counted with a small constant per element;
they are asymptotically irrelevant but we include them for honesty.
"""

from __future__ import annotations

SOFTMAX_FLOPS_PER_ELEMENT = 5  # exp + max-subtract + sum + divide, roughly


def standard_attention_flops(n: int, d_k: int, heads: int = 1, batch: int = 1) -> dict[str, float]:
    """softmax(Q K^T / sqrt(d_k)) V for one attention layer.

    Q K^T : [n, d_k] @ [d_k, n] -> 2 n^2 d_k
    A V   : [n, n]   @ [n, d_k] -> 2 n^2 d_k
    """
    bh = batch * heads
    scores = 2.0 * n * n * d_k
    softmax = SOFTMAX_FLOPS_PER_ELEMENT * n * n
    context = 2.0 * n * n * d_k
    total = bh * (scores + softmax + context)
    return {
        "scores_QKt": bh * scores,
        "softmax": bh * softmax,
        "context_AV": bh * context,
        "total": total,
        "attention_matrix_elements": bh * n * n,
    }


def csat_attention_flops(n: int, m: int, d_k: int, heads: int = 1, batch: int = 1,
                         share_phi: bool = False,
                         phi_per_head: bool = False) -> dict[str, float]:
    """Compressed attention: K~ = Phi_K K, V~ = Phi_V V, A~ = softmax(Q K~^T), Z = A~ V~.

    Projections : [m, n] @ [n, d_k] -> 2 m n d_k, twice (or once if Phi shared
                  AND K is V, which it is not -- sharing Phi saves storage, not
                  these FLOPs, so both projections are always counted).
    Q K~^T      : [n, d_k] @ [d_k, m] -> 2 n m d_k
    A~ V~       : [n, m]   @ [m, d_k] -> 2 n m d_k
    """
    bh = batch * heads
    project = 2.0 * (2.0 * m * n * d_k)     # Phi_K K and Phi_V V
    scores = 2.0 * n * m * d_k
    softmax = SOFTMAX_FLOPS_PER_ELEMENT * n * m
    context = 2.0 * n * m * d_k
    total = bh * (project + scores + softmax + context)
    return {
        "projections": bh * project,
        "scores_QKt": bh * scores,
        "softmax": bh * softmax,
        "context_AV": bh * context,
        "total": total,
        "attention_matrix_elements": bh * n * m,
    }


def ista_decode_flops(n: int, d_k: int, k_atoms: int, n_iters: int,
                      p_meas: int, heads: int = 1, batch: int = 1) -> dict[str, float]:
    """Per-token ISTA decoding cost for n tokens.

    One ISTA step on a single vector with A in R^{p x k}:
        A alpha            : 2 p k
        A^T residual       : 2 p k
        thresholding etc.  : ~3 k
    Repeated ``n_iters`` times for each of ``n`` token rows.
    The final synthesis C_hat = Psi alpha costs 2 * d_k * k per token.
    """
    bh = batch * heads
    per_step = 4.0 * p_meas * k_atoms + 3.0 * k_atoms
    iterate = n * n_iters * per_step
    synth = n * 2.0 * d_k * k_atoms
    return {
        "iterations": bh * iterate,
        "synthesis": bh * synth,
        "total": bh * (iterate + synth),
    }


def lista_decode_flops(n: int, d_k: int, k_atoms: int, n_layers: int,
                       p_meas: int, heads: int = 1, batch: int = 1,
                       ) -> dict[str, float]:
    """LISTA: alpha^{t+1} = eta(S alpha^t + B Z).

    B Z : 2 p k  (computed once, reused every layer in the standard formulation)
    S a : 2 k^2  per layer  <- note this is QUADRATIC in the number of atoms,
          which is why LISTA is not automatically cheaper than ISTA.
    """
    bh = batch * heads
    b_term = n * 2.0 * p_meas * k_atoms
    layers = n * n_layers * (2.0 * k_atoms * k_atoms + 3.0 * k_atoms)
    synth = n * 2.0 * d_k * k_atoms
    return {
        "B_projection": bh * b_term,
        "layers": bh * layers,
        "synthesis": bh * synth,
        "total": bh * (b_term + layers + synth),
    }


def complexity_table(n_values, m: int, d_k: int, heads: int = 8, batch: int = 1,
                     k_atoms: int | None = None, ista_iters: int = 20,
                     lista_layers: int = 8):
    """Build the standard-vs-CSAT FLOP comparison used in the notebook."""
    k_atoms = k_atoms or d_k
    rows = []
    for n in n_values:
        std = standard_attention_flops(n, d_k, heads, batch)["total"]
        csat = csat_attention_flops(n, m, d_k, heads, batch)["total"]
        ista = ista_decode_flops(n, d_k, k_atoms, ista_iters, d_k, heads, batch)["total"]
        lista = lista_decode_flops(n, d_k, k_atoms, lista_layers, d_k, heads, batch)["total"]
        rows.append({
            "n": n, "m": m, "d_k": d_k, "heads": heads,
            "standard_GFLOPs": std / 1e9,
            "csat_attention_GFLOPs": csat / 1e9,
            "ista_decode_GFLOPs": ista / 1e9,
            "lista_decode_GFLOPs": lista / 1e9,
            "csat_plus_ista_GFLOPs": (csat + ista) / 1e9,
            "speedup_attention_only": std / max(csat, 1e-9),
            "speedup_with_ista_decode": std / max(csat + ista, 1e-9),
            "speedup_with_lista_decode": std / max(csat + lista, 1e-9),
        })
    return rows
