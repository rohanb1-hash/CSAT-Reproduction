"""CSAT compressed attention -- the paper's core mechanism (Section 3).

PAPER EQUATIONS, verbatim in symbols
    K~ = Phi_K K  in R^{m x d_k}          Phi_K in R^{m x n}
    V~ = Phi_V V  in R^{m x d_k}          Phi_V in R^{m x n}
    A~ = softmax( Q K~^T / sqrt(d_k) )  in R^{n x m}
    Z  = A~ V~                           in R^{n x d_k}

TENSOR LAYOUT HERE ([B, H, N, D] multi-head generalisation)
    Q          : [B, H, N, D]
    K, V       : [B, H, N, D]
    Phi_K,Phi_V: [M, N]           (or [H, M, N] if per_head_phi=True)
    K~, V~     : [B, H, M, D]
    A~         : [B, H, N, M]     <- the n x n attention matrix never materialises
    Z          : [B, H, N, D]

TWO PROPERTIES OF Z THAT THE PAPER DOES NOT DISCUSS, both testable and both
implemented as diagnostics below:

(1) Z is a linear map of V with a NON-STOCHASTIC effective attention matrix.
        Z = A~ V~ = A~ (Phi_V V) = (A~ Phi_V) V  =:  M_eff V,  M_eff in R^{n x n}
    Standard attention gives C = A V with A row-stochastic and non-negative.
    M_eff = A~ Phi_V is generally NOT non-negative and its rows do NOT sum to 1,
    because Phi_V has negative entries. So Z is not a weighted average of value
    vectors at all. ``effective_attention_matrix`` below computes M_eff so this
    can be measured rather than assumed.

(2) Z_i in R^{d_k} and C_i in R^{d_k} have THE SAME dimension.
    The compression is along the token axis (n -> m), and that axis is summed
    over by the attention weighting. So Z_i is not a lower-dimensional
    measurement of C_i; it is a same-dimensional approximation of it. This is
    the root of the dimensional inconsistency analysed in models/bridges.py.

MASKING
    Causal masking is NOT expressible in this mechanism. After K~ = Phi_K K,
    compressed key slot j is a linear combination of ALL n keys, including
    future ones, so no mask over the m compressed slots can enforce
    "token i may not see token > i". The forward pass therefore refuses a causal
    mask instead of silently applying a meaningless one. (Linformer has the same
    limitation and states it; this paper reports autoregressive language
    modelling on WikiText-103 without addressing it.)
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .measurement import EnsembleName, MeasurementMatrix

ScalingName = Literal["paper_sqrt_dk", "row_normalized", "variance_calibrated"]


class CompressedAttention(nn.Module):
    """Compressed-sensing attention exactly as written in the paper.

    Args:
        seq_len: n, the token count Phi is built for. Token-axis compression
            requires a FIXED n at construction time (same constraint as
            Linformer); variable-length batches must be padded to n.
        m: number of measurements (paper: m << n; the paper never gives a value).
        d_k: head dimension.
        n_heads: used only when ``per_head_phi`` is True.
        ensemble: sub-Gaussian ensemble for Phi (paper names gaussian /
            rademacher / structured hadamard).
        share_phi: use one matrix for both Phi_K and Phi_V. [CHOICE] The paper
            writes two symbols, but reuses Phi_V at decode time; sharing has a
            concrete consequence tested in experiments/exp02.
        learnable_phi: make Phi a trained parameter (paper allows both).
        scaling: how the compressed logits are scaled.
            - 'paper_sqrt_dk'       : divide by sqrt(d_k). THE PAPER'S FORMULA.
            - 'row_normalized'      : L2-normalise K~ rows first.   [OUR DIAGNOSTIC]
            - 'variance_calibrated' : divide by the empirical std of the logits.
                                      [OUR DIAGNOSTIC]
            The two diagnostics exist because Phi changes the scale of the
            logits (each K~ row is a sum of n key rows), which shifts the softmax
            temperature. The paper does not mention this.
    """

    def __init__(self, seq_len: int, m: int, d_k: int, n_heads: int = 1,
                 ensemble: EnsembleName = "gaussian", normalize_phi: bool = True,
                 share_phi: bool = False, learnable_phi: bool = False,
                 per_head_phi: bool = False, scaling: ScalingName = "paper_sqrt_dk",
                 dropout: float = 0.0,
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if m > seq_len:
            raise ValueError(f"CSAT requires m <= n; got m={m}, n={seq_len}")
        self.n, self.m, self.d_k = seq_len, m, d_k
        self.share_phi, self.scaling, self.dropout = share_phi, scaling, dropout

        heads = n_heads if per_head_phi else None
        mk = {"m": m, "n": seq_len, "ensemble": ensemble, "normalize": normalize_phi,
                  "learnable": learnable_phi, "n_heads": heads, "device": device, "dtype": dtype}
        self.phi_k = MeasurementMatrix(**mk)
        self.phi_v = self.phi_k if share_phi else MeasurementMatrix(**mk)

    # ------------------------------------------------------------------ #
    def compress(self, k: torch.Tensor, v: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """K~ = Phi_K K, V~ = Phi_V V.  [B,H,N,D] -> [B,H,M,D] each."""
        return self.phi_k(k), self.phi_v(v)

    def _scaled_logits(self, q: torch.Tensor, k_tilde: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(q, k_tilde.transpose(-2, -1))          # [B,H,N,M]
        if self.scaling == "paper_sqrt_dk":
            return logits / math.sqrt(self.d_k)
        if self.scaling == "row_normalized":
            kt = k_tilde / k_tilde.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return torch.matmul(q, kt.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.scaling == "variance_calibrated":
            return logits / logits.std(dim=-1, keepdim=True).clamp_min(1e-12)
        raise ValueError(f"unknown scaling '{self.scaling}'")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor | None = None,
                return_weights: bool = False,
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (Z [B,H,N,D], A~ [B,H,N,M] or None).

        ``mask`` may only mask QUERY positions (shape broadcastable to
        [B, H, N, 1]); a mask over the compressed axis of size m, or a causal
        [N, N] mask, is rejected -- see the module docstring.
        """
        assert q.shape[-1] == self.d_k, f"expected d_k={self.d_k}, got {q.shape[-1]}"
        assert k.shape[-2] == self.n, (
            f"Phi was built for n={self.n} tokens but K has {k.shape[-2]}; "
            "token-axis compression needs a fixed, padded sequence length."
        )
        if mask is not None and mask.shape[-1] not in (1, self.m):
            raise ValueError(
                f"CompressedAttention received a mask whose last dim is {mask.shape[-1]}. "
                f"After compression the key axis has length m={self.m} and each compressed "
                "slot mixes all n original tokens, so per-token (e.g. causal) masks are "
                "mathematically undefined here. Mask query positions instead."
            )

        k_tilde, v_tilde = self.compress(k, v)
        logits = self._scaled_logits(q, k_tilde)
        if mask is not None:
            logits = (logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
                      if mask.dtype == torch.bool else logits + mask)

        a_tilde = torch.softmax(logits, dim=-1)                       # [B,H,N,M]
        if self.dropout > 0 and self.training:
            a_tilde = F.dropout(a_tilde, p=self.dropout, training=True)

        z = torch.matmul(a_tilde, v_tilde)                            # [B,H,N,D]
        return (z, a_tilde) if return_weights else (z, None)

    # ------------------------------------------------------------------ #
    # Diagnostics (not part of the paper; used by experiments/exp02)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def effective_attention_matrix(self, q: torch.Tensor, k: torch.Tensor
                                   ) -> torch.Tensor:
        """M_eff = A~ Phi_V in R^{n x n}, so that Z = M_eff V exactly.

        Materialises an n x n matrix on purpose: this is a DIAGNOSTIC, it
        defeats the efficiency of the method and is never used in the forward
        pass or in any timing measurement.
        """
        k_tilde = self.phi_k(k)
        a_tilde = torch.softmax(self._scaled_logits(q, k_tilde), dim=-1)  # [B,H,N,M]
        phi_v = self.phi_v.phi
        if phi_v.dim() == 2:
            return torch.einsum("bhnm,mj->bhnj", a_tilde, phi_v)
        return torch.einsum("bhnm,hmj->bhnj", a_tilde, phi_v)

    def compression_ratio(self) -> float:
        """n / m -- how many times fewer key/value slots attention sees."""
        return self.n / self.m

    def extra_repr(self) -> str:
        return (f"n={self.n}, m={self.m}, d_k={self.d_k}, "
                f"share_phi={self.share_phi}, scaling={self.scaling}")


class LinearCompressedAttention(nn.Module):
    """Softmax-free control: Z_lin = (Q K~^T / sqrt(d_k)) V~.

    NOT in the paper. It exists to isolate one specific mechanism. With a SINGLE
    shared Phi (Phi_K = Phi_V = Phi, Gaussian, 1/sqrt(m) normalised) we have

        Q K~^T V~ = Q K^T Phi^T Phi V   and   E[Phi^T Phi] = I_n,

    so the softmax-free compressed product is an unbiased estimator of the
    softmax-free full product Q K^T V. Applying the softmax to the compressed
    logits destroys that identity, because softmax does not commute with Phi^T.
    Comparing this module against CompressedAttention therefore attributes
    approximation error to the softmax rather than to the projection.
    """

    def __init__(self, seq_len: int, m: int, d_k: int, ensemble: EnsembleName = "gaussian",
                 normalize_phi: bool = True, share_phi: bool = True,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.n, self.m, self.d_k = seq_len, m, d_k
        mk = {"m": m, "n": seq_len, "ensemble": ensemble, "normalize": normalize_phi,
                  "learnable": False, "n_heads": None, "device": device, "dtype": dtype}
        self.phi_k = MeasurementMatrix(**mk)
        self.phi_v = self.phi_k if share_phi else MeasurementMatrix(**mk)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        k_tilde, v_tilde = self.phi_k(k), self.phi_v(v)
        logits = torch.matmul(q, k_tilde.transpose(-2, -1)) / math.sqrt(self.d_k)
        return torch.matmul(logits, v_tilde), None
