"""End-to-end CSAT attention block: Figure 1 of the paper, as far as it is defined.

PIPELINE (paper Figure 1)
    tokens -> Q,K,V projections -> Phi_K/Phi_V compression -> compressed attention
           -> sparse decoder (ISTA/LISTA) -> context vector Psi*alpha -> fusion/output

This module assembles the pieces. The decoder stage uses one of the bridges in
models/bridges.py; the default is 'denoise', which is the only reading that is
simultaneously dimensionally consistent AND uses the paper's own Psi and
C_hat_i = Psi alpha_hat_i. Choosing 'none' disables decoding and gives plain
compressed attention -- the ablation that isolates what the decoder contributes.

TRAINABILITY NOTE
    With decoder='lista' the whole block is differentiable, so gradients reach
    W^Q/W^K/W^V, Phi (if learnable), Psi (if learnable), and the LISTA weights.
    With decoder='ista' the decoder is a no-grad fixed-point iteration, so
    gradients do NOT flow through it; the block is then usable at inference but
    the attention projections receive no gradient through the decoder path. The
    paper does not discuss how the analytic decoder is trained end to end -- a
    gap worth stating plainly.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .bridges import BridgeSpec, build_bridge
from .compressed_attention import CompressedAttention
from .ista import ISTADecoder
from .lista import LISTA, LISTADecoder

DecoderName = Literal["none", "ista", "lista"]


class CSATBlock(nn.Module):
    """Multi-head CSAT: X -> compressed attention -> optional sparse decode -> W^O.

    Args mirror ``MultiHeadSelfAttention`` so the two can be swapped in a model.
    """

    def __init__(self, d_model: int, n_heads: int, seq_len: int, m: int,
                 ensemble: str = "gaussian", share_phi: bool = False,
                 learnable_phi: bool = False, per_head_phi: bool = False,
                 scaling: str = "paper_sqrt_dk", dropout: float = 0.0,
                 decoder: DecoderName = "none", bridge: str = "denoise",
                 dictionary: str = "dct", dict_atoms: int | None = None,
                 ista_lam: float = 0.05, ista_iters: int = 30,
                 lista_layers: int = 8, learn_psi: bool = False,
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.decoder_name = decoder

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attn = CompressedAttention(
            seq_len=seq_len, m=m, d_k=self.d_k, n_heads=n_heads, ensemble=ensemble,
            share_phi=share_phi, learnable_phi=learnable_phi,
            per_head_phi=per_head_phi, scaling=scaling, dropout=dropout,
            device=device, dtype=dtype,
        )

        self.bridge_spec: BridgeSpec | None = None
        self.decoder: nn.Module | None = None
        if decoder != "none":
            self.bridge_spec = build_bridge(
                bridge, d_k=self.d_k, n=seq_len, m=m, dictionary=dictionary,
                k_atoms=dict_atoms, ensemble=ensemble, device=device, dtype=dtype)
            if self.bridge_spec.A.shape[0] != self.d_k:
                raise ValueError(
                    f"bridge '{bridge}' produces measurements of dim "
                    f"{self.bridge_spec.A.shape[0]}, which cannot consume the "
                    f"d_k={self.d_k} rows of Z. Only bridge='denoise' plugs directly "
                    "into the attention output; see models/bridges.py.")
            if decoder == "ista":
                self.decoder = ISTADecoder(self.bridge_spec.A, self.bridge_spec.psi,
                                           lam=ista_lam, n_iters=ista_iters)
            else:
                lista = LISTA(p=self.bridge_spec.A.shape[0],
                              k=self.bridge_spec.A.shape[1],
                              n_layers=lista_layers, A=self.bridge_spec.A,
                              lam=ista_lam)
                self.decoder = LISTADecoder(lista, self.bridge_spec.psi,
                                            learn_psi=learn_psi)

    # ------------------------------------------------------------------ #
    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, self.d_k).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * D)

    def forward(self, x: torch.Tensor, return_internals: bool = False
                ) -> tuple[torch.Tensor, dict | None]:
        q = self._split(self.w_q(x))
        k = self._split(self.w_k(x))
        v = self._split(self.w_v(x))

        z, a_tilde = self.attn(q, k, v, return_weights=return_internals)   # [B,H,N,D]
        decoded = self.decoder(z) if self.decoder is not None else z
        out = self.w_o(self._merge(decoded))

        if not return_internals:
            return out, None
        return out, {"Z": z, "A_tilde": a_tilde, "decoded": decoded}

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, heads={self.n_heads}, "
                f"m={self.attn.m}, decoder={self.decoder_name}")
