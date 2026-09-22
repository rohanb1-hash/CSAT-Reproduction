"""EXPERIMENT 06 -- Runtime and memory: standard attention vs CSAT.

WHAT THE PAPER REPORTS (Table 5, sequence length 4096)
    Transformer (Full)  18.4 GB   1113 ms
    Linformer            5.8 GB    395 ms
    Performer            6.4 GB    412 ms
    CSAT (ours)          6.9 GB    439 ms

WHAT THE PAPER DOES NOT REPORT, and which makes those numbers unreproducible
    * the GPU model and precision;
    * the batch size;
    * the number of layers measured (one attention layer, or a whole model?);
    * the value of m;
    * the decoder configuration (ISTA iterations or LISTA depth);
    * whether the decoder is included in the 439 ms at all.
We therefore do NOT attempt to match those absolute numbers. We measure the
SCALING behaviour on whatever device this notebook runs on, which is the part
that can be checked, and we report the decoder overhead separately so that it
cannot be hidden inside an attention-only comparison.

FAIRNESS RULES (see utils/benchmarking.py)
    warm-up, CUDA synchronisation, repeated timing, mean +- std, separate
    reporting of allocated and reserved memory, OOM recorded rather than crashed.
"""

from __future__ import annotations

import json
import os

import torch

from csat.models.bridges import build_bridge
from csat.models.compressed_attention import CompressedAttention
from csat.models.ista import ISTADecoder
from csat.models.lista import LISTA, LISTADecoder
from csat.models.standard_attention import scaled_dot_product_attention
from csat.utils.benchmarking import benchmark, enable_benchmark_mode, reset_memory_stats
from csat.utils.flops import csat_attention_flops, standard_attention_flops
from csat.utils.tensor_utils import make_qkv


def run(seq_lens: list[int] = (512, 1024, 2048, 4096),
        m_values: list[int] = (64, 128, 256),
        batch: int = 1, n_heads: int = 8, d_k: int = 64,
        ista_iters: int = 20, lista_layers: int = 8,
        warmup: int = 5, repeats: int = 20,
        device: torch.device | str = "cpu",
        save_path: str | None = None) -> list[dict]:
    device = torch.device(device)
    if device.type == "cuda":
        enable_benchmark_mode()

    rows: list[dict] = []
    for n in seq_lens:
        reset_memory_stats(device)
        try:
            q, k, v = make_qkv(batch, n_heads, n, d_k, device=device)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            rows.append({"n": n, "method": "allocate_inputs", "status": "OOM"})
            reset_memory_stats(device)
            continue

        # ---- baseline: full attention -------------------------------------- #
        # Closures bind q/k/v as default arguments so they capture the CURRENT
        # tensors by value. A bare `lambda: f(q, k, v)` would look the names up at
        # call time, which breaks once the loop rebinds or `del`etes them -- a real
        # latent bug in a benchmark harness, not a style preference.
        res = benchmark(lambda q=q, k=k, v=v: scaled_dot_product_attention(q, k, v),
                        device, warmup, repeats, label="standard_attention")
        res.update(n=n, m=None, method="standard_attention", batch=batch,
                   heads=n_heads, d_k=d_k,
                   theoretical_GFLOPs=standard_attention_flops(n, d_k, n_heads, batch)["total"] / 1e9,
                   attention_matrix_MB=batch * n_heads * n * n * 4 / 1024 ** 2)
        rows.append(res)

        # ---- CSAT, attention stage only ------------------------------------ #
        for m in m_values:
            if m > n:
                continue
            attn = CompressedAttention(seq_len=n, m=m, d_k=d_k, n_heads=n_heads,
                                       device=device).to(device)
            res = benchmark(lambda attn=attn, q=q, k=k, v=v: attn(q, k, v),
                            device, warmup, repeats, label=f"csat_attention_m{m}")
            res.update(n=n, m=m, method="csat_attention", batch=batch,
                       heads=n_heads, d_k=d_k,
                       theoretical_GFLOPs=csat_attention_flops(n, m, d_k, n_heads, batch)["total"] / 1e9,
                       attention_matrix_MB=batch * n_heads * n * m * 4 / 1024 ** 2)
            rows.append(res)

            # ---- decoder overhead, measured separately --------------------- #
            with torch.no_grad():
                z, _ = attn(q, k, v)
            spec = build_bridge("denoise", d_k=d_k, dictionary="dct", device=device)

            ista_dec = ISTADecoder(spec.A, spec.psi, lam=0.05, n_iters=ista_iters).to(device)
            res = benchmark(lambda dec=ista_dec, z=z: dec(z),
                            device, warmup, max(repeats // 2, 3),
                            label=f"ista_decoder_m{m}")
            res.update(n=n, m=m, method="ista_decoder", batch=batch, heads=n_heads,
                       d_k=d_k, decoder_iters=ista_iters)
            rows.append(res)

            lista_dec = LISTADecoder(
                LISTA(p=spec.A.shape[0], k=spec.A.shape[1], n_layers=lista_layers,
                      A=spec.A, lam=0.05), spec.psi).to(device)
            res = benchmark(lambda dec=lista_dec, z=z: dec(z),
                            device, warmup, max(repeats // 2, 3),
                            label=f"lista_decoder_m{m}")
            res.update(n=n, m=m, method="lista_decoder", batch=batch, heads=n_heads,
                       d_k=d_k, decoder_layers=lista_layers)
            rows.append(res)

            # ---- full pipeline: compressed attention + ISTA decode --------- #
            res = benchmark(
                lambda dec=ista_dec, attn=attn, q=q, k=k, v=v: dec(attn(q, k, v)[0]),
                device, warmup, max(repeats // 2, 3), label=f"csat_pipeline_m{m}")
            res.update(n=n, m=m, method="csat_full_pipeline", batch=batch,
                       heads=n_heads, d_k=d_k, decoder_iters=ista_iters)
            rows.append(res)

            del attn, z, ista_dec, lista_dec
            reset_memory_stats(device)

        del q, k, v
        reset_memory_stats(device)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(rows, f, indent=2)
    return rows
