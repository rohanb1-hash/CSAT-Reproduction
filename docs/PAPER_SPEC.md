# Paper specification

Everything here is read off arXiv:2507.02957v1. Each equation is given as the paper
writes it, then its tensor dimensions, then what the operation does, then how this
repository implements it, then what is ambiguous or missing.

**Provenance tags**

| Tag | Meaning |
|---|---|
| `[PAPER]` | stated explicitly in the paper |
| `[CHOICE]` | the paper leaves this free; we picked a standard value and say so |
| `[MISSING]` | the paper *needs* this quantity but never reports it |
| `[INCONSIST]` | the paper's own statement is mathematically inconsistent here |

---

## 1. The problem

Self-attention costs `O(n²d)` in time and memory. The paper (§1) targets vision-language
models, where attention is computed within *and across* modalities, making the quadratic
term dominant for long video sequences and high-resolution image-token streams.

The hypothesis (§1, §2): **attention context vectors are sparse or compressible in some
fixed or learned basis.** If so, they can be recovered from far fewer linear measurements
than their ambient dimension, so attention can be computed in a compressed space and the
full output recovered by sparse decoding.

This is an empirical claim about data. `exp05` measures it rather than assuming it.

## 2. Standard attention `[PAPER, §3]`

```
Q = XW^Q,   K = XW^K,   V = XW^V
Attn(Q,K,V) = softmax(Q Kᵀ / √d_k) V
```

| Symbol | Shape | Meaning |
|---|---|---|
| `X` | `n × d` | input token sequence |
| `W^Q, W^K, W^V` | `d × d_k` | learned projections |
| `Q, K, V` | `n × d_k` | queries, keys, values |
| `QKᵀ` | `n × n` | the quadratic term |
| `A = softmax(·)` | `n × n` | row-stochastic attention matrix |
| `C = AV` | `n × d_k` | context vectors; row `i` is `C_i ∈ ℝ^{d_k}` |

Each query scores every key; softmax turns each row into a probability distribution;
`C_i` is the resulting **convex combination** of value vectors. Two properties matter
later: `A ≥ 0`, and every row of `A` sums to 1.

**Implemented in** [`models/standard_attention.py`](../src/csat/models/standard_attention.py),
in the multi-head layout `[B, H, N, D]`.

`[MISSING]` The paper writes the single-head case and never states how heads compose. We
use `d_model = H · d_k` (Vaswani convention), consistent with the paper's own "512 hidden
dimensions and 8 attention heads" (§4).

## 3. Compression `[PAPER, §3]`

```
K̃ = Φ_K K ∈ ℝ^{m×d_k}      Φ_K, Φ_V ∈ ℝ^{m×n},  m ≪ n
Ṽ = Φ_V V ∈ ℝ^{m×d_k}
```

`Φ` acts on the **token axis**: it replaces `n` key/value rows with `m` random *linear
mixtures* of them. This is not selection or pooling — compressed slot `j` is a weighted
sum of **all** `n` tokens.

The paper says `Φ` are "measurement matrices satisfying the RIP", "typically drawn from
sub-Gaussian ensembles (e.g., random Gaussian, Rademacher, or structured Hadamard
matrices) that exhibit low coherence with sparse bases". §7 adds that they "can be fixed
post-training or made learnable".

**What the paper does not say** — all `[MISSING]`:

1. **The normalisation constant.** `[CHOICE]` We use entries `N(0, 1/m)` so that
   `E[ΦᵀΦ] = I_n`, the convention under which standard RIP results hold. This matters:
   the scale of `K̃` directly changes the softmax temperature.
2. **The value of `m`** — in any experiment, including Table 5.
3. Whether `Φ` is shared across heads, layers or modalities.
4. Whether `Φ` is re-drawn per batch. We fix it at initialisation, which is what "fixed
   post-training" implies.

**RIP, stated precisely.** `Φ` satisfies the RIP of order `s` with constant `δ_s` if for
every `s`-sparse `x`: `(1−δ_s)‖x‖² ≤ ‖Φx‖² ≤ (1+δ_s)‖x‖²`. For i.i.d. sub-Gaussian
entries this holds w.h.p. once `m = O(s log(n/s))`.

**A question the paper does not answer.** `Φ_K` and `Φ_V` act along the *token* axis, so
the RIP requirement ties `m` to the sparsity of *something along the token axis*. The
paper's sparsity assumption, however, is about **context vectors in feature space**
(`α_i ∈ ℝ^{d_k}`). The object whose sparsity would justify `m ≪ n` is never identified.
We therefore report computable proxies — mutual coherence and a Monte-Carlo **lower
bound** on `δ_s` — and mark exact RIP verification (NP-hard in general) as
*Cannot reproduce exactly*.

**Implemented in** [`models/measurement.py`](../src/csat/models/measurement.py).

## 4. Compressed attention `[PAPER, §3]`

```
Ã = softmax(Q K̃ᵀ / √d_k) ∈ ℝ^{n×m}
Z = Ã Ṽ ∈ ℝ^{n×d_k}
```

| Tensor | Shape | Note |
|---|---|---|
| `Q` | `n × d_k` | queries are **not** compressed |
| `Ã` | `n × m` | the `n×n` matrix is never formed — this is the saving |
| `Z` | `n × d_k` | the compressed attention output |

**Implemented in** [`models/compressed_attention.py`](../src/csat/models/compressed_attention.py),
exactly as written.

Two consequences the paper does not state, both measured here:

**(a)** `Z = (Ã Φ_V) V`, so the effective mixing matrix `M_eff = Ã Φ_V` is generally
neither non-negative nor row-stochastic. See [DIMENSIONAL_ANALYSIS.md](DIMENSIONAL_ANALYSIS.md) §6.

**(b)** `Z_i` and `C_i` both live in `ℝ^{d_k}`. A single row is not an undersampled
measurement of `C_i` — it is a same-dimensional approximation of it.

`[MISSING]` **Logit scaling.** The paper keeps `√d_k` unchanged after projection, but
each row of `K̃` is a sum of `n` key rows, so the logits have a different variance, which
shifts the softmax temperature. We implement the paper's literal formula as the
**default** and offer two re-scalings as explicitly labelled diagnostics.

`[MISSING]` **Masking.** The paper reports autoregressive language modelling (Table 1)
but never discusses causal masking. After `K̃ = Φ_K K`, compressed slot `j` mixes **all**
`n` keys including future ones, so no mask over `m` slots can enforce "token `i` may not
see token `j > i`". Our implementation **refuses** a causal mask rather than applying a
meaningless one.

## 5. Sparse representation `[PAPER, §3]`

> "Suppose there exists a dictionary `Ψ ∈ ℝ^{d_k×d_k}` such that the true context vector
> `C_i` admits a sparse representation: `C_i = Ψα_i`, where `α_i ∈ ℝ^{d_k}` is sparse."

**A logical point that governs the experiment design.** `Ψ` is *square*. If invertible,
`C_i = Ψα_i` has an exact solution `α_i = Ψ⁻¹C_i` for **every** `C_i` whatsoever. The
existence of a representation is therefore vacuous; only its **sparsity** carries
content, and sparsity is an empirical property of the data, not a consequence of the
model.

`[MISSING]` How `Ψ` is obtained (fixed? learned? from what data?), whether it is shared,
the sparsity level `s` actually observed, and any evidence at all that context vectors
are sparse in any basis. We default to the DCT, since the paper's own motivation is the
JPEG analogy, and additionally *learn* a dictionary on the test data — the most
favourable case possible — in `exp05`.

**Implemented in** [`models/dictionary.py`](../src/csat/models/dictionary.py).

## 6. Decoding `[INCONSIST]`

```
Z_i = Φ Ψ α_i,   Φ = Φ_V reused
α̂_i = argmin ‖α‖₁  s.t.  Z_i = ΦΨα
Ĉ_i = Ψ α̂_i
```

**This does not type-check.** Full analysis in
[DIMENSIONAL_ANALYSIS.md](DIMENSIONAL_ANALYSIS.md); three consistent alternatives in
[`models/bridges.py`](../src/csat/models/bridges.py).

## 7. Solvers `[PAPER, §2–3, §5]`

### ISTA
```
F(α) = ½‖y − Aα‖² + λ‖α‖₁
α_{t+1} = S_θ(α_t − η Aᵀ(Aα_t − y)),   θ = ηλ
S_θ(x) = sign(x)·max(|x| − θ, 0)
```
`[CHOICE]` `η = 1/L`, `L = σ_max(A)²`, by power iteration — the classical choice
guaranteeing monotone descent.

### LISTA `[PAPER, §3]`
```
α_i^{(t+1)} = η_θ(S α_i^{(t)} + B Z_i)
```
Expanding the ISTA step shows this is a re-parameterisation:
```
α_{t+1} = S_θ((I − ηAᵀA)α_t + ηAᵀy)
W_s = I − ηAᵀA ∈ ℝ^{k×k}      W_e = ηAᵀ ∈ ℝ^{k×p}
```
so the paper's `(S, B)` are exactly `(W_s, W_e)`. `[CHOICE]` We **initialise LISTA at
these values**, so the network starts as exact ISTA and any gain is attributable to
learning rather than to a weak baseline. Asserted by `tests/test_lista.py`.

**Cost note.** `W_s` is `k × k`, so one LISTA layer costs `O(k²)` per token while one
ISTA iteration costs `O(pk)`. LISTA is cheaper only because it uses far fewer layers than
ISTA needs iterations.

**A structural detail.** The first layer's `W_s` can never receive gradient, since
`α^{(0)} = 0`. A `t`-layer untied LISTA has `t−1` effective `W_s` blocks. Pinned down by a
unit test.

`[MISSING]` `λ`, step size, iteration count, stopping rule, sparsity level `s`, LISTA
depth `t`, weight tying, threshold parametrisation, training loss, training data, and
optimiser are **none of them reported**.

A further gap: classical ISTA is a non-differentiable fixed-point iteration, so a CSAT
block with an ISTA decoder cannot be trained end-to-end through the decoder. The paper
does not address how the analytic-decoder variant is trained.

**Implemented in** [`models/ista.py`](../src/csat/models/ista.py),
[`models/lista.py`](../src/csat/models/lista.py), [`models/omp.py`](../src/csat/models/omp.py).

## 8. Reported results `[PAPER, §4]` — transcribed, not reproduced

**Table 1 — WikiText-103** (12 layers, 512 hidden, 8 heads, 151M params, 300k-step cap)

| Model | Perplexity ↓ |
|---|---|
| Transformer (Full) | 17.5 |
| Linformer | 19.9 |
| Performer | 20.5 |
| Longformer | 19.1 |
| **CSAT** | **18.7** |

**Table 2 — LRA Pathfinder-X** (length 4096): Transformer 85.0 · Linformer 78.3 ·
Performer 80.4 · Longformer 81.6 · **CSAT 84.2**

**Table 3 — Flickr30k retrieval** (R@1/R@5/R@10): BLIP 82.1/95.5/98.1 ·
+Linformer 78.9/94.1/97.2 · +Performer 80.3/94.8/97.4 · **+CSAT 82.4/95.7/98.3**

**Table 4 — MS-COCO captioning** (CIDEr/BLEU-4): BLIP 121.4/38.2 ·
+Linformer 117.5/36.8 · +Performer 119.0/37.1 · **+CSAT 122.3/38.7**

**Table 5 — Efficiency at n = 4096**

| Model | GPU memory (GB) | Inference (ms) |
|---|---|---|
| Transformer (Full) | 18.4 | 1113 |
| Linformer | 5.8 | 395 |
| Performer | 6.4 | 412 |
| CSAT | 6.9 | 439 |

`[MISSING]` for Table 5, which is why it cannot be reproduced: GPU model, numeric
precision, batch size, how many layers were measured, the value of `m`, the decoder
configuration, and whether the sparse decoding step is included in the 439 ms at all.

These values are stored in `csat.config.PAPER_REPORTED` for reference only.

## 9. Claims

**Complexity `[PAPER, §1]`** — "a significant reduction in complexity from `O(n²d)` to
`O(nmd + decoding)`". The "decoding" term is never expanded. For row-wise ISTA with `T`
iterations it is `O(n·T·p·k)`; for a `t`-layer LISTA it is `O(n·t·k²)`. `exp06` counts
both stages analytically and measures both empirically.

**Efficiency `[PAPER, §4]`** — "Although the sparse decoding step introduces a small
overhead, it is amortized across layers and does not dominate runtime." Directly
testable; `exp06` tests it.

## 10. Limitations the authors state `[PAPER, §7]`

1. The sparsity assumption "may not generalize to tasks involving densely entangled
   representations" — fine-grained video captioning, dense object detection.
2. Iterative recovery "may require multiple matrix-vector multiplications per token,
   which can become a bottleneck if not properly amortized".
3. Learned decoders such as LISTA "may sacrifice some generalization or require
   retraining when sparsity levels or modalities change".
4. Modality mismatch: "projection noise from one modality could corrupt alignment signals
   in the other".
5. Random projections "may introduce non-determinism and variability in performance".

Limitations 3 and 5 are measured here (`exp04`, `exp01`/`exp07`); 1 is probed in `exp05`;
2 is measured in `exp06`.
