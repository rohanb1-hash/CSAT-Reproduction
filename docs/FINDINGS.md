# Findings — Phase 4 (preliminary)

**Every result on this page is preliminary.** It comes from synthetic data, mostly at
random initialisation, at small scale and with a single attention layer. None of it is a
Phase 5 conclusion, and none of it refutes the paper's Tables 1–5, which were not run
here. The scope of each result is stated with it.

Numbers below come from the committed reference run in
[`results/reference/`](../results/reference), produced on CPU with `seed=1234` in QUICK
mode. Regenerate with `csat run all`.

---

## Summary

| # | Finding | Evidence | Confidence |
|---|---|---|---|
| 1 | The paper's decoding equation is dimensionally inconsistent for every `m ≪ n` | `csat check` | **Certain** — arithmetic on the paper's own shapes |
| 2 | No fixed operator relates `Z` to `C`, so the CS problem has no measurement operator | [analysis](DIMENSIONAL_ANALYSIS.md) §4 | **Certain** — follows from the definitions |
| 3 | `M_eff = Ã Φ_V` is ~50% negative with non-unit row sums | `exp02` | High — property of the mechanism, not of our choices |
| 4 | At initialisation `Z` is nearly orthogonal to `C` | `exp01` | Medium — synthetic tokens, untrained |
| 5 | The softmax, not the projection, destroys the correspondence | `exp01` control | Medium-high — matches theory |
| 6 | Context vectors are not sparse in the bases tested | `exp05(a)` | **Low for real models** — synthetic features only |
| 7 | Decoding `Z` does not improve directional agreement with `C` | `exp05(b)` | Medium |
| 8 | LISTA beats ISTA by a wide margin at matched depth | `exp04` | High — standard result, cleanly reproduced |
| 9 | LISTA degrades under sparsity shift, as the paper's §7 warns | `exp04` | High |
| 10 | A fixed random `Φ` fails at content-addressed retrieval even at `m ≈ n` | `exp07` | Medium — one task, one layer |
| 11 | A learnable `Φ` succeeds by learning *not to mix* | `exp07` + Φ diagnostic | Medium |
| 12 | The attention stage scales as claimed; the decoder does not amortise | `exp06` | High for the configurations tested |

---

## 1–2. The decoding equation

See [DIMENSIONAL_ANALYSIS.md](DIMENSIONAL_ANALYSIS.md). These two are not empirical
claims — they follow from the paper's definitions, and CI runs `csat check` on every
push to keep them verified.

## 3. CSAT does not compute a weighted average (`exp02`)

| m | negative entry fraction | mean row sum | true `A` row sum |
|---|---|---|---|
| 16 | 0.501 | 0.725 | 1.000 |
| 32 | 0.481 | 1.215 | 1.000 |
| 64 | 0.498 | 0.060 | 1.000 |
| 128 | 0.506 | −0.051 | 1.000 |

Sharing `Φ_K = Φ_V` changes nothing material. Standard attention's `A` is non-negative
with row sums of exactly 1; `M_eff` is neither. CSAT computes a **signed** linear
combination of value vectors.

That is a legitimate thing for a layer to do. What it rules out is reading "`Z_i` is a
compressed version of `C_i`" in the averaging sense, which is the intuition that would
motivate treating `Z_i` as a noisy observation of `C_i` for decoding.

## 4–5. Fidelity, and what actually breaks it (`exp01`)

Cosine similarity is the column to read: it is scale-free, so it reports directional
agreement independently of norm mismatch.

| tokens | scaling | m | rel. L2 | **cosine** | ‖Z‖/‖C‖ |
|---|---|---|---|---|---|
| i.i.d. | paper `√d_k` | 16 | 29.79 | **0.028** | 29.79 |
| i.i.d. | paper `√d_k` | 64 | 7.89 | **0.014** | 7.84 |
| i.i.d. | paper `√d_k` | 128 | 3.00 | **0.022** | 2.85 |
| i.i.d. | **no softmax (control)** | 16 | 4.05 | **0.233** | 4.17 |
| i.i.d. | **no softmax (control)** | 64 | 1.95 | **0.433** | 2.16 |
| i.i.d. | **no softmax (control)** | 128 | 1.39 | **0.580** | 1.71 |
| redundant | paper `√d_k` | 64 | 2.73 | 0.033 | 2.60 |
| redundant | **no softmax (control)** | 128 | 1.08 | **0.734** | 1.66 |

**The norm mismatch is structural, not a bug.** Each row of `Ṽ = Φ_V V` is a sum of `n`
value rows, so `‖Ṽ_j‖ ~ √(n/m)·‖V‖`, while `Ã` averages only `m` of them. True attention
averages `~n` nearly-independent value rows, so `‖C_i‖` *shrinks* by roughly
`1/√n_eff`. The two scales diverge as `n/m` grows.

**The softmax is what breaks the correspondence.** With a single shared Gaussian `Φ` and
no softmax, `QK̃ᵀṼ = QKᵀΦᵀΦV` is an unbiased estimator of `QKᵀV` because `E[ΦᵀΦ] = I`.
The control's cosine rises steadily with `m`, exactly as that predicts. Applying softmax
to the compressed logits destroys the identity — softmax does not commute with `Φᵀ`.

Redundant tokens do better than i.i.d. ones, so the paper's redundancy intuition is
directionally right, but not nearly enough to close the gap.

**Scope.** Untrained, synthetic, single layer. This does not show that a *trained* CSAT
model fails.

## 6–7. The premise, and whether decoding helps (`exp05`)

### (a) Are context vectors sparse?

| basis | exact? | coefficients for 95% energy | ratio | reconstruction error |
|---|---|---|---|---|
| DCT | yes | 34.8 of 64 | 0.544 | 7e-7 |
| random orthogonal | yes | 37.2 of 64 | 0.582 | 7e-7 |
| learned overcomplete (128 atoms) | no | 16.8 of 128 | 0.131 | **0.125** |

In the DCT basis, context vectors need roughly **half** of all coefficients to carry 95%
of their energy — where 1.0 would mean no compressibility at all. Even a dictionary
**learned on the test data** with twice as many atoms as dimensions reaches useful
concentration only by accepting ~12% reconstruction error, and still puts ~41 non-zeros
per row. These are not `‖α_i‖₀ ≪ d_k` signals.

### (b) Does decoding reduce the error to `C`?

| m | λ | rel. L2 `Z` vs `C` | rel. L2 `Ĉ` vs `C` | **cosine `Z`** | **cosine `Ĉ`** |
|---|---|---|---|---|---|
| 32 | 0.001 | 7.136 | 7.133 | 0.0174 | 0.0174 |
| 32 | 0.200 | 7.136 | 6.582 | 0.0174 | 0.0174 |
| 64 | 0.200 | 3.128 | 2.646 | 0.0555 | 0.0516 |
| 128 | 0.200 | 1.495 | 1.223 | 0.1535 | 0.1341 |

L2 error falls slightly as `λ` grows, but **cosine similarity is unchanged** — pinned at
0.0174 for `m=32` across a 200× change in `λ`. That pattern has one explanation: `Z` is
badly over-scaled, and shrinking an over-scaled estimate toward zero reduces L2 error
*without improving direction*. The apparent "improvement" is a scale artefact. A decoder
that genuinely recovered `C` would move the cosine column.

Bridges 2 and 3 fail for the same underlying reason (rel. L2 0.58–0.96 and 0.80–0.98).
The solvers are not at fault — `exp03` shows they recover truly sparse signals to ~1e-3.
**The data is not sparse.**

### Three reasons finding 6 is low-confidence for real models

1. Our tokens are synthetic Gaussian/prototype constructions, **not** real ViT or BERT
   features. Real features may be far more compressible — the paper cites neural collapse
   for exactly this reason. **Measuring compressibility on real pretrained features is
   the single highest-priority Phase 5 experiment.**
2. Everything is at random initialisation; training could reshape the representation
   toward compressibility.
3. Our `Ψ` choices may simply be the wrong basis, and the paper never says which basis it
   used.

## 8–9. LISTA (`exp04`)

LISTA at initialisation equals ISTA to ~1e-7, which makes the comparison honest by
construction. After training, an 8-layer LISTA matches what classical ISTA needs
hundreds to thousands of iterations to reach. This is the strongest support in this
repository for the paper's choice of a learned decoder.

Trained at `s=8`, LISTA degrades as test sparsity moves away from 8, while the
long-running classical solver does not care. The paper states this risk in §7 and does
not measure it; here it is measured.

## 10–11. Does a CSAT block train? (`exp07`)

Content-addressed associative recall, solvable by one attention layer, chance = 1/32.

| method | m | Φ learnable | accuracy | Φ participation before → after |
|---|---|---|---|---|
| standard attention | — | — | **1.000** | — |
| CSAT | 2 | no | 0.080 | 0.440 → 0.440 |
| CSAT | 8 | no | 0.041 | 0.426 → 0.426 |
| CSAT | 16 | no | 0.074 | 0.375 → 0.375 |
| CSAT | 2 | **yes** | 0.167 | 0.440 → 0.416 |
| CSAT | 8 | **yes** | 0.198 | 0.426 → 0.400 |
| CSAT | 16 | **yes** | **0.918** | 0.375 → **0.219** |

(Full mode, 1200 steps, gives a cleaner monotone progression: ≈0.17 → 0.32 → 0.73 → 1.00
across `m = 2, 4, 8, 16`. QUICK mode under-trains the intermediate points.)

**A fixed random `Φ` fails at every `m`** — barely above chance, *even at `m=16` with
`n=17`, where there is essentially no compression at all*. So the failure is not about
the compression ratio. Mixing the token axis with a fixed random operator destroys
content-addressed retrieval, and the model cannot undo it by adapting `W^Q` and `W^K`,
because the mixture is fixed in *position* space while the content it must retrieve sits
at a data-dependent position.

**A learnable `Φ` recovers** — and the participation-ratio diagnostic says why. The
learned `Φ`'s rows become markedly more concentrated (0.375 → 0.219; the row-max share
rises 0.166 → 0.263). **The learnable variant succeeds by learning not to mix**, drifting
toward a selection/pooling operator.

But a concentrated, data-adapted `Φ` is no longer an incoherent random measurement
operator, so it forfeits the RIP guarantees that are the paper's entire theoretical
contribution. The variant that works is, in effect, a learned token-pooling scheme —
close to Linformer, which is precisely the family the paper positions CSAT against.

**Scope.** One task, one layer, small scale. This cannot refute Tables 1–4. It does
establish that the mechanism trains, and identifies a concrete tension between *working*
and *being compressed sensing*.

## 12. Efficiency (`exp06`, CPU reference run)

| n | standard (ms) | CSAT attn (ms) | ISTA decode (ms) | LISTA decode (ms) | full pipeline (ms) | speedup attn-only | **speedup full** |
|---|---|---|---|---|---|---|---|
| 256 | 1.21 | 1.13 | 10.43 | 4.07 | 11.72 | 1.1× | **0.10×** |
| 512 | 15.06 | 1.52 | 20.79 | 7.25 | 22.81 | 9.9× | **0.66×** |
| 1024 | 62.05 | 3.81 | 53.27 | 15.18 | 55.94 | 16.3× | **1.11×** |
| 2048 | 231.74 | 7.04 | 153.99 | 53.72 | 162.86 | 32.9× | **1.42×** |

(CPU wall-clock varies by tens of percent between runs on a shared machine; the
*ratios* and their direction are the stable part, not the absolute milliseconds.
`m = 64`, 20 ISTA iterations, 8 LISTA layers, batch 1, 8 heads, `d_k = 64`.)

**The attention stage alone scales as advertised** — measured time and analytic FLOPs
both flatten from quadratic toward linear once `m` is fixed, and the stored attention
matrix shrinks by exactly `n/m`. The `O(n²d) → O(nmd)` claim, *for the attention stage*,
is supported.

**The decoder is not a "small overhead."** With 20 ISTA iterations the decoding stage
costs more than the compressed attention it follows. The full-pipeline speedup is a small
fraction of the attention-only speedup, and at small `n` the pipeline is **slower** than
full attention. The paper's statement that decoding "does not dominate runtime" is not
supported at any configuration we can construct — and the paper supplies no configuration
of its own to check against.

LISTA is materially cheaper than ISTA, consistent with using far fewer layers than ISTA
needs iterations.

**Caveat.** These are CPU numbers. GPU ratios will differ — the compressed kernels are
small and may be launch-bound rather than compute-bound. Re-run with `csat run exp06` on
your own hardware; every parameter is recorded in the output.

**Table 5 cannot be reproduced** — not "did not match", but *cannot be attempted*. The
GPU, precision, batch size, layer count, `m`, and decoder configuration are all
unreported, and it is not even stated whether the 439 ms includes decoding.

---

## What would change these conclusions

Ordered by impact — this is the short form of [PHASE5_PLAN.md](PHASE5_PLAN.md).

1. **Real features.** Extract `C` from a pretrained ViT-B/16 and BERT-base and measure
   compressibility. If real context vectors *are* sparse, finding 6 is confined to our
   synthetic setting and findings 7 and 4 need re-testing.
2. **Trained models.** Re-measure `cosine(Z, C)` during training of a small CSAT
   transformer. If it rises, finding 4 describes initialisation only.
3. **Author clarification** of the decoding equation. Findings 1–2 stand regardless, but
   which bridge is intended determines what should be tested next.
