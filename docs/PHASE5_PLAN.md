# Phase 5 — verification plan

Phase 4 (this repository) implements the paper and records preliminary measurements.
Phase 5 is where conclusions get drawn. The items below are ordered by how much each
would change what [FINDINGS.md](FINDINGS.md) currently says.

Each item states the experiment, what it would settle, and what a result in either
direction would mean — so that nothing gets run without a prior commitment about how it
will be read.

---

## Tier 1 — directly tests the paper's premise

### 1. Compressibility of *real* context vectors

**Do:** extract `C = softmax(QKᵀ/√d_k)V` from a pretrained ViT-B/16 and BERT-base across
layers, heads and both modalities. Measure the 95%-energy coefficient count in (a) the
DCT, (b) a PCA basis fitted on the same features, (c) a K-SVD / online-learned
overcomplete dictionary. Report the distribution, not just the mean, and break it down by
layer depth.

**Settles:** the paper's central assumption, and the confidence level of findings 6 and 7.

**Either way:**
- If real `C` *is* sparse (say ≲10% of coefficients for 95% energy), then finding 6 is
  confined to our synthetic setting, the decoder stage deserves re-testing on real
  features, and the paper's premise is vindicated even though its decoding equation still
  needs repair.
- If real `C` is *not* sparse, the paper's motivation does not hold for the
  representations it targets, and the CS framing is decorative rather than load-bearing.

**Cost:** low. Inference only, no training. This is the single highest-value experiment
in the plan and should be done first.

### 2. Fidelity during training

**Do:** train a small transformer (4–6 layers) with CSAT blocks on a real task; log
`cosine(Z, C)`, `‖Z‖/‖C‖` and the compressibility of `C` at intervals throughout
training.

**Settles:** whether finding 4 (near-orthogonality of `Z` and `C`) is a property of
initialisation or of the mechanism.

**Either way:** if cosine rises substantially during training, our untrained measurements
describe a transient and the paper's results become plausible by a route the paper does
not describe. If it stays flat, CSAT's output is best understood as a *different* mixing
operator that happens to work, not as an approximation of attention.

### 3. Ask the authors

**Do:** contact the authors about `Z_i = ΦΨα_i`; check for a v2 or a published version.

**Settles:** which of the three bridges (if any) was intended.

**Note:** findings 1 and 2 stand regardless — the published v1 is inconsistent as
written. But the answer determines which experiments in Tier 2 are worth running.

---

## Tier 2 — tests the mechanism's claims

### 4. RIP applicability

**Do:** state precisely what, along the token axis, would have to be sparse for
`Φ ∈ ℝ^{m×n}` to be a meaningful measurement operator. Then estimate `δ_2s` for the
required `s` at plausible `m` using a proper combinatorial search over small instances
plus the Monte-Carlo lower bound at scale. Check against `δ_2s < √2 − 1 ≈ 0.414`.

**Settles:** whether the RIP assertion in §3 is satisfiable at any `m` the paper could
have used.

### 5. Does a learned `Φ` always stop mixing?

**Do:** extend the participation-ratio diagnostic from `exp07` to longer training, larger
`n`, more layers and a real task. Track whether `Φ` converges toward a
selection/pooling operator in every case, or whether some regimes keep a dense, incoherent
`Φ` *and* good accuracy.

**Settles:** finding 11, which is the sharpest tension in the reproduction — if the
variant that works is always the one that abandons incoherence, then "CSAT that works" is
a learned pooling method and belongs in a head-to-head with Linformer on those terms,
not on compressed-sensing terms.

### 6. Causal masking

**Do:** determine how the WikiText-103 result (Table 1) was obtained given that
compressed keys mix future tokens. Implement and evaluate a causal-compatible variant:
block-wise `Φ` over a prefix, or a prefix-restricted measurement operator recomputed per
position.

**Settles:** whether the autoregressive result is attainable at all with token-axis
compression, and at what cost.

### 7. Full-pipeline efficiency frontier

**Do:** sweep ISTA iterations `T ∈ {5, 10, 20, 50, 100}` and LISTA depth
`t ∈ {2, 4, 8, 16}` against downstream accuracy on a real task, on GPU, measuring the
complete pipeline. Plot accuracy against wall-clock.

**Settles:** whether an operating point exists where CSAT is both accurate and faster than
full attention — finding 12 says it does not at the configurations we could construct,
but we had no accuracy axis to trade against.

---

## Tier 3 — the paper's own benchmarks

These require training budgets well beyond a notebook and configuration details the paper
does not supply, so each needs assumptions documented before it is run.

8. **WikiText-103** (Table 1), with `m` and the decoder configuration stated.
9. **LRA Pathfinder-X** (Table 2) — note this task is notoriously recipe-sensitive.
10. **BLIP + CSAT** on Flickr30k / MS-COCO (Tables 3–4), stating which layers were replaced.
11. **Linformer, Performer, Longformer** baselines under identical conditions.
12. **Table 5 re-measured** with every parameter reported: GPU, precision, batch size,
    layer count, `m`, decoder config, and whether decoding is included.

A negative result in Tier 1 item 1 would make Tier 3 less interesting but not pointless:
a method can work for reasons other than the ones claimed for it, and Tier 3 would then
be measuring *that*.

---

## Tier 4 — failure characterisation

13. **Where the sparsity assumption breaks.** Dense prediction and fine-grained
    captioning — the regimes the paper's own §7 flags.
14. **Variance across `Φ` draws.** The paper's §5 non-determinism caveat, measured over
    20+ seeds at several `m`, reporting the spread and not just the mean.
15. **Modality mismatch.** Separate vs shared `Φ` for visual and textual tokens, testing
    the §7 claim that "projection noise from one modality could corrupt alignment signals
    in the other".

---

## Reporting standard for Phase 5

Whatever the outcome:

- every number reported with the hardware, seed, and full configuration that produced it;
- claims separated into *what the paper specifies*, *what we chose*, and *what we measured*;
- negative results reported as prominently as positive ones;
- the tracking table in `csat.utils.reporting` updated, and
  `python scripts/gen_tracking_doc.py` re-run, so the documentation cannot drift from the
  code.
