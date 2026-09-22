<div align="center">

# CSAT / CS-VLM — reproduction and analysis

**A faithful PyTorch implementation of *Compressed Sensing Attention* — and a record of
where the paper's mathematics does not close.**

[![CI](https://github.com/OWNER/csat-reproduction/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/csat-reproduction/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-60%20passing-brightgreen.svg)](tests/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.02957-b31b1b.svg)](https://arxiv.org/abs/2507.02957)

</div>

Reproduction of **"CS-VLM: Compressed Sensing Attention for Efficient Vision-Language
Representation Learning"** — Kiruluta, Raju & Burity, [arXiv:2507.02957v1](https://arxiv.org/abs/2507.02957) —
which proposes the **Compressed Sensing Attention Transformer (CSAT)**.

This repository implements what the paper specifies, measures what it asserts, and
documents what it leaves undefined. Every claim is tagged with its provenance, and no
component is marked reproduced merely because the code runs.

---

## The headline finding

The paper's decoding equation

```
Z_i = Φ Ψ α_i        with Φ = Φ_V reused as the measurement matrix
```

**does not type-check under the paper's own declared shapes.** `Φ_V` is `m × n` (token
axis) while `Ψα_i` and `Z_i` are `d_k` (feature axis). The product requires `n = d_k`,
and even then lands in `ℝ^m` rather than `ℝ^{d_k}`. It closes only when `m = n = d_k` —
precisely the case of *no compression at all*, contradicting the paper's own `m ≪ n`.

Verify it in one command:

```bash
csat check
```

There is a second, independent problem: since `C = AV` but `Z = Ã Φ_V V` with *different*
mixing matrices, **no fixed operator relates `Z` to `C`**, so the recovery problem has no
measurement operator regardless of shapes.

Rather than silently repairing the equation, this repository implements the compressed
attention **exactly as written** and implements reconstruction **separately**, as three
labelled, dimension-checked bridges — none of which is claimed to be the paper's method.

📄 Full analysis: **[docs/DIMENSIONAL_ANALYSIS.md](docs/DIMENSIONAL_ANALYSIS.md)**

---

## Install

```bash
git clone https://github.com/OWNER/csat-reproduction.git
cd csat-reproduction

# CPU-only PyTorch (skip if you already have a CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
```

Everything falls back to CPU automatically — nothing here requires a GPU.

## Quick start

```bash
csat info            # device, versions, where results are written
csat check           # the dimensional consistency check on the paper's equation
csat list            # the seven experiments
csat run exp05       # "are context vectors sparse, and does decoding help?"
csat run all         # everything (quick sweeps, a few minutes on CPU)
csat run all --full  # the full sweeps
csat track           # the reproduction tracking table
pytest               # 60 unit tests
```

As a library:

```python
import torch
from csat.models import CompressedAttention, scaled_dot_product_attention
from csat.utils import make_qkv, all_metrics

q, k, v = make_qkv(batch=1, heads=4, seq_len=256, d_k=64)

C, A = scaled_dot_product_attention(q, k, v, return_weights=True)   # ground truth
attn = CompressedAttention(seq_len=256, m=32, d_k=64, n_heads=4)     # the paper's method
Z, A_tilde = attn(q, k, v, return_weights=True)

A.shape, A_tilde.shape        # (1,4,256,256) vs (1,4,256,32) -- the saving
all_metrics(Z, C)             # how far Z is from true attention

# The effective mixing matrix CSAT actually applies (diagnostic only)
M_eff = attn.effective_attention_matrix(q, k)
(M_eff < 0).float().mean()    # ~0.5 -- NOT a weighted average of value vectors
```

## The Kaggle notebook

[`notebooks/csat_reproduction_phase4.ipynb`](notebooks/csat_reproduction_phase4.ipynb)
is a self-contained tutorial: 119 cells that rebuild the entire project inside
`/kaggle/working` with `%%writefile`, run the tests, and work through every experiment
with the mathematics explained alongside. Upload it to Kaggle and run top to bottom; it
installs nothing.

It is **generated from this repository's sources** — the code in the notebook is the code
the test suite covers, modulo the import prefix:

```bash
python scripts/build_notebook.py           # regenerate
python scripts/build_notebook.py --check   # fail if out of date (used in CI)
```

---

## What the reproduction found

All results are **preliminary**: synthetic data, mostly untrained, single-layer. They do
not refute the paper's Tables 1–5, which were not run here. Full detail, with numbers and
confidence levels, in **[docs/FINDINGS.md](docs/FINDINGS.md)**.

| # | Finding | Where |
|---|---|---|
| 1 | The decoding equation is dimensionally inconsistent for every `m ≪ n` | `csat check` |
| 2 | No fixed operator relates `Z` to `C` — the CS problem has no measurement operator | [analysis](docs/DIMENSIONAL_ANALYSIS.md) |
| 3 | `M_eff = Ã Φ_V` is ~50% negative with non-unit row sums: CSAT computes a *signed* combination, not a weighted average | `exp02` |
| 4 | At initialisation `Z` is nearly orthogonal to `C` (cosine ≈ 0.02) | `exp01` |
| 5 | The **softmax**, not the projection, destroys the correspondence — the softmax-free control reaches cosine 0.58–0.73 | `exp01` |
| 6 | Context vectors are **not sparse**: ~35 of 64 DCT coefficients for 95% energy | `exp05` |
| 7 | Decoding lowers L2 error but leaves cosine **exactly unchanged** — it shrinks an over-scaled estimate rather than recovering `C` | `exp05` |
| 8 | LISTA at matched depth beats ISTA by a wide margin (initialised *at* ISTA, so the comparison is honest) | `exp04` |
| 9 | LISTA degrades under sparsity shift — the paper's own §7 caveat, measured | `exp04` |
| 10 | With a **fixed** random `Φ`, a CSAT block fails at content-addressed retrieval even at `m ≈ n` | `exp07` |
| 11 | With a **learnable** `Φ` it succeeds — by learning *not to mix*, forfeiting the incoherence RIP requires | `exp07` |
| 12 | The attention stage scales as claimed, but the decoder does not amortise: full-pipeline speedup 0.10×–1.42× vs 1.1×–32.9× for attention alone | `exp06` |

The solvers themselves are correct — `exp03` reproduces the textbook compressed-sensing
phase transition and recovers truly sparse signals to ~1e-3. **The data is not sparse**,
which is a different problem.

> **The most important caveat.** Findings 4, 6, 7, 10 and 11 rest on *synthetic* tokens,
> not real ViT or BERT features. Real features may be far more compressible — the paper
> cites neural collapse for exactly this reason. Re-testing on real pretrained features
> is the top item in [docs/PHASE5_PLAN.md](docs/PHASE5_PLAN.md).

## Reproduction status

| Status | Count |
|---|---|
| ✅ Implemented as specified | 3 |
| 🟡 Partially implemented (our documented choices) | 8 |
| ❓ Not specified by the paper | 1 |
| ⬜ Not yet implemented (out of Phase 4 scope) | 4 |
| ❌ Cannot reproduce exactly | 4 |

Tables 1–5 are **not** reproduced. The training budgets (300k steps, BLIP fine-tuning)
and the missing configuration — the value of `m` in *any* experiment, the decoder
settings, the hardware, batch size, precision and layer count behind Table 5 — put them
out of reach. Component-by-component detail in
**[docs/REPRODUCTION_TRACKING.md](docs/REPRODUCTION_TRACKING.md)**.

---

## Repository layout

```
csat-reproduction/
├── src/csat/
│   ├── config.py                  every hyper-parameter, each tagged
│   │                              [PAPER] / [CHOICE] / [MISSING] / [INCONSIST]
│   ├── cli.py                     the `csat` command
│   ├── models/
│   │   ├── standard_attention.py  baseline softmax(QKᵀ/√d_k)V
│   │   ├── measurement.py         Φ ensembles + coherence / empirical-RIP diagnostics
│   │   ├── compressed_attention.py CSAT attention + effective-attention diagnostic
│   │   ├── dictionary.py          Ψ: identity / DCT / random orthogonal / overcomplete / learned
│   │   ├── ista.py                ISTA, FISTA, soft threshold, debiasing
│   │   ├── lista.py               LISTA, initialised to be exactly ISTA
│   │   ├── omp.py                 batched orthogonal matching pursuit
│   │   ├── bridges.py             the dimension check + the three decoding readings
│   │   └── csat_block.py          end-to-end block matching the paper's Figure 1
│   ├── utils/                     seeding, metrics, FLOP counting, benchmarking,
│   │                              plotting, reproduction tracking
│   └── experiments/               exp01 … exp07
├── tests/                         60 unit tests
├── notebooks/                     the Kaggle notebook (generated)
├── scripts/                       notebook + tracking-doc generators
├── docs/                          paper spec, dimensional analysis, findings, Phase 5 plan
└── results/reference/             committed reference run (CPU, seed 1234)
```

### Design decisions worth knowing

- **Provenance tags everywhere.** `[PAPER]`, `[CHOICE]`, `[MISSING]`, `[INCONSIST]`.
  Nothing is attributed to the paper unless it is in the paper.
- **LISTA is initialised at exact ISTA**, so any measured gain comes from learning rather
  than from a weak random baseline. A unit test asserts the equality.
- **Causal masks are refused, not faked.** After `K̃ = Φ_K K` each compressed slot mixes
  all `n` keys including future ones, so a causal mask is undefined. The paper reports
  autoregressive language modelling without addressing this.
- **Benchmarks are fair by construction**: warm-up, CUDA synchronisation, repeated
  measurement with mean ± std, allocated *and* reserved memory reported separately, and
  OOM recorded as a result rather than a crash.
- **Theoretical FLOPs and measured runtime are never conflated**, and an attention-only
  speedup is never reported as a pipeline speedup.
- **The tracking table lives in code** (`csat.utils.reporting.TRACKING_ROWS`) so the CLI,
  the notebook, the CSV exports and the docs cannot drift apart.

## The experiments

| | Question |
|---|---|
| `exp01` | How close is the compressed output `Z` to true attention `C`? Includes a softmax-free control that separates projection error from softmax error. |
| `exp02` | Is the effective matrix `Ã Φ_V` still a weighted average of value vectors? |
| `exp03` | Do ISTA / FISTA / OMP recover sparse signals, and in which regime? |
| `exp04` | LISTA vs ISTA at matched budget, and generalisation under sparsity shift. |
| `exp05` | Are context vectors sparse at all, and does decoding `Z` reduce the error to `C`? |
| `exp06` | Runtime and memory — attention stage vs the full pipeline, decoder counted. |
| `exp07` | Does a CSAT block train, on a task that needs precise retrieval? |

---

## Before you push this to GitHub

1. Replace `OWNER` with your GitHub username in `README.md`, `pyproject.toml` and
   `CITATION.cff`.
2. Put your name in `LICENSE` and `CITATION.cff` (both say `<YOUR NAME>`).
3. Push — CI runs tests on Python 3.9/3.11/3.12, lints, re-checks the dimensional
   analysis, and executes the notebook end to end.

```bash
git remote add origin https://github.com/OWNER/csat-reproduction.git
git push -u origin main
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The one rule specific to this project: **findings
must be separable into what the paper specifies, what we chose, and what we measured.**
A pull request that blurs those categories will be asked to un-blur them.

## Citing

If you use this reproduction, please cite both it and the original paper — see
[CITATION.cff](CITATION.cff).

```bibtex
@misc{kiruluta2025csvlm,
  title  = {CS-VLM: Compressed Sensing Attention for Efficient Vision-Language Representation Learning},
  author = {Kiruluta, Andrew and Raju, Preethi and Burity, Priscilla},
  year   = {2025},
  eprint = {2507.02957},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## License

MIT — see [LICENSE](LICENSE). The paper's text, equations and reported results remain the
property of its authors and are quoted here only for analysis and commentary.
