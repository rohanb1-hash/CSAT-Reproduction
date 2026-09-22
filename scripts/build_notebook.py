#!/usr/bin/env python3
"""Regenerate the Kaggle notebook from this repository's package sources.

The repository is the single source of truth. The notebook is a *self-contained*
teaching artefact: it recreates the whole project inside ``/kaggle/working`` with
``%%writefile`` cells, so it must run on Kaggle with nothing installed. That means
its files use flat imports (``from models.ista import ista``) while the installed
package uses ``csat.``-prefixed ones. This script performs that rewrite, so the
code in the notebook is byte-for-byte the code that the test suite covers, modulo
the import prefix.

Usage
-----
    python scripts/build_notebook.py                 # regenerate the notebook
    python scripts/build_notebook.py --check         # fail if it is out of date
    python scripts/build_notebook.py -o other.ipynb  # write elsewhere
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KROOT = "/kaggle/working/csat-reproduction"
DEFAULT_OUT = os.path.join(REPO, "notebooks", "csat_reproduction_phase4.ipynb")

# notebook path -> path inside the repository
SOURCE_MAP = {
    "configs/config.py": "src/csat/config.py",
    **{f"models/{n}.py": f"src/csat/models/{n}.py" for n in (
        "standard_attention", "measurement", "compressed_attention", "dictionary",
        "ista", "lista", "omp", "bridges", "csat_block")},
    **{f"utils/{n}.py": f"src/csat/utils/{n}.py" for n in (
        "seed", "metrics", "tensor_utils", "flops", "benchmarking", "plotting",
        "reporting")},
    **{f"experiments/{n}.py": f"src/csat/experiments/{n}.py" for n in (
        "exp01_attention_fidelity", "exp02_effective_attention",
        "exp03_sparse_recovery", "exp04_lista_training", "exp05_decoder_bridge",
        "exp06_efficiency", "exp07_learnability")},
    **{f"tests/{n}.py": f"tests/{n}.py" for n in (
        "test_standard_attention", "test_compressed_attention", "test_ista",
        "test_lista", "test_measurement_dictionary")},
}

# csat.* -> flat imports, for the notebook's self-contained layout
IMPORT_REWRITES = [
    (r"\bfrom csat\.utils\.", "from utils."),
    (r"\bfrom csat\.models\.", "from models."),
    (r"\bfrom csat\.experiments import\b", "from experiments import"),
    (r"\bfrom csat\.experiments\.", "from experiments."),
    (r"\bfrom csat import config\b", "from configs import config"),
]

# The notebook's package __init__ files perform NO eager imports: %%writefile
# creates the modules one at a time, in the paper's own order, so an __init__ that
# imported every submodule would fail on the first execution. The installed
# package has no such constraint and re-exports eagerly.
INIT_FILES = {
    "configs/__init__.py": '''"""Configuration package. Import the module explicitly:

    from configs import config as cfg_mod
    cfg = cfg_mod.quick(cfg_mod.ProjectConfig())
"""
''',
    "models/__init__.py": '''"""Model components for the CSAT (CS-VLM) reproduction.

NOTE ON IMPORT STYLE
    This file deliberately performs NO eager imports. The notebook creates the
    modules one at a time with %%writefile, in the order the paper builds them
    up, and an __init__ that imported every submodule would fail on the first
    import simply because a later file did not exist yet. Import submodules
    explicitly instead:

        from models.standard_attention import scaled_dot_product_attention
        from models.compressed_attention import CompressedAttention
        from models.ista import ista, soft_threshold
"""

__all__ = [
    "standard_attention", "measurement", "compressed_attention", "dictionary",
    "ista", "lista", "omp", "bridges", "csat_block",
]
''',
    "utils/__init__.py": '''"""Utilities for the CSAT (CS-VLM) reproduction.

No eager imports: the notebook writes these modules one at a time, so importing
submodules explicitly is what keeps every cell runnable in order.

    from utils.seed import set_seed, get_device
    from utils.metrics import relative_l2, cosine_similarity
    from utils.benchmarking import benchmark
"""

__all__ = [
    "seed", "metrics", "tensor_utils", "flops", "benchmarking", "plotting", "reporting",
]
''',
    "experiments/__init__.py": '''"""Experiment scripts.

Each module exposes ``run(...)`` and returns plain dicts/lists so the notebook
can turn results straight into DataFrames, CSV and JSON. No eager imports, for
the same reason as models/__init__.py -- import what you need:

    from experiments import exp01_attention_fidelity
"""

__all__ = [
    "exp01_attention_fidelity", "exp02_effective_attention", "exp03_sparse_recovery",
    "exp04_lista_training", "exp05_decoder_bridge", "exp06_efficiency",
    "exp07_learnability",
]
''',
    "tests/__init__.py": '''"""Unit tests for the CSAT reproduction.

Run with:  PYTHONPATH=<project root> python -m pytest tests/ -q
"""
''',
}

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n")})


def _read_source(relpath: str) -> str:
    """Return the notebook-flavoured source for one project file."""
    if relpath in INIT_FILES:
        return INIT_FILES[relpath].rstrip("\n")
    src_path = SOURCE_MAP.get(relpath)
    if src_path is None:
        raise KeyError(f"no source mapped for notebook file {relpath!r}")
    body = open(os.path.join(REPO, src_path)).read()
    for pattern, replacement in IMPORT_REWRITES:
        body = re.sub(pattern, replacement, body)
    return body.rstrip("\n")


def writefile(relpath):
    """Emit a %%writefile cell carrying the package source for ``relpath``."""
    code(f"%%writefile {KROOT}/{relpath}\n{_read_source(relpath)}")


# ============================================================================ #
# 0. TITLE
# ============================================================================ #
md(r"""
# CSAT / CS-VLM — Phase 4: PyTorch Implementation and Paper Reproduction

Reproduction of **"CS-VLM: Compressed Sensing Attention for Efficient Vision-Language
Representation Learning"** — Andrew Kiruluta, Preethi Raju, Priscilla Burity,
arXiv:2507.02957v1 (30 June 2025), which proposes the **Compressed Sensing Attention
Transformer (CSAT)**.

---

## What this notebook is

A complete, modular, executable implementation of everything the paper specifies,
built up in the order the paper builds it, with every component tested and every
claim either measured or explicitly marked unmeasurable.

## What this notebook is **not**

It is not a claim that the paper's Tables 1–5 have been reproduced. They have not,
and §12 explains exactly why they cannot be from the information the paper provides.
All observations here are labelled **preliminary**; comprehensive verification is
Phase 5.

## The one thing to read before anything else

The paper's decoding equation

$$Z_i = \Phi\,\Psi\,\alpha_i$$

**does not type-check under the paper's own declared shapes.** §2.6 works this out in
full and `models/bridges.py` verifies it arithmetically. Rather than silently
"repairing" the equation, this notebook implements the compressed attention exactly as
specified, and implements three separate, clearly labelled, dimensionally consistent
reconstruction experiments — none of which is claimed to be the paper's method.

---

## Provenance tags used everywhere in this notebook and codebase

| Tag | Meaning |
|---|---|
| `[PAPER]` | Stated explicitly in arXiv:2507.02957v1 |
| `[CHOICE]` | The paper leaves this free; we picked a standard value and say so |
| `[MISSING]` | The paper *needs* this quantity but never reports it |
| `[INCONSIST]` | The paper's own statement is mathematically inconsistent here |

Nothing is attributed to the paper unless it is actually in the paper.

---

## Contents

| § | Section |
|---|---|
| 1 | How to run this notebook |
| 2 | **Paper specification** — every equation, its dimensions, and its ambiguities |
| 3 | Phase A — environment, seeds, hardware |
| 4 | Phase B — standard attention |
| 5 | Phase C — measurement matrices and compressed attention |
| 6 | Phase D — sparse representation and the three decoding bridges |
| 7 | Phase E — ISTA / FISTA / OMP |
| 8 | Phase F — LISTA |
| 9 | The end-to-end CSAT block (paper Figure 1) |
| 10 | Unit tests |
| 11 | Phase G — synthetic experiments |
| 12 | Phase H — complexity, runtime and memory |
| 13 | Reproduction tracking table |
| 14 | Summary, preliminary observations, and the Phase 5 plan |
""")

md(r"""
## 1. How to run this notebook

* **Runtime:** Kaggle *GPU T4 x2* or *GPU P100*. Everything falls back to CPU
  automatically — no cell requires a GPU.
* **Order:** run top to bottom. Every cell depends only on cells above it; there is no
  hidden state and no cell that must be re-run.
* **Speed:** the first code cell sets `CSAT_QUICK=1`, which shrinks every sweep so the
  whole notebook finishes in a few minutes. Set it to `"0"` for the full sweeps
  (roughly 20–40 minutes on a T4).
* **Outputs:** all tables land in `/kaggle/working/csat-reproduction/results/` as both
  CSV and JSON; figures land in `figures/`.
* **Packages:** nothing is installed. The notebook uses only `torch`, `numpy`,
  `pandas`, `matplotlib` and `pytest`, all present in the Kaggle image.
""")

# ============================================================================ #
# 2. PAPER SPECIFICATION
# ============================================================================ #
md(r"""
---
# 2. Paper Specification

Everything in this section is read off the paper. Each equation is given as the paper
writes it, then its tensor dimensions, then what the operation does, then how we
implement it, then what is ambiguous or missing.

## 2.1 The problem being solved

Self-attention costs $\mathcal{O}(n^2 d)$ in time and memory for sequence length $n$
and embedding dimension $d$. The paper (§1) targets vision-language models, where
"attention must be computed not only within modalities but also across them", making
the quadratic term the dominant cost for long video sequences and high-resolution
image-token streams.

The paper's hypothesis (§1, §2): *attention context vectors — the weighted sums of
value vectors produced by attention — are sparse or compressible in some fixed or
learned basis.* If so, compressed sensing says they can be recovered from far fewer
linear measurements than their ambient dimension, so attention can be computed in a
compressed space and the full output recovered by sparse decoding.

This hypothesis is an **empirical claim about data**, and §11.5 of this notebook
measures it rather than assuming it.
""")

md(r"""
## 2.2 Standard attention `[PAPER, §3]`

$$Q = XW^Q,\qquad K = XW^K,\qquad V = XW^V$$
$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

**Dimensions (as the paper writes them, single head)**

| Symbol | Shape | Meaning |
|---|---|---|
| $X$ | $n \times d$ | input token sequence |
| $W^Q, W^K, W^V$ | $d \times d_k$ | learned projections |
| $Q, K, V$ | $n \times d_k$ | queries, keys, values |
| $QK^{\top}$ | $n \times n$ | the quadratic term |
| $A = \mathrm{softmax}(\cdot)$ | $n \times n$ | row-stochastic attention matrix |
| $C = AV$ | $n \times d_k$ | context vectors; row $i$ is $C_i \in \mathbb{R}^{d_k}$ |

**Operation.** Each query scores every key; the softmax turns each score row into a
probability distribution; the context vector $C_i$ is the resulting **convex
combination** of value vectors. Two properties matter later: $A \ge 0$ and each row of
$A$ sums to $1$.

**Implementation.** `models/standard_attention.py`, in the multi-head layout
`[B, H, N, D]`.

**Ambiguity.** `[MISSING]` The paper writes the single-head case and never states how
heads compose. We use the standard Vaswani convention $d_{\text{model}} = H \cdot d_k$,
consistent with the paper's own "512 hidden dimensions and 8 attention heads" (§4).
""")

md(r"""
## 2.3 The compression mechanism `[PAPER, §3]`

$$\widetilde{K} = \Phi_K K \in \mathbb{R}^{m \times d_k},
\qquad \widetilde{V} = \Phi_V V \in \mathbb{R}^{m \times d_k},
\qquad \Phi_K, \Phi_V \in \mathbb{R}^{m \times n},\quad m \ll n$$

**Operation.** $\Phi$ acts on the **token axis**: it replaces $n$ key/value rows with
$m$ random *linear mixtures* of them. This is not selection or pooling — compressed
slot $j$ is a weighted sum of **all** $n$ tokens.

**What the paper says about $\Phi$.** They are "measurement matrices satisfying the
RIP" and "typically drawn from sub-Gaussian ensembles (e.g., random Gaussian,
Rademacher, or structured Hadamard matrices) that exhibit low coherence with sparse
bases". §7 adds that they "can be fixed post-training or made learnable".

**What the paper does not say** — all `[MISSING]`:

1. **The normalisation constant.** We use entries $\mathcal{N}(0, 1/m)$ so that
   $\mathbb{E}[\Phi^{\top}\Phi] = I_n$, the convention under which the standard RIP
   results hold. This matters: the scale of $\widetilde{K}$ directly changes the softmax
   temperature (§2.4).
2. **The value of $m$ — in any experiment, including Table 5.** Every $m$ in this
   notebook is ours and is always reported alongside its result.
3. Whether $\Phi$ is shared across heads, layers or modalities.
4. Whether $\Phi$ is re-drawn per batch. We fix it at initialisation, which is what
   "fixed post-training" implies.

**RIP, stated precisely.** $\Phi$ satisfies the RIP of order $s$ with constant
$\delta_s$ if for every $s$-sparse $x$,
$$(1-\delta_s)\|x\|_2^2 \le \|\Phi x\|_2^2 \le (1+\delta_s)\|x\|_2^2 .$$
For i.i.d. sub-Gaussian entries this holds w.h.p. once $m = \mathcal{O}(s\log(n/s))$.

**A question the paper does not answer.** $\Phi_K$ and $\Phi_V$ act along the *token*
axis, so the RIP requirement ties $m$ to the sparsity of *something along the token
axis*. The paper's sparsity assumption, however, is about **context vectors in feature
space** ($\alpha_i \in \mathbb{R}^{d_k}$). The object whose sparsity would justify
$m \ll n$ is never identified. We therefore report computable proxies (mutual
coherence, a Monte-Carlo lower bound on $\delta_s$) and mark exact RIP verification —
which is NP-hard in general — as `Cannot reproduce exactly`.
""")

md(r"""
## 2.4 Compressed attention `[PAPER, §3]`

$$\widetilde{A} = \mathrm{softmax}\!\left(\frac{Q\widetilde{K}^{\top}}{\sqrt{d_k}}\right)
\in \mathbb{R}^{n \times m},
\qquad Z = \widetilde{A}\,\widetilde{V} \in \mathbb{R}^{n \times d_k}$$

| Tensor | Shape | Note |
|---|---|---|
| $Q$ | $n \times d_k$ | queries are **not** compressed |
| $\widetilde{K}^{\top}$ | $d_k \times m$ | |
| $\widetilde{A}$ | $n \times m$ | the $n\times n$ matrix is never formed — this is the saving |
| $Z$ | $n \times d_k$ | the compressed attention output |

**Implementation.** `models/compressed_attention.py`, exactly as written.

### Two consequences the paper does not state, both of which we measure

**(a) The effective attention matrix is not a weighted average.** Since
$\widetilde{V} = \Phi_V V$,

$$Z \;=\; \widetilde{A}\,\widetilde{V} \;=\; \widetilde{A}\,\Phi_V V \;=\; \underbrace{\left(\widetilde{A}\,\Phi_V\right)}_{=:\,M_{\text{eff}} \in \mathbb{R}^{n\times n}} V .$$

So CSAT applies an effective $n \times n$ mixing matrix $M_{\text{eff}}$ in place of
$A$. But $\Phi_V$ has negative entries, so $M_{\text{eff}}$ is in general **neither
non-negative nor row-stochastic**, while $A$ is both. Experiment 02 measures this.

**(b) $Z_i$ and $C_i$ live in the same space.** Both are in $\mathbb{R}^{d_k}$. The
compression is along the token axis, and the attention weighting *sums over* that axis.
So a single row $Z_i$ is **not** an undersampled measurement of $C_i$ — it is a
same-dimensional approximation of it. This is the root of §2.6.

**Ambiguity.** `[MISSING]` The paper keeps the $\sqrt{d_k}$ scaling unchanged after
projection. But each row of $\widetilde{K}$ is a sum of $n$ key rows, so the logits
$Q\widetilde{K}^\top$ have a different variance than $QK^\top$, which shifts the softmax
temperature. We implement the paper's literal formula as the **default** and offer two
re-scalings as explicitly labelled diagnostics.

**Masking.** `[MISSING]` The paper reports autoregressive language modelling on
WikiText-103 (Table 1) but never discusses causal masking. After $\widetilde{K} =
\Phi_K K$, compressed slot $j$ mixes **all** $n$ keys including future ones, so no mask
over $m$ slots can enforce "token $i$ may not see token $j>i$". Our implementation
**refuses** a causal mask rather than applying a meaningless one.
""")

md(r"""
## 2.5 Sparse representation `[PAPER, §3]`

> "Suppose there exists a dictionary $\Psi \in \mathbb{R}^{d_k \times d_k}$ such that the
> true context vector $C_i$ admits a sparse representation: $C_i = \Psi\alpha_i$, where
> $\alpha_i \in \mathbb{R}^{d_k}$ is sparse."

| Symbol | Shape |
|---|---|
| $\Psi$ | $d_k \times d_k$ |
| $\alpha_i$ | $d_k$, with $\|\alpha_i\|_0 \ll d_k$ |
| $C_i = \Psi\alpha_i$ | $d_k$ |

**A logical point that governs the whole experiment design.** $\Psi$ is *square*. If it
is invertible, then $C_i = \Psi\alpha_i$ has an exact solution $\alpha_i = \Psi^{-1}C_i$
for **every** $C_i$ whatsoever. The existence of a representation is therefore vacuous;
only its **sparsity** carries content, and sparsity is an empirical property of the data,
not a consequence of the model. Experiment 05 measures it.

**Ambiguity** — all `[MISSING]`: how $\Psi$ is obtained (fixed? learned? from what
data?), whether it is shared across heads/layers/modalities, the sparsity level $s$
actually observed, and any evidence at all that context vectors are sparse in any basis.
We default to the DCT, since the paper's own motivation is the JPEG analogy ("natural
images are known to be sparse in wavelet, DCT, or learned convolutional bases", §3), and
we additionally *learn* a dictionary on the test data — the most favourable case
possible — to give the assumption its best chance.
""")

md(r"""
## 2.6 `[INCONSIST]` The decoding equation does not type-check

This is the central obstacle to reproducing the paper's full method, so it is worked
through in full.

### What the paper writes (§3)

> "Then the observed compressed output $Z_i$ can be written as:
> $$Z_i = \Phi\Psi\alpha_i, \quad\text{with } \|\alpha_i\|_0 \ll d_k,$$
> where $\Phi = \Phi_V$ is reused as the measurement matrix for decoding."

### The declared shapes, collected

| Object | Shape | Source |
|---|---|---|
| $\Phi_V$ | $m \times n$ | §3, compression |
| $\Psi$ | $d_k \times d_k$ | §3, dictionary |
| $\alpha_i$ | $d_k$ | §3, dictionary |
| $\Psi\alpha_i$ | $d_k$ | matrix–vector product |
| $Z_i$ | $d_k$ | row of $Z \in \mathbb{R}^{n \times d_k}$ |

### Substituting

$$\underbrace{\Phi_V}_{m \times n}\ \underbrace{(\Psi\alpha_i)}_{d_k}$$

* The product is **defined only if** $n = d_k$.
* Even then, the result lies in $\mathbb{R}^{m}$, whereas $Z_i \in \mathbb{R}^{d_k}$,
  so we also need $m = d_k$.
* Together: $m = n = d_k$ — which **contradicts the paper's own $m \ll n$**.

So the equation closes only in the degenerate case of *no compression at all*. This is
checked arithmetically by `check_paper_equation()` in §6, and asserted by a unit test.

### A second, independent problem: there is no measurement operator

Set shapes aside. Write $A = \mathrm{softmax}(QK^{\top}/\sqrt{d_k})$ and
$\widetilde{A} = \mathrm{softmax}(Q\widetilde{K}^{\top}/\sqrt{d_k})$. Then

$$C = A V \qquad\text{but}\qquad Z = \widetilde{A}\,\Phi_V V .$$

These use **different mixing matrices**, and $\widetilde{A}$ depends on $\Phi_K$, $Q$
and $K$. **There is no fixed $\Phi$ for which $Z = \Phi C$.** Compressed sensing needs a
known linear measurement operator relating the observation to the signal; here none
exists. Shapes could be patched; this cannot.

### A third: nothing is undersampled per row

$\dim(Z_i) = \dim(C_i) = d_k$. A single row poses no underdetermined inverse problem,
so RIP and $\ell_1$ recovery have nothing to act on at the row level.

### What we do about it

Per the reproduction protocol: implement the explicitly defined mechanism, implement
mathematically consistent recovery **separately**, label both, explain the gap, and claim
nothing about the combination.

`models/bridges.py` provides three readings, each dimension-checked:

| Bridge | Construction | Well-posed? | Honest label |
|---|---|---|---|
| **1 `denoise`** | $Z_i \approx C_i + e$; solve $\min \tfrac12\|Z_i-\Psi\alpha\|^2 + \lambda\|\alpha\|_1$ | Yes | Consistent with the paper's symbols and with $\hat{C}_i = \Psi\hat{\alpha}_i$, but **not compressed sensing** — $A=\Psi$ is square, nothing is undersampled, RIP is irrelevant |
| **2 `feature_cs`** | new $\Phi_f \in \mathbb{R}^{p \times d_k}$, $y_i = \Phi_f C_i$, $A = \Phi_f\Psi$ | Yes | Genuine CS, but $\Phi_f$ **appears nowhere in the paper** and $y_i$ is not the paper's $Z_i$ |
| **3 `token_cs`** | per feature column, $\widetilde{V}_{:,j} = \Phi_V V_{:,j}$, $V_{:,j} = \Psi_{\text{tok}}\beta_j$ | Yes | The **only** reading where $\Phi\in\mathbb{R}^{m\times n}$ composes and $m \ll n$ is meaningful — but it recovers $V$, not $C$, and recovering $V$ then running full attention costs $\mathcal{O}(n^2d)$ again |

**None of these is the paper's method.** Bridge 1 is used in the end-to-end block
because it is the only one that consumes $Z$ directly.
""")

md(r"""
## 2.7 The solvers `[PAPER, §2–3, §5]`

The paper poses basis pursuit,
$$\hat{\alpha}_i = \arg\min_{\alpha}\|\alpha\|_1 \quad\text{s.t.}\quad Z_i = \Phi\Psi\alpha,$$
then says exact convex solvers are "often computationally expensive" and that CSAT
"instead leverages fast approximate solvers, such as ISTA … or its learned variant
LISTA"; §2 also names OMP.

### ISTA
We solve the standard unconstrained relaxation
$$F(\alpha) = \tfrac{1}{2}\|y - A\alpha\|_2^2 + \lambda\|\alpha\|_1,$$
$$\alpha_{t+1} = \mathcal{S}_{\theta}\!\left(\alpha_t - \eta A^{\top}(A\alpha_t - y)\right),
\qquad \theta = \eta\lambda,$$
$$\mathcal{S}_{\theta}(x) = \mathrm{sign}(x)\max(|x|-\theta, 0).$$
With $\eta \le 1/L$, $L = \sigma_{\max}(A)^2$, the objective is non-increasing. We take
$\eta = 1/L$ by power iteration.

### LISTA `[PAPER, §3]`
$$\alpha_i^{(t+1)} = \eta_{\theta}\!\left(S\alpha_i^{(t)} + BZ_i\right)$$
"where $S, B$ are learned weight matrices, $\eta_\theta$ is a learned soft-thresholding
function, and $t$ is the number of iterations (layers)."

This is a re-parameterisation of ISTA. Expanding the ISTA step:
$$\alpha_{t+1} = \mathcal{S}_{\theta}\big((I - \eta A^{\top}A)\alpha_t + \eta A^{\top}y\big),$$
so the paper's $S$ and $B$ are
$$W_s = I - \eta A^{\top}A \in \mathbb{R}^{k\times k}, \qquad
W_e = \eta A^{\top} \in \mathbb{R}^{k\times p}.$$
We **initialise LISTA at exactly these values**, so the network starts as exact ISTA and
any measured gain is attributable to learning rather than to a weak baseline. A unit
test asserts the equality at initialisation.

### Reconstruction `[PAPER, §3]`
$$\hat{C}_i = \Psi\hat{\alpha}_i,$$
applied row-wise to $Z$ to give $\hat{C} \in \mathbb{R}^{n \times d_k}$.

### `[MISSING]` — everything numerical
$\lambda$, step size, iteration count, stopping rule, sparsity level $s$, LISTA depth
$t$, weight tying, threshold parametrisation, training loss, training data, and
optimiser are **none of them reported**. Every value we use is ours, is stated in
`configs/config.py`, and is swept where it matters.

A further gap: classical ISTA is a non-differentiable fixed-point iteration, so a CSAT
block with an ISTA decoder cannot be trained end-to-end through the decoder. The paper
does not address how the analytic-decoder variant is trained.
""")

md(r"""
## 2.8 Datasets, experiments and reported results `[PAPER, §4]`

Transcribed from the paper. **This notebook does not attempt to reproduce these
numbers** — §12 and §13 explain why.

**Table 1 — WikiText-103 language modelling** (12 layers, 512 hidden, 8 heads, 151M
params, 300k-step cap, early stopping on validation perplexity)

| Model | Perplexity ↓ |
|---|---|
| Transformer (Full) | 17.5 |
| Linformer | 19.9 |
| Performer | 20.5 |
| Longformer | 19.1 |
| **CSAT (ours)** | **18.7** |

**Table 2 — LRA Pathfinder-X**, sequence length 4096

| Model | Accuracy ↑ |
|---|---|
| Transformer (Full) | 85.0 |
| Linformer | 78.3 |
| Performer | 80.4 |
| Longformer | 81.6 |
| **CSAT (ours)** | **84.2** |

**Table 3 — Flickr30k retrieval** (R@1 / R@5 / R@10): BLIP 82.1/95.5/98.1 ·
+Linformer 78.9/94.1/97.2 · +Performer 80.3/94.8/97.4 · **+CSAT 82.4/95.7/98.3**

**Table 4 — MS-COCO captioning** (CIDEr / BLEU-4): BLIP 121.4/38.2 ·
+Linformer 117.5/36.8 · +Performer 119.0/37.1 · **+CSAT 122.3/38.7**

**Table 5 — Efficiency at $n=4096$**

| Model | GPU memory (GB) | Inference (ms) |
|---|---|---|
| Transformer (Full) | 18.4 | 1113 |
| Linformer | 5.8 | 395 |
| Performer | 6.4 | 412 |
| CSAT (ours) | 6.9 | 439 |

**`[MISSING]` for Table 5, which is why it cannot be reproduced:** GPU model, numeric
precision, batch size, how many layers were measured, the value of $m$, the decoder
configuration, and whether the sparse decoding step is included in the 439 ms at all.
""")

md(r"""
## 2.9 Claims and author-stated limitations

### Complexity claim `[PAPER, §1]`
> "a significant reduction in complexity from $\mathcal{O}(n^2d)$ to
> $\mathcal{O}(nmd + \text{decoding})$, where $m \ll n$"

The "decoding" term is never expanded. For row-wise ISTA with $T$ iterations, operator
$A \in \mathbb{R}^{p\times k}$ and $n$ tokens it is $\mathcal{O}(n\,T\,p\,k)$; for a
$t$-layer LISTA it is $\mathcal{O}(n\,t\,k^2)$ — note the $k^2$, so a LISTA *layer* is
not intrinsically cheaper than an ISTA *iteration*; LISTA wins only by using far fewer
of them. §12 counts both stages analytically and measures both empirically, and never
reports an attention-only speedup as a pipeline speedup.

### Efficiency claim `[PAPER, §4]`
> "Although the sparse decoding step introduces a small overhead, it is amortized across
> layers and does not dominate runtime."

Directly testable; §12 tests it.

### Limitations the authors themselves state `[PAPER, §7]`
1. The sparsity assumption "may not generalize to tasks involving densely entangled
   representations", e.g. fine-grained video captioning or dense object detection.
2. Iterative recovery "may require multiple matrix-vector multiplications per token,
   which can become a bottleneck if not properly amortized".
3. Learned decoders such as LISTA "may sacrifice some generalization or require
   retraining when sparsity levels or modalities change".
4. Modality mismatch: text and visual tokens differ statistically, complicating the
   design of shared measurement matrices; "projection noise from one modality could
   corrupt alignment signals in the other".
5. Random projections "may introduce non-determinism and variability in performance".

Limitations 3 and 5 are measured in this notebook (Experiments 04 and 01/07); 1 is
probed in Experiment 05; 2 is measured in §12.
""")

md(r"""
## 2.10 Component classification

Following the reproduction protocol, every component is placed in one of four classes.

### A. Explicitly described in the paper — implemented as specified
* Standard attention and the $Q,K,V$ projections
* $\widetilde{K} = \Phi_K K$, $\widetilde{V} = \Phi_V V$ with $\Phi \in \mathbb{R}^{m\times n}$
* $\widetilde{A} = \mathrm{softmax}(Q\widetilde{K}^\top/\sqrt{d_k})$, $Z = \widetilde{A}\widetilde{V}$
* The families of $\Phi$ (Gaussian / Rademacher / structured Hadamard)
* The LISTA recurrence $\alpha^{(t+1)} = \eta_\theta(S\alpha^{(t)} + BZ_i)$
* $\hat{C}_i = \Psi\hat{\alpha}_i$, applied row-wise

### B. Requiring reasonable implementation choices — implemented, choices documented
* $\Phi$ normalisation ($1/\sqrt{m}$), head sharing, fixed-vs-learnable
* Multi-head layout and $d_{\text{model}} = H d_k$
* The dictionary $\Psi$ (DCT default; identity / random orthogonal / overcomplete / learned also provided)
* ISTA step size ($1/L$), $\lambda$, iteration budget; LISTA depth, tying, threshold, optimiser
* Which bridge connects $Z$ to the decoder

### C. Missing or underspecified in the paper — swept, never invented
* The value of $m$ — in every experiment, including Table 5
* The sparsity level $s$, and $\lambda$, and the ISTA iteration count
* How $\Psi$ is obtained; whether $\alpha_i$ is empirically sparse
* LISTA's training data, loss and schedule
* Batch size, precision, hardware and layer count behind Table 5
* Causal masking for the autoregressive WikiText-103 result

### D. Cannot be reproduced exactly from the available information
* **The decoding equation $Z_i = \Phi\Psi\alpha_i$** — inconsistent as written (§2.6)
* **Exact RIP verification** — NP-hard in general; the paper asserts RIP without stating
  $s$ or $\delta_s$, and without identifying what is sparse along the token axis
* **Tables 1–4** — training recipes are not given and the budgets (300k steps, BLIP
  fine-tuning) exceed a notebook
* **Table 5's absolute numbers** — every parameter needed to reproduce them is absent
""")

# ============================================================================ #
# 3. PHASE A — ENVIRONMENT
# ============================================================================ #
md(r"""
---
# 3. Phase A — Environment, directories, hardware and seeds
""")

code(r'''
# --- Project layout and run mode ------------------------------------------- #
import os, sys, json, platform, subprocess, textwrap

PROJECT_ROOT = "/kaggle/working/csat-reproduction"
for sub in ("", "configs", "models", "utils", "experiments", "tests", "results", "figures"):
    os.makedirs(os.path.join(PROJECT_ROOT, sub), exist_ok=True)

# Make the project importable, and tell the modules where to write results.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ["CSAT_ROOT"] = PROJECT_ROOT

# QUICK mode shrinks every sweep so this notebook runs end to end in minutes.
# Set to "0" for the full sweeps.
os.environ["CSAT_QUICK"] = "1"

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
print("project root :", PROJECT_ROOT)
print("quick mode   :", os.environ["CSAT_QUICK"] == "1")
print("directories  :", sorted(os.listdir(PROJECT_ROOT)))
''')

code(r'''
# --- Hardware, versions and package availability ---------------------------- #
import torch, numpy as np

print(f"python        : {platform.python_version()}")
print(f"torch         : {torch.__version__}")
print(f"numpy         : {np.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"cuda version  : {torch.version.cuda}")
    print(f"gpu name      : {torch.cuda.get_device_name(0)}")
    print(f"gpu count     : {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    print(f"gpu memory    : {props.total_memory / 1024**3:.1f} GB")
    print(f"capability    : {props.major}.{props.minor}")
    DEVICE = torch.device("cuda")
else:
    print("no GPU visible -- every cell below falls back to CPU "
          "(slower, but nothing in this notebook requires a GPU)")
    DEVICE = torch.device("cpu")

# Optional packages: we check rather than install.
for pkg in ("pandas", "matplotlib", "pytest", "scipy"):
    try:
        __import__(pkg)
        print(f"{pkg:12s}: available")
    except ImportError:
        print(f"{pkg:12s}: MISSING (only needed for tables/plots/tests)")

print(f"\ndevice selected: {DEVICE}")
''')

md(r"""
### 3.1 `configs/config.py` — every hyper-parameter in one place, each tagged

Nothing in this project uses a magic number that is not declared here with its
provenance. This is what makes the `[PAPER]` / `[CHOICE]` / `[MISSING]` distinction
auditable rather than rhetorical.
""")

writefile("configs/__init__.py")
writefile("configs/config.py")

code(r'''
import importlib
importlib.invalidate_caches()
from configs import config as cfg_mod

cfg = cfg_mod.quick(cfg_mod.ProjectConfig())
cfg_mod.ensure_dirs()

print("QUICK_MODE     :", cfg.quick)
print("seed           :", cfg.seed)
print("attention      :", cfg.attention)
print("m (ours)       :", cfg.csat.m, " <- [MISSING] the paper never states m")
print("ISTA           : lam=%.3f, iters=%d  <- [MISSING] both" % (cfg.ista.lam, cfg.ista.n_iters))
print("LISTA layers   :", cfg.lista.n_layers, " <- [MISSING]")
print("\nPaper-reported numbers we are NOT reproducing here:")
for k, v in cfg_mod.PAPER_REPORTED["efficiency_n4096"].items():
    print(f"   {k:20s} {v[0]:>5} GB   {v[1]:>5} ms")
''')

md(r"""
### 3.2 Seeding and environment capture

`set_seed` seeds Python, NumPy and Torch, and enables deterministic kernels. Note the
deliberate separation: deterministic algorithms are used for *correctness* work, while
`utils/benchmarking.py` re-enables cuDNN autotuning for *timing* work, because
determinism changes kernel selection and therefore runtime. Mixing the two would make
the benchmark numbers meaningless.
""")

writefile("utils/__init__.py")
writefile("utils/seed.py")

code(r'''
importlib.invalidate_caches()
from utils.seed import set_seed, get_device, device_report

set_seed(cfg.seed)
DEVICE = get_device()
env = device_report(DEVICE)
env["python"] = platform.python_version()
env["quick_mode"] = cfg.quick

with open(os.path.join(RESULTS_DIR, "environment.json"), "w") as f:
    json.dump(env, f, indent=2)

print(json.dumps(env, indent=2))
print("\nsaved -> results/environment.json")
''')

md(r"""
### 3.3 Shared helpers: synthetic data and metrics

`tensor_utils` generates the inputs. Note `make_redundant_tokens`: the paper motivates
CSAT with *redundant* visual tokens, but i.i.d. Gaussian tokens have no redundancy at
all, so a method that exploits redundancy cannot possibly look good on them. Sweeping
both regimes is the only fair test.

`metrics` includes `sparsity_profile`, which answers "how many coefficients carry 95% of
a vector's energy?" — the direct empirical test of the paper's central assumption.
""")

writefile("utils/tensor_utils.py")
writefile("utils/metrics.py")

code(r'''
importlib.invalidate_caches()
from utils.tensor_utils import make_qkv, make_redundant_tokens, make_sparse_signals
from utils.metrics import relative_l2, cosine_similarity, sparsity_profile, all_metrics

set_seed(cfg.seed)
q, k, v = make_qkv(2, 4, 64, 16, device=DEVICE)
print("make_qkv        ->", tuple(q.shape), "(B, H, N, D)")

red = make_redundant_tokens(2, 4, 64, 16, n_clusters=8, device=DEVICE)
u, s, _ = torch.linalg.svd(red[0, 0])
print("redundant tokens-> effective rank (95% energy):",
      int((torch.cumsum(s**2, 0) / (s**2).sum() < 0.95).sum().item()) + 1, "of", red.shape[-1])
print("iid tokens      -> effective rank (95% energy):",
      int((torch.cumsum(torch.linalg.svdvals(q[0,0])**2, 0) /
          (torch.linalg.svdvals(q[0,0])**2).sum() < 0.95).sum().item()) + 1, "of", q.shape[-1])

alpha = make_sparse_signals(4, 32, sparsity=3, device=DEVICE)
print("sparse signal   -> nnz per row:", (alpha != 0).sum(1).tolist())
print("sparsity_profile of a generic Gaussian vector:",
      {kk: round(vv, 2) for kk, vv in sparsity_profile(torch.randn(64, 32, device=DEVICE)).items()})
''')

# ============================================================================ #
# 4. PHASE B — STANDARD ATTENTION
# ============================================================================ #
md(r"""
---
# 4. Phase B — Standard attention

**Objective.** Implement the baseline the paper compares against, with enough care that
it can serve as *ground truth* for every fidelity measurement later.

**Equation `[PAPER, §3]`.**
$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

**Dimensions.** `[B, H, N, D]` throughout: batch, heads, tokens, head dimension.
Scores are `[B, H, N, N]` — the quadratic term the paper removes.

**Implementation decisions.**
* Softmax over the **last** axis (the key axis), so each query row is a distribution.
* We rely on `torch.softmax`'s internal max-subtraction for numerical stability rather
  than hand-rolling it; a unit test checks logits of magnitude $10^4$ produce no NaN.
* `causal_mask` is provided here to make the contrast in §5 concrete: masking is
  perfectly well defined for standard attention and becomes undefined once keys are
  mixed across the token axis.
""")

writefile("models/__init__.py")
writefile("models/standard_attention.py")

code(r'''
importlib.invalidate_caches()
from models.standard_attention import (
    scaled_dot_product_attention, StandardAttention, MultiHeadSelfAttention, causal_mask,
)

set_seed(cfg.seed)
q, k, v = make_qkv(2, 4, 32, 16, device=DEVICE)
ctx, w = scaled_dot_product_attention(q, k, v, return_weights=True)

print("Q, K, V          :", tuple(q.shape))
print("attention weights:", tuple(w.shape), "<- the [N, N] term, O(n^2)")
print("context C        :", tuple(ctx.shape))
print("row sums of A    : %.6f (must be 1.0)" % w.sum(-1).mean().item())
print("min weight       : %.6e (must be >= 0)" % w.min().item())

# Agreement with PyTorch's own fused kernel.
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
print("max |ours - torch.nn.functional|: %.2e" % (ctx - ref).abs().max().item())

# Gradient flow.
q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q, k, v))
scaled_dot_product_attention(q2, k2, v2)[0].sum().backward()
print("gradients reach Q/K/V:", all(t.grad is not None and t.grad.abs().sum() > 0
                                    for t in (q2, k2, v2)))

# Numerical stability at extreme logits.
big_ctx, big_w = scaled_dot_product_attention(q * 100, k * 100, v, return_weights=True)
print("large-logit output finite:", bool(torch.isfinite(big_ctx).all()))

# Causal masking works here -- and will be REFUSED by compressed attention.
mask = causal_mask(32, 32, device=DEVICE).view(1, 1, 32, 32)
_, wc = scaled_dot_product_attention(q, k, v, mask=mask, return_weights=True)
print("causal mask: max weight above diagonal = %.2e (must be 0)"
      % wc[0, 0].triu(1).abs().max().item())
''')

md(r"""
**What these outputs establish.** The implementation matches PyTorch's reference kernel
to $10^{-7}$, produces a genuinely row-stochastic non-negative attention matrix,
propagates gradients to all three inputs, survives extreme logits, and honours causal
masks exactly. It is therefore trustworthy as the ground truth $C$ against which every
CSAT measurement below is made.

**What remains for Phase 5.** Nothing for this component — it is the baseline, not a
claim under test. Its role in Phase 5 is to supply $C$ for fidelity and to provide the
`Transformer (Full)` row if the paper's benchmarks are ever run.
""")

# ============================================================================ #
# 5. PHASE C — COMPRESSED ATTENTION
# ============================================================================ #
md(r"""
---
# 5. Phase C — Measurement matrices and compressed attention

## 5.1 Measurement matrices $\Phi \in \mathbb{R}^{m \times n}$

**Objective.** Build the sub-Gaussian and structured ensembles the paper names, plus the
diagnostics that make its RIP assertion checkable.

**Implementation decisions.**
* Entries scaled by $1/\sqrt{m}$ so $\mathbb{E}[\Phi^{\top}\Phi] = I_n$ `[CHOICE]` — the
  paper gives no constant, and this is the convention the RIP literature uses.
* Four ensembles: Gaussian, Rademacher, row-orthogonal, and subsampled randomised
  Hadamard (the "structured" option; requires $n$ a power of two, and the code says so
  rather than silently falling back).
* Two diagnostics: **mutual coherence** (computable) and a **Monte-Carlo lower bound**
  on $\delta_s$. The latter is named `delta_s_lower_bound`, not `delta_s`, because random
  sampling can only under-estimate a worst case — reporting it as $\delta_s$ would be a
  false claim.
""")

writefile("models/measurement.py")

code(r'''
importlib.invalidate_caches()
from models.measurement import (
    make_measurement_matrix, mutual_coherence, empirical_rip_constant, MeasurementMatrix,
)
import pandas as pd

set_seed(cfg.seed)
n_tok, m_meas = 128, 32
rows = []
for ens in ("gaussian", "rademacher", "orthogonal", "hadamard"):
    phi = make_measurement_matrix(m_meas, n_tok, ens, device=DEVICE)
    rip = empirical_rip_constant(phi, sparsity=4, n_trials=2000)
    rows.append({
        "ensemble": ens, "shape": tuple(phi.shape),
        "mutual_coherence": round(mutual_coherence(phi), 4),
        "delta_s_lower_bound (s=4)": round(rip["delta_s_lower_bound"], 4),
        "||Phi x||/||x|| min": round(rip["min_ratio"] ** 0.5, 3),
        "||Phi x||/||x|| max": round(rip["max_ratio"] ** 0.5, 3),
    })
display(pd.DataFrame(rows))

# E[Phi^T Phi] = I_n under the 1/sqrt(m) convention.
acc = torch.zeros(32, 32, device=DEVICE)
for _ in range(200):
    p = make_measurement_matrix(16, 32, "gaussian", device=DEVICE)
    acc += p.T @ p
acc /= 200
print("E[Phi^T Phi]: mean diagonal = %.3f (target 1.0), max |off-diagonal| = %.3f"
      % (acc.diag().mean().item(), (acc - torch.diag(acc.diag())).abs().max().item()))
''')

md(r"""
**What this establishes.** All four ensembles build correctly and the $1/\sqrt{m}$
convention does give $\mathbb{E}[\Phi^{\top}\Phi]\approx I$. Note the
`delta_s_lower_bound` column: even at $s=4$ the bound is already large — for a valid RIP
one needs $\delta_{2s} < \sqrt{2}-1 \approx 0.414$, and these are *lower* bounds on the
true constant. The paper asserts RIP without reporting any such number.

**What remains for Phase 5.** Whether any $m$ the paper could plausibly have used
satisfies RIP at the sparsity its method needs — which first requires the paper to
identify what is sparse along the token axis (§2.3).
""")

md(r"""
## 5.2 Compressed attention

**Objective.** Implement the paper's core mechanism exactly.

**Equations `[PAPER, §3]`.**
$$\widetilde{K} = \Phi_K K,\quad \widetilde{V} = \Phi_V V,\quad
\widetilde{A} = \mathrm{softmax}\!\left(\frac{Q\widetilde{K}^{\top}}{\sqrt{d_k}}\right),\quad
Z = \widetilde{A}\widetilde{V}$$

**Dimensions.** `Q:[B,H,N,D]`, `Φ:[M,N]` (or `[H,M,N]`), `K̃,Ṽ:[B,H,M,D]`,
`Ã:[B,H,N,M]`, `Z:[B,H,N,D]`. The $n\times n$ matrix is never materialised.

**Implementation decisions.**
* Configurable $n$, $m$, $d_k$, batch, heads, ensemble — all as required.
* `share_phi` (use one $\Phi$ for keys and values), `learnable_phi`, `per_head_phi`,
  all defaulting to the reading closest to the paper's text.
* `scaling="paper_sqrt_dk"` is the **default and is the paper's literal formula**. The
  two alternatives are labelled diagnostics, not the paper's implementation.
* A causal mask raises a `ValueError` with an explanation rather than being applied.
* `effective_attention_matrix()` builds $M_{\text{eff}} = \widetilde{A}\Phi_V$ for
  diagnosis only — it is never used in the forward pass or in any timing.
* `LinearCompressedAttention` is a **control that is not in the paper**: with a single
  shared Gaussian $\Phi$ and no softmax, $Q\widetilde{K}^\top\widetilde{V} =
  QK^\top\Phi^\top\Phi V$ is an unbiased estimator of $QK^\top V$ because
  $\mathbb{E}[\Phi^\top\Phi]=I$. Comparing against it attributes error to the softmax
  rather than to the projection.
""")

writefile("models/compressed_attention.py")

code(r'''
importlib.invalidate_caches()
from models.compressed_attention import CompressedAttention, LinearCompressedAttention

set_seed(cfg.seed)
B, H, N, D, M = 2, 4, 256, 64, 32
q, k, v = make_qkv(B, H, N, D, device=DEVICE)

attn = CompressedAttention(seq_len=N, m=M, d_k=D, n_heads=H, device=DEVICE).to(DEVICE)
z, a_tilde = attn(q, k, v, return_weights=True)
ctx_true, a_true = scaled_dot_product_attention(q, k, v, return_weights=True)

print(f"n = {N}, m = {M}, compression ratio n/m = {attn.compression_ratio():.1f}")
print("K~, V~           :", tuple(attn.compress(k, v)[0].shape), "(B, H, M, D)")
print("A~               :", tuple(a_tilde.shape), " vs standard A:", tuple(a_true.shape))
print(f"attention elements: {a_tilde.numel():,} vs {a_true.numel():,} "
      f"({a_true.numel() / a_tilde.numel():.1f}x fewer)")
print("Z                :", tuple(z.shape), " vs C:", tuple(ctx_true.shape), "(same shape!)")
print("A~ rows sum to   : %.6f" % a_tilde.sum(-1).mean().item())

print("\n--- fidelity of Z against the true context C ---")
for kk, vv in all_metrics(z, ctx_true).items():
    print(f"   {kk:20s}: {vv: .4f}")
''')

code(r'''
# --- The two structural consequences derived in section 2.4 ----------------- #
m_eff = attn.effective_attention_matrix(q, k)          # [B,H,N,N]; DIAGNOSTIC ONLY
print("M_eff = A~ Phi_V :", tuple(m_eff.shape))
print()
print("                          standard A        CSAT M_eff")
print("  fraction of negatives:  %8.3f   %15.3f"
      % ((a_true < 0).float().mean().item(), (m_eff < 0).float().mean().item()))
print("  mean row sum         :  %8.3f   %15.3f"
      % (a_true.sum(-1).mean().item(), m_eff.sum(-1).mean().item()))
print("  std of row sums      :  %8.3f   %15.3f"
      % (a_true.sum(-1).std().item(), m_eff.sum(-1).std().item()))

# Verify the identity Z = M_eff V exactly.
z_via = torch.einsum("bhnj,bhjd->bhnd", m_eff, v)
print("\nmax |Z - M_eff V| = %.2e  (the identity Z = (A~ Phi_V) V holds exactly)"
      % (z - z_via).abs().max().item())

# Causal masking is refused, not faked.
try:
    attn(q, k, v, mask=torch.tril(torch.ones(N, N, dtype=torch.bool, device=DEVICE)).view(1,1,N,N))
except ValueError as exc:
    print("\ncausal mask refused, as it must be:\n   " + str(exc)[:200] + " ...")
''')

md(r"""
**What these outputs establish.**

1. The mechanism is implemented exactly: $\widetilde{A}$ is $n\times m$ and row-stochastic,
   $Z$ has the same shape as $C$, and the identity $Z = (\widetilde{A}\Phi_V)V$ holds to
   floating-point precision.
2. **$M_{\text{eff}}$ is roughly 50% negative and its row sums are not 1**, while the
   standard $A$ is non-negative with row sums exactly 1. CSAT's output is therefore *not*
   a weighted average of value vectors, contrary to the reading that $Z_i$ is "a
   compressed version of $C_i$" in an averaging sense. This is a property of the
   mechanism as defined, not of our implementation choices.
3. The fidelity numbers printed above are the first preliminary quantitative result —
   Experiment 01 sweeps them properly.

**What remains for Phase 5.** Whether a *trained* network compensates for (2). Experiment
07 gives a first, narrow answer; a full answer needs the paper's benchmarks.
""")

# ============================================================================ #
# 6. PHASE D — SPARSE REPRESENTATION AND BRIDGES
# ============================================================================ #
md(r"""
---
# 6. Phase D — Sparse representation, and the dimensional consistency check

## 6.1 The dictionary $\Psi$

**Equation `[PAPER, §3]`.** $C_i = \Psi\alpha_i$, $\Psi \in \mathbb{R}^{d_k\times d_k}$,
$\alpha_i$ sparse.

**Implementation decisions.** Five options — identity, DCT, random orthogonal,
overcomplete random, and learnable — with DCT the default because the paper's own
motivation is the JPEG/DCT analogy. `fit_dictionary` additionally *learns* $\Psi$ by
alternating ISTA sparse-coding and gradient dictionary updates, so that §11.5 can give
the sparsity assumption its most favourable possible test.

**The point to keep in view.** A square invertible $\Psi$ represents *every* vector
exactly, so the representation's existence is vacuous and only its sparsity has content.
""")

writefile("models/dictionary.py")

code(r'''
importlib.invalidate_caches()
from models.dictionary import make_dictionary, dct_matrix, Dictionary

set_seed(cfg.seed)
psi = dct_matrix(64, device=DEVICE)
print("DCT Psi          :", tuple(psi.shape))
print("orthonormality   : max |Psi^T Psi - I| = %.2e" % (psi.T @ psi - torch.eye(64, device=DEVICE)).abs().max().item())

# The vacuity point, demonstrated: any vector has an exact DCT representation ...
generic = torch.randn(256, 64, device=DEVICE)
alpha_generic = generic @ torch.linalg.inv(psi).T
print("\nexact representation error for a GENERIC vector: %.2e"
      % relative_l2(alpha_generic @ psi.T, generic))
# ... but it is not remotely sparse.
prof = sparsity_profile(alpha_generic, energy=0.95)
print("coefficients needed for 95%% of energy: %.1f of %d  ->  NOT sparse"
      % (prof["k_for_95pct_mean"], int(prof["dim"])))

# Contrast: a genuinely sparse signal.
sparse_sig = make_sparse_signals(256, 64, sparsity=5, device=DEVICE) @ psi.T
prof_s = sparsity_profile(sparse_sig @ torch.linalg.inv(psi).T, energy=0.95)
print("same measure for a truly 5-sparse signal: %.1f of %d  ->  sparse"
      % (prof_s["k_for_95pct_mean"], int(prof_s["dim"])))
''')

md(r"""
## 6.2 The dimensional consistency check, run as code

Section 2.6 argued on paper that $Z_i = \Phi\Psi\alpha_i$ does not type-check. Here the
argument is executed on the paper's own declared shapes, so it can be inspected rather
than taken on trust — and `models/bridges.py` then builds the three consistent
alternatives.
""")

writefile("models/bridges.py")

code(r'''
importlib.invalidate_caches()
from models.bridges import check_paper_equation, build_bridge, decode_context

print("=" * 78)
print("PAPER'S EQUATION  Z_i = Phi_V Psi alpha_i, checked at the paper's own shapes")
print("=" * 78)
for n_tokens, d_k_dim, m_meas in [(4096, 64, 64), (512, 64, 32), (64, 64, 64)]:
    rep = check_paper_equation(m=m_meas, n=n_tokens, d_k=d_k_dim)
    print(f"\n  n={n_tokens}, d_k={d_k_dim}, m={m_meas}")
    for kk, vv in rep.items():
        print(f"     {kk:38s}: {vv}")
print("\n  => the equation closes ONLY in the degenerate case m = n = d_k,")
print("     i.e. exactly when there is no compression at all.")
''')

code(r'''
# --- The three dimensionally consistent readings ---------------------------- #
set_seed(cfg.seed)
d_k_dim, n_tokens, m_meas = 64, 256, 32

specs = {
    "denoise":    build_bridge("denoise", d_k=d_k_dim, dictionary="dct", device=DEVICE),
    "feature_cs": build_bridge("feature_cs", d_k=d_k_dim, p_features=32,
                               dictionary="dct", device=DEVICE),
    "token_cs":   build_bridge("token_cs", d_k=d_k_dim, n=n_tokens, m=m_meas,
                               dictionary="dct", device=DEVICE),
}
rows = []
for name, spec in specs.items():
    p_dim, k_dim = spec.A.shape
    rows.append({
        "bridge": name, "A (p x k)": tuple(spec.A.shape),
        "Psi (d x k)": tuple(spec.psi.shape),
        "Phi": tuple(spec.phi.shape) if spec.phi is not None else "none",
        "underdetermined (p<k)": spec.is_underdetermined,
        "genuine compressed sensing?": "yes" if spec.is_underdetermined else "NO",
    })
display(pd.DataFrame(rows))

for name, spec in specs.items():
    print(f"\n[{name}]\n" + textwrap.fill(spec.description, 92, subsequent_indent="   "))
''')

md(r"""
**What this establishes.** The inconsistency is arithmetic, not interpretation: with the
paper's own shapes the equation is ill-formed for every $m \ll n$, and closes only when
$m=n=d_k$. The three bridges are each well-formed, and the table makes explicit that only
two of them are actually compressed sensing — the one that plugs into $Z$ (`denoise`) is
**not**.

**What remains for Phase 5.** Whether the authors intended one of these readings, or a
fourth we have not identified. That is a question for the authors; nothing in the paper
resolves it.
""")

# ============================================================================ #
# 7. PHASE E — ISTA
# ============================================================================ #
md(r"""
---
# 7. Phase E — ISTA, FISTA and OMP

**Objective `[PAPER, §3]`.**
$$F(\alpha) = \tfrac12\|y - A\alpha\|_2^2 + \lambda\|\alpha\|_1,\qquad
\alpha_{t+1} = \mathcal{S}_{\theta}\!\left(\alpha_t - \eta A^{\top}(A\alpha_t - y)\right)$$

**Dimensions.** $A \in \mathbb{R}^{p\times k}$ shared; $y \in \mathbb{R}^{N\times p}$,
one measurement vector per token row; $\alpha \in \mathbb{R}^{N\times k}$. All updates
are batched matmuls, so $n$ tokens decode in parallel — which is what the FLOP counter in
§12 assumes.

**Implementation decisions.**
* $\eta = 1/L$ with $L = \sigma_{\max}(A)^2$ by power iteration `[CHOICE]` — the paper
  gives no step size; this is the classical choice guaranteeing monotone descent.
* Fixed iteration budget with early stopping **off by default**, so that runtime
  comparisons against LISTA are honest.
* `@torch.no_grad()` on purpose: classical ISTA is a fixed-point iteration, not a
  differentiable layer. This is precisely the limitation LISTA exists to remove, and it
  means a CSAT block with an ISTA decoder cannot be trained through the decoder.
* FISTA (Nesterov momentum) and `debias` (least-squares refit on the detected support)
  are **ours, not the paper's**, and are always reported as separate columns so the raw
  ISTA number is never quietly improved.
""")

writefile("models/ista.py")
writefile("models/omp.py")

code(r'''
importlib.invalidate_caches()
from models.ista import (ista, fista, debias, soft_threshold, lasso_objective,
                         estimate_step_size, ISTADecoder)
from models.omp import omp
from utils.metrics import support_f1

# --- soft thresholding: the only non-linearity in the algorithm ------------- #
xs = torch.tensor([-3.0, -1.0, -0.4, 0.0, 0.4, 1.0, 3.0])
print("x              :", xs.tolist())
print("S_1.0(x)       :", soft_threshold(xs, 1.0).tolist())
print("  -> note SHRINKAGE: 3.0 becomes 2.0, not 3.0. This bias is what debias() undoes.")

# --- a genuinely underdetermined CS problem --------------------------------- #
set_seed(cfg.seed)
k_atoms, p_meas, s_true, n_sig = 128, 64, 8, 256
A = torch.randn(p_meas, k_atoms, device=DEVICE) / p_meas ** 0.5
A = A / A.norm(dim=0, keepdim=True)
alpha_true = make_sparse_signals(n_sig, k_atoms, s_true, device=DEVICE)
y = alpha_true @ A.T

print(f"\nproblem: A is [{p_meas} x {k_atoms}], alpha is {s_true}-sparse "
      f"-> {p_meas} measurements for {k_atoms} unknowns (underdetermined)")
print("step size eta = 1/L = %.5f" % estimate_step_size(A))

out = ista(A, y, lam=0.005, n_iters=2000, track_objective=True)
obj = out["objective"]
print("\nISTA objective: %.4f -> %.4f" % (obj[0], obj[-1]))
print("monotonically non-increasing:",
      all(obj[i+1] <= obj[i] + 1e-6 for i in range(len(obj)-1)), " <- guaranteed by eta <= 1/L")
print("relative error ||alpha_hat - alpha|| / ||alpha||: %.4f" % relative_l2(out["alpha"], alpha_true))
print("support F1                                     : %.3f" % support_f1(out["alpha"], alpha_true))
print("after debiasing (least-squares refit on support): %.4f"
      % relative_l2(debias(A, y, out["alpha"]), alpha_true))

out_f = fista(A, y, lam=0.005, n_iters=2000)
print("\nFISTA [ours, not the paper's] relative error    : %.4f" % relative_l2(out_f["alpha"], alpha_true))
out_o = omp(A, y, sparsity=s_true)
print("OMP (given the true s) relative error / F1      : %.4f / %.3f"
      % (relative_l2(out_o["alpha"], alpha_true), support_f1(out_o["alpha"], alpha_true)))
''')

code(r'''
# --- The convergence curve, and why the iteration count matters ------------- #
import matplotlib.pyplot as plt

curves = {}
for label, fn in (("ISTA", ista), ("FISTA", fista)):
    r = fn(A, y, lam=0.005, n_iters=2000, track_objective=True)
    curves[label] = r["objective"]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for label, obj in curves.items():
    axes[0].plot(obj, label=label)
axes[0].set_xlabel("iteration"); axes[0].set_ylabel(r"$F(\alpha)$")
axes[0].set_yscale("log"); axes[0].legend(); axes[0].set_title("Objective")

its = [10, 25, 50, 100, 250, 500, 1000, 2000]
for label, fn in (("ISTA", ista), ("FISTA", fista)):
    errs = [relative_l2(fn(A, y, lam=0.005, n_iters=t)["alpha"], alpha_true) for t in its]
    axes[1].plot(its, errs, marker="o", label=label)
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlabel("iterations"); axes[1].set_ylabel("relative error")
axes[1].legend(); axes[1].set_title("Accuracy vs decoder compute")
plt.tight_layout(); plt.show()
''')

md(r"""
**What these outputs establish.**

* Soft thresholding is the exact proximal operator, and it *shrinks* surviving
  coefficients — the $\ell_1$ bias that `debias` removes.
* With $\eta = 1/L$ the objective is monotonically non-increasing, as the theory
  requires.
* Recovery is demonstrated **empirically**, not assumed: the notebook never claims exact
  recovery without a measurement.
* The right-hand plot is directly relevant to the paper's efficiency claim: ISTA needs
  roughly an order of magnitude more iterations than FISTA for the same accuracy, and the
  decoder's cost is linear in that count. **The paper reports neither the count nor the
  resulting error.**

**What remains for Phase 5.** The recovery regime of the *actual* operator in a trained
CSAT model — which cannot be pinned down until the measurement equation is (§2.6).
""")

# ============================================================================ #
# 8. PHASE F — LISTA
# ============================================================================ #
md(r"""
---
# 8. Phase F — LISTA

**The derivation the paper's equation rests on.** Expand the ISTA update:

$$\alpha_{t+1} = \mathcal{S}_{\theta}\!\left(\alpha_t - \eta A^{\top}(A\alpha_t - y)\right)
= \mathcal{S}_{\theta}\!\left(\underbrace{(I - \eta A^{\top}A)}_{W_s}\alpha_t
+ \underbrace{\eta A^{\top}}_{W_e} y\right)$$

which is exactly the paper's
$\alpha_i^{(t+1)} = \eta_{\theta}(S\alpha_i^{(t)} + BZ_i)$ with $S = W_s$, $B = W_e$.
LISTA drops the constraint that $W_s$ and $W_e$ derive from a single $A$, and learns them
(and $\theta$) by back-propagation through $t$ unrolled layers.

**Implementation decisions.**
* **Initialise at exact ISTA** `[CHOICE]`, so the network *starts* equal to the analytic
  solver. A unit test asserts this equality. Any measured gain is then attributable to
  learning rather than to a weak random baseline — the usual way LISTA comparisons are
  inflated.
* Untied weights by default `[MISSING]`; per-coordinate learned thresholds; Adam.
* `[MISSING]` The paper never states what LISTA is trained against. We support
  supervision on $\alpha$ (classical, needs ground truth) and on $\Psi\alpha$ (the
  setting that would apply inside a transformer).

**Cost note.** $W_s$ is $k\times k$, so one LISTA layer costs $\mathcal{O}(k^2)$ per token
while one ISTA iteration costs $\mathcal{O}(pk)$. LISTA is cheaper only because it uses
far fewer layers than ISTA needs iterations — not because a layer is cheaper. §12 counts
this explicitly.

**A structural detail worth knowing.** The first layer's $W_s$ can never receive
gradient, because $\alpha^{(0)}=0$ makes $W_s\alpha^{(0)}$ identically zero. A $t$-layer
untied LISTA therefore has $t-1$ effective $W_s$ blocks. A unit test pins this down.
""")

writefile("models/lista.py")

code(r'''
importlib.invalidate_caches()
from models.lista import LISTA, LISTADecoder, train_lista

set_seed(cfg.seed)
layers, lam = 8, 0.005
net = LISTA(p=p_meas, k=k_atoms, n_layers=layers, A=A, lam=lam, init_from_ista=True).to(DEVICE)

# 1. At initialisation LISTA IS ISTA -- the check that makes the comparison fair.
with torch.no_grad():
    lista_init = net(y)
ista_matched = ista(A, y, lam=lam, n_iters=layers, step_size=estimate_step_size(A))["alpha"]
print("LISTA at init vs %d ISTA iterations: max |difference| = %.2e"
      % (layers, (lista_init - ista_matched).abs().max().item()))
print("   -> the unrolled network is a re-parameterisation of ISTA, not a new algorithm")

n_lista_par = sum(p.numel() for p in net.parameters())
print(f"\nparameters: W_s is [{k_atoms} x {k_atoms}] per layer, "
      f"W_e is [{k_atoms} x {p_meas}] per layer, total {n_lista_par:,}")

# 2. Train it (supervised on alpha; synthetic data has ground truth).
n_tr = 2048 if cfg.quick else 4096
alpha_tr = make_sparse_signals(n_tr + 512, k_atoms, s_true, device=DEVICE)
y_tr = alpha_tr @ A.T
tr, va = slice(0, n_tr), slice(n_tr, n_tr + 512)
print("\ntraining LISTA (%d epochs):" % (15 if cfg.quick else 40))
hist = train_lista(net, y_tr[tr], alpha_tr[tr], y_tr[va], alpha_tr[va],
                   supervision="alpha", n_epochs=15 if cfg.quick else 40,
                   batch_size=128, lr=1e-3, verbose=True, log_every=5)

with torch.no_grad():
    after = net(y_tr[va])
before_err = relative_l2(ista(A, y_tr[va], lam=lam, n_iters=layers)["alpha"], alpha_tr[va])
print("\n%d-iteration ISTA   relative error: %.4f" % (layers, before_err))
print("%d-layer trained LISTA relative error: %.4f" % (layers, relative_l2(after, alpha_tr[va])))
''')

md(r"""
**What these outputs establish.** The equivalence at initialisation is exact to machine
precision, which validates the derivation above and makes the ISTA-vs-LISTA comparison
meaningful. Training then improves on that starting point at a fixed depth.

**What this does *not* establish.** That LISTA helps *in CSAT*. It was trained on
synthetic signals that are exactly sparse in a known dictionary — the ideal case.
Experiment 04 tests the paper's own caveat that learned decoders "may sacrifice some
generalization … when sparsity levels … change", and Experiment 05 shows the data CSAT
actually produces is not sparse in the first place.

**What remains for Phase 5.** Training LISTA inside a full model against a task loss,
which is what the paper implies but never describes.
""")

# ============================================================================ #
# 9. CSAT BLOCK
# ============================================================================ #
md(r"""
---
# 9. The end-to-end CSAT block (paper Figure 1)

Figure 1 of the paper runs: tokens → $Q,K,V$ → $\Phi$ compression → compressed
attention → sparse decoder (ISTA/LISTA) → context vector $\Psi\alpha$ → fusion +
residual + norm → prediction head.

`CSATBlock` assembles exactly that, with `decoder="none"` available as the ablation that
isolates what the decoding stage contributes. The decoder uses **Bridge 1**, the only
reading that consumes $Z$ directly; the block raises an explicit error if a bridge is
requested whose measurement dimension cannot consume $Z$'s rows.
""")

writefile("models/csat_block.py")

code(r'''
importlib.invalidate_caches()
from models.csat_block import CSATBlock

set_seed(cfg.seed)
d_model, heads, seq, m_blk = 128, 4, 64, 16
x = torch.randn(2, seq, d_model, device=DEVICE)

for dec in ("none", "ista", "lista"):
    blk = CSATBlock(d_model=d_model, n_heads=heads, seq_len=seq, m=m_blk,
                    decoder=dec, bridge="denoise", dictionary="dct",
                    ista_iters=20, lista_layers=6, device=DEVICE).to(DEVICE)
    out, internals = blk(x, return_internals=True)
    n_par = sum(p.numel() for p in blk.parameters() if p.requires_grad)
    print(f"decoder={dec:6s} -> out {tuple(out.shape)}, Z {tuple(internals['Z'].shape)}, "
          f"trainable params {n_par:,}")

# Gradient flow through the differentiable (LISTA) variant.
blk = CSATBlock(d_model=d_model, n_heads=heads, seq_len=seq, m=m_blk, decoder="lista",
                learnable_phi=True, device=DEVICE).to(DEVICE)
blk(x)[0].sum().backward()
print("\ngradients reach W^Q          :", blk.w_q.weight.grad.abs().sum().item() > 0)
print("gradients reach Phi_K        :", blk.attn.phi_k.phi.grad.abs().sum().item() > 0)
print("gradients reach LISTA weights:", blk.decoder.lista.W_e[0].grad.abs().sum().item() > 0)

# The ISTA variant is NOT differentiable through the decoder -- stated, not hidden.
blk_i = CSATBlock(d_model=d_model, n_heads=heads, seq_len=seq, m=m_blk,
                  decoder="ista", device=DEVICE).to(DEVICE)
out_i, _ = blk_i(x)
print("\nISTA-decoder block output requires_grad:", out_i.requires_grad,
      "\n   -> the analytic decoder blocks gradients; the paper does not say how this",
      "\n      variant is trained end to end.")
''')

md(r"""
**What this establishes.** The full pipeline of Figure 1 runs, in all three decoder
configurations, and gradients reach the projections, a learnable $\Phi$, and the LISTA
weights. It also makes visible a gap the paper leaves: with the analytic decoder the
block is **not** differentiable end to end.

**What remains for Phase 5.** Stacking these blocks into a full transformer and training
on the paper's benchmarks.
""")

# ============================================================================ #
# 10. SUPPORTING UTILITIES + TESTS
# ============================================================================ #
md(r"""
---
# 10. Supporting utilities and the unit-test suite

## 10.1 FLOP counting, benchmarking, plotting, reporting

`flops.py` counts multiply-accumulates analytically so the *theoretical* complexity claim
can be separated from *measured* runtime — these are different things and §12 never
conflates them.

`benchmarking.py` enforces the fairness rules: warm-up iterations, CUDA synchronisation
around every timed region, repeated measurement with mean and standard deviation,
separate reporting of `max_memory_allocated` (tensor bytes) and `max_memory_reserved`
(allocator bytes), and OOM recorded as a result rather than a crash — because full
attention at $n=8192$ genuinely does not fit on a 16 GB card.
""")

writefile("utils/flops.py")
writefile("utils/benchmarking.py")
writefile("utils/plotting.py")
writefile("utils/reporting.py")

md(r"""
## 10.2 The test suite

Sixty tests across five files. They check the usual things — shapes, normalisation,
gradients, agreement with PyTorch — and also pin down the structural facts this
reproduction turns on:

* compressed attention's $M_{\text{eff}}$ is **not** row-stochastic;
* causal masks are **refused**;
* LISTA at initialisation **equals** ISTA;
* layer-0 $W_s$ **cannot** receive gradient;
* the paper's decoding equation **fails** its dimension check;
* a square dictionary represents *any* vector, so sparsity is never implied by the model.
""")

for t in ("tests/__init__.py", "tests/test_standard_attention.py",
          "tests/test_compressed_attention.py", "tests/test_ista.py",
          "tests/test_lista.py", "tests/test_measurement_dictionary.py"):
    writefile(t)

code(r'''
# Run the suite. Kaggle has pytest; if it is missing we say so rather than failing.
import subprocess, sys
proc = subprocess.run(
    [sys.executable, "-m", "pytest", os.path.join(PROJECT_ROOT, "tests"),
     "-q", "--no-header", "-p", "no:cacheprovider"],
    capture_output=True, text=True, cwd=PROJECT_ROOT,
    env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
)
print(proc.stdout[-4000:])
if proc.returncode != 0:
    print("STDERR:\n", proc.stderr[-3000:])
print("pytest exit code:", proc.returncode, "(0 = all tests passed)")
''')

# ============================================================================ #
# 11. PHASE G — EXPERIMENTS
# ============================================================================ #
md(r"""
---
# 11. Phase G — Synthetic experiments

Every experiment below is **controlled and synthetic**: fixed seeds, known ground truth,
all parameters reported. None of them is one of the paper's benchmarks, and none of them
is evidence about Tables 1–5. They answer narrower questions that *can* be answered here.

Results are written to `results/` as CSV and JSON.
""")

writefile("experiments/__init__.py")

md(r"""
## 11.1 Experiment 01 — Fidelity: how close is $Z$ to $C$?

The paper claims CSAT maintains "semantic fidelity" and that $Z_i$ is "a compressed
version of the true context vector $C_i$", but never measures the gap. We sweep $m$, the
ensemble, the token distribution (i.i.d. vs redundant) and the logit scaling, and include
the **softmax-free control** that isolates the softmax from the projection.

Cosine similarity is the key column: it is scale-free, so it reports directional
agreement independently of any norm mismatch.
""")

writefile("experiments/exp01_attention_fidelity.py")

code(r'''
importlib.invalidate_caches()
from experiments import exp01_attention_fidelity as exp01
from utils.reporting import save_table
from utils.plotting import plot_fidelity

set_seed(cfg.seed)
N_EXP = 256 if cfg.quick else 512
rows01 = exp01.run(seq_len=N_EXP, d_k=64, n_heads=4, batch=1,
                   m_values=[16, 32, 64, 128] + ([256] if not cfg.quick else []),
                   ensembles=["gaussian", "rademacher"],
                   token_modes=["iid", "redundant"],
                   scalings=["paper_sqrt_dk", "variance_calibrated"],
                   device=DEVICE)
df01 = pd.DataFrame(rows01)
save_table(rows01, "exp01_attention_fidelity", RESULTS_DIR)

view = df01[df01.ensemble.eq("gaussian") | df01.scaling.eq("NO_SOFTMAX_control")]
display(view[["token_mode", "scaling", "m", "compression_ratio", "relative_l2",
              "cosine_similarity", "norm_ratio"]]
        .round(4).reset_index(drop=True))
plot_fidelity(rows01, os.path.join(FIGURES_DIR, "exp01_fidelity.png"))
plt.show()
''')

md(r"""
### Preliminary observations — Experiment 01

Read the `cosine_similarity` column first, because it is scale-free.

1. **For the paper's formulation, $Z$ is close to orthogonal to $C$** — cosine similarity
   sits near zero (typically $|\cos| < 0.05$) at every compression ratio that would
   actually be worth using, on both token distributions, while the relative L2 error runs
   to several hundred per cent. It lifts only as $m$ approaches $n$, i.e. exactly where
   there is no longer any compression to speak of.
2. **`norm_ratio` $\gg 1$** explains part of it and is structural, not a bug. Each row of
   $\widetilde{V} = \Phi_V V$ is a sum of $n$ value rows, so
   $\|\widetilde{V}_j\| \sim \sqrt{n/m}\,\|V\|$, while $\widetilde{A}$ averages only $m$
   of them. Meanwhile true attention averages $\sim n$ nearly-independent value rows, so
   $\|C_i\|$ *shrinks* by roughly $1/\sqrt{n_{\text{eff}}}$. The two scales diverge as
   $n/m$ grows.
3. **The softmax-free control behaves completely differently**: cosine similarity rises
   steadily with $m$ (to $\approx 0.6$–$0.7$ at the largest $m$), exactly as the
   $\mathbb{E}[\Phi^{\top}\Phi]=I$ argument predicts. **So the projection preserves
   information; applying the softmax to the compressed logits is what destroys the
   correspondence with $C$.** Softmax does not commute with $\Phi^{\top}$.
4. Redundant tokens give uniformly better numbers than i.i.d. ones — the paper's
   redundancy intuition is directionally right — but not nearly enough to close the gap.

**Strictly preliminary.** This is untrained, randomly initialised, synthetic. It does not
show that a *trained* CSAT model fails; Experiment 07 begins that question.
""")

md(r"""
## 11.2 Experiment 02 — Is $M_{\text{eff}}$ still a weighted average?
""")

writefile("experiments/exp02_effective_attention.py")

code(r'''
importlib.invalidate_caches()
from experiments import exp02_effective_attention as exp02

set_seed(cfg.seed)
rows02 = exp02.run(seq_len=256, d_k=64, n_heads=4, batch=1,
                   m_values=[16, 32, 64, 128], share_phi_options=[False, True],
                   device=DEVICE)
save_table(rows02, "exp02_effective_attention", RESULTS_DIR)
display(pd.DataFrame(rows02)[["m", "share_phi", "compression_ratio",
                              "negative_entry_fraction", "row_sum_mean", "row_sum_std",
                              "rel_error_vs_true_A", "true_A_row_sum_mean"]].round(4))
''')

md(r"""
### Preliminary observations — Experiment 02

About **half** of $M_{\text{eff}}$'s entries are negative and its row sums scatter widely
around values that are not 1, while the true $A$ has row sums of exactly 1 and no
negative entries. Sharing $\Phi_K = \Phi_V$ changes nothing material.

This says something specific: **CSAT does not compute a weighted average of value
vectors.** It computes a signed linear combination. That is a legitimate thing for a layer
to do — but it means the phrase "$Z_i$ is a compressed version of $C_i$" cannot be read
in the averaging sense, and it removes the intuition that would motivate treating $Z_i$ as
a noisy observation of $C_i$ for decoding.
""")

md(r"""
## 11.3 Experiment 03 — Do the solvers actually recover sparse signals?

This is the experiment the paper's recovery guarantee deserves and never gets. It is a
clean, fully-specified CS problem — **our** construction, not the paper's pipeline.
""")

writefile("experiments/exp03_sparse_recovery.py")

code(r'''
importlib.invalidate_caches()
from experiments import exp03_sparse_recovery as exp03
from utils.plotting import plot_recovery_sweeps

set_seed(cfg.seed)
res03 = exp03.run(device=DEVICE, quick=cfg.quick, save_dir=RESULTS_DIR)
for name in ("measurements", "sparsity", "iterations", "noise"):
    print(f"\n--- sweep: {name} ---")
    display(pd.DataFrame(res03[name]).round(5))
plot_recovery_sweeps(res03, os.path.join(FIGURES_DIR, "exp03_recovery.png"))
plt.show()
''')

md(r"""
### Preliminary observations — Experiment 03

1. **The classic phase transition appears.** At $k=128$, $s=8$, recovery is essentially
   exact once $p \gtrsim 96$ and degrades sharply below $p \approx 48$ — consistent with
   $m = \mathcal{O}(s\log(k/s))$.
2. **Sparsity is the binding constraint.** At fixed $p=48$, error rises steeply past
   $s\approx 8$. CS works when the signal really is sparse; that premise is tested for
   CSAT's actual data in Experiment 05.
3. **Iteration count dominates ISTA's accuracy**, and FISTA reaches the same error roughly
   an order of magnitude sooner. Since decoding cost is linear in iterations, any
   efficiency claim about a CSAT pipeline is meaningless without stating this number —
   and the paper does not state it.
4. Debiasing helps when the support is right and *hurts* when it is not, which is why it
   is reported as a separate column rather than folded into the headline number.

**What this establishes:** our solvers are correct and behave as CS theory predicts. **What
it does not establish:** anything about the paper's attention pipeline, since the paper's
$Z_i$ is not a measurement of $C_i$ (§2.6).
""")

md(r"""
## 11.4 Experiment 04 — LISTA vs ISTA, and the paper's own generalisation caveat
""")

writefile("experiments/exp04_lista_training.py")

code(r'''
importlib.invalidate_caches()
from experiments import exp04_lista_training as exp04

set_seed(cfg.seed)
res04 = exp04.run(k=128, p=64, s=8,
                  n_train=2048 if cfg.quick else 4096, n_val=512,
                  layers=8, lam=0.005,
                  n_epochs=15 if cfg.quick else 40,
                  device=DEVICE, verbose=False, save_dir=RESULTS_DIR)

print("--- matched budget: 8 LISTA layers vs 8 ISTA iterations ---")
for kk, vv in res04["matched_budget"].items():
    print(f"   {kk:42s}: {vv:.6f}" if isinstance(vv, float) else f"   {kk:42s}: {vv}")
print("\n--- how many classical iterations match the trained LISTA? ---")
for kk, vv in res04["iteration_equivalence"].items():
    print(f"   {kk:28s}: {vv}")
print("\n--- generalisation under sparsity shift (trained at s=8) ---")
display(pd.DataFrame(res04["sparsity_shift"]).round(5))
''')

md(r"""
### Preliminary observations — Experiment 04

1. **LISTA starts exactly at ISTA** (difference $\sim 10^{-7}$), so the comparison is
   honest by construction.
2. At matched budget the trained 8-layer LISTA is far better than 8 ISTA iterations, and
   matches what classical ISTA needs *hundreds to thousands* of iterations to reach. That
   is a real and large win, and it is the strongest support in this notebook for the
   paper's choice of a learned decoder.
3. **The paper's own caveat is confirmed.** Trained at $s=8$, LISTA degrades as the test
   sparsity moves away from 8, while the long-running classical solver does not care. The
   paper states this risk in §7 and does not measure it; here it is measured.

**What remains for Phase 5.** Whether the win survives when the decoder is trained on
real attention outputs rather than exactly-sparse synthetic signals.
""")

md(r"""
## 11.5 Experiment 05 — The decisive test: is the paper's premise true, and does decoding help?

Two questions, in the right order.

**(a) Are context vectors sparse in any basis?** We test the DCT, a random orthogonal
basis, and a dictionary **learned on the test data itself** — deliberately the most
favourable case, since fitting on test data is otherwise indefensible.

**(b) Given $Z$, does sparse decoding move it closer to $C$?** If
$\|\hat{C}-C\| \ge \|Z-C\|$, the decoder is a distortion, not a decoder.

Bridges 2 and 3 are then run for completeness, clearly labelled as **not** the paper's
pipeline.
""")

writefile("experiments/exp05_decoder_bridge.py")

code(r'''
importlib.invalidate_caches()
from experiments import exp05_decoder_bridge as exp05

set_seed(cfg.seed)
res05 = exp05.run(seq_len=256, d_k=64, n_heads=4, batch=1,
                  m_values=[32, 64, 128], lam_values=[0.001, 0.01, 0.05, 0.2],
                  token_mode="redundant", n_iters=200 if cfg.quick else 500,
                  device=DEVICE, save_dir=RESULTS_DIR)

print("=" * 86)
print("(a) IS THE TRUE CONTEXT MATRIX C SPARSE IN ANY BASIS?  [the paper's core premise]")
print("=" * 86)
display(pd.DataFrame(res05["compressibility_of_C"]).round(4))

print("\n" + "=" * 86)
print("(b) DOES SPARSE DECODING OF Z REDUCE THE ERROR TO C?  [BRIDGE 1, 'denoise']")
print("=" * 86)
display(pd.DataFrame(res05["bridge1_denoise"])[
    ["m", "lam", "rel_l2_Z_vs_C", "rel_l2_Chat_vs_C", "improvement",
     "cosine_Z_vs_C", "cosine_Chat_vs_C", "mean_nnz_alpha"]].round(4))

print("\n--- BRIDGE 2 (feature-space CS; OUR construction, not the paper's) ---")
display(pd.DataFrame(res05["bridge2_feature_cs"]).drop(columns=["note"]).round(4))
print("--- BRIDGE 3 (token-axis CS; recovers V, not C) ---")
display(pd.DataFrame(res05["bridge3_token_cs"]).drop(columns=["note"]).round(4))
''')

md(r"""
### Preliminary observations — Experiment 05 *(the most important in this notebook)*

**(a) The sparsity premise is not supported by the data we can generate.**
In the DCT basis, context vectors need roughly **half** of all coefficients to carry 95%
of their energy — a compressibility ratio near 0.54, where 1.0 means no compressibility at
all. A random orthogonal basis is slightly worse. Even a dictionary **learned on the test
data** and given twice as many atoms as dimensions reaches a useful concentration only by
accepting $\sim$12% reconstruction error, and still puts $\sim$40 non-zeros per row. These
are not $\|\alpha_i\|_0 \ll d_k$ signals.

**(b) The decoder does not rescue $Z$.** Relative error falls slightly as $\lambda$ grows,
but **cosine similarity is unchanged** — pinned near 0.02–0.15. That pattern has a single
explanation: $Z$ is badly over-scaled (Experiment 01), and shrinking an over-scaled
estimate toward zero reduces $L_2$ error *without improving direction*. The "improvement"
column is a scale artefact, not recovery. A decoder that genuinely recovered $C$ would move
the cosine column.

**Bridges 2 and 3** fail for the same underlying reason: relative errors of 0.58–0.96 and
0.80–0.98 respectively. The solvers are not at fault — Experiment 03 shows they recover
truly sparse signals to $10^{-3}$. **The data is not sparse.**

**The honest bottom line for Phase 4.** On synthetic data, the chain
*compress → attend → sparsely decode* does not reconstruct the true attention output, and
the reason is traceable to a premise (compressibility of context vectors) that the paper
asserts and never measures.

**Three reasons this is preliminary, not a refutation.**
1. Our tokens are synthetic Gaussian/prototype constructions, **not** real ViT or BERT
   features. Real features may be far more compressible — the paper cites neural collapse
   for exactly this. **Measuring compressibility on real pretrained features is the single
   highest-priority Phase 5 experiment.**
2. Everything here is at random initialisation. Training could reshape the representation
   toward compressibility.
3. Our $\Psi$ choices may simply be the wrong basis, and the paper never says which basis
   it used.
""")

md(r"""
## 11.6 Experiment 07 — Does a CSAT block train?

The stress test: content-addressed associative recall, which a single attention layer can
solve exactly and which *requires* retrieving one specific token. Mixing the token axis
should damage it. We compare full attention against CSAT with fixed $\Phi$ and CSAT with
learnable $\Phi$ (the variant §7 of the paper permits).

We also record a diagnostic on $\Phi$ itself: the **participation ratio** of its rows,
which measures how many tokens each row actually reads. $1/n$ means the row selects a
single token (no mixing); $1$ means it spreads over everything.
""")

writefile("experiments/exp07_learnability.py")

code(r'''
importlib.invalidate_caches()
from experiments import exp07_learnability as exp07
from utils.plotting import plot_learnability

set_seed(cfg.seed)
rows07 = exp07.run(n_pairs=16, vocab=32, d_model=64, n_heads=4,
                   m_values=[2, 4, 8, 16],
                   steps=400 if cfg.quick else 1200, batch=64, lr=3e-3,
                   learnable_phi_options=[False, True],
                   device=DEVICE, verbose=False,
                   save_path=os.path.join(RESULTS_DIR, "exp07_learnability.json"))
df07 = pd.DataFrame(rows07)
save_table([{kk: vv for kk, vv in r.items() if kk != "history"} for r in rows07],
           "exp07_learnability", RESULTS_DIR)
display(df07[["method", "m", "learnable_phi", "compression_ratio", "eval_accuracy",
              "chance_accuracy", "final_train_loss",
              "phi_participation_before", "phi_participation_after"]].round(4))
plot_learnability(rows07, os.path.join(FIGURES_DIR, "exp07_learnability.png"))
plt.show()
''')

md(r"""
### Preliminary observations — Experiment 07

1. **Full attention solves the task** (accuracy $\approx 1.0$ against a chance level of
   $1/32$). The baseline is sound, so the comparison means something.
2. **CSAT with a fixed random $\Phi$ fails at every $m$** — barely above chance, *even at
   $m=16$ with $n=17$, where there is essentially no compression at all.* So the failure is
   not about the compression ratio. Mixing the token axis with a fixed random operator
   destroys content-addressed retrieval, and the model cannot undo it by adapting $W^Q$ and
   $W^K$, because the mixture is fixed in *position* space while the content it must
   retrieve sits at a data-dependent position.
3. **CSAT with a learnable $\Phi$ recovers** as $m$ grows, reaching near-perfect accuracy
   at the largest $m$. (In QUICK mode the intermediate $m$ are visibly under-trained and
   the trend is not clean; with `CSAT_QUICK="0"` the progression is monotone —
   roughly 0.17 -> 0.32 -> 0.73 -> 1.00 across $m = 2, 4, 8, 16$.)
4. **The $\Phi$ diagnostic says why.** The learned $\Phi$'s participation ratio drops
   sharply (from $\approx 0.38$ toward $\approx 0.18$) and its rows become far more
   concentrated. **The learnable variant succeeds by learning *not* to mix** — by drifting
   toward a selection/pooling operator. But a concentrated, data-adapted $\Phi$ is no longer
   an incoherent random measurement operator, so it forfeits the RIP guarantees that are the
   paper's entire theoretical contribution. The variant that works is, in effect, a learned
   token-pooling scheme — which is close to Linformer, and is precisely the family the paper
   positions CSAT against.

**Scope.** One task, one layer, small scale. It cannot refute Tables 1–4. It does establish
that the mechanism trains, and it identifies a concrete tension between *working* and
*being compressed sensing* that Phase 5 should pursue directly.
""")

# ============================================================================ #
# 12. PHASE H — BENCHMARKING
# ============================================================================ #
md(r"""
---
# 12. Phase H — Complexity, runtime and memory

Three quantities that must never be conflated:

| Quantity | What it is | Where it comes from |
|---|---|---|
| **Theoretical FLOPs** | analytic multiply-accumulate count | `utils/flops.py` |
| **Measured runtime** | wall-clock on *this* device | `utils/benchmarking.py` |
| **Memory** | peak allocated vs peak reserved | `torch.cuda` counters |

Runtime is hardware-, precision- and kernel-dependent; FLOPs are not. A large FLOP
reduction can produce a small speedup (or none) when the smaller kernels are
memory-bound or launch-bound.
""")

code(r'''
# --- Analytic complexity: attention stage AND decoder, side by side ---------- #
importlib.invalidate_caches()
from utils.flops import complexity_table, standard_attention_flops, csat_attention_flops

tbl = complexity_table(n_values=[512, 1024, 2048, 4096, 8192], m=64, d_k=64,
                       heads=8, batch=1, ista_iters=20, lista_layers=8)
df_flops = pd.DataFrame(tbl)
display(df_flops.round(3))
save_table(tbl, "complexity_flops_m64", RESULTS_DIR)

print("Reading the last three columns:")
print("  speedup_attention_only    -- what the paper's O(n^2 d) -> O(nmd) claim describes")
print("  speedup_with_ista_decode  -- the SAME pipeline once row-wise ISTA decoding is counted")
print("  speedup_with_lista_decode -- with an 8-layer LISTA instead")
print("\nThe decoder term the paper writes only as '+ decoding' is O(n * T * p * k) for ISTA")
print("and O(n * t * k^2) for LISTA -- linear in n, but with a constant that can dominate.")
''')

writefile("experiments/exp06_efficiency.py")

code(r'''
# --- Measured runtime and memory -------------------------------------------- #
importlib.invalidate_caches()
from experiments import exp06_efficiency as exp06
from utils.plotting import plot_efficiency

set_seed(cfg.seed)
seq_lens = ([256, 512, 1024, 2048] if cfg.quick else [512, 1024, 2048, 4096, 8192])
rows06 = exp06.run(seq_lens=seq_lens, m_values=[64, 128], batch=1, n_heads=8, d_k=64,
                   ista_iters=20, lista_layers=8,
                   warmup=3 if cfg.quick else 5, repeats=10 if cfg.quick else 20,
                   device=DEVICE,
                   save_path=os.path.join(RESULTS_DIR, "exp06_efficiency.json"))
df06 = pd.DataFrame(rows06)
save_table(rows06, "exp06_efficiency", RESULTS_DIR)

cols = ["n", "m", "method", "status", "time_ms_mean", "time_ms_std",
        "peak_allocated_MB", "peak_reserved_MB", "theoretical_GFLOPs", "attention_matrix_MB"]
display(df06[[c for c in cols if c in df06.columns]].round(3))
plot_efficiency(rows06, os.path.join(FIGURES_DIR, "exp06_efficiency.png"))
plt.show()
''')

code(r'''
# --- The comparison that matters: attention-only vs the FULL pipeline -------- #
piv = df06[df06.status.eq("ok")].pivot_table(
    index="n", columns="method", values="time_ms_mean", aggfunc="min")
if "standard_attention" in piv.columns:
    summary = pd.DataFrame({"standard_attention_ms": piv["standard_attention"]})
    for col, label in (("csat_attention", "csat_attention_ms"),
                       ("ista_decoder", "ista_decoder_ms"),
                       ("lista_decoder", "lista_decoder_ms"),
                       ("csat_full_pipeline", "csat_full_pipeline_ms")):
        if col in piv.columns:
            summary[label] = piv[col]
    if "csat_attention" in piv.columns:
        summary["speedup_attention_only"] = piv["standard_attention"] / piv["csat_attention"]
    if "csat_full_pipeline" in piv.columns:
        summary["speedup_FULL_pipeline"] = piv["standard_attention"] / piv["csat_full_pipeline"]
    display(summary.round(3))
    save_table(summary.reset_index().to_dict("records"), "efficiency_summary", RESULTS_DIR)

print("\nPaper Table 5 for reference (n=4096, hardware/batch/m/decoder all UNREPORTED):")
for kk, vv in cfg_mod.PAPER_REPORTED["efficiency_n4096"].items():
    print(f"   {kk:20s}  {vv[0]:>5} GB   {vv[1]:>5} ms")
print("\nWe do NOT claim to reproduce those absolute numbers -- see section 13.")
''')

md(r"""
### Preliminary observations — Phase H

1. **The attention stage alone scales as advertised.** Measured time and the analytic FLOP
   count both flatten from quadratic toward linear once $m$ is fixed, and the stored
   attention matrix shrinks by exactly $n/m$. The $\mathcal{O}(n^2d)\to\mathcal{O}(nmd)$
   claim, *for the attention stage*, is supported.
2. **The decoder is not a "small overhead" at the settings we can test.** With 20 ISTA
   iterations the decoding stage costs more than the compressed attention it follows, and
   the full pipeline's speedup over standard attention is a small fraction of the
   attention-only speedup — at some settings the pipeline is *slower*. The paper's
   statement that decoding "does not dominate runtime" is not supported at any
   configuration we can construct, and the paper supplies no configuration of its own to
   check against.
3. **LISTA is materially cheaper than ISTA**, consistent with §8: far fewer layers than
   ISTA needs iterations.
4. **Table 5 cannot be reproduced.** Not "did not match" — *cannot be attempted*. The GPU,
   precision, batch size, layer count, $m$, and decoder configuration are all unreported,
   and it is not even stated whether the 439 ms includes decoding. Our numbers describe
   this notebook's device and every parameter is printed above.
""")

# ============================================================================ #
# 13. TRACKING TABLE
# ============================================================================ #
md(r"""
---
# 13. Reproduction tracking table

A component is marked reproduced only if the paper specifies it well enough to implement
**and** our implementation matches that specification. Code that runs is not evidence of
reproduction.
""")

code(r'''
importlib.invalidate_caches()
from utils.reporting import tracking_table, status_summary, TRACKING_ROWS, STATUSES

pd.set_option("display.max_colwidth", 96)
track = tracking_table()
display(track[["component", "status", "implemented", "exact_or_approx"]])
save_table(TRACKING_ROWS, "reproduction_tracking", RESULTS_DIR)

print("\nStatus counts:")
for s, c in status_summary().items():
    print(f"   {s:26s}: {c}")
''')

code(r'''
# Full detail for the components that are NOT cleanly reproduced.
for row in TRACKING_ROWS:
    if row["status"] in ("Cannot reproduce exactly", "Not specified by paper",
                         "Not yet implemented"):
        print("=" * 92)
        print(f"COMPONENT : {row['component']}")
        print(f"STATUS    : {row['status']}")
        print("PAPER     : " + textwrap.fill(row["paper_spec"], 78, subsequent_indent=" " * 12))
        print("MISSING   : " + textwrap.fill(row["missing_details"], 78, subsequent_indent=" " * 12))
        print("WE DID    : " + textwrap.fill(row["assumption"], 78, subsequent_indent=" " * 12))
print("=" * 92)
''')

# ============================================================================ #
# 14. FINAL SUMMARY
# ============================================================================ #
md(r"""
---
# 14. Summary, exported artefacts, and the Phase 5 plan
""")

code(r'''
# --- 14.1 Project tree and created files ------------------------------------ #
from utils.reporting import project_tree, list_result_files

print(project_tree(PROJECT_ROOT))
print()
py_files = [os.path.join(dp, f) for dp, _, fs in os.walk(PROJECT_ROOT)
            for f in fs if f.endswith(".py")]
print(f"Python modules created: {len(py_files)}")
print(f"Total lines of code   : {sum(len(open(p).readlines()) for p in py_files):,}")
''')

code(r'''
# --- 14.2 Exported result files --------------------------------------------- #
files = list_result_files(RESULTS_DIR)
print(f"{len(files)} result files in results/:\n")
for f in files:
    print(f"   {os.path.basename(f):42s} {os.path.getsize(f)/1024:8.1f} KB")

figs = sorted(os.listdir(FIGURES_DIR)) if os.path.isdir(FIGURES_DIR) else []
print(f"\n{len(figs)} figures in figures/:")
for f in figs:
    print("   " + f)
''')

md(r"""
## 14.3 How to run the project outside this notebook

```bash
export PYTHONPATH=/kaggle/working/csat-reproduction
cd /kaggle/working/csat-reproduction

python -m pytest tests/ -q                     # 60 unit tests

python - <<'PY'
from experiments import exp01_attention_fidelity as e1, exp06_efficiency as e6
print(e1.run(seq_len=512, m_values=[32, 64, 128]))
print(e6.run(seq_lens=[512, 1024], m_values=[64]))
PY
```

`CSAT_QUICK=0` enables the full sweeps; `CSAT_ROOT` relocates the output directory.
""")

md(r"""
## 14.4 What is implemented

**Fully, exactly as the paper specifies**
* Standard scaled dot-product attention and the $Q,K,V$ projections
* $\widetilde{K}=\Phi_K K$, $\widetilde{V}=\Phi_V V$ with $\Phi\in\mathbb{R}^{m\times n}$,
  in four ensembles
* $\widetilde{A}=\mathrm{softmax}(Q\widetilde{K}^\top/\sqrt{d_k})$,
  $Z=\widetilde{A}\widetilde{V}$
* The LISTA recurrence, initialised to be exactly ISTA
* $\hat{C}_i=\Psi\hat{\alpha}_i$ applied row-wise
* The end-to-end block of Figure 1, in three decoder configurations

**Implemented with documented choices the paper leaves open**
* $\Phi$ normalisation, head sharing, fixed vs learnable
* $\Psi$: identity / DCT / random orthogonal / overcomplete / learned
* ISTA ($\eta=1/L$, swept $\lambda$ and iterations), FISTA, OMP, debiasing
* LISTA depth, tying, threshold parametrisation, optimiser

**Implemented as diagnostics that are not in the paper**
* The effective attention matrix $M_{\text{eff}}=\widetilde{A}\Phi_V$
* The softmax-free control that separates projection error from softmax error
* Mutual coherence and a Monte-Carlo lower bound on $\delta_s$
* Compressibility profiling of context vectors, including a learned dictionary
* The $\Phi$ participation-ratio diagnostic

## 14.5 What is **not** implemented, and why

| Not implemented | Why |
|---|---|
| WikiText-103 LM (Table 1) | 300k-step budget; tokeniser, schedule, $m$, decoder config all unreported |
| LRA Pathfinder-X (Table 2) | Full training recipe unreported; the task is recipe-sensitive |
| BLIP retrieval / captioning (Tables 3–4) | Needs pretrained BLIP weights and multimodal datasets; which layers were replaced is unreported |
| Linformer / Performer / Longformer baselines | Baseline configurations unreported |
| Table 5's absolute numbers | Hardware, precision, batch size, layer count, $m$, decoder config all unreported |
| The decoding equation as literally written | `[INCONSIST]` — ill-formed for every $m \ll n$ (§2.6) |
| Exact RIP verification | NP-hard in general; the paper states neither $s$ nor $\delta_s$ |
| Causal masking for CSAT | Undefined after token-axis mixing (§2.4) |

## 14.6 Underspecified details, collected

1. The value of $m$ — **in every experiment in the paper**
2. $\lambda$, ISTA step size, iteration count, stopping rule
3. The sparsity level $s = \|\alpha_i\|_0$
4. How $\Psi$ is obtained, and whether it is shared or learned
5. LISTA depth $t$, weight tying, threshold form, training data, loss, optimiser
6. $\Phi$ normalisation; sharing across heads, layers and modalities; fixed vs learnable in the reported runs
7. Batch size, precision, GPU and layer count behind Table 5; whether decoding is included in it
8. How causal masking is handled for the autoregressive result
9. How the non-differentiable ISTA-decoder variant is trained end to end
10. What, along the token axis, is sparse enough to justify the RIP requirement on $\Phi$
""")

md(r"""
## 14.7 Preliminary observations, collected

**Every item here is preliminary**: synthetic data, mostly untrained, single-layer,
small-scale. None is a Phase 5 conclusion.

| # | Observation | Evidence |
|---|---|---|
| 1 | The compressed attention mechanism is implementable exactly as specified and behaves as described at the level of shapes and cost | §5, Exp 06 |
| 2 | CSAT's effective matrix $M_{\text{eff}}$ is $\approx$50% negative with non-unit row sums, so $Z$ is not a weighted average of value vectors | Exp 02 |
| 3 | At initialisation $Z$ is nearly orthogonal to $C$ (cosine $\approx 0$), with a large norm mismatch that follows structurally from $\sqrt{n/m}$ scaling | Exp 01 |
| 4 | The softmax on compressed logits — not the random projection — is what destroys the correspondence; the softmax-free control behaves as CS theory predicts | Exp 01 control |
| 5 | The solvers are correct and show the textbook phase transition on genuinely sparse signals | Exp 03 |
| 6 | Context vectors are **not** sparse in the DCT, a random orthogonal basis, or a dictionary learned on the test data | Exp 05(a) |
| 7 | Sparse decoding of $Z$ lowers $L_2$ error only by shrinking an over-scaled estimate; cosine similarity is unchanged, so it is not recovering $C$ | Exp 05(b) |
| 8 | LISTA at matched depth beats ISTA by a wide margin, and matches ISTA runs hundreds of times longer | Exp 04 |
| 9 | LISTA degrades under sparsity shift, exactly as the paper's §7 warns | Exp 04 |
| 10 | With a **fixed** random $\Phi$, a CSAT block fails at content-addressed retrieval even at $m \approx n$ | Exp 07 |
| 11 | With a **learnable** $\Phi$ it succeeds — by learning *not to mix*, which forfeits the incoherence that RIP requires | Exp 07 + $\Phi$ diagnostic |
| 12 | The attention stage alone scales as claimed, but the decoder's cost is not a small overhead at any configuration we can construct | §12 |

**The single most important caveat.** Observations 3, 6, 7, 10 and 11 rest on *synthetic*
tokens. Real ViT/BERT features may be substantially more compressible — the paper cites
neural collapse for exactly that reason. Until observation 6 is re-tested on real
pretrained features, it constrains this notebook's synthetic setting only.
""")

md(r"""
## 14.8 Phase 5 verification plan

Ordered by how much each would change the conclusions.

**Tier 1 — directly tests the paper's premise**
1. **Compressibility of real context vectors.** Extract $C = \mathrm{softmax}(QK^\top/\sqrt{d_k})V$
   from a pretrained ViT-B/16 and BERT-base across layers, heads and modalities; measure
   the 95%-energy coefficient count in DCT, PCA, and K-SVD-learned dictionaries. *This
   single experiment decides whether the paper's central assumption holds.*
2. **Fidelity after training.** Fine-tune a small transformer with CSAT blocks and re-measure
   $\mathrm{cosine}(Z, C)$ during training. Does the representation adapt toward
   compressibility?
3. **Ask the authors** to disambiguate $Z_i = \Phi\Psi\alpha_i$, or locate a corrected
   version.

**Tier 2 — tests the mechanism's claims**
4. **RIP applicability.** Establish what, along the token axis, would have to be sparse, and
   estimate $\delta_{2s}$ for the required $s$ at plausible $m$. Check $\delta_{2s}<\sqrt2-1$.
5. **Learnable-$\Phi$ convergence.** Extend the $\Phi$ participation-ratio diagnostic to
   longer training and larger $n$: does a learned $\Phi$ always drift toward selection? If
   so, CSAT-that-works is a learned pooling method and should be compared to Linformer on
   those terms.
6. **Causal masking.** Determine how the WikiText-103 result was obtained given that
   compressed keys mix future tokens; a causal CSAT variant would need block-wise or
   prefix-restricted $\Phi$.
7. **Full-pipeline efficiency.** Sweep ISTA iterations and LISTA depth against downstream
   accuracy to find the operating point where CSAT is both accurate and fast — and check
   whether one exists.

**Tier 3 — the paper's own benchmarks**
8. WikiText-103 (Table 1), with a stated $m$ and decoder configuration.
9. LRA Pathfinder-X (Table 2).
10. BLIP + CSAT on Flickr30k / MS-COCO (Tables 3–4).
11. Linformer, Performer and Longformer baselines under identical conditions.
12. Table 5 re-measured with every parameter reported.

**Tier 4 — failure characterisation**
13. Where the sparsity assumption breaks: dense prediction, fine-grained captioning —
    the regimes §7 itself flags.
14. Variance across $\Phi$ draws (§7's non-determinism caveat), over many seeds.
15. Modality mismatch: separate vs shared $\Phi$ for visual and textual tokens.
""")

code(r'''
# --- 14.9 Final manifest ---------------------------------------------------- #
manifest = {
    "paper": "CS-VLM: Compressed Sensing Attention for Efficient Vision-Language "
             "Representation Learning (arXiv:2507.02957v1)",
    "phase": "Phase 4 -- PyTorch implementation and paper reproduction",
    "environment": env,
    "quick_mode": cfg.quick,
    "seed": cfg.seed,
    "python_modules": len(py_files),
    "lines_of_code": sum(len(open(p).readlines()) for p in py_files),
    "unit_tests_exit_code": proc.returncode,
    "status_counts": status_summary(),
    "result_files": [os.path.basename(f) for f in list_result_files(RESULTS_DIR)],
    "figures": figs,
    "reproduced_tables_1_to_5": False,
    "reason_tables_not_reproduced":
        "Training budgets and configuration details (m, decoder settings, hardware, "
        "batch size, precision, layer count, schedules) are not reported in the paper.",
    "central_finding":
        "The paper's decoding equation Z_i = Phi Psi alpha_i is dimensionally "
        "inconsistent with its own declared shapes for every m << n, and no fixed "
        "operator relates Z to C. Compressed attention is implemented exactly; "
        "reconstruction is implemented separately in three labelled bridges.",
}
with open(os.path.join(RESULTS_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, default=str)

print(json.dumps(manifest, indent=2, default=str))
print("\n" + "=" * 78)
print("PHASE 4 COMPLETE -- all artefacts in", RESULTS_DIR)
print("Conclusions are deliberately deferred to Phase 5 (section 14.8).")
print("=" * 78)
''')

# ============================================================================ #
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [],
                   "isGpuEnabled": True, "isInternetEnabled": False,
                   "language": "python", "sourceType": "notebook"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the notebook on disk is out of date")
    args = parser.parse_args()

    rendered = json.dumps(nb, indent=1)

    if args.check:
        if not os.path.exists(args.output):
            print(f"{args.output} does not exist", file=sys.stderr)
            return 1
        if open(args.output).read() != rendered:
            print(f"{args.output} is out of date; run python scripts/build_notebook.py",
                  file=sys.stderr)
            return 1
        print(f"{args.output} is up to date")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        handle.write(rendered)

    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    n_wf = sum(1 for c in cells if c["cell_type"] == "code"
               and c["source"].startswith("%%writefile"))
    print(f"wrote {args.output}")
    print(f"  {len(cells)} cells ({n_md} markdown, {n_code} code, {n_wf} %%writefile)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
