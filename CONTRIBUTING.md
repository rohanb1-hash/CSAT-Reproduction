# Contributing

Thanks for considering a contribution. This is a *reproduction* repository, so it has one
rule that ordinary projects do not, described under "Provenance discipline" below.

## Setup

```bash
git clone https://github.com/OWNER/csat-reproduction.git
cd csat-reproduction
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only
pip install -e ".[dev,notebook]"
python -m pytest
```

## Before opening a pull request

```bash
python -m pytest                              # 60 tests, all must pass
python -m ruff check src tests                # lint
csat check                                    # the dimensional analysis still holds
python scripts/gen_tracking_doc.py --check    # tracking doc matches the code
python scripts/build_notebook.py --check      # notebook matches the sources
```

`make check` runs all of these.

---

## Provenance discipline

Every statement about the method must be traceable to one of four categories, and the tag
must appear in the code or docstring:

| Tag | Meaning | Example |
|---|---|---|
| `[PAPER]` | stated explicitly in arXiv:2507.02957v1 | `Ã = softmax(QK̃ᵀ/√d_k)` |
| `[CHOICE]` | the paper leaves it free; we picked a standard value and said so | `η = 1/σ_max(A)²` |
| `[MISSING]` | the paper needs the quantity but never reports it | the value of `m` |
| `[INCONSIST]` | the paper's own statement is mathematically inconsistent | `Z_i = ΦΨα_i` |

**Never attribute something to the paper that is not in the paper.** If you add a
mechanism the paper does not describe — an optimisation, a better solver, a different
normalisation — label it as yours and keep the paper's version as the default. Two
existing examples to follow:

- FISTA and `debias()` are ours and are reported as *separate columns*, so the raw ISTA
  number is never quietly improved.
- `scaling="paper_sqrt_dk"` is the default because it is the paper's literal formula; the
  re-scalings are offered as labelled diagnostics.

## Adding an experiment

1. Create `src/csat/experiments/expNN_short_name.py` with a `run(...)` that returns plain
   dicts/lists and accepts `device` and `save_dir`/`save_path`.
2. Register it in `src/csat/experiments/__init__.py` (`REGISTRY` and `DESCRIPTIONS`).
3. Add a dispatch branch in `cli.py::_run_one` with sensible quick/full parameters.
4. Open the module docstring with the **question it answers** and what the answer would
   and would not establish. Look at `exp05` for the house style.
5. Add the observations to `docs/FINDINGS.md` with a confidence level and an explicit
   scope statement.

## Adding a unit test

Every test's docstring should say **what it establishes**, not just what it does. The
tests here carry real analytical weight — for instance:

- `test_lista_at_initialisation_equals_ista` proves LISTA is a re-parameterisation of
  ISTA, which is what makes the benchmark fair;
- `test_causal_mask_is_rejected` pins down that token-axis mixing makes causal masking
  undefined;
- `test_paper_equation_does_not_type_check` is the repository's central finding, asserted
  in CI.

Tests must run on CPU in seconds. Mark anything slower with `@pytest.mark.slow`, and
anything needing CUDA with `@pytest.mark.gpu`.

## Changing the tracking table

`csat.utils.reporting.TRACKING_ROWS` is the single source of truth for what counts as
reproduced. Edit it, then run:

```bash
python scripts/gen_tracking_doc.py
```

A component may only move to `Implemented` when the paper specifies it well enough to
implement **and** the implementation matches that specification. Code that runs is not
evidence of reproduction.

## Changing the notebook

Do not edit `notebooks/csat_reproduction_phase4.ipynb` by hand — it is generated. Edit
the package sources (for code) or `scripts/build_notebook.py` (for the markdown
narrative), then:

```bash
python scripts/build_notebook.py
```

CI executes the notebook end to end on every push, so a change that breaks it will be
caught.

## Reporting a disagreement with a finding

Disagreements are welcome and are the point of a reproduction. The most useful form is:

1. which finding, by number from `docs/FINDINGS.md`;
2. what you ran, with seed, hardware and full configuration;
3. what you got;
4. why you think the difference arises.

Findings 6, 7, 10 and 11 rest on synthetic data and are explicitly flagged as
low-to-medium confidence for real models. A result on real pretrained features would be
the single most valuable contribution this repository could receive — see
[docs/PHASE5_PLAN.md](docs/PHASE5_PLAN.md), Tier 1.

## Code style

- `ruff check src tests` must pass; line length 100.
- Type hints where they clarify; docstrings on every public function.
- Docstrings explain the *mathematics* and the *decision*, not just the signature. This
  code is meant to be read by someone learning the method.
- No new runtime dependencies without discussion — the project deliberately needs only
  torch, numpy, pandas and matplotlib, all present in the Kaggle image.
