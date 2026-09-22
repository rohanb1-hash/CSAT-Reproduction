# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-09-22

Initial release: Phase 4, *PyTorch implementation and paper reproduction*.

### Added
- `csat.models` — standard attention, measurement operators (Gaussian /
  Rademacher / orthogonal / Hadamard), compressed attention, dictionaries
  (identity / DCT / random orthogonal / overcomplete / learned), ISTA, FISTA,
  OMP, LISTA, and the end-to-end `CSATBlock` of the paper's Figure 1.
- `csat.models.bridges` — `check_paper_equation()` plus the three dimensionally
  consistent readings of the paper's decoding equation (`denoise`,
  `feature_cs`, `token_cs`).
- `csat.experiments` — seven experiments (exp01–exp07) covering fidelity,
  effective-attention structure, sparse recovery, LISTA vs ISTA, the decoder
  bridges, efficiency, and a learnability probe.
- `csat.utils` — seeding, metrics, synthetic data, analytic FLOP counting,
  benchmarking with warm-up/synchronisation/repeats, plotting and reproduction
  tracking.
- `csat` CLI: `info`, `check`, `list`, `run`, `track`.
- 60 unit tests, GitHub Actions CI across Python 3.9 / 3.11 / 3.12, and a CI job
  that executes the Kaggle notebook end to end.
- `notebooks/csat_reproduction_phase4.ipynb` — the self-contained tutorial
  notebook, regenerable from package sources via `scripts/build_notebook.py`.
- Documentation: paper specification, dimensional analysis, findings,
  reproduction tracking, and the Phase 5 plan.

### Findings recorded
- The paper's decoding equation `Z_i = Phi Psi alpha_i` does not type-check
  under the paper's own declared shapes for any `m << n`.
- No fixed operator relates `Z` to the true context `C`, so the recovery problem
  has no measurement operator independently of shapes.
- Measured: `M_eff = A~ Phi_V` is ~50% negative with non-unit row sums; context
  vectors are not sparse in the bases tested; decoding `Z` does not improve
  directional agreement with `C`; a fixed random `Phi` fails at content-addressed
  retrieval while a learned `Phi` succeeds by learning not to mix.

### Not included
- The paper's Tables 1–5 (WikiText-103, LRA Pathfinder-X, Flickr30k, MS-COCO,
  and the efficiency table). See `docs/REPRODUCTION_TRACKING.md` for why each is
  out of reach from the information the paper provides.
