# Reference run

Committed output of:

```bash
CSAT_ROOT=$(pwd) csat run all --cpu     # QUICK mode, seed 1234
```

produced on CPU so that anyone can reproduce it without a GPU. It exists so the
numbers quoted in [`docs/FINDINGS.md`](../../docs/FINDINGS.md) can be checked
without re-running anything.

**What varies if you re-run it**

- `exp06` wall-clock times vary by tens of percent between runs and substantially
  between devices. The *ratios* and their direction are the stable part.
- Everything else is seeded and should reproduce to within floating-point noise
  on the same PyTorch version.
- `--full` (instead of QUICK) widens every sweep and trains `exp04`/`exp07` for
  longer; `exp07` in particular needs the full run before its intermediate `m`
  values are properly trained.

Figures for the same run are in [`../../figures/reference/`](../../figures/reference).
