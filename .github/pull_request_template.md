## What this changes

## Checklist

- [ ] `pytest` passes
- [ ] `ruff check src tests` passes
- [ ] `csat check` still reports the equation as ill-formed
- [ ] `python scripts/gen_tracking_doc.py --check` passes
- [ ] `python scripts/build_notebook.py --check` passes (regenerate if code changed)
- [ ] New claims are tagged `[PAPER]` / `[CHOICE]` / `[MISSING]` / `[INCONSIST]`
- [ ] Nothing is attributed to the paper that is not in the paper
- [ ] Any new finding in `docs/FINDINGS.md` states its confidence level and scope
