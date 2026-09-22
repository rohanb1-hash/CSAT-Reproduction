---
name: Disagreement with a finding
about: You ran something and got a different result
title: "[finding N] "
labels: reproduction
---

**Which finding** (number from `docs/FINDINGS.md`):

**What you ran** (command, seed, hardware, full configuration):

```
csat run expNN ...
```

**What you got** (paste the relevant rows of results/):

**Why you think the difference arises:**

**Is this synthetic or real data?**
Findings 4, 6, 7, 10 and 11 are explicitly flagged as low-to-medium confidence
for real models. A result on real pretrained features is the most valuable kind
of contribution here — see `docs/PHASE5_PLAN.md`, Tier 1.
