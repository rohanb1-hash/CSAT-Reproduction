#!/usr/bin/env python3
"""Regenerate docs/REPRODUCTION_TRACKING.md from the tracking table in code.

The table lives in ``csat.utils.reporting.TRACKING_ROWS`` so that the CLI
(``csat track``), the notebook, the CSV/JSON exports and this document can never
disagree with each other. Edit the Python list, then run:

    python scripts/gen_tracking_doc.py
"""

from __future__ import annotations

import os
import sys

from csat.utils.reporting import TRACKING_ROWS, status_summary

EMOJI = {
    "Implemented": "✅",
    "Partially implemented": "🟡",
    "Not specified by paper": "❓",
    "Not yet implemented": "⬜",
    "Cannot reproduce exactly": "❌",
}

MEANING = {
    "Implemented": "specified by the paper and implemented as specified",
    "Partially implemented": "implemented, but some sub-detail is our documented choice",
    "Not specified by paper": "the paper needs it but never states it",
    "Not yet implemented": "out of scope for Phase 4",
    "Cannot reproduce exactly": "specified but unreproducible from the information given",
}

HEADER = """# Reproduction tracking

A component counts as reproduced only when the paper specifies it well enough to
implement **and** this implementation matches that specification. Code that runs is
not evidence of reproduction.

This file is generated from `csat.utils.reporting.TRACKING_ROWS`, which is the single
source of truth. Regenerate with `python scripts/gen_tracking_doc.py`, or print the
table with `csat track`.
"""


def build() -> str:
    lines = [HEADER, "## Summary", "", "| Status | Count | Meaning |", "|---|---|---|"]
    for status, count in status_summary().items():
        lines.append(f"| {EMOJI[status]} {status} | {count} | {MEANING[status]} |")

    lines += ["", "## Components", "",
              "| | Component | Status | Exact/approx |", "|---|---|---|---|"]
    for row in TRACKING_ROWS:
        lines.append(f"| {EMOJI[row['status']]} | {row['component']} | "
                     f"{row['status']} | {row['exact_or_approx']} |")

    lines += ["", "---", "", "## Detail", ""]
    for row in TRACKING_ROWS:
        lines += [
            f"### {EMOJI[row['status']]} {row['component']}", "",
            f"- **Status:** {row['status']}",
            f"- **Paper says:** {row['paper_spec']}",
            f"- **Implemented:** {row['implemented']}",
            f"- **Exact or approximate:** {row['exact_or_approx']}",
            f"- **Missing from the paper:** {row['missing_details']}",
            f"- **Our assumption:** {row['assumption']}", "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(repo_root, "docs", "REPRODUCTION_TRACKING.md")
    content = build()

    if "--check" in sys.argv:
        existing = open(out_path).read() if os.path.exists(out_path) else ""
        if existing != content:
            print("docs/REPRODUCTION_TRACKING.md is out of date; "
                  "run python scripts/gen_tracking_doc.py", file=sys.stderr)
            return 1
        print("docs/REPRODUCTION_TRACKING.md is up to date")
        return 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write(content)
    print(f"wrote {out_path} ({len(TRACKING_ROWS)} components)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
