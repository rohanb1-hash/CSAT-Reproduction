"""Command-line interface for the CSAT reproduction.

    csat info                 # environment, device, package version
    csat check                # run the paper's decoding equation dimension check
    csat list                 # list available experiments
    csat run exp01            # run one experiment
    csat run all --full       # run everything with the full sweeps
    csat track                # print the reproduction tracking table

Every subcommand writes CSV/JSON into ``$CSAT_ROOT/results`` (see
``csat.config``) and figures into ``$CSAT_ROOT/figures``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from csat import __paper__, __version__


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_info(args: argparse.Namespace) -> int:
    import torch

    from csat.config import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR
    from csat.utils.seed import device_report, get_device

    device = get_device(prefer_cuda=not args.cpu)
    info = device_report(device)
    info.update(
        csat_version=__version__,
        paper=__paper__,
        project_root=PROJECT_ROOT,
        results_dir=RESULTS_DIR,
        figures_dir=FIGURES_DIR,
        quick_mode=os.environ.get("CSAT_QUICK", "1") == "1",
    )
    print(json.dumps(info, indent=2))
    if not torch.cuda.is_available():
        print("\nNo GPU visible — everything falls back to CPU. "
              "Nothing in this project requires a GPU.", file=sys.stderr)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Run the dimensional consistency check on the paper's decoding equation."""
    from csat.models.bridges import build_bridge, check_paper_equation

    print("=" * 78)
    print("PAPER'S EQUATION  Z_i = Phi_V Psi alpha_i")
    print("checked against the paper's OWN declared shapes")
    print("=" * 78)
    cases = [(4096, 64, 64), (512, 64, 32), (64, 64, 64)]
    ok_any = False
    for n, d_k, m in cases:
        rep = check_paper_equation(m=m, n=n, d_k=d_k)
        print(f"\n  n={n}, d_k={d_k}, m={m}")
        for key, value in rep.items():
            print(f"     {key:38s}: {value}")
        ok_any = ok_any or bool(rep["equation well-formed?"])

    print("\n  => the equation closes ONLY when m == n == d_k,")
    print("     i.e. exactly when there is no compression at all.\n")

    print("The three dimensionally consistent alternatives implemented here:")
    for name, kwargs in (("denoise", {}),
                         ("feature_cs", {"p_features": 32}),
                         ("token_cs", {"n": 256, "m": 32})):
        spec = build_bridge(name, d_k=64, **kwargs)
        p, k = spec.A.shape
        print(f"\n  [{name}]  A={tuple(spec.A.shape)}  Psi={tuple(spec.psi.shape)}  "
              f"underdetermined={spec.is_underdetermined}")
        print(f"     {'genuine compressed sensing' if spec.is_underdetermined else 'NOT compressed sensing (square system)'}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    from csat.experiments import DESCRIPTIONS

    print(f"csat {__version__}  —  reproduction of {__paper__}\n")
    for name, desc in DESCRIPTIONS.items():
        print(f"  {name}   {desc}")
    print("\nRun one with:  csat run exp01")
    print("Run all with:  csat run all")
    return 0


def cmd_track(args: argparse.Namespace) -> int:
    from csat.utils.reporting import TRACKING_ROWS, status_summary

    try:
        import pandas as pd
        pd.set_option("display.max_colwidth", 60)
        pd.set_option("display.width", 200)
        print(pd.DataFrame(TRACKING_ROWS)[
            ["component", "status", "implemented", "exact_or_approx"]].to_string(index=False))
    except ImportError:
        for row in TRACKING_ROWS:
            print(f"{row['status']:26s} | {row['component']}")

    print("\nStatus counts:")
    for status, count in status_summary().items():
        print(f"   {status:26s}: {count}")
    print("\nA component counts as reproduced only when the paper specifies it well")
    print("enough to implement AND the implementation matches. Code that runs is not")
    print("evidence of reproduction.")
    return 0


# --------------------------------------------------------------------------- #
def _run_one(name: str, device, quick: bool, results_dir: str,
             figures_dir: str) -> dict[str, object]:
    """Dispatch a single experiment with sensible quick/full parameters."""
    import matplotlib
    matplotlib.use("Agg")

    from csat.experiments import REGISTRY
    from csat.utils import plotting
    from csat.utils.reporting import save_table

    mod = REGISTRY[name]
    started = time.time()

    if name == "exp01":
        rows = mod.run(
            seq_len=256 if quick else 512, d_k=64, n_heads=4, batch=1,
            m_values=[16, 32, 64, 128] + ([256] if not quick else []),
            ensembles=["gaussian", "rademacher"],
            token_modes=["iid", "redundant"],
            scalings=["paper_sqrt_dk", "variance_calibrated"],
            device=device)
        save_table(rows, "exp01_attention_fidelity", results_dir)
        plotting.plot_fidelity(rows, os.path.join(figures_dir, "exp01_fidelity.png"))
        out = {"rows": len(rows)}

    elif name == "exp02":
        rows = mod.run(seq_len=256, d_k=64, n_heads=4, batch=1,
                       m_values=[16, 32, 64, 128],
                       share_phi_options=[False, True], device=device)
        save_table(rows, "exp02_effective_attention", results_dir)
        out = {"rows": len(rows)}

    elif name == "exp03":
        res = mod.run(device=device, quick=quick, save_dir=results_dir)
        plotting.plot_recovery_sweeps(res, os.path.join(figures_dir, "exp03_recovery.png"))
        out = {"sweeps": list(res)}

    elif name == "exp04":
        res = mod.run(k=128, p=64, s=8, n_train=2048 if quick else 4096, n_val=512,
                      layers=8, lam=0.005, n_epochs=15 if quick else 40,
                      device=device, verbose=not quick, save_dir=results_dir)
        out = {"matched_budget": res["matched_budget"]}

    elif name == "exp05":
        res = mod.run(seq_len=256, d_k=64, n_heads=4, batch=1,
                      m_values=[32, 64, 128], lam_values=[0.001, 0.01, 0.05, 0.2],
                      token_mode="redundant", n_iters=200 if quick else 500,
                      device=device, save_dir=results_dir)
        out = {"compressibility": res["compressibility_of_C"]}

    elif name == "exp06":
        rows = mod.run(
            seq_lens=[256, 512, 1024, 2048] if quick else [512, 1024, 2048, 4096, 8192],
            m_values=[64, 128], batch=1, n_heads=8, d_k=64,
            ista_iters=20, lista_layers=8,
            warmup=3 if quick else 5, repeats=10 if quick else 20,
            device=device,
            save_path=os.path.join(results_dir, "exp06_efficiency.json"))
        save_table(rows, "exp06_efficiency", results_dir)
        plotting.plot_efficiency(rows, os.path.join(figures_dir, "exp06_efficiency.png"))
        out = {"rows": len(rows)}

    elif name == "exp07":
        rows = mod.run(n_pairs=16, vocab=32, d_model=64, n_heads=4,
                       m_values=[2, 4, 8, 16], steps=400 if quick else 1200,
                       batch=64, lr=3e-3, learnable_phi_options=[False, True],
                       device=device, verbose=not quick,
                       save_path=os.path.join(results_dir, "exp07_learnability.json"))
        save_table([{k: v for k, v in r.items() if k != "history"} for r in rows],
                   "exp07_learnability", results_dir)
        plotting.plot_learnability(rows, os.path.join(figures_dir, "exp07_learnability.png"))
        out = {"accuracies": {f"m={r['m']},learnable={r['learnable_phi']}":
                              round(r["eval_accuracy"], 3) for r in rows}}
    else:
        raise ValueError(f"unknown experiment '{name}'")

    out["elapsed_s"] = round(time.time() - started, 2)
    return out


def cmd_run(args: argparse.Namespace) -> int:
    from csat.config import FIGURES_DIR, RESULTS_DIR, ensure_dirs
    from csat.experiments import REGISTRY
    from csat.utils.seed import get_device, set_seed

    quick = not args.full
    os.environ["CSAT_QUICK"] = "0" if args.full else "1"
    ensure_dirs()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    names: list[str] = sorted(REGISTRY) if args.name == "all" else [args.name]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        print(f"unknown experiment(s): {unknown}. Try `csat list`.", file=sys.stderr)
        return 2

    device = get_device(prefer_cuda=not args.cpu)
    print(f"device: {device} | mode: {'FULL' if args.full else 'QUICK'} | "
          f"seed: {args.seed} | results -> {RESULTS_DIR}\n")

    summary: dict[str, object] = {}
    for name in names:
        set_seed(args.seed)            # every experiment starts from the same seed
        print(f"--- {name} ---", flush=True)
        try:
            summary[name] = _run_one(name, device, quick, RESULTS_DIR, FIGURES_DIR)
            print(f"    done in {summary[name]['elapsed_s']}s\n", flush=True)
        except Exception as exc:                      # noqa: BLE001 - report, don't crash a batch
            summary[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"    FAILED: {type(exc).__name__}: {exc}\n", file=sys.stderr, flush=True)

    path = os.path.join(RESULTS_DIR, "cli_run_summary.json")
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nsummary -> {path}")
    return 1 if any("error" in v for v in summary.values() if isinstance(v, dict)) else 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="csat",
        description=f"CSAT / CS-VLM reproduction ({__paper__}) — Phase 4 toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"csat {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="print environment and device information")
    p_info.add_argument("--cpu", action="store_true", help="force CPU")
    p_info.set_defaults(func=cmd_info)

    p_check = sub.add_parser(
        "check", help="run the dimensional consistency check on the paper's decoding equation")
    p_check.set_defaults(func=cmd_check)

    p_list = sub.add_parser("list", help="list the available experiments")
    p_list.set_defaults(func=cmd_list)

    p_track = sub.add_parser("track", help="print the reproduction tracking table")
    p_track.set_defaults(func=cmd_track)

    p_run = sub.add_parser("run", help="run one experiment, or all of them")
    p_run.add_argument("name", help="experiment name (exp01..exp07) or 'all'")
    p_run.add_argument("--full", action="store_true",
                       help="full sweeps instead of the quick ones")
    p_run.add_argument("--cpu", action="store_true", help="force CPU")
    p_run.add_argument("--seed", type=int, default=1234)
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
