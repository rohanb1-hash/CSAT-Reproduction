"""Matplotlib helpers.

Kept deliberately plain: no styling beyond the defaults, no seaborn dependency,
and every axis labelled with the quantity it shows. Figures are written to
``FIGURES_DIR`` so that the CLI and the notebook export them alongside the
numeric results.

The matplotlib backend is deliberately NOT forced here. Matplotlib already
selects Agg when there is no display (a plain script run), and calling
``matplotlib.use("Agg")`` inside a notebook would disable inline rendering. The
CLI sets Agg explicitly before importing this module, which is the right place
for that decision.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt


def _save(fig, path: str | None):
    """Save the figure when a path is given, and return it either way."""
    if path:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def plot_fidelity(rows: list[dict], save_path: str | None = None):
    """Relative L2 and cosine similarity of Z vs C, against the number of measurements.

    The right-hand panel is the one that matters: cosine similarity is scale-free,
    so it reports directional agreement independently of the large norm mismatch
    between Z and C.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for (mode, scaling), grp in df.groupby(["token_mode", "scaling"]):
        grp = grp.sort_values("m")
        label = f"{mode} / {scaling}"
        axes[0].plot(grp["m"], grp["relative_l2"], marker="o", label=label)
        axes[1].plot(grp["m"], grp["cosine_similarity"], marker="o", label=label)

    axes[0].set_xlabel("m (number of measurements)")
    axes[0].set_ylabel(r"$\|Z-C\|_F / \|C\|_F$")
    axes[0].set_yscale("log")
    axes[0].set_title("Compressed-attention error vs true attention")

    axes[1].set_xlabel("m (number of measurements)")
    axes[1].set_ylabel("mean cosine similarity")
    axes[1].axhline(0.0, color="k", lw=0.8, ls="--")
    axes[1].set_title("Directional agreement (scale-free)")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    return _save(fig, save_path)


def plot_recovery_sweeps(sweeps: dict[str, list[dict]], save_path: str | None = None):
    """Four-panel summary of the sparse-recovery experiment (exp03)."""
    import pandas as pd

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # --- phase transition vs number of measurements ------------------------ #
    meas = pd.DataFrame(sweeps["measurements"])
    for col, lab in (("ista_rel_l2", "ISTA"), ("fista_rel_l2", "FISTA"),
                     ("omp_rel_l2", "OMP")):
        axes[0, 0].plot(meas["measurement_ratio_p_over_k"], meas[col], marker="o", label=lab)
    axes[0, 0].set_xlabel("p / k (measurement ratio)")
    axes[0, 0].set_ylabel(r"$\|\hat\alpha-\alpha\|/\|\alpha\|$")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()
    axes[0, 0].set_title("Phase transition vs measurements")

    # --- breakdown vs sparsity --------------------------------------------- #
    sparsity = pd.DataFrame(sweeps["sparsity"])
    for col, lab in (("ista_rel_l2", "ISTA"), ("fista_rel_l2", "FISTA"),
                     ("omp_rel_l2", "OMP")):
        axes[0, 1].plot(sparsity["sparsity_s"], sparsity[col], marker="o", label=lab)
    axes[0, 1].set_xlabel("sparsity s")
    axes[0, 1].set_ylabel("relative error")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()
    axes[0, 1].set_title("Breakdown vs sparsity")

    # --- accuracy vs decoder compute --------------------------------------- #
    iters = pd.DataFrame(sweeps["iterations"])
    axes[1, 0].plot(iters["n_iters"], iters["ista_rel_l2"], marker="o", label="ISTA")
    axes[1, 0].plot(iters["n_iters"], iters["fista_rel_l2"], marker="s", label="FISTA")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("iterations")
    axes[1, 0].set_ylabel("relative error")
    axes[1, 0].legend()
    axes[1, 0].set_title("Accuracy vs decoder compute")

    # --- stability under noise --------------------------------------------- #
    noise = pd.DataFrame(sweeps["noise"])
    axes[1, 1].plot(noise["noise_std"], noise["fista_rel_l2"], marker="o")
    axes[1, 1].set_xlabel("measurement noise std")
    axes[1, 1].set_ylabel("relative error")
    axes[1, 1].set_title("Stability under noise")

    fig.tight_layout()
    return _save(fig, save_path)


def plot_efficiency(rows: list[dict], save_path: str | None = None):
    """Measured runtime and analytic FLOPs vs sequence length.

    Both panels are shown side by side on purpose: a large FLOP reduction can
    produce a small speedup (or none) when the smaller kernels are memory-bound
    or launch-bound, and conflating the two is how efficiency claims go wrong.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    df = df[df.get("status", "ok") == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # --- measured runtime --------------------------------------------------- #
    for method, grp in df.groupby("method"):
        if method == "standard_attention":
            ordered = grp.sort_values("n")
            axes[0].plot(ordered["n"], ordered["time_ms_mean"], marker="o", lw=2,
                         label="standard attention")
        else:
            for m_val, sub in grp.groupby("m"):
                ordered = sub.sort_values("n")
                axes[0].plot(ordered["n"], ordered["time_ms_mean"], marker=".", ls="--",
                             label=f"{method} (m={int(m_val)})")
    axes[0].set_xlabel("sequence length n")
    axes[0].set_ylabel("time (ms)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=6)
    axes[0].set_title("Measured runtime")

    # --- theoretical FLOPs -------------------------------------------------- #
    flops = df.dropna(subset=["theoretical_GFLOPs"])
    for method, grp in flops.groupby("method"):
        if method == "standard_attention":
            ordered = grp.sort_values("n")
            axes[1].plot(ordered["n"], ordered["theoretical_GFLOPs"], marker="o", lw=2,
                         label="standard")
        else:
            for m_val, sub in grp.groupby("m"):
                ordered = sub.sort_values("n")
                axes[1].plot(ordered["n"], ordered["theoretical_GFLOPs"], marker=".",
                             ls="--", label=f"CSAT attn (m={int(m_val)})")
    axes[1].set_xlabel("sequence length n")
    axes[1].set_ylabel("GFLOPs (analytic)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=6)
    axes[1].set_title("Theoretical FLOPs (attention stage)")

    fig.tight_layout()
    return _save(fig, save_path)


def plot_learnability(rows: list[dict], save_path: str | None = None):
    """Associative-recall accuracy vs m, for a fixed versus a learnable Phi."""
    import pandas as pd

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4))

    baseline = df[df["method"] == "standard_attention"]["eval_accuracy"].iloc[0]
    chance = df["chance_accuracy"].iloc[0]

    csat = df[df["method"] == "csat_attention"]
    for learnable, grp in csat.groupby("learnable_phi"):
        ordered = grp.sort_values("m")
        ax.plot(ordered["m"], ordered["eval_accuracy"], marker="o",
                label=f"CSAT, learnable Phi={learnable}")

    ax.axhline(baseline, color="k", ls="-", lw=1.5, label="standard attention")
    ax.axhline(chance, color="r", ls=":", lw=1.2, label="chance")
    ax.set_xlabel("m (compressed token slots)")
    ax.set_ylabel("recall accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Associative recall after training")

    fig.tight_layout()
    return _save(fig, save_path)
