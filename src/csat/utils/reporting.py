"""Reproduction tracking, result persistence and plotting.

The tracking table is the honest bookkeeping the notebook is judged by: a
component counts as reproduced only when the paper specifies it well enough to
implement AND our implementation matches that specification -- never merely
because code runs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

STATUSES = (
    "Implemented",              # specified by the paper and implemented as specified
    "Partially implemented",    # implemented, but some sub-detail is our choice
    "Not specified by paper",   # the paper needs it but never states it
    "Not yet implemented",      # out of scope for this notebook
    "Cannot reproduce exactly", # specified but unreproducible from the information given
)


# --------------------------------------------------------------------------- #
# The tracking table
# --------------------------------------------------------------------------- #
TRACKING_ROWS: list[dict[str, str]] = [
    {
        "component": "Standard attention baseline",
        "paper_spec": "Attn(Q,K,V) = softmax(QK^T/sqrt(d_k))V, Sec. 3",
        "implemented": "Yes",
        "exact_or_approx": "Exact",
        "missing_details": "None",
        "assumption": "[B,H,N,D] multi-head layout (paper writes the single-head case)",
        "status": "Implemented",
    },
    {
        "component": "Q/K/V projections",
        "paper_spec": "Q=XW^Q, K=XW^K, V=XW^V with W in R^{d x d_k}, Sec. 3",
        "implemented": "Yes",
        "exact_or_approx": "Exact",
        "missing_details": "How heads compose; bias terms",
        "assumption": "d_model = H*d_k (Vaswani convention); bias enabled",
        "status": "Implemented",
    },
    {
        "component": "Measurement matrices Phi_K, Phi_V",
        "paper_spec": "Phi in R^{m x n} from sub-Gaussian ensembles, RIP-satisfying, Sec. 3",
        "implemented": "Yes (gaussian / rademacher / orthogonal / hadamard)",
        "exact_or_approx": "Approximate",
        "missing_details": "Normalisation constant; sharing across heads/layers/modalities; whether re-drawn per batch",
        "assumption": "1/sqrt(m) scaling so E[Phi^T Phi]=I; one Phi shared across heads; fixed at init",
        "status": "Partially implemented",
    },
    {
        "component": "Value of m",
        "paper_spec": "'m << n'; Table 5 benchmarks n=4096",
        "implemented": "Swept as a free parameter",
        "exact_or_approx": "N/A",
        "missing_details": "The paper never states m for ANY experiment",
        "assumption": "We sweep m in {16..256} and report m explicitly everywhere",
        "status": "Not specified by paper",
    },
    {
        "component": "Compressed attention A~ = softmax(QK~^T/sqrt(d_k)), Z = A~V~",
        "paper_spec": "Sec. 3, explicit equations",
        "implemented": "Yes",
        "exact_or_approx": "Exact",
        "missing_details": "Whether the sqrt(d_k) scale is re-calibrated after projection",
        "assumption": "Paper's literal sqrt(d_k) is the default; two re-scalings offered as labelled diagnostics",
        "status": "Implemented",
    },
    {
        "component": "RIP assumption on Phi",
        "paper_spec": "'measurement matrices satisfying the RIP'",
        "implemented": "Diagnostics only (coherence, Monte-Carlo lower bound)",
        "exact_or_approx": "Approximate",
        "missing_details": "RIP order s and constant delta_s are never stated; what is sparse along the TOKEN axis is never identified",
        "assumption": "We report coherence and a Monte-Carlo LOWER bound; exact RIP verification is NP-hard",
        "status": "Cannot reproduce exactly",
    },
    {
        "component": "Sparse representation C_i = Psi alpha_i",
        "paper_spec": "Psi in R^{d_k x d_k}, alpha_i sparse, Sec. 3",
        "implemented": "Yes (identity / DCT / random orthogonal / overcomplete / learned)",
        "exact_or_approx": "Approximate",
        "missing_details": "How Psi is obtained; sparsity level s; any evidence that context vectors are sparse",
        "assumption": "DCT default (the paper's own JPEG analogy); we MEASURE compressibility rather than assume it",
        "status": "Partially implemented",
    },
    {
        "component": "Decoding equation Z_i = Phi Psi alpha_i",
        "paper_spec": "Sec. 3, with Phi = Phi_V reused",
        "implemented": "Cannot be implemented as written",
        "exact_or_approx": "N/A",
        "missing_details": "Phi_V is [m x n] while Psi alpha_i is [d_k]; the product needs n = d_k and still lands in R^m, not R^{d_k}. Closes only if m = n = d_k, contradicting m << n",
        "assumption": "Three labelled, dimensionally consistent alternatives implemented instead (models/bridges.py)",
        "status": "Cannot reproduce exactly",
    },
    {
        "component": "Basis pursuit / l1 recovery",
        "paper_spec": "alpha_hat = argmin ||alpha||_1 s.t. Z_i = Phi Psi alpha",
        "implemented": "Unconstrained LASSO relaxation via ISTA/FISTA, plus OMP",
        "exact_or_approx": "Approximate",
        "missing_details": "lambda, step size, iteration count, stopping rule",
        "assumption": "eta = 1/sigma_max(A)^2; lambda and iterations swept and reported",
        "status": "Partially implemented",
    },
    {
        "component": "ISTA decoder",
        "paper_spec": "Named as the analytic solver, Sec. 3 and Sec. 5",
        "implemented": "Yes, batched over tokens, plus FISTA",
        "exact_or_approx": "Approximate",
        "missing_details": "All hyper-parameters; how the analytic decoder is trained end to end (it is not differentiable)",
        "assumption": "Fixed iteration budget, no early stop, so timings stay comparable",
        "status": "Partially implemented",
    },
    {
        "component": "OMP decoder",
        "paper_spec": "Named alongside ISTA, Sec. 2/5",
        "implemented": "Yes, batched",
        "exact_or_approx": "Approximate",
        "missing_details": "Sparsity level s is required by OMP and never stated",
        "assumption": "s supplied as an explicit experiment parameter",
        "status": "Partially implemented",
    },
    {
        "component": "LISTA decoder",
        "paper_spec": "alpha^{t+1} = eta_theta(S alpha^t + B Z_i), Sec. 3",
        "implemented": "Yes, with W_s/W_e initialised from ISTA",
        "exact_or_approx": "Approximate",
        "missing_details": "Depth t, weight tying, threshold form, training data, loss, optimiser",
        "assumption": "Untied by default, learned per-coordinate threshold, Adam, supervised on alpha for synthetic data",
        "status": "Partially implemented",
    },
    {
        "component": "C_hat_i = Psi alpha_hat_i (row-wise decoding)",
        "paper_spec": "Sec. 3",
        "implemented": "Yes",
        "exact_or_approx": "Exact given a bridge",
        "missing_details": "Depends on the ill-posed measurement equation above",
        "assumption": "Applied row-wise to Z under BRIDGE 1",
        "status": "Partially implemented",
    },
    {
        "component": "Causal / autoregressive masking",
        "paper_spec": "Not discussed; WikiText-103 LM results are reported (Table 1)",
        "implemented": "Rejected with an explicit error",
        "exact_or_approx": "N/A",
        "missing_details": "How CSAT performs autoregressive LM when each compressed key slot mixes all n tokens, including future ones",
        "assumption": "We refuse causal masks rather than apply a meaningless one",
        "status": "Cannot reproduce exactly",
    },
    {
        "component": "Complexity claim O(n^2 d) -> O(nmd + decoding)",
        "paper_spec": "Sec. 1 and Sec. 3",
        "implemented": "Analytic FLOP counter + measured runtime, reported separately",
        "exact_or_approx": "Approximate",
        "missing_details": "The 'decoding' term is never expanded; it is O(n * iters * p * k) for ISTA",
        "assumption": "We count both stages and never report attention-only speedups as pipeline speedups",
        "status": "Partially implemented",
    },
    {
        "component": "Table 5 efficiency numbers (18.4GB/1113ms vs 6.9GB/439ms)",
        "paper_spec": "n = 4096",
        "implemented": "Scaling measured on this notebook's device",
        "exact_or_approx": "Cannot match absolutes",
        "missing_details": "GPU model, precision, batch size, layer count, m, decoder configuration, whether decoding is included",
        "assumption": "We report our own hardware and every parameter, and do not claim to match the paper's absolute numbers",
        "status": "Cannot reproduce exactly",
    },
    {
        "component": "WikiText-103 language modelling (Table 1)",
        "paper_spec": "12 layers, 512 hidden, 8 heads, 151M params, 300k steps, perplexity 18.7",
        "implemented": "No",
        "exact_or_approx": "N/A",
        "missing_details": "Tokeniser, context length, optimiser, LR schedule, m, decoder config; 300k steps is far beyond a notebook budget",
        "assumption": "Out of scope for Phase 4; noted as a Phase 5 item requiring a multi-GPU run",
        "status": "Not yet implemented",
    },
    {
        "component": "LRA Pathfinder-X (Table 2)",
        "paper_spec": "Sequence length 4096, accuracy 84.2%",
        "implemented": "No",
        "exact_or_approx": "N/A",
        "missing_details": "Full training recipe; Pathfinder-X is notoriously sensitive to it",
        "assumption": "Out of scope for Phase 4",
        "status": "Not yet implemented",
    },
    {
        "component": "BLIP retrieval / captioning (Tables 3-4)",
        "paper_spec": "Flickr30k, MS-COCO, CSAT blocks replacing BLIP attention",
        "implemented": "No",
        "exact_or_approx": "N/A",
        "missing_details": "Which layers were replaced, fine-tuning schedule, m, decoder config, checkpoint",
        "assumption": "Out of scope for Phase 4; requires pretrained BLIP weights and multimodal datasets",
        "status": "Not yet implemented",
    },
    {
        "component": "Comparison with Linformer / Performer / Longformer",
        "paper_spec": "Baselines throughout Tables 1-5",
        "implemented": "No",
        "exact_or_approx": "N/A",
        "missing_details": "Baseline configurations are not given",
        "assumption": "Deferred to Phase 5",
        "status": "Not yet implemented",
    },
]


def tracking_table():
    """Return the tracking table as a pandas DataFrame (import kept local)."""
    import pandas as pd
    return pd.DataFrame(TRACKING_ROWS)


def status_summary() -> dict[str, int]:
    counts = dict.fromkeys(STATUSES, 0)
    for row in TRACKING_ROWS:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_results(obj, path: str) -> str:
    """Write JSON (dict/list) or CSV (DataFrame) and return the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
    elif path.endswith(".csv"):
        obj.to_csv(path, index=False)
    else:
        raise ValueError("path must end in .json or .csv")
    return path


def save_table(rows: Sequence[dict], stem: str, results_dir: str) -> list[str]:
    """Save a list of dicts as BOTH csv and json; returns the two paths."""
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, f"{stem}.csv")
    json_path = os.path.join(results_dir, f"{stem}.json")
    save_results(df, csv_path)
    save_results(list(rows), json_path)
    return [csv_path, json_path]


def list_result_files(results_dir: str) -> list[str]:
    if not os.path.isdir(results_dir):
        return []
    return sorted(
        os.path.join(results_dir, f) for f in os.listdir(results_dir)
        if f.endswith((".csv", ".json"))
    )


def project_tree(root: str, skip: Sequence[str] = ("__pycache__", ".ipynb_checkpoints")) -> str:
    """ASCII tree of the project directory, for the notebook's final section."""
    lines = []
    root = root.rstrip("/")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip)
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if rel != ".":
            lines.append("    " * (depth - 1) + f"|-- {os.path.basename(dirpath)}/")
        for fn in sorted(filenames):
            if fn.endswith(".pyc"):
                continue
            lines.append("    " * depth + f"|-- {fn}")
    return f"{os.path.basename(root)}/\n" + "\n".join(lines)
