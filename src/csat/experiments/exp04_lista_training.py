"""EXPERIMENT 04 -- LISTA vs ISTA at a matched budget.

QUESTION
    The paper replaces the analytic decoder with LISTA to make decoding cheap:
    "While learned decoders such as LISTA significantly reduce this cost and
    allow for parallel execution, they may sacrifice some generalization or
    require retraining when sparsity levels or modalities change."
    Both halves of that sentence are testable, and the paper tests neither.

WHAT WE MEASURE
    1. Accuracy at matched depth: t LISTA layers vs t ISTA iterations. Because
       LISTA is INITIALISED at exact ISTA (see models/lista.py), the comparison
       starts from equality and any gain is attributable to learning.
    2. How many ISTA iterations are needed to match a trained t-layer LISTA --
       the honest form of "LISTA is cheaper".
    3. Generalisation under distribution shift: train at sparsity s_train, test
       at other sparsity levels. This is the paper's own stated caveat, measured.

STATUS: preliminary (Phase 4).
"""

from __future__ import annotations

import json
import os

import torch

from csat.models.ista import estimate_step_size, fista, ista
from csat.models.lista import LISTA, train_lista
from csat.utils.metrics import relative_l2, support_f1
from csat.utils.tensor_utils import make_sparse_signals


def _problem(k: int, p: int, s: int, n: int, device: torch.device, seed: int = 0):
    torch.manual_seed(seed)
    A = torch.randn(p, k, device=device) / (p ** 0.5)
    A = A / A.norm(dim=0, keepdim=True)
    alpha = make_sparse_signals(n, k, s, device=device)
    return A, alpha, alpha @ A.T


def run(k: int = 128, p: int = 64, s: int = 8, n_train: int = 4096, n_val: int = 512,
        layers: int = 8, lam: float = 0.005, n_epochs: int = 40, lr: float = 1e-3,
        batch_size: int = 128, shift_sparsities: list[int] = (2, 4, 8, 16, 24),
        device: torch.device | str = "cpu", verbose: bool = True,
        save_dir: str | None = None) -> dict[str, object]:
    device = torch.device(device)
    A, alpha, y = _problem(k, p, s, n_train + n_val, device)
    tr, va = slice(0, n_train), slice(n_train, n_train + n_val)

    net = LISTA(p=p, k=k, n_layers=layers, A=A, lam=lam, init_from_ista=True)
    net.to(device)

    # --- 1. matched-budget comparison, before and after training ------------ #
    eta = estimate_step_size(A)
    with torch.no_grad():
        lista_before = net(y[va])
    ista_matched = ista(A, y[va], lam=lam, n_iters=layers, step_size=eta)["alpha"]
    init_gap = (lista_before - ista_matched).abs().max().item()

    history = train_lista(net, y[tr], alpha[tr], y[va], alpha[va],
                          supervision="alpha", n_epochs=n_epochs,
                          batch_size=batch_size, lr=lr, verbose=verbose)

    with torch.no_grad():
        lista_after = net(y[va])

    matched = {
        "layers": layers,
        "lista_init_equals_ista_max_abs_diff": init_gap,
        "ista_at_t_iters_rel_l2": relative_l2(ista_matched, alpha[va]),
        "lista_before_training_rel_l2": relative_l2(lista_before, alpha[va]),
        "lista_after_training_rel_l2": relative_l2(lista_after, alpha[va]),
        "lista_after_training_support_f1": support_f1(lista_after, alpha[va]),
        "ista_at_t_iters_support_f1": support_f1(ista_matched, alpha[va]),
    }

    # --- 2. how many ISTA/FISTA iterations match the trained LISTA? --------- #
    target = matched["lista_after_training_rel_l2"]
    equivalence = {"target_rel_l2": target, "ista_iters_to_match": None,
                   "fista_iters_to_match": None}
    for it in (layers, 25, 50, 100, 250, 500, 1000, 2000, 5000):
        if equivalence["ista_iters_to_match"] is None:
            e = relative_l2(ista(A, y[va], lam=lam, n_iters=it)["alpha"], alpha[va])
            if e <= target:
                equivalence["ista_iters_to_match"] = it
        if equivalence["fista_iters_to_match"] is None:
            e = relative_l2(fista(A, y[va], lam=lam, n_iters=it)["alpha"], alpha[va])
            if e <= target:
                equivalence["fista_iters_to_match"] = it
        if equivalence["ista_iters_to_match"] and equivalence["fista_iters_to_match"]:
            break

    # --- 3. generalisation under sparsity shift ----------------------------- #
    shift = []
    for s_test in shift_sparsities:
        # seed=0 regenerates the SAME operator A the network was trained on, so
        # only the test-time sparsity changes. Using a different A would confound
        # distribution shift with operator mismatch.
        A_test, a_test, y_test = _problem(k, p, s_test, n_val, device, seed=0)
        assert torch.allclose(A_test, A), "the shift test must reuse the training operator"
        with torch.no_grad():
            pred = net(y_test)
        shift.append({
            "train_sparsity": s, "test_sparsity": s_test,
            "lista_rel_l2": relative_l2(pred, a_test),
            "ista_same_budget_rel_l2": relative_l2(
                ista(A_test, y_test, lam=lam, n_iters=layers)["alpha"], a_test),
            "fista_long_rel_l2": relative_l2(
                fista(A_test, y_test, lam=lam, n_iters=1000)["alpha"], a_test),
        })

    out = {"matched_budget": matched, "iteration_equivalence": equivalence,
           "sparsity_shift": shift, "history": history,
           "config": {"k": k, "p": p, "s": s, "layers": layers, "lam": lam,
                      "n_train": n_train, "n_epochs": n_epochs, "lr": lr}}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "exp04_lista.json"), "w") as f:
            json.dump(out, f, indent=2)
    return out
