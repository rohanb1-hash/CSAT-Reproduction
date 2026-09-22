"""EXPERIMENT 07 -- Can a CSAT block be trained? (a learnability stress test)

QUESTION
    Every measurement so far evaluates CSAT at random initialisation. The
    paper's numbers come from TRAINED models, and training could in principle
    let W^Q/W^K/W^V (and a learnable Phi) adapt to the projection. This probe
    asks whether the mechanism trains at all, and what token-axis compression
    costs on a task that genuinely requires retrieving one specific token.

TASK: content-addressed associative recall (solvable by ONE attention layer)
    Each sample holds P key-value pairs and a query key.
      position j < P : embedding = E_key(key_j) + E_val(value_j)
      position P     : embedding = E_query(query)
    The model must attend from the query position to the position whose key
    matches, and read that position's value out of the value component.
    There are no positional embeddings: retrieval is purely content-based, so
    the task isolates exactly the capability that mixing the token axis should
    damage. Chance accuracy is 1/vocab.

WHY THIS TASK AND NOT THE PAPER'S
    This is NOT WikiText-103, LRA Pathfinder-X, Flickr30k or MS-COCO, and it
    cannot confirm or refute Tables 1-4. Those require pretraining budgets far
    beyond a notebook. What it can establish is narrow and real: whether the
    block optimises, and how accuracy moves as m falls -- on a task where the
    correct answer is known by construction.

VARIANTS COMPARED
    standard attention | CSAT with fixed Phi | CSAT with learnable Phi
    (a learnable Phi is allowed by the paper's Section 7 and is the variant most
    likely to recover accuracy, since it can learn WHICH tokens to keep.)

STATUS: preliminary (Phase 4).
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from csat.models.csat_block import CSATBlock
from csat.models.standard_attention import MultiHeadSelfAttention


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #
def make_recall_batch(batch: int, n_pairs: int, vocab: int, device: torch.device):
    """Returns (keys [B,P], values [B,P], query [B], target [B]).

    Keys are distinct within a sample so the answer is unambiguous.
    """
    keys = torch.stack([torch.randperm(vocab, device=device)[:n_pairs]
                        for _ in range(batch)])
    values = torch.randint(0, vocab, (batch, n_pairs), device=device)
    pick = torch.randint(0, n_pairs, (batch,), device=device)
    rows = torch.arange(batch, device=device)
    return keys, values, keys[rows, pick], values[rows, pick]


class RecallModel(nn.Module):
    """Embed pairs -> one attention block -> residual+norm -> classify last position."""

    def __init__(self, attention: nn.Module, vocab: int, d_model: int):
        super().__init__()
        self.key_embed = nn.Embedding(vocab, d_model)
        self.val_embed = nn.Embedding(vocab, d_model)
        self.query_embed = nn.Embedding(vocab, d_model)
        self.attn = attention
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, keys, values, query):
        pairs = self.key_embed(keys) + self.val_embed(values)      # [B, P, d]
        q = self.query_embed(query).unsqueeze(1)                   # [B, 1, d]
        x = torch.cat([pairs, q], dim=1)                           # [B, P+1, d]
        out, _ = self.attn(x)
        x = self.norm(x + out)                                     # residual, as in Fig. 1
        return self.head(x[:, -1])


# --------------------------------------------------------------------------- #
def train_model(model: nn.Module, n_pairs: int, vocab: int, steps: int,
                batch: int, lr: float, device: torch.device,
                eval_batches: int = 16, log_every: int = 100,
                verbose: bool = True) -> dict[str, object]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    history = {"step": [], "loss": [], "train_acc": []}

    model.train()
    for step in range(steps):
        k, v, q, target = make_recall_batch(batch, n_pairs, vocab, device)
        logits = model(k, v, q)
        loss = lossf(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            acc = (logits.argmax(-1) == target).float().mean().item()
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["train_acc"].append(acc)
            if verbose:
                print(f"    step {step:4d} | loss {loss.item():.4f} | acc {acc:.3f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(eval_batches):
            k, v, q, target = make_recall_batch(batch, n_pairs, vocab, device)
            correct += (model(k, v, q).argmax(-1) == target).sum().item()
            total += target.numel()
    return {"history": history, "eval_accuracy": correct / total,
            "chance_accuracy": 1.0 / vocab, "final_train_loss": history["loss"][-1]}


@torch.no_grad()
def phi_concentration(phi: torch.Tensor) -> dict[str, float]:
    """How concentrated is each row of Phi on a few tokens?

    Participation ratio per row:  PR = (sum phi^2)^2 / sum phi^4, normalised by n.
    PR/n = 1/n  -> the row reads a single token (a selection matrix, no mixing).
    PR/n = 1    -> the row spreads uniformly over all n tokens (maximal mixing).

    A Gaussian Phi starts near PR/n ~ 1/3 (the value for i.i.d. normal weights).
    If a LEARNED Phi drifts towards 1/n, it is learning NOT to mix -- i.e. it is
    discarding the compressed-sensing character of the operator and behaving like
    a learned token-selection/pooling matrix instead.
    """
    p = phi.detach().reshape(-1, phi.shape[-1])
    s2 = p.pow(2).sum(dim=-1)
    s4 = p.pow(4).sum(dim=-1)
    pr = (s2 ** 2) / s4.clamp_min(1e-20)
    n = p.shape[-1]
    return {
        "participation_ratio_mean": pr.mean().item(),
        "participation_fraction": (pr / n).mean().item(),
        "row_max_share": (p.abs().max(dim=-1).values /
                          p.abs().sum(dim=-1).clamp_min(1e-20)).mean().item(),
    }


def run(n_pairs: int = 16, vocab: int = 32, d_model: int = 64, n_heads: int = 4,
        m_values: list[int] = (2, 4, 8), steps: int = 800, batch: int = 64,
        lr: float = 3e-3, learnable_phi_options: list[bool] = (False, True),
        device: torch.device | str = "cpu", verbose: bool = True,
        save_path: str | None = None) -> list[dict]:
    device = torch.device(device)
    seq_len = n_pairs + 1
    results: list[dict] = []

    if verbose:
        print("  [baseline] full attention")
    torch.manual_seed(0)
    base = RecallModel(MultiHeadSelfAttention(d_model, n_heads), vocab, d_model)
    out = train_model(base, n_pairs, vocab, steps, batch, lr, device, verbose=verbose)
    results.append({"method": "standard_attention", "m": None, "learnable_phi": None,
                    "n": seq_len, "compression_ratio": 1.0,
                    "eval_accuracy": out["eval_accuracy"],
                    "chance_accuracy": out["chance_accuracy"],
                    "final_train_loss": out["final_train_loss"],
                    "history": out["history"]})

    for learnable in learnable_phi_options:
        for m in m_values:
            if m > seq_len:
                continue
            if verbose:
                print(f"  [CSAT] m={m}, learnable_phi={learnable}")
            torch.manual_seed(0)
            blk = CSATBlock(d_model=d_model, n_heads=n_heads, seq_len=seq_len, m=m,
                            learnable_phi=learnable, decoder="none", device=device)
            model = RecallModel(blk, vocab, d_model)
            phi_before = phi_concentration(blk.attn.phi_k.phi)
            out = train_model(model, n_pairs, vocab, steps, batch, lr, device,
                              verbose=verbose)
            phi_after = phi_concentration(blk.attn.phi_k.phi)
            results.append({"method": "csat_attention", "m": m,
                            "learnable_phi": learnable, "n": seq_len,
                            "compression_ratio": seq_len / m,
                            "eval_accuracy": out["eval_accuracy"],
                            "chance_accuracy": out["chance_accuracy"],
                            "final_train_loss": out["final_train_loss"],
                            "phi_participation_before": phi_before["participation_fraction"],
                            "phi_participation_after": phi_after["participation_fraction"],
                            "phi_row_max_share_before": phi_before["row_max_share"],
                            "phi_row_max_share_after": phi_after["row_max_share"],
                            "history": out["history"]})

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
    return results
