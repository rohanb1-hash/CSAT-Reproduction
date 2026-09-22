"""Experiments.

Each module exposes ``run(...)`` and returns plain dicts/lists, so results go
straight into DataFrames, CSV and JSON.

=========  ====================================================================
Module     Question it answers
=========  ====================================================================
exp01      How close is the compressed output Z to true attention C?
exp02      Is the effective matrix A~ Phi_V still a weighted average of values?
exp03      Do ISTA / FISTA / OMP recover sparse signals, and in which regime?
exp04      LISTA vs ISTA at matched budget, and under sparsity shift.
exp05      Are context vectors sparse at all, and does decoding Z help?
exp06      Runtime and memory, with decoder overhead reported separately.
exp07      Does a CSAT block train on a task that needs precise retrieval?
=========  ====================================================================

``REGISTRY`` maps short names to modules so the CLI can run them by name.
"""

from csat.experiments import (
    exp01_attention_fidelity,
    exp02_effective_attention,
    exp03_sparse_recovery,
    exp04_lista_training,
    exp05_decoder_bridge,
    exp06_efficiency,
    exp07_learnability,
)

REGISTRY = {
    "exp01": exp01_attention_fidelity,
    "exp02": exp02_effective_attention,
    "exp03": exp03_sparse_recovery,
    "exp04": exp04_lista_training,
    "exp05": exp05_decoder_bridge,
    "exp06": exp06_efficiency,
    "exp07": exp07_learnability,
}

DESCRIPTIONS = {
    "exp01": "Fidelity of compressed attention Z against true attention C",
    "exp02": "Structure of the effective attention matrix M_eff = A~ Phi_V",
    "exp03": "Sparse recovery with ISTA / FISTA / OMP (a genuine CS problem)",
    "exp04": "LISTA vs ISTA at matched budget, and generalisation under shift",
    "exp05": "Is C sparse? Does sparse decoding of Z reduce the error to C?",
    "exp06": "Runtime and memory: attention stage vs the full pipeline",
    "exp07": "Learnability probe on content-addressed associative recall",
}

__all__ = [
    "REGISTRY", "DESCRIPTIONS",
    "exp01_attention_fidelity", "exp02_effective_attention",
    "exp03_sparse_recovery", "exp04_lista_training", "exp05_decoder_bridge",
    "exp06_efficiency", "exp07_learnability",
]
