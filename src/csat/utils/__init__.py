"""Utilities: seeding, metrics, synthetic data, FLOP counting, benchmarking,
plotting and reproduction reporting."""

from csat.utils.benchmarking import (
    benchmark,
    enable_benchmark_mode,
    measure_peak_memory,
    reset_memory_stats,
)
from csat.utils.flops import (
    complexity_table,
    csat_attention_flops,
    ista_decode_flops,
    lista_decode_flops,
    standard_attention_flops,
)
from csat.utils.metrics import (
    all_metrics,
    cosine_similarity,
    nmse_db,
    per_row_relative_l2,
    relative_l2,
    sparsity_profile,
    support_f1,
)
from csat.utils.reporting import (
    STATUSES,
    TRACKING_ROWS,
    list_result_files,
    project_tree,
    save_results,
    save_table,
    status_summary,
    tracking_table,
)
from csat.utils.seed import device_report, get_device, set_seed
from csat.utils.tensor_utils import (
    add_noise,
    count_parameters,
    make_qkv,
    make_redundant_tokens,
    make_sparse_signals,
    spectral_norm,
)

__all__ = [
    "set_seed", "get_device", "device_report",
    "relative_l2", "per_row_relative_l2", "cosine_similarity", "nmse_db",
    "support_f1", "sparsity_profile", "all_metrics",
    "make_qkv", "make_redundant_tokens", "make_sparse_signals", "add_noise",
    "spectral_norm", "count_parameters",
    "benchmark", "measure_peak_memory", "reset_memory_stats", "enable_benchmark_mode",
    "standard_attention_flops", "csat_attention_flops", "ista_decode_flops",
    "lista_decode_flops", "complexity_table",
    "tracking_table", "status_summary", "save_results", "save_table",
    "list_result_files", "project_tree", "TRACKING_ROWS", "STATUSES",
]
