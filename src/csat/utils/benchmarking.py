"""Runtime and memory benchmarking utilities.

Rules this module enforces so that comparisons are fair:

1.  GPU work is asynchronous. Every timed region is wrapped in
    ``torch.cuda.synchronize()`` so we measure completed work, not queue time.
2.  The first call to a CUDA kernel includes allocation and autotuning cost, so
    we always run ``warmup`` untimed iterations first.
3.  We report mean AND standard deviation over ``repeats`` runs; a single
    measurement on a shared Kaggle GPU is not a measurement.
4.  Peak memory is read with ``torch.cuda.max_memory_allocated`` (tensor bytes)
    and ``max_memory_reserved`` (bytes the caching allocator holds). They differ,
    and quoting only one of them is how benchmark numbers get inflated.
5.  Out-of-memory is caught and recorded as a result ("OOM"), not as a crash,
    because full attention at n = 8192 genuinely does not fit on a 16 GB card.
"""

from __future__ import annotations

import gc
import time
from typing import Callable

import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_memory_stats(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def benchmark(fn: Callable[[], object], device: torch.device,
              warmup: int = 5, repeats: int = 20,
              measure_memory: bool = True,
              label: str = "") -> dict[str, object]:
    """Time ``fn`` and record peak memory. Returns a dict, never raises on OOM.

    ``fn`` must be a zero-argument closure that performs exactly the work to be
    measured (allocate inputs OUTSIDE the closure, or input allocation will be
    counted as part of the measurement).
    """
    result: dict[str, object] = {"label": label, "status": "ok", "device": str(device)}
    try:
        with torch.no_grad():
            for _ in range(warmup):
                fn()
            _sync(device)

            reset_memory_stats(device)
            timings = []
            for _ in range(repeats):
                _sync(device)
                t0 = time.perf_counter()
                fn()
                _sync(device)
                timings.append((time.perf_counter() - t0) * 1000.0)  # ms

        t = torch.tensor(timings)
        result.update(
            time_ms_mean=t.mean().item(),
            time_ms_std=t.std(unbiased=False).item(),
            time_ms_median=t.median().item(),
            time_ms_min=t.min().item(),
            repeats=repeats,
        )
        if measure_memory and device.type == "cuda":
            result.update(
                peak_allocated_MB=torch.cuda.max_memory_allocated() / 1024 ** 2,
                peak_reserved_MB=torch.cuda.max_memory_reserved() / 1024 ** 2,
            )
        else:
            result.update(peak_allocated_MB=float("nan"),
                          peak_reserved_MB=float("nan"))
    except torch.cuda.OutOfMemoryError:
        result.update(status="OOM", time_ms_mean=float("nan"),
                      time_ms_std=float("nan"), time_ms_median=float("nan"),
                      time_ms_min=float("nan"),
                      peak_allocated_MB=float("nan"), peak_reserved_MB=float("nan"))
        reset_memory_stats(device)
    except RuntimeError as exc:              # CPU OOM surfaces as RuntimeError
        if "out of memory" not in str(exc).lower():
            raise
        result.update(status="OOM", time_ms_mean=float("nan"),
                      time_ms_std=float("nan"), time_ms_median=float("nan"),
                      time_ms_min=float("nan"),
                      peak_allocated_MB=float("nan"), peak_reserved_MB=float("nan"))
        reset_memory_stats(device)
    return result


def measure_peak_memory(fn: Callable[[], object], device: torch.device
                        ) -> dict[str, float] | None:
    """Peak memory of a single forward call, with no timing loop."""
    reset_memory_stats(device)
    with torch.no_grad():
        fn()
    _sync(device)
    if device.type != "cuda":
        return None
    return {
        "peak_allocated_MB": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "peak_reserved_MB": torch.cuda.max_memory_reserved() / 1024 ** 2,
    }


def enable_benchmark_mode() -> None:
    """Turn ON cuDNN autotuning for timing runs.

    Deterministic mode (set during correctness tests) can change kernel choice
    and therefore runtime, so timing and determinism are deliberately separated.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
