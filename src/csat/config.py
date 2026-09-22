"""
Central configuration for the CSAT (CS-VLM) reproduction.

Every hyper-parameter used anywhere in the project is declared here so that the
notebook never hides "magic numbers" inside experiment code.

PROVENANCE TAGS used throughout this project
--------------------------------------------
[PAPER]     : value or mechanism explicitly stated in arXiv:2507.02957v1.
[CHOICE]    : the paper leaves this free; we picked a standard, documented value.
[MISSING]   : the paper needs this quantity but never reports it. Our value is a
              placeholder for experimentation, NOT a reproduction of the paper.
[INCONSIST] : the paper's own statement is mathematically inconsistent here.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Literal


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _default_root() -> str:
    """Where results and figures are written.

    Resolution order:
      1. ``$CSAT_ROOT`` if set (what CI and the notebook use);
      2. ``/kaggle/working/csat-reproduction`` when running on Kaggle;
      3. the repository root, found by walking up from this file.
    """
    env = os.environ.get("CSAT_ROOT")
    if env:
        return env
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/csat-reproduction"
    # src/csat/config.py -> src/csat -> src -> <repo root>
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


PROJECT_ROOT: str = _default_root()
RESULTS_DIR: str = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR: str = os.path.join(PROJECT_ROOT, "figures")

# QUICK_MODE shrinks every sweep so the whole notebook runs in a few minutes.
# Set CSAT_QUICK=0 in the environment for the full sweeps.
QUICK_MODE: bool = os.environ.get("CSAT_QUICK", "1") == "1"

GLOBAL_SEED: int = 1234


def ensure_dirs() -> None:
    """Create every directory the project writes to."""
    for d in (PROJECT_ROOT, RESULTS_DIR, FIGURES_DIR):
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------- #
# Attention / CSAT configuration
# --------------------------------------------------------------------------- #
EnsembleName = Literal["gaussian", "rademacher", "orthogonal", "hadamard"]
ScalingName = Literal["paper_sqrt_dk", "row_normalized", "variance_calibrated"]
DictionaryName = Literal["identity", "dct", "random_orthogonal", "overcomplete", "learnable"]


@dataclass
class AttentionConfig:
    """Shapes for the attention tensors, in the [B, H, N, D] layout.

    Paper notation -> code notation
        n   (number of tokens)      -> N  (seq_len)
        d_k (per-head key dim)      -> D  (d_k)
        d   (model/embedding dim)   -> d_model = H * d_k  [CHOICE: the paper writes
            W^Q in R^{d x d_k} for a single head and never states how heads compose;
            we use the standard Vaswani convention d_model = H * d_k.]
    """

    batch_size: int = 2          # [MISSING] paper never reports batch sizes
    n_heads: int = 8             # [PAPER] "8 attention heads" (WikiText-103 setup)
    seq_len: int = 512           # [PAPER] sequence lengths 512..8192 are benchmarked
    d_k: int = 64                # [CHOICE] 512 hidden / 8 heads = 64 (paper's LM config)
    dropout: float = 0.0         # [MISSING] paper never mentions attention dropout

    @property
    def d_model(self) -> int:
        return self.n_heads * self.d_k


@dataclass
class CSATConfig:
    """Configuration of the compressed-sensing attention block (paper Section 3)."""

    # ---- compression ------------------------------------------------------ #
    m: int = 64
    """[PAPER] 'number of measurements m, with m << n'.
    [MISSING] The paper NEVER states the value of m used in any experiment,
    including Table 5. Any specific m here is ours, not the paper's."""

    ensemble: EnsembleName = "gaussian"
    """[PAPER] 'Phi_K and Phi_V are typically drawn from sub-Gaussian ensembles
    (e.g., random Gaussian, Rademacher, or structured Hadamard matrices)'."""

    normalize_phi: bool = True
    """[CHOICE] Scale entries by 1/sqrt(m) so that E[Phi^T Phi] = I_n, the standard
    normalisation under which RIP results are stated. The paper does not give
    a normalisation constant."""

    share_phi: bool = False
    """[CHOICE] If True use Phi_K = Phi_V. The paper writes them as two separate
    matrices (so default False) but also says Phi = Phi_V is 'reused' at decode
    time, and sharing has a concrete theoretical consequence we test in exp02."""

    learnable_phi: bool = False
    """[PAPER, both options] Section 7: 'measurement matrices can be fixed
    post-training or made learnable'. Default follows the CS framing (fixed)."""

    per_head_phi: bool = False
    """[MISSING] The paper's math is single-head; it never says whether heads share
    Phi. False = one Phi shared by all heads ('a shared measurement matrix Phi',
    Section 3, VLM paragraph)."""

    scaling: ScalingName = "paper_sqrt_dk"
    """[PAPER] The paper divides the compressed logits by sqrt(d_k), exactly as in
    standard attention. The other two options are OUR diagnostics (exp01) because
    Phi changes the variance of the logits; they are NOT the paper's formulation."""

    # ---- sparse decoding -------------------------------------------------- #
    decoder: Literal["none", "ista", "lista", "omp"] = "none"
    """Which sparse decoder reconstructs C_hat from Z. 'none' = the compressed
    attention output is used directly (ablation baseline)."""

    dictionary: DictionaryName = "dct"
    """[MISSING] The paper posits a dictionary Psi in R^{d_k x d_k} such that
    C_i = Psi alpha_i with alpha_i sparse, but never says how Psi is obtained
    (fixed? learned? from what data?). DCT is a standard compressible basis."""

    dict_atoms: int | None = None
    """Number of dictionary columns k. None -> square (k = d_k), which is what the
    paper's Psi in R^{d_k x d_k} implies. Overcomplete (k > d_k) is a [CHOICE]."""

    bridge: Literal["denoise", "feature_cs", "token_cs"] = "denoise"
    """Which mathematical reading of 'Z_i = Phi Psi alpha_i' is used.
    [INCONSIST] The literal equation does not type-check (see notebook Section 2.6).
    - 'denoise'    : Z_i ~ C_i + noise, solve min 1/2||Z_i - Psi a||^2 + lam||a||_1.
                     Dimensionally consistent, matches C_hat_i = Psi alpha_hat_i,
                     but it is NOT compressed sensing (no undersampling).
    - 'feature_cs' : introduce a NEW feature-space matrix Phi_f in R^{p x d_k}.
                     Genuine CS, but Phi_f appears nowhere in the paper.
    - 'token_cs'   : recover V from V_tilde = Phi_V V along the token axis.
                     The only reading in which Phi in R^{m x n} composes with a
                     dictionary and m << n, but it recovers V, not C."""


@dataclass
class ISTAConfig:
    """Iterative Shrinkage-Thresholding Algorithm.

    [PAPER] names ISTA/OMP/LISTA as the decoders but gives NO step size, NO
    lambda, NO iteration count, and NO sparsity level. Everything below is
    [MISSING] -> our own defaults.
    """

    lam: float = 0.1            # l1 weight
    n_iters: int = 100
    step_size: float | None = None   # None -> 1/L with L = sigma_max(A)^2
    use_fista: bool = False     # [CHOICE] Nesterov acceleration, not in the paper
    track_objective: bool = True
    tol: float = 0.0            # 0 disables early stopping (keeps timing honest)


@dataclass
class LISTAConfig:
    """Learned ISTA (Gregor & LeCun 2010), the paper's reference [37].

    [PAPER] gives the recurrence alpha^{t+1} = eta_theta(S alpha^t + B Z_i) and
    calls t the number of layers. [MISSING] depth, tying, threshold
    parametrisation, optimiser, learning rate, training data, and loss are all
    unreported.
    """

    n_layers: int = 8           # [MISSING]
    tied_weights: bool = False  # [MISSING] LISTA is classically untied
    learn_threshold: bool = True
    init_from_ista: bool = True # [CHOICE] W_e = eta A^T, W_s = I - eta A^T A
    lr: float = 1e-3
    n_epochs: int = 40
    batch_size: int = 128
    n_train: int = 4096
    n_val: int = 512
    supervision: Literal["alpha", "signal"] = "alpha"
    """[MISSING] The paper never says what LISTA is trained against. 'alpha'
    = supervise the sparse code (classical LISTA); 'signal' = supervise Psi*alpha."""


@dataclass
class SparseSignalConfig:
    """Synthetic sparse-signal generator used by the reconstruction experiments."""

    n_signals: int = 512
    dim: int = 128              # ambient dimension of alpha (dictionary atoms k)
    measurements: int = 48      # p, number of linear measurements
    sparsity: int = 8           # s = ||alpha||_0
    noise_std: float = 0.0
    amplitude: tuple[float, float] = (0.5, 1.5)


@dataclass
class BenchmarkConfig:
    """Runtime / memory benchmark settings (paper Table 5 is at n = 4096)."""

    seq_lens: tuple[int, ...] = (512, 1024, 2048, 4096)
    m_values: tuple[int, ...] = (64, 128, 256)
    batch_size: int = 1         # [MISSING] paper never states the batch size
    n_heads: int = 8            # [PAPER] 8 heads
    d_k: int = 64               # [CHOICE] 512/8
    warmup: int = 5
    repeats: int = 20
    dtype: str = "float32"      # [MISSING] paper never states precision


@dataclass
class ProjectConfig:
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    csat: CSATConfig = field(default_factory=CSATConfig)
    ista: ISTAConfig = field(default_factory=ISTAConfig)
    lista: LISTAConfig = field(default_factory=LISTAConfig)
    signal: SparseSignalConfig = field(default_factory=SparseSignalConfig)
    bench: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    seed: int = GLOBAL_SEED
    quick: bool = QUICK_MODE

    def to_dict(self) -> dict:
        return asdict(self)


def quick(cfg: ProjectConfig) -> ProjectConfig:
    """Shrink every sweep for a fast end-to-end notebook run."""
    if not cfg.quick:
        return cfg
    cfg.attention.seq_len = 256
    cfg.attention.batch_size = 2
    cfg.ista.n_iters = 60
    cfg.lista.n_epochs = 15
    cfg.lista.n_train = 2048
    cfg.signal.n_signals = 256
    cfg.bench.seq_lens = (256, 512, 1024, 2048)
    cfg.bench.m_values = (64, 128)
    cfg.bench.repeats = 10
    cfg.bench.warmup = 3
    return cfg


# --------------------------------------------------------------------------- #
# Values the paper REPORTS (for the reproduction-tracking table only).
# These are transcribed from the PDF; we do not attempt to reproduce them here.
# --------------------------------------------------------------------------- #
PAPER_REPORTED = {
    "wikitext103_perplexity": {           # Table 1, all models 151M params
        "Transformer (Full)": 17.5, "Linformer": 19.9, "Performer": 20.5,
        "Longformer": 19.1, "CSAT (ours)": 18.7,
    },
    "lra_pathfinder_x_accuracy": {        # Table 2, sequence length 4096
        "Transformer (Full)": 85.0, "Linformer": 78.3, "Performer": 80.4,
        "Longformer": 81.6, "CSAT (ours)": 84.2,
    },
    "flickr30k_retrieval": {              # Table 3, R@1 / R@5 / R@10
        "BLIP (baseline)": (82.1, 95.5, 98.1), "BLIP + Linformer": (78.9, 94.1, 97.2),
        "BLIP + Performer": (80.3, 94.8, 97.4), "BLIP + CSAT (ours)": (82.4, 95.7, 98.3),
    },
    "mscoco_captioning": {                # Table 4, CIDEr / BLEU-4
        "BLIP (baseline)": (121.4, 38.2), "BLIP + Linformer": (117.5, 36.8),
        "BLIP + Performer": (119.0, 37.1), "BLIP + CSAT (ours)": (122.3, 38.7),
    },
    "efficiency_n4096": {                 # Table 5, GPU memory (GB) / inference (ms)
        "Transformer (Full)": (18.4, 1113), "Linformer": (5.8, 395),
        "Performer": (6.4, 412), "CSAT (ours)": (6.9, 439),
    },
}
