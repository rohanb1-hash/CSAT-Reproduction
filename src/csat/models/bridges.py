"""The decoding bridge: three readings of the paper's equation Z_i = Phi Psi alpha_i.

=============================================================================
THE PROBLEM, STATED PRECISELY
=============================================================================
The paper defines, all in Section 3:

    (a)  Phi_K, Phi_V in R^{m x n},   m << n            [token-axis operators]
    (b)  Z = A~ V~ in R^{n x d_k},    Z_i in R^{d_k}    [rows of the output]
    (c)  C_i in R^{d_k}                                 [true context vector]
    (d)  Psi in R^{d_k x d_k},  alpha_i in R^{d_k},  C_i = Psi alpha_i
    (e)  "Z_i = Phi Psi alpha_i,  with ||alpha_i||_0 << d_k",
         where "Phi = Phi_V is reused as the measurement matrix for decoding"

Substituting (a) and (d) into (e):

        Phi_V  @  (Psi alpha_i)
      [m x n]  @  [d_k]

    The product is defined only if n == d_k, and even then its result lies in
    R^m, whereas (b) says Z_i lies in R^{d_k}. So (e) requires

        n == d_k   AND   m == d_k   =>   m == n,

    which contradicts the paper's own requirement m << n. Equation (e) does not
    type-check under the paper's own definitions.

A SECOND, INDEPENDENT PROBLEM
    Even setting shapes aside, Z_i is not a linear measurement of C_i by any
    fixed operator. Writing A = softmax(QK^T/sqrt(d_k)) and A~ = softmax(QK~^T/sqrt(d_k)):

        C = A V           (true context)
        Z = A~ Phi_V V = (A~ Phi_V) V

    The two use DIFFERENT mixing matrices, and A~ depends on Phi_K, Q and K.
    There is no fixed Phi with Z = Phi C. So "recover C_i from Z_i by compressed
    sensing" has no well-posed measurement operator, whatever the shapes.

A THIRD OBSERVATION
    dim(Z_i) = dim(C_i) = d_k. Compression happens along the token axis, which
    the attention weighting sums over. Nothing is undersampled at the level of a
    single row, so a single row presents no compressed-sensing problem at all.

=============================================================================
WHAT THIS MODULE DOES ABOUT IT
=============================================================================
It implements THREE mathematically consistent readings, each clearly labelled,
and never claims any of them is what the paper wrote. Experiment exp05 compares
them side by side.

  BRIDGE 1 -- 'denoise'  (closest to the paper's text, NOT compressed sensing)
      Treat Z_i as a corrupted observation of C_i in the same space:
          Z_i ~ C_i + e_i,   solve  min_a 1/2||Z_i - Psi a||^2 + lam||a||_1,
          then C_hat_i = Psi a_hat.
      Consistent with (b), (c), (d) and with the paper's C_hat_i = Psi alpha_hat_i.
      Requires no new matrix. But A = Psi is square/invertible, so this is LASSO
      denoising, NOT an underdetermined CS recovery: RIP is irrelevant to it and
      no undersampling occurs. Honest label: "sparse denoising of the compressed
      attention output".

  BRIDGE 2 -- 'feature_cs'  (genuine CS, but with a matrix the paper never defines)
      Introduce a NEW feature-axis measurement matrix Phi_f in R^{p x d_k}, p < d_k,
      and define the measurement explicitly as y_i = Phi_f C_i. Then
          y_i = Phi_f Psi alpha_i
      is a textbook CS problem with m-like undersampling and RIP applying to
      Phi_f Psi. This is the only reading in which the paper's sentence "Z_i =
      Phi Psi alpha_i" becomes literally true -- but only after replacing Phi_V
      in R^{m x n} with a different operator that appears nowhere in the paper,
      and after redefining what is measured. Honest label: "feature-space CS,
      our construction, tests the SOLVER not the paper's pipeline".

  BRIDGE 3 -- 'token_cs'  (the only reading where Phi in R^{m x n} composes)
      Compress along the token axis and recover the VALUE matrix:
          V~[:, j] = Phi_V V[:, j]   for each feature column j,
          assume V[:, j] = Psi_tok beta_j with Psi_tok in R^{n x n} and beta_j sparse.
      Shapes work, m << n is meaningful, and RIP applies to Phi_V Psi_tok.
      But it recovers V, not the context vector C, and recovering V then running
      full attention costs O(n^2 d) again -- defeating the method's purpose.
      Honest label: "token-axis CS, consistent but off-purpose".

None of the three is 'the paper's method'. What the paper explicitly specifies
and what we therefore reproduce exactly is the compressed attention of
models/compressed_attention.py; the decoder stage cannot be reproduced as
written because as written it is inconsistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .dictionary import make_dictionary
from .measurement import make_measurement_matrix

# NOTE: models.ista is imported lazily inside decode_context() rather than here.
# This module only needs a solver at decode time, and the notebook builds the
# project in the paper's own order -- sparse representation (Phase D) before the
# solvers (Phase E) -- so a module-level import would fail on first execution.

BridgeName = Literal["denoise", "feature_cs", "token_cs"]


@dataclass
class BridgeSpec:
    """A fully-specified, dimension-checked recovery problem y = A alpha."""

    name: BridgeName
    A: torch.Tensor           # [p, k] operator fed to ISTA/LISTA/OMP
    psi: torch.Tensor         # [d, k] synthesis dictionary
    phi: torch.Tensor | None  # measurement matrix, or None for 'denoise'
    is_underdetermined: bool  # True only when p < k (i.e. genuine CS)
    description: str

    def shapes(self) -> dict[str, tuple]:
        return {
            "A (p x k)": tuple(self.A.shape),
            "psi (d x k)": tuple(self.psi.shape),
            "phi": tuple(self.phi.shape) if self.phi is not None else None,
        }


def check_paper_equation(m: int, n: int, d_k: int) -> dict[str, object]:
    """Machine-checked verification that Z_i = Phi_V Psi alpha_i does not type-check.

    Returns a dict of the shapes involved and whether each required equality
    holds, so the notebook can PRINT the contradiction rather than assert it in
    prose. Nothing here depends on our interpretation: it is arithmetic on the
    paper's own declared shapes.
    """
    phi_shape = (m, n)
    psi_alpha_shape = (d_k,)
    product_defined = (n == d_k)
    product_shape = (m,) if product_defined else None
    z_shape = (d_k,)
    output_matches = product_defined and (m == d_k)
    return {
        "Phi_V shape (paper)": phi_shape,
        "Psi alpha_i shape (paper)": psi_alpha_shape,
        "Z_i shape (paper)": z_shape,
        "inner dims agree (n == d_k)?": product_defined,
        "Phi_V (Psi alpha_i) shape": product_shape,
        "output dim matches Z_i (m == d_k)?": output_matches,
        "equation well-formed?": bool(product_defined and output_matches),
        "implied constraint if forced": "m == n == d_k, contradicting m << n",
    }


def build_bridge(name: BridgeName, d_k: int, n: int = 0, m: int = 0,
                 p_features: int | None = None,
                 dictionary: str = "dct", k_atoms: int | None = None,
                 ensemble: str = "gaussian",
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32) -> BridgeSpec:
    """Construct one of the three recovery problems, with all shapes checked."""
    k_atoms = k_atoms or d_k

    if name == "denoise":
        psi = make_dictionary(dictionary, d_k, k_atoms, device, dtype)   # [d_k, k]
        return BridgeSpec(
            name="denoise", A=psi, psi=psi, phi=None,
            is_underdetermined=(psi.shape[0] < psi.shape[1]),
            description=(
                "BRIDGE 1 (denoise): solve min 1/2||Z_i - Psi a||^2 + lam||a||_1 with "
                "A = Psi. Dimensionally consistent with the paper's Z_i, C_i, Psi and "
                "with C_hat_i = Psi alpha_hat_i, but it is sparse DENOISING, not "
                "compressed sensing: no undersampling occurs and RIP is not invoked."),
        )

    if name == "feature_cs":
        p = p_features or max(1, d_k // 2)
        psi = make_dictionary(dictionary, d_k, k_atoms, device, dtype)   # [d_k, k]
        phi_f = make_measurement_matrix(p, d_k, ensemble, True, device, dtype)  # [p, d_k]
        A = phi_f @ psi                                                  # [p, k]
        return BridgeSpec(
            name="feature_cs", A=A, psi=psi, phi=phi_f,
            is_underdetermined=(A.shape[0] < A.shape[1]),
            description=(
                f"BRIDGE 2 (feature_cs): y_i = Phi_f C_i with a NEW Phi_f in R^{{{p} x {d_k}}} "
                "applied along the FEATURE axis. A = Phi_f Psi is genuinely "
                "underdetermined, so this is real compressed sensing -- but Phi_f does "
                "not appear in the paper and y_i is not the paper's Z_i."),
        )

    if name == "token_cs":
        assert n > 0 and m > 0, "token_cs needs the token count n and measurements m"
        psi_tok = make_dictionary(dictionary, n, n, device, dtype)       # [n, n]
        phi_v = make_measurement_matrix(m, n, ensemble, True, device, dtype)  # [m, n]
        A = phi_v @ psi_tok                                             # [m, n]
        return BridgeSpec(
            name="token_cs", A=A, psi=psi_tok, phi=phi_v,
            is_underdetermined=(m < n),
            description=(
                f"BRIDGE 3 (token_cs): recover a value COLUMN in R^{n} from its m={m} "
                "token-axis measurements. This is the only reading in which Phi in "
                "R^{m x n} composes with a dictionary and m << n is meaningful -- but it "
                "recovers V, not the context vector C."),
        )

    raise ValueError(f"unknown bridge '{name}'")


@torch.no_grad()
def decode_context(z: torch.Tensor, spec: BridgeSpec, lam: float = 0.05,
                   n_iters: int = 100) -> dict[str, torch.Tensor]:
    """Apply the paper's row-wise decoding C_hat_i = Psi alpha_hat_i to Z.

    z: [..., d] for 'denoise'/'feature_cs' (feature-axis rows) or [..., n] for
    'token_cs' (token-axis columns). Leading dimensions are flattened and restored.
    """
    from .ista import ista  # lazy import; see the note at the top of this file

    lead, p = z.shape[:-1], z.shape[-1]
    assert p == spec.A.shape[0], (
        f"bridge '{spec.name}' expects measurements of dimension {spec.A.shape[0]}, "
        f"but received {p}")
    out = ista(spec.A, z.reshape(-1, p), lam=lam, n_iters=n_iters)
    alpha = out["alpha"]
    c_hat = alpha @ spec.psi.T
    return {
        "alpha": alpha.reshape(*lead, alpha.shape[-1]),
        "c_hat": c_hat.reshape(*lead, spec.psi.shape[0]),
    }
