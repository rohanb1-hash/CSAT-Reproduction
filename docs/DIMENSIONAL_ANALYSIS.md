# The decoding equation does not type-check

This is the central finding of the reproduction. It is arithmetic on the paper's own
declared shapes, not a matter of interpretation, and it is verified in code by
[`check_paper_equation()`](../src/csat/models/bridges.py) and asserted by
[`tests/test_measurement_dictionary.py`](../tests/test_measurement_dictionary.py).

Reproduce it yourself:

```bash
csat check
```

---

## 1. What the paper writes

From §3 of arXiv:2507.02957v1:

> "Then the observed compressed output `Z_i` can be written as:
> `Z_i = ΦΨα_i`, with `‖α_i‖₀ ≪ d_k`,
> where `Φ = Φ_V` is reused as the measurement matrix for decoding."

## 2. The shapes, all from the same section

| Object | Shape | Where the paper declares it |
|---|---|---|
| `Φ_V` | `m × n` | §3, compression: "Let `Φ_K, Φ_V ∈ ℝ^{m×n}` … where `m ≪ n`" |
| `Ψ` | `d_k × d_k` | §3, dictionary |
| `α_i` | `d_k` | §3, dictionary |
| `Ψα_i` | `d_k` | matrix–vector product of the two above |
| `Z_i` | `d_k` | row of `Z = ÃṼ ∈ ℝ^{n×d_k}` |

## 3. Substituting

```
    Φ_V      ·   (Ψ α_i)
  [m × n]        [d_k]
```

* The product is **defined only if** `n = d_k`.
* Even then the result lies in `ℝ^m`, whereas `Z_i ∈ ℝ^{d_k}`, so we also need `m = d_k`.
* Together: **`m = n = d_k`** — which contradicts the paper's own `m ≪ n`.

The equation closes only in the degenerate case where there is **no compression at all**.

What `csat check` prints:

```
  n=4096, d_k=64, m=64
     Phi_V shape (paper)                   : (64, 4096)
     Psi alpha_i shape (paper)             : (64,)
     Z_i shape (paper)                     : (64,)
     inner dims agree (n == d_k)?          : False
     Phi_V (Psi alpha_i) shape             : None
     output dim matches Z_i (m == d_k)?    : False
     equation well-formed?                 : False
     implied constraint if forced          : m == n == d_k, contradicting m << n
```

## 4. A second, independent problem: there is no measurement operator

Set the shapes aside entirely. Write

```
A  = softmax(Q Kᵀ  / √d_k)        Ã = softmax(Q K̃ᵀ / √d_k)
C  = A V                           Z = Ã Φ_V V
```

The two expressions use **different mixing matrices**, and `Ã` depends on `Φ_K`, `Q` and
`K`. **There is no fixed `Φ` for which `Z = Φ C`.**

Compressed sensing requires a known linear measurement operator relating the observation
to the signal. Here none exists. A shape mismatch could be patched by redefining a
symbol; this cannot.

## 5. A third: nothing is undersampled per row

`dim(Z_i) = dim(C_i) = d_k`.

The compression is along the *token* axis, and the attention weighting sums over exactly
that axis. A single row therefore poses no underdetermined inverse problem, so RIP and
ℓ₁ recovery have nothing to act on at the row level.

## 6. A related structural fact

Because `Ṽ = Φ_V V`,

```
Z = Ã Ṽ = Ã Φ_V V = M_eff V,     M_eff := Ã Φ_V ∈ ℝ^{n×n}
```

So CSAT applies an effective `n × n` mixing matrix in place of `A`. But `Φ_V` has
negative entries, so `M_eff` is in general **neither non-negative nor row-stochastic**,
while `A` is both. Measured in `exp02`: roughly **50% of `M_eff`'s entries are negative**
and its row sums scatter around values that are not 1.

This removes the intuition that would motivate treating `Z_i` as a noisy observation of
`C_i`: `Z_i` is not a weighted average of value vectors at all.

---

## 7. What this repository does about it

Rather than silently repairing the equation, the code implements the compressed
attention **exactly as specified** and implements reconstruction **separately**, in three
labelled, dimension-checked readings. None is claimed to be the paper's method.

| Bridge | Construction | Underdetermined? | Honest label |
|---|---|---|---|
| **1 `denoise`** | `Z_i ≈ C_i + e`; solve `min ½‖Z_i − Ψα‖² + λ‖α‖₁` | **No** (`A = Ψ` is square) | Consistent with the paper's symbols and with `Ĉ_i = Ψα̂_i`, but **not compressed sensing** — nothing is undersampled and RIP is irrelevant |
| **2 `feature_cs`** | new `Φ_f ∈ ℝ^{p×d_k}`, `y_i = Φ_f C_i`, `A = Φ_f Ψ` | Yes | Genuine CS, but `Φ_f` **appears nowhere in the paper** and `y_i` is not the paper's `Z_i` |
| **3 `token_cs`** | per feature column, `Ṽ_{:,j} = Φ_V V_{:,j}`, `V_{:,j} = Ψ_tok β_j` | Yes | The **only** reading in which `Φ ∈ ℝ^{m×n}` composes with a dictionary and `m ≪ n` is meaningful — but it recovers `V`, not `C`, and then running full attention costs `O(n²d)` again |

Bridge 1 is the one used inside `CSATBlock`, because it is the only reading that consumes
`Z` directly. `CSATBlock` raises an explicit error if a bridge is requested whose
measurement dimension cannot consume `Z`'s rows.

```python
from csat.models import build_bridge

spec = build_bridge("feature_cs", d_k=64, p_features=32)
spec.A.shape              # (32, 64) -> genuinely underdetermined
spec.is_underdetermined   # True

spec = build_bridge("denoise", d_k=64)
spec.A.shape              # (64, 64) -> square
spec.is_underdetermined   # False  <- NOT compressed sensing
```

## 8. What would resolve this

Only the authors can say which reading was intended. The open possibilities are:

1. `Ψ` was meant to be `n × n` (a token-axis dictionary), making bridge 3 the intended
   method — but then the recovered object is `V`, not the context vector `C`, and §3's
   `Ψ ∈ ℝ^{d_k×d_k}` and `Ĉ_i = Ψα̂_i` would both need restating.
2. A second, feature-axis measurement matrix exists but was not written down, making
   bridge 2 the intended method — but then `Φ = Φ_V` is not "reused", and the compressed
   attention output `Z` is not what gets decoded.
3. The decoder is a denoiser rather than a CS recovery, making bridge 1 the intended
   method — but then the RIP framing, the basis-pursuit formulation and the paper's
   central theoretical contribution do not apply to it.

Each option costs the paper something it claims. Asking the authors is item 3 of
[the Phase 5 plan](PHASE5_PLAN.md).
