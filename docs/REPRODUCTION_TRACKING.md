# Reproduction tracking

A component counts as reproduced only when the paper specifies it well enough to
implement **and** this implementation matches that specification. Code that runs is
not evidence of reproduction.

This file is generated from `csat.utils.reporting.TRACKING_ROWS`, which is the single
source of truth. Regenerate with `python scripts/gen_tracking_doc.py`, or print the
table with `csat track`.

## Summary

| Status | Count | Meaning |
|---|---|---|
| ✅ Implemented | 3 | specified by the paper and implemented as specified |
| 🟡 Partially implemented | 8 | implemented, but some sub-detail is our documented choice |
| ❓ Not specified by paper | 1 | the paper needs it but never states it |
| ⬜ Not yet implemented | 4 | out of scope for Phase 4 |
| ❌ Cannot reproduce exactly | 4 | specified but unreproducible from the information given |

## Components

| | Component | Status | Exact/approx |
|---|---|---|---|
| ✅ | Standard attention baseline | Implemented | Exact |
| ✅ | Q/K/V projections | Implemented | Exact |
| 🟡 | Measurement matrices Phi_K, Phi_V | Partially implemented | Approximate |
| ❓ | Value of m | Not specified by paper | N/A |
| ✅ | Compressed attention A~ = softmax(QK~^T/sqrt(d_k)), Z = A~V~ | Implemented | Exact |
| ❌ | RIP assumption on Phi | Cannot reproduce exactly | Approximate |
| 🟡 | Sparse representation C_i = Psi alpha_i | Partially implemented | Approximate |
| ❌ | Decoding equation Z_i = Phi Psi alpha_i | Cannot reproduce exactly | N/A |
| 🟡 | Basis pursuit / l1 recovery | Partially implemented | Approximate |
| 🟡 | ISTA decoder | Partially implemented | Approximate |
| 🟡 | OMP decoder | Partially implemented | Approximate |
| 🟡 | LISTA decoder | Partially implemented | Approximate |
| 🟡 | C_hat_i = Psi alpha_hat_i (row-wise decoding) | Partially implemented | Exact given a bridge |
| ❌ | Causal / autoregressive masking | Cannot reproduce exactly | N/A |
| 🟡 | Complexity claim O(n^2 d) -> O(nmd + decoding) | Partially implemented | Approximate |
| ❌ | Table 5 efficiency numbers (18.4GB/1113ms vs 6.9GB/439ms) | Cannot reproduce exactly | Cannot match absolutes |
| ⬜ | WikiText-103 language modelling (Table 1) | Not yet implemented | N/A |
| ⬜ | LRA Pathfinder-X (Table 2) | Not yet implemented | N/A |
| ⬜ | BLIP retrieval / captioning (Tables 3-4) | Not yet implemented | N/A |
| ⬜ | Comparison with Linformer / Performer / Longformer | Not yet implemented | N/A |

---

## Detail

### ✅ Standard attention baseline

- **Status:** Implemented
- **Paper says:** Attn(Q,K,V) = softmax(QK^T/sqrt(d_k))V, Sec. 3
- **Implemented:** Yes
- **Exact or approximate:** Exact
- **Missing from the paper:** None
- **Our assumption:** [B,H,N,D] multi-head layout (paper writes the single-head case)

### ✅ Q/K/V projections

- **Status:** Implemented
- **Paper says:** Q=XW^Q, K=XW^K, V=XW^V with W in R^{d x d_k}, Sec. 3
- **Implemented:** Yes
- **Exact or approximate:** Exact
- **Missing from the paper:** How heads compose; bias terms
- **Our assumption:** d_model = H*d_k (Vaswani convention); bias enabled

### 🟡 Measurement matrices Phi_K, Phi_V

- **Status:** Partially implemented
- **Paper says:** Phi in R^{m x n} from sub-Gaussian ensembles, RIP-satisfying, Sec. 3
- **Implemented:** Yes (gaussian / rademacher / orthogonal / hadamard)
- **Exact or approximate:** Approximate
- **Missing from the paper:** Normalisation constant; sharing across heads/layers/modalities; whether re-drawn per batch
- **Our assumption:** 1/sqrt(m) scaling so E[Phi^T Phi]=I; one Phi shared across heads; fixed at init

### ❓ Value of m

- **Status:** Not specified by paper
- **Paper says:** 'm << n'; Table 5 benchmarks n=4096
- **Implemented:** Swept as a free parameter
- **Exact or approximate:** N/A
- **Missing from the paper:** The paper never states m for ANY experiment
- **Our assumption:** We sweep m in {16..256} and report m explicitly everywhere

### ✅ Compressed attention A~ = softmax(QK~^T/sqrt(d_k)), Z = A~V~

- **Status:** Implemented
- **Paper says:** Sec. 3, explicit equations
- **Implemented:** Yes
- **Exact or approximate:** Exact
- **Missing from the paper:** Whether the sqrt(d_k) scale is re-calibrated after projection
- **Our assumption:** Paper's literal sqrt(d_k) is the default; two re-scalings offered as labelled diagnostics

### ❌ RIP assumption on Phi

- **Status:** Cannot reproduce exactly
- **Paper says:** 'measurement matrices satisfying the RIP'
- **Implemented:** Diagnostics only (coherence, Monte-Carlo lower bound)
- **Exact or approximate:** Approximate
- **Missing from the paper:** RIP order s and constant delta_s are never stated; what is sparse along the TOKEN axis is never identified
- **Our assumption:** We report coherence and a Monte-Carlo LOWER bound; exact RIP verification is NP-hard

### 🟡 Sparse representation C_i = Psi alpha_i

- **Status:** Partially implemented
- **Paper says:** Psi in R^{d_k x d_k}, alpha_i sparse, Sec. 3
- **Implemented:** Yes (identity / DCT / random orthogonal / overcomplete / learned)
- **Exact or approximate:** Approximate
- **Missing from the paper:** How Psi is obtained; sparsity level s; any evidence that context vectors are sparse
- **Our assumption:** DCT default (the paper's own JPEG analogy); we MEASURE compressibility rather than assume it

### ❌ Decoding equation Z_i = Phi Psi alpha_i

- **Status:** Cannot reproduce exactly
- **Paper says:** Sec. 3, with Phi = Phi_V reused
- **Implemented:** Cannot be implemented as written
- **Exact or approximate:** N/A
- **Missing from the paper:** Phi_V is [m x n] while Psi alpha_i is [d_k]; the product needs n = d_k and still lands in R^m, not R^{d_k}. Closes only if m = n = d_k, contradicting m << n
- **Our assumption:** Three labelled, dimensionally consistent alternatives implemented instead (models/bridges.py)

### 🟡 Basis pursuit / l1 recovery

- **Status:** Partially implemented
- **Paper says:** alpha_hat = argmin ||alpha||_1 s.t. Z_i = Phi Psi alpha
- **Implemented:** Unconstrained LASSO relaxation via ISTA/FISTA, plus OMP
- **Exact or approximate:** Approximate
- **Missing from the paper:** lambda, step size, iteration count, stopping rule
- **Our assumption:** eta = 1/sigma_max(A)^2; lambda and iterations swept and reported

### 🟡 ISTA decoder

- **Status:** Partially implemented
- **Paper says:** Named as the analytic solver, Sec. 3 and Sec. 5
- **Implemented:** Yes, batched over tokens, plus FISTA
- **Exact or approximate:** Approximate
- **Missing from the paper:** All hyper-parameters; how the analytic decoder is trained end to end (it is not differentiable)
- **Our assumption:** Fixed iteration budget, no early stop, so timings stay comparable

### 🟡 OMP decoder

- **Status:** Partially implemented
- **Paper says:** Named alongside ISTA, Sec. 2/5
- **Implemented:** Yes, batched
- **Exact or approximate:** Approximate
- **Missing from the paper:** Sparsity level s is required by OMP and never stated
- **Our assumption:** s supplied as an explicit experiment parameter

### 🟡 LISTA decoder

- **Status:** Partially implemented
- **Paper says:** alpha^{t+1} = eta_theta(S alpha^t + B Z_i), Sec. 3
- **Implemented:** Yes, with W_s/W_e initialised from ISTA
- **Exact or approximate:** Approximate
- **Missing from the paper:** Depth t, weight tying, threshold form, training data, loss, optimiser
- **Our assumption:** Untied by default, learned per-coordinate threshold, Adam, supervised on alpha for synthetic data

### 🟡 C_hat_i = Psi alpha_hat_i (row-wise decoding)

- **Status:** Partially implemented
- **Paper says:** Sec. 3
- **Implemented:** Yes
- **Exact or approximate:** Exact given a bridge
- **Missing from the paper:** Depends on the ill-posed measurement equation above
- **Our assumption:** Applied row-wise to Z under BRIDGE 1

### ❌ Causal / autoregressive masking

- **Status:** Cannot reproduce exactly
- **Paper says:** Not discussed; WikiText-103 LM results are reported (Table 1)
- **Implemented:** Rejected with an explicit error
- **Exact or approximate:** N/A
- **Missing from the paper:** How CSAT performs autoregressive LM when each compressed key slot mixes all n tokens, including future ones
- **Our assumption:** We refuse causal masks rather than apply a meaningless one

### 🟡 Complexity claim O(n^2 d) -> O(nmd + decoding)

- **Status:** Partially implemented
- **Paper says:** Sec. 1 and Sec. 3
- **Implemented:** Analytic FLOP counter + measured runtime, reported separately
- **Exact or approximate:** Approximate
- **Missing from the paper:** The 'decoding' term is never expanded; it is O(n * iters * p * k) for ISTA
- **Our assumption:** We count both stages and never report attention-only speedups as pipeline speedups

### ❌ Table 5 efficiency numbers (18.4GB/1113ms vs 6.9GB/439ms)

- **Status:** Cannot reproduce exactly
- **Paper says:** n = 4096
- **Implemented:** Scaling measured on this notebook's device
- **Exact or approximate:** Cannot match absolutes
- **Missing from the paper:** GPU model, precision, batch size, layer count, m, decoder configuration, whether decoding is included
- **Our assumption:** We report our own hardware and every parameter, and do not claim to match the paper's absolute numbers

### ⬜ WikiText-103 language modelling (Table 1)

- **Status:** Not yet implemented
- **Paper says:** 12 layers, 512 hidden, 8 heads, 151M params, 300k steps, perplexity 18.7
- **Implemented:** No
- **Exact or approximate:** N/A
- **Missing from the paper:** Tokeniser, context length, optimiser, LR schedule, m, decoder config; 300k steps is far beyond a notebook budget
- **Our assumption:** Out of scope for Phase 4; noted as a Phase 5 item requiring a multi-GPU run

### ⬜ LRA Pathfinder-X (Table 2)

- **Status:** Not yet implemented
- **Paper says:** Sequence length 4096, accuracy 84.2%
- **Implemented:** No
- **Exact or approximate:** N/A
- **Missing from the paper:** Full training recipe; Pathfinder-X is notoriously sensitive to it
- **Our assumption:** Out of scope for Phase 4

### ⬜ BLIP retrieval / captioning (Tables 3-4)

- **Status:** Not yet implemented
- **Paper says:** Flickr30k, MS-COCO, CSAT blocks replacing BLIP attention
- **Implemented:** No
- **Exact or approximate:** N/A
- **Missing from the paper:** Which layers were replaced, fine-tuning schedule, m, decoder config, checkpoint
- **Our assumption:** Out of scope for Phase 4; requires pretrained BLIP weights and multimodal datasets

### ⬜ Comparison with Linformer / Performer / Longformer

- **Status:** Not yet implemented
- **Paper says:** Baselines throughout Tables 1-5
- **Implemented:** No
- **Exact or approximate:** N/A
- **Missing from the paper:** Baseline configurations are not given
- **Our assumption:** Deferred to Phase 5

