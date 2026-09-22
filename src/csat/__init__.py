"""csat — a reproduction of CS-VLM / CSAT (arXiv:2507.02957v1).

Reproduction of *"CS-VLM: Compressed Sensing Attention for Efficient
Vision-Language Representation Learning"* (Kiruluta, Raju & Burity, 2025), which
proposes the Compressed Sensing Attention Transformer (CSAT).

The package implements what the paper specifies, measures what it asserts, and
documents what it leaves undefined. See ``docs/DIMENSIONAL_ANALYSIS.md`` for the
central finding: the paper's decoding equation ``Z_i = Phi Psi alpha_i`` does not
type-check under the paper's own declared shapes.

Quick start
-----------
>>> import torch
>>> from csat.models.compressed_attention import CompressedAttention
>>> from csat.utils.tensor_utils import make_qkv
>>> q, k, v = make_qkv(1, 4, 256, 64)
>>> attn = CompressedAttention(seq_len=256, m=32, d_k=64, n_heads=4)
>>> z, a_tilde = attn(q, k, v, return_weights=True)
>>> z.shape, a_tilde.shape
(torch.Size([1, 4, 256, 64]), torch.Size([1, 4, 256, 32]))

Provenance tags used throughout the source
------------------------------------------
``[PAPER]``     stated explicitly in the paper
``[CHOICE]``    the paper leaves this free; we picked a standard value
``[MISSING]``   the paper needs this quantity but never reports it
``[INCONSIST]`` the paper's own statement is mathematically inconsistent here
"""

__version__ = "0.1.0"
__paper__ = "arXiv:2507.02957v1"
__all__ = ["config", "models", "utils", "experiments", "__version__", "__paper__"]
