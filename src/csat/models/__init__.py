"""Model components: attention, measurement operators, dictionaries, solvers."""

from csat.models.bridges import (
    BridgeSpec,
    build_bridge,
    check_paper_equation,
    decode_context,
)
from csat.models.compressed_attention import (
    CompressedAttention,
    LinearCompressedAttention,
)
from csat.models.csat_block import CSATBlock
from csat.models.dictionary import (
    Dictionary,
    dct_matrix,
    fit_dictionary,
    make_dictionary,
)
from csat.models.ista import (
    ISTADecoder,
    debias,
    estimate_step_size,
    fista,
    ista,
    lasso_objective,
    soft_threshold,
)
from csat.models.lista import LISTA, LISTADecoder, train_lista
from csat.models.measurement import (
    MeasurementMatrix,
    empirical_rip_constant,
    make_measurement_matrix,
    mutual_coherence,
)
from csat.models.omp import omp
from csat.models.standard_attention import (
    MultiHeadSelfAttention,
    StandardAttention,
    causal_mask,
    scaled_dot_product_attention,
)

__all__ = [
    # standard attention
    "StandardAttention", "MultiHeadSelfAttention", "scaled_dot_product_attention",
    "causal_mask",
    # measurement operators
    "MeasurementMatrix", "make_measurement_matrix", "mutual_coherence",
    "empirical_rip_constant",
    # compressed attention
    "CompressedAttention", "LinearCompressedAttention",
    # dictionaries
    "Dictionary", "make_dictionary", "dct_matrix", "fit_dictionary",
    # solvers
    "ista", "fista", "debias", "soft_threshold", "lasso_objective",
    "estimate_step_size", "ISTADecoder", "omp",
    "LISTA", "LISTADecoder", "train_lista",
    # the decoding bridges
    "build_bridge", "check_paper_equation", "decode_context", "BridgeSpec",
    # end-to-end block
    "CSATBlock",
]
