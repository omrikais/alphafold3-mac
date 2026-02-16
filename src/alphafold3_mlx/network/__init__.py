"""Network components for AlphaFold 3 MLX implementation.

This subpackage contains the neural network building blocks including:
- Attention mechanisms (GatedSelfAttention, GridSelfAttention)
- Transition blocks (TransitionBlock)
- Triangle operations (TriangleMultiplication)
- Outer product mean (OuterProductMean)
- PairFormer iteration (PairFormerIteration)
- Evoformer stack (Evoformer)
- Diffusion components (DiffusionTransformer, DiffusionHead)
- Confidence prediction (ConfidenceHead)
"""

# Phase 2: Foundational components
from alphafold3_mlx.network.attention import (
    GatedSelfAttention,
    GridSelfAttention,
    ChunkedGatedSelfAttention,
    boolean_to_additive_mask,
    detect_fully_masked_rows,
    chunked_attention,
    AttentionConfig,
)
from alphafold3_mlx.network.transition import TransitionBlock
from alphafold3_mlx.network.triangle_ops import (
    TriangleMultiplication,
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from alphafold3_mlx.network.outer_product import OuterProductMean, OuterProductMeanMSA

# Phase 3: PairFormer and Evoformer
from alphafold3_mlx.network.pairformer import PairFormerIteration, PairFormerStack
from alphafold3_mlx.network.evoformer import (
    Evoformer,
    EvoformerIteration,
    RelativePositionEmbedding,
)

# Phase 3: MSA stack
from alphafold3_mlx.network.msa_attention import (
    MSARowAttention,
    MSATransition,
)

# Phase 3: Diffusion components
from alphafold3_mlx.network.noise_schedule import (
    karras_schedule,
    NoiseLevelEmbedding,
    NoiseScheduleSampler,
)
from alphafold3_mlx.network.noise_level_embeddings import noise_embeddings
from alphafold3_mlx.network.diffusion_transformer import (
    AdaptiveLayerNorm,
    DiffusionTransformerBlock,
    DiffusionTransformer,
)
from alphafold3_mlx.network.diffusion_head import DiffusionHead
from alphafold3_mlx.network.atom_cross_attention import (
    AtomCrossAttEncoder,
    AtomCrossAttDecoder,
)
from alphafold3_mlx.network.featurization import create_relative_encoding
from alphafold3_mlx.network.template_modules import (
    DistogramFeaturesConfig,
    dgram_from_positions,
)

# Phase 3: Confidence head
from alphafold3_mlx.network.confidence_head import ConfidenceHead

__all__ = [
    # Attention
    "GatedSelfAttention",
    "GridSelfAttention",
    "ChunkedGatedSelfAttention",
    "boolean_to_additive_mask",
    "detect_fully_masked_rows",
    "chunked_attention",
    "AttentionConfig",
    # Transition
    "TransitionBlock",
    # Triangle operations
    "TriangleMultiplication",
    "TriangleMultiplicationOutgoing",
    "TriangleMultiplicationIncoming",
    # Outer product
    "OuterProductMean",
    "OuterProductMeanMSA",
    # PairFormer
    "PairFormerIteration",
    "PairFormerStack",
    # Evoformer
    "Evoformer",
    "EvoformerIteration",
    "RelativePositionEmbedding",
    # MSA stack
    "MSARowAttention",
    "MSATransition",
    # Noise schedule
    "karras_schedule",
    "NoiseLevelEmbedding",
    "NoiseScheduleSampler",
    "noise_embeddings",
    # Diffusion transformer
    "AdaptiveLayerNorm",
    "DiffusionTransformerBlock",
    "DiffusionTransformer",
    # Diffusion head
    "DiffusionHead",
    # Atom cross-attention
    "AtomCrossAttEncoder",
    "AtomCrossAttDecoder",
    # Featurization
    "create_relative_encoding",
    # Template modules
    "DistogramFeaturesConfig",
    "dgram_from_positions",
    # Confidence head
    "ConfidenceHead",
]
