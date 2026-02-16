"""Configuration dataclasses for AlphaFold 3 MLX.

This module defines all configuration dataclasses including:
- AttentionConfig: Low-level attention computation config (Phase 0)
- PairFormerConfig: PairFormer block configuration
- TemplateConfig: Template embedding configuration
- MSAStackConfig: MSA stack configuration
- SampleConfig: Diffusion sampling configuration
- EvoformerConfig: Evoformer configuration
- DiffusionConfig: Diffusion head configuration
- ConfidenceConfig: Confidence head configuration
- GlobalConfig: Global model settings
- ModelConfig: Complete model configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for scaled dot-product attention.

    Attributes:
        batch_size: Batch dimension (default: 1)
        num_heads: Number of attention heads (default: 4)
        seq_q: Query sequence length (default: 256)
        seq_k: Key/Value sequence length (default: 256)
        head_dim: Dimension per head (default: 64)
        dtype: Data type string - "float32", "float16", or "bfloat16"
        mask_value: Large negative value for masking (default: 1e9)
        seed: Random seed for reproducibility (default: 42)
    """

    batch_size: int = 1
    num_heads: int = 4
    seq_q: int = 256
    seq_k: int = 256
    head_dim: int = 64
    dtype: str = "float32"
    mask_value: float = 1e9
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {self.num_heads}")
        if self.seq_q < 1:
            raise ValueError(f"seq_q must be >= 1, got {self.seq_q}")
        if self.seq_k < 1:
            raise ValueError(f"seq_k must be >= 1, got {self.seq_k}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {self.head_dim}")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"dtype must be float32/float16/bfloat16, got {self.dtype}")
        if self.mask_value <= 1e6:
            raise ValueError(f"mask_value must be > 1e6, got {self.mask_value}")

    @property
    def q_shape(self) -> tuple[int, int, int, int]:
        """Return shape of query tensor [batch, heads, seq_q, head_dim]."""
        return (self.batch_size, self.num_heads, self.seq_q, self.head_dim)

    @property
    def k_shape(self) -> tuple[int, int, int, int]:
        """Return shape of key tensor [batch, heads, seq_k, head_dim]."""
        return (self.batch_size, self.num_heads, self.seq_k, self.head_dim)

    @property
    def v_shape(self) -> tuple[int, int, int, int]:
        """Return shape of value tensor [batch, heads, seq_k, head_dim]."""
        return self.k_shape

    @property
    def mask_shape(self) -> tuple[int, int]:
        """Return shape of boolean mask [batch, seq_k]."""
        return (self.batch_size, self.seq_k)

    @property
    def bias_shape(self) -> tuple[int, int, int, int]:
        """Return shape of additive bias [batch, heads, seq_q, seq_k]."""
        return (self.batch_size, self.num_heads, self.seq_q, self.seq_k)

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        """Return shape of output tensor [batch, heads, seq_q, head_dim]."""
        return self.q_shape

    def to_dict(self) -> dict:
        """Serialize configuration to dictionary."""
        return {
            "batch_size": self.batch_size,
            "num_heads": self.num_heads,
            "seq_q": self.seq_q,
            "seq_k": self.seq_k,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
            "mask_value": self.mask_value,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttentionConfig":
        """Create configuration from dictionary."""
        return cls(**data)


@dataclass
class PairFormerConfig:
    """PairFormer block configuration.

    Attributes:
        num_attention_heads: Number of attention heads for pair attention.
        single_attention_heads: Number of attention heads for single attention.
        attention_key_dim: Key dimension per attention head.
        dropout_rate: Dropout rate (0 for inference).
    """

    num_attention_heads: int = 4
    single_attention_heads: int = 16
    attention_key_dim: int = 32
    dropout_rate: float = 0.0


@dataclass
class TemplateConfig:
    """Template embedding configuration.

    Attributes:
        template_pair_channel: Channel dimension for template pair features.
        num_template_blocks: Number of PairFormer layers in template stack.
        enabled: Whether template embedding is enabled.
        dgram_min_bin: Left edge of first distogram bin.
        dgram_max_bin: Left edge of final distogram bin.
        dgram_num_bins: Number of distogram bins.
    """

    template_pair_channel: int = 64
    num_template_blocks: int = 2
    enabled: bool = True
    dgram_min_bin: float = 3.25
    dgram_max_bin: float = 50.75
    dgram_num_bins: int = 39


@dataclass
class MSAStackConfig:
    """MSA stack configuration.

    Attributes:
        num_layers: Number of MSA attention layers.
        num_attention_heads: Number of attention heads in MSA attention blocks.
        msa_channel: MSA representation channel dimension.
        pair_channel: Pair representation channel dimension.
    """

    num_layers: int = 4
    num_attention_heads: int = 8
    msa_channel: int = 64
    pair_channel: int = 128


@dataclass
class SampleConfig:
    """Diffusion sampling configuration.

    Attributes:
        steps: Number of diffusion denoising steps.
        gamma_0: Initial gamma for stochastic sampling.
        gamma_min: Minimum gamma value.
        noise_scale: Noise scale factor.
        step_scale: Step scale factor.
        num_samples: Number of structure samples to generate.
        return_trajectories: Whether to return intermediate trajectories.
    """

    steps: int = 200
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    num_samples: int = 5
    return_trajectories: bool = False


@dataclass
class EvoformerConfig:
    """Evoformer configuration.

    Attributes:
        seq_channel: Single representation channel dimension.
        pair_channel: Pair representation channel dimension.
        num_pairformer_layers: Number of PairFormer layers (default: 48).
        num_msa_layers: Number of MSA layers (default: 4).
        msa_channel: MSA representation channel dimension.
        num_msa: Maximum number of MSA sequences.
        max_relative_idx: Maximum relative position index.
        max_relative_chain: Maximum relative chain index.
        use_msa_stack: Whether to use MSA stack when MSA features provided.
        pairformer: PairFormer block configuration.
        template: Template embedding configuration.
        msa_stack: MSA stack configuration.
    """

    seq_channel: int = 384
    pair_channel: int = 128
    num_pairformer_layers: int = 48
    num_msa_layers: int = 4
    msa_channel: int = 64
    num_msa: int = 1024
    max_relative_idx: int = 32
    max_relative_chain: int = 2
    use_msa_stack: bool = True
    pairformer: PairFormerConfig = field(default_factory=PairFormerConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)
    msa_stack: MSAStackConfig = field(default_factory=MSAStackConfig)


@dataclass
class DiffusionConfig:
    """Diffusion head configuration.

    Attributes:
        num_steps: Number of diffusion denoising steps.
        num_samples: Number of structure samples to generate.
        gamma_0: Initial gamma for stochastic sampling.
        gamma_min: Minimum gamma value.
        noise_scale: Noise scale factor.
        step_scale: Step scale factor.
        num_transformer_blocks: Number of diffusion transformer blocks.
        transformer_heads: Number of attention heads in diffusion transformer.
        key_dim: Key dimension per head (None = default to head_dim).
        value_dim: Value dimension per head (None = default to head_dim).
        conditioning_prob: Probability of using conditioning during training.
        conditioning_pair_channel: Pair conditioning channel dimension.
        conditioning_seq_channel: Sequence conditioning channel dimension.
    """

    num_steps: int = 200
    num_samples: int = 5
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    num_transformer_blocks: int = 24
    transformer_heads: int = 16
    key_dim: int | None = None
    value_dim: int | None = None
    conditioning_prob: float = 0.8
    conditioning_pair_channel: int = 128
    conditioning_seq_channel: int = 384

    # Atom cross-attention conditioning (AF3 parity)
    per_token_channels: int = 768
    per_atom_channels: int = 128
    per_atom_pair_channels: int = 16
    atom_transformer_num_blocks: int = 3
    atom_transformer_num_intermediate_factor: int = 2
    atom_transformer_num_head: int = 4
    atom_transformer_key_dim: int = 128
    atom_transformer_value_dim: int = 128

    # Diffusion transformer stack (AF3 parity)
    transformer_super_block_size: int = 4
    transformer_num_intermediate_factor: int = 2


@dataclass
class ConfidenceConfig:
    """Confidence head configuration.

    Attributes:
        num_pairformer_layers: Number of PairFormer layers for confidence refinement.
        num_plddt_bins: Number of bins for pLDDT prediction.
        num_bins: Number of bins for PAE/PDE prediction.
        num_pae_bins: Alias for num_bins (backward compatibility).
        max_error_bin: Maximum error bin value in Angstroms.
        no_embedding_prob: Probability of not using embeddings during training.
        dgram_min_bin: Minimum distogram bin in Angstroms.
        dgram_max_bin: Maximum distogram bin in Angstroms.
        dgram_num_bins: Number of distogram bins.
    """

    num_pairformer_layers: int = 4
    num_plddt_bins: int = 50
    num_bins: int = 64
    num_pae_bins: int = 64
    max_error_bin: float = 31.0
    no_embedding_prob: float = 0.2
    dgram_min_bin: float = 3.25
    dgram_max_bin: float = 50.75
    dgram_num_bins: int = 39


@dataclass
class GlobalConfig:
    """Global model configuration.

    Attributes:
        precision: Compute precision - 'float32', 'float16', or 'bfloat16'.
        use_compile: Whether to use mx.compile for performance.
        flash_attention: Whether to use flash attention optimization.
        chunk_size: Chunk size for memory-efficient computation (None = no chunking).
    """

    precision: str = "float32"
    use_compile: bool = True
    flash_attention: bool = True
    chunk_size: int | None = None

    # AF3 parity flags
    bfloat16: str = "all"
    final_init: str = "zeros"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.precision not in ("float32", "float16", "bfloat16"):
            raise ValueError(
                f"precision must be 'float32', 'float16', or 'bfloat16', "
                f"got '{self.precision}'"
            )
        if self.bfloat16 not in ("all", "none", "intermediate"):
            raise ValueError(
                f"bfloat16 must be 'all', 'none', or 'intermediate', got '{self.bfloat16}'"
            )
        if self.final_init not in ("zeros", "linear"):
            raise ValueError(
                f"final_init must be 'zeros' or 'linear', got '{self.final_init}'"
            )


@dataclass
class ModelConfig:
    """Complete model configuration.

    Attributes:
        evoformer: Evoformer configuration.
        diffusion: Diffusion head configuration.
        confidence: Confidence head configuration.
        global_config: Global model settings.
        num_recycles: Number of recycling iterations.
        return_embeddings: Whether to return Evoformer embeddings in ModelResult.
    """

    evoformer: EvoformerConfig = field(default_factory=EvoformerConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    num_recycles: int = 10
    return_embeddings: bool = False

    @classmethod
    def default(cls) -> "ModelConfig":
        """Return default configuration matching AF3 paper."""
        return cls(
            evoformer=EvoformerConfig(),
            diffusion=DiffusionConfig(),
            confidence=ConfidenceConfig(),
            global_config=GlobalConfig(),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            ModelConfig instance loaded from file.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If YAML is invalid.
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        evoformer_data = data.get("evoformer", {})
        diffusion_data = data.get("diffusion", {})
        confidence_data = data.get("confidence", {})
        global_data = data.get("global_config", {})

        # Handle nested configs in evoformer
        pairformer_data = evoformer_data.pop("pairformer", {})
        template_data = evoformer_data.pop("template", {})
        msa_stack_data = evoformer_data.pop("msa_stack", {})

        return cls(
            evoformer=EvoformerConfig(
                **evoformer_data,
                pairformer=PairFormerConfig(**pairformer_data),
                template=TemplateConfig(**template_data),
                msa_stack=MSAStackConfig(**msa_stack_data),
            ),
            diffusion=DiffusionConfig(**diffusion_data),
            confidence=ConfidenceConfig(**confidence_data),
            global_config=GlobalConfig(**global_data),
            num_recycles=data.get("num_recycles", 10),
            return_embeddings=data.get("return_embeddings", False),
        )
