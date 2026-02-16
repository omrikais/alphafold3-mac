"""Constants for alphafold3_mlx package."""

# Tolerance thresholds by precision
# Note: Reduced precision tolerances account for error accumulation in attention
# (softmax + matrix multiply chain). These are validated with identical inputs.
TOLERANCES: dict[str, dict[str, float]] = {
    "float32": {"rtol": 1e-4, "atol": 1e-5},
    "float16": {"rtol": 2e-3, "atol": 5e-4},  # Looser due to FP16 accumulation
    "bfloat16": {"rtol": 5e-3, "atol": 5e-3},  # Loosest - only 7 mantissa bits
}

# AF3 representative shapes for validation
AF3_SHAPES: list[dict[str, int]] = [
    {"batch": 1, "heads": 4, "seq": 256, "head_dim": 64},
    {"batch": 1, "heads": 4, "seq": 512, "head_dim": 64},
    {"batch": 1, "heads": 4, "seq": 1024, "head_dim": 64},
]

# Memory threshold for go/no-go
# Peak memory / theoretical minimum must be <= this value
MEMORY_RATIO_THRESHOLD: float = 2.0

# Default mask value for boolean mask conversion
DEFAULT_MASK_VALUE: float = 1e9

# =============================================================================
# Model Architecture Constants (Phase 3)
# =============================================================================

# Diffusion model constants
SIGMA_DATA: float = 16.0  # Training data noise scale
SIGMA_MAX: float = 160.0  # Maximum noise level
SIGMA_MIN: float = 0.0004  # Minimum noise level
RHO: float = 7.0  # Noise schedule rho parameter

# Atom representation
MAX_ATOMS: int = 37  # Maximum atoms per residue (atom37 representation)

# Embedding dimensions (default values matching AF3 paper)
SEQ_CHANNEL: int = 384  # Single representation dimension
PAIR_CHANNEL: int = 128  # Pair representation dimension
MSA_CHANNEL: int = 64  # MSA representation dimension

# Evoformer architecture
NUM_PAIRFORMER_LAYERS: int = 48  # Number of PairFormer layers
NUM_MSA_LAYERS: int = 4  # Number of MSA layers

# Diffusion architecture
NUM_DIFFUSION_STEPS: int = 200  # Default number of denoising steps
NUM_DIFFUSION_TRANSFORMER_BLOCKS: int = 24  # Transformer blocks in diffusion
NUM_SAMPLES: int = 5  # Default number of structure samples

# Confidence head
NUM_PLDDT_BINS: int = 50  # Number of pLDDT bins
NUM_PAE_BINS: int = 64  # Number of PAE bins
MAX_ERROR_BIN: float = 31.0  # Maximum error bin value in Angstroms

# Recycling
DEFAULT_NUM_RECYCLES: int = 10  # Default number of recycling iterations

# Memory management
DIFFUSION_EVAL_INTERVAL: int = 10 # mx.eval every N diffusion steps

# Physical constants for structure validation
BOND_LENGTH_TOLERANCE: float = 0.05  # Angstroms
BOND_ANGLE_TOLERANCE: float = 5.0  # Degrees
MIN_VALID_FRACTION: float = 0.95  # 95% of bonds/angles must be within tolerance
