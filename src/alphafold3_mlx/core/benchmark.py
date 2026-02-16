"""Benchmark result dataclass for performance measurement."""

from dataclasses import dataclass, asdict
from typing import Any

from alphafold3_mlx.core.config import AttentionConfig
from alphafold3_mlx.core.constants import MEMORY_RATIO_THRESHOLD


@dataclass
class BenchmarkResult:
    """Performance benchmark result for a single configuration.

    Attributes:
        config: Attention configuration used
        execution_time_s: Execution time in seconds
        peak_memory_bytes: Peak memory usage in bytes
        theoretical_minimum_bytes: Theoretical minimum memory required
        memory_ratio: Ratio of peak to theoretical minimum
        memory_within_threshold: True if ratio <= MEMORY_RATIO_THRESHOLD (2.0)
    """

    config: AttentionConfig
    execution_time_s: float
    peak_memory_bytes: int
    theoretical_minimum_bytes: int
    memory_ratio: float
    memory_within_threshold: bool

    @classmethod
    def create(
        cls,
        config: AttentionConfig,
        execution_time_s: float,
        peak_memory_bytes: int,
        *,
        use_bias: bool = False,
    ) -> "BenchmarkResult":
        """Create benchmark result with computed metrics.

        Args:
            config: Attention configuration
            execution_time_s: Measured execution time
            peak_memory_bytes: Measured peak memory
            use_bias: Whether additive bias is used (affects theoretical minimum)

        Returns:
            BenchmarkResult with computed memory ratio and threshold check
        """
        theoretical_min = theoretical_minimum_bytes(
            config,
            include_logits=True,
            include_bias=use_bias,
        )
        ratio = peak_memory_bytes / theoretical_min if theoretical_min > 0 else float("inf")

        return cls(
            config=config,
            execution_time_s=execution_time_s,
            peak_memory_bytes=peak_memory_bytes,
            theoretical_minimum_bytes=theoretical_min,
            memory_ratio=ratio,
            memory_within_threshold=ratio <= MEMORY_RATIO_THRESHOLD,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "config": self.config.to_dict(),
            "execution_time_s": self.execution_time_s,
            "peak_memory_bytes": self.peak_memory_bytes,
            "theoretical_minimum_bytes": self.theoretical_minimum_bytes,
            "memory_ratio": self.memory_ratio,
            "memory_within_threshold": self.memory_within_threshold,
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        status = "PASS" if self.memory_within_threshold else "FAIL"
        return (
            f"BenchmarkResult(seq={self.config.seq_q}, "
            f"time={self.execution_time_s:.3f}s, "
            f"memory_ratio={self.memory_ratio:.2f}x, "
            f"{status})"
        )


def theoretical_minimum_bytes(
    config: AttentionConfig,
    *,
    include_logits: bool = True,
    include_bias: bool = False,
) -> int:
    """Calculate theoretical minimum memory for attention.

    Updated:
    - Base: Q + K + V + output = 4 × batch × heads × seq × head_dim
    - Logits: batch × heads × seq_q × seq_k (inherent to SDPA)
    - Bias: batch × heads × seq_q × seq_k (if using additive bias)

    For cross-attention (seq_q != seq_k), shapes are handled correctly.

    Note: Always uses float32 (4 bytes/element) regardless of config.dtype
    because The policy requires float32 internal computation for numerical
    stability. The attention implementation upcasts Q, K, V to float32
    before SDPA execution.

    Args:
        config: Attention configuration
        include_logits: Include logits matrix in calculation (default True)
        include_bias: Include bias tensor in calculation (default False)

    Returns:
        Theoretical minimum bytes required
    """
    # Internal computation is always float32 for numerical stability.
    # The attention implementation upcasts all inputs to float32 before SDPA,
    # so memory baseline must reflect actual float32 allocations.
    bytes_per_element = 4  # float32

    # Q and output use seq_q
    q_size = config.batch_size * config.num_heads * config.seq_q * config.head_dim
    # K and V use seq_k
    kv_size = config.batch_size * config.num_heads * config.seq_k * config.head_dim

    # Base: Q + K + V + output = 2*q_size + 2*kv_size
    total_elements = 2 * q_size + 2 * kv_size

    # Logits matrix: batch × heads × seq_q × seq_k
    if include_logits:
        logits_size = config.batch_size * config.num_heads * config.seq_q * config.seq_k
        total_elements += logits_size

    # Bias tensor: batch × heads × seq_q × seq_k
    if include_bias:
        bias_size = config.batch_size * config.num_heads * config.seq_q * config.seq_k
        total_elements += bias_size

    return total_elements * bytes_per_element
