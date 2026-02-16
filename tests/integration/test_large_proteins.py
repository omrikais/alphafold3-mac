"""Large protein memory tests.

Tests memory requirements for large proteins:
- 1000 residues: <100GB peak memory
- 2000 residues: graceful rejection or handling within limits
"""

from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx

from alphafold3_mlx.core.validation import (
    estimate_peak_memory_gb,
    check_memory_requirements,
    get_available_memory_gb,
)
from alphafold3_mlx.core.exceptions import MemoryError


class TestMemoryEstimation:
    """Test memory estimation accuracy."""

    def test_estimation_scales_quadratically_with_residues(self):
        """Test that memory estimation scales ~O(N²) with sequence length."""
        mem_500 = estimate_peak_memory_gb(500)
        mem_1000 = estimate_peak_memory_gb(1000)
        mem_2000 = estimate_peak_memory_gb(2000)

        # Should roughly quadruple when doubling sequence length
        ratio_1000_500 = mem_1000 / mem_500
        ratio_2000_1000 = mem_2000 / mem_1000

        # Allow some deviation from perfect quadratic due to other terms
        assert 2.5 < ratio_1000_500 < 5.0, f"Ratio 1000/500 = {ratio_1000_500}"
        assert 2.5 < ratio_2000_1000 < 5.0, f"Ratio 2000/1000 = {ratio_2000_1000}"

    def test_estimation_accounts_for_samples(self):
        """Test that memory estimation accounts for number of samples."""
        mem_1_sample = estimate_peak_memory_gb(500, num_samples=1)
        mem_5_samples = estimate_peak_memory_gb(500, num_samples=5)

        # More samples should require more memory
        assert mem_5_samples > mem_1_sample

    @pytest.mark.parametrize("num_residues", [100, 200, 500, 1000])
    def test_estimation_returns_positive(self, num_residues: int):
        """Test that memory estimation returns positive values."""
        estimate = estimate_peak_memory_gb(num_residues)
        assert estimate > 0, f"Estimate for {num_residues} residues is non-positive"


class TestLargeProteinMemory:
    """Test memory handling for large proteins."""

    def test_1000_residue_memory_estimate(self):
        """Test that 1000-residue estimate is reasonable. requirement: 1000 residues should require <100GB.
        """
        estimate = estimate_peak_memory_gb(1000, num_samples=5)

        # Should be under 100GB for 128GB Mac
        assert estimate < 100, (
            f"1000 residue estimate {estimate:.1f}GB exceeds 100GB limit"
        )

    def test_2000_residue_detection(self):
        """Test that 2000-residue proteins are handled appropriately. requirement: 2000 residues should either work within memory
        limits or be gracefully rejected.
        """
        estimate = estimate_peak_memory_gb(2000, num_samples=5)

        # This is an informational test - large proteins may exceed limits
        print(f"2000 residue estimate: {estimate:.1f}GB")

        # Either it fits (estimate < available) or rejection works
        available = get_available_memory_gb

        if estimate > available * 0.8:
            # Should raise MemoryError when checked
            with pytest.raises(MemoryError):
                check_memory_requirements(
                    num_residues=2000,
                    available_gb=available,
                    num_samples=5,
                )
        else:
            # Should pass the check
            check_memory_requirements(
                num_residues=2000,
                available_gb=available,
                num_samples=5,
            )

    def test_memory_check_raises_for_insufficient_memory(self):
        """Test that memory check raises appropriate error."""
        # Simulate low memory scenario
        with pytest.raises(MemoryError) as exc_info:
            check_memory_requirements(
                num_residues=5000, # Very large
                available_gb=32, # Limited memory
                num_samples=5,
            )

        error = exc_info.value
        assert error.estimated_gb > 0
        assert error.available_gb == 32
        assert error.num_residues == 5000

    def test_memory_error_message_helpful(self):
        """Test that memory error provides actionable information."""
        try:
            check_memory_requirements(
                num_residues=3000,
                available_gb=64,
                num_samples=5,
            )
        except MemoryError as e:
            error_str = str(e)
            # Should mention residue count
            assert "3000" in error_str or "residue" in error_str.lower
            # Should mention memory
            assert "memory" in error_str.lower or "GB" in error_str


class TestLargeProteinAllocation:
    """Test actual memory allocation for large proteins."""

    @pytest.mark.slow
    def test_500_residue_allocation(self):
        """Test that 500-residue allocations work."""
        num_residues = 500
        pair_dim = 128
        seq_dim = 384

        # Allocate core tensors
        pair = mx.zeros((1, num_residues, num_residues, pair_dim))
        single = mx.zeros((1, num_residues, seq_dim))
        mx.eval(pair, single)

        # Should succeed
        assert pair.shape == (1, num_residues, num_residues, pair_dim)
        assert single.shape == (1, num_residues, seq_dim)

        # Clean up
        del pair, single
        try:
            mx.metal.clear_cache
        except AttributeError:
            pass

    @pytest.mark.slow
    def test_1000_residue_allocation(self):
        """Test that 1000-residue allocations work."""
        num_residues = 1000
        pair_dim = 128
        seq_dim = 384

        try:
            pair = mx.zeros((1, num_residues, num_residues, pair_dim))
            single = mx.zeros((1, num_residues, seq_dim))
            mx.eval(pair, single)

            assert pair.shape == (1, num_residues, num_residues, pair_dim)

            del pair, single

        except (MemoryError, RuntimeError) as e:
            # If allocation fails, that's informative but not a test failure
            # on memory-limited systems
            pytest.skip(f"Allocation failed (likely memory limited): {e}")

    @pytest.mark.slow
    def test_chunked_processing_enabled(self):
        """Test that chunked processing can handle large sequences."""
        from alphafold3_mlx.network.attention import chunked_attention

        # Simulate large sequence attention
        batch = 1
        heads = 4
        seq = 512 # Moderate size for testing
        head_dim = 64

        q = mx.random.normal(shape=(batch, heads, seq, head_dim), key=mx.random.key(0))
        k = mx.random.normal(shape=(batch, heads, seq, head_dim), key=mx.random.key(1))
        v = mx.random.normal(shape=(batch, heads, seq, head_dim), key=mx.random.key(2))

        scale = 1.0 / (head_dim ** 0.5)

        # Run with chunking
        output = chunked_attention(q, k, v, scale=scale, chunk_size=128)
        mx.eval(output)

        # Verify output shape
        assert output.shape == (batch, heads, seq, head_dim)

        # Verify no NaN
        output_np = np.array(output)
        assert not np.any(np.isnan(output_np))


class TestGracefulDegradation:
    """Test graceful degradation for resource limits."""

    def test_reduce_samples_for_large_proteins(self):
        """Test reducing sample count for memory efficiency."""
        # For large proteins, reducing samples can help fit in memory
        mem_5_samples = estimate_peak_memory_gb(1500, num_samples=5)
        mem_1_sample = estimate_peak_memory_gb(1500, num_samples=1)

        # Single sample should use less memory
        assert mem_1_sample < mem_5_samples

        # Calculate savings
        savings = (mem_5_samples - mem_1_sample) / mem_5_samples * 100
        print(f"Memory savings with 1 sample vs 5: {savings:.1f}%")

    def test_get_available_memory(self):
        """Test that available memory detection works."""
        available = get_available_memory_gb

        # Should return positive value
        assert available > 0

        # Should be reasonable for modern Mac (>8GB)
        assert available >= 8.0, f"Available memory {available}GB seems too low"

        # Should be reasonable maximum (<1TB)
        assert available < 1024, f"Available memory {available}GB seems too high"
