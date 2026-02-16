"""Unit tests for weight loading and conversion.

Tests the convert_array function, dtype handling, and shape preservation.
"""

import numpy as np
import pytest

# Import MLX - skip tests if not available
mlx = pytest.importorskip("mlx.core")

from alphafold3_mlx.weights import (
    PlatformError,
    PlatformInfo,
    UnsupportedDtypeError,
    WeightsNotFoundError,
    convert_array,
    get_platform_info,
)


class TestConvertArray:
    """Tests for the convert_array function."""

    def test_float32_conversion(self):
        """Test float32 array conversion preserves values and shape."""
        np_arr = np.random.randn(64, 128).astype(np.float32)
        mlx_arr = convert_array(np_arr)

        assert mlx_arr.shape == (64, 128)
        assert mlx_arr.dtype == mlx.float32
        np.testing.assert_allclose(
            np.array(mlx_arr), np_arr, rtol=1e-6, atol=1e-6
        )

    def test_float16_conversion(self):
        """Test float16 array conversion preserves values and shape."""
        np_arr = np.random.randn(32, 64).astype(np.float16)
        mlx_arr = convert_array(np_arr)

        assert mlx_arr.shape == (32, 64)
        assert mlx_arr.dtype == mlx.float16
        np.testing.assert_allclose(
            np.array(mlx_arr).astype(np.float32),
            np_arr.astype(np.float32),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_int32_conversion(self):
        """Test int32 array conversion preserves values and shape."""
        np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        mlx_arr = convert_array(np_arr)

        assert mlx_arr.shape == (2, 3)
        assert mlx_arr.dtype == mlx.int32
        np.testing.assert_array_equal(np.array(mlx_arr), np_arr)

    def test_dtype_override(self):
        """Test explicit dtype conversion."""
        np_arr = np.random.randn(16, 16).astype(np.float32)
        mlx_arr = convert_array(np_arr, dtype=mlx.float16)

        assert mlx_arr.dtype == mlx.float16
        # Values should be approximately equal after conversion
        np.testing.assert_allclose(
            np.array(mlx_arr).astype(np.float32),
            np_arr,
            rtol=1e-2,
            atol=1e-3,
        )

    def test_1d_array(self):
        """Test 1D array conversion."""
        np_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        mlx_arr = convert_array(np_arr)

        assert mlx_arr.shape == (4,)
        assert mlx_arr.dtype == mlx.float32

    def test_3d_array(self):
        """Test 3D array conversion."""
        np_arr = np.random.randn(2, 3, 4).astype(np.float32)
        mlx_arr = convert_array(np_arr)

        assert mlx_arr.shape == (2, 3, 4)
        assert mlx_arr.dtype == mlx.float32

    def test_large_array(self):
        """Test large array conversion (memory efficiency)."""
        # 256 x 256 x 64 = ~4MB float32
        np_arr = np.random.randn(256, 256, 64).astype(np.float32)
        mlx_arr = convert_array(np_arr)

        assert mlx_arr.shape == (256, 256, 64)
        # Spot check values
        np.testing.assert_allclose(
            np.array(mlx_arr[:2, :2, :2]),
            np_arr[:2, :2, :2],
            rtol=1e-6,
            atol=1e-6,
        )


class TestShapePreservation:
    """Tests for shape preservation (0% mismatch)."""

    @pytest.mark.parametrize(
        "shape",
        [
            (1,),
            (10,),
            (100,),
            (10, 10),
            (32, 64),
            (128, 256),
            (8, 16, 32),
            (4, 8, 16, 32),
            (2, 3, 4, 5, 6),
        ],
    )
    def test_shape_preserved(self, shape):
        """Test that various shapes are preserved exactly."""
        np_arr = np.random.randn(*shape).astype(np.float32)
        mlx_arr = convert_array(np_arr)

        assert mlx_arr.shape == shape, f"Shape mismatch: {mlx_arr.shape} != {shape}"


class TestBfloat16:
    """Tests for bfloat16 handling (requires M3+ for efficient inference)."""

    def test_bfloat16_creation(self):
        """Test that bfloat16 arrays can be created."""
        # Create a float32 array and convert to bfloat16
        np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mlx_arr = convert_array(np_arr, dtype=mlx.bfloat16)

        assert mlx_arr.dtype == mlx.bfloat16
        # Convert back to float32 for comparison
        result = np.array(mlx_arr.astype(mlx.float32))
        np.testing.assert_allclose(result, np_arr, rtol=1e-2, atol=1e-2)

    def test_bfloat16_shape_preserved(self):
        """Test that shape is preserved in bfloat16 conversion."""
        np_arr = np.random.randn(64, 128).astype(np.float32)
        mlx_arr = convert_array(np_arr, dtype=mlx.bfloat16)

        assert mlx_arr.shape == (64, 128)
        assert mlx_arr.dtype == mlx.bfloat16


class TestPlatformInfo:
    """Tests for platform information."""

    def test_get_platform_info(self):
        """Test that platform info is returned correctly."""
        info = get_platform_info()

        assert isinstance(info, PlatformInfo)
        assert info.system == "Darwin"
        assert info.machine == "arm64"
        # Per API contract: M2, M3, M4, or Unknown (M1 maps to Unknown)
        assert info.chip_family in ("M2", "M3", "M4", "Unknown")
        assert isinstance(info.supports_bfloat16, bool)
        assert isinstance(info.memory_gb, int)
        assert info.memory_gb > 0

    def test_bfloat16_support_logic(self):
        """Test that bfloat16 support is correctly determined."""
        info = get_platform_info()

        if info.chip_family in ("M3", "M4"):
            assert info.supports_bfloat16 is True
        elif info.chip_family in ("M2", "Unknown"):
            # M2 and Unknown (including M1) don't have efficient bf16
            assert info.supports_bfloat16 is False


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_contiguous_array(self):
        """Test that non-contiguous arrays are handled correctly."""
        np_arr = np.random.randn(10, 10).astype(np.float32)
        # Create non-contiguous view
        non_contiguous = np_arr[::2, ::2]
        assert not non_contiguous.flags["C_CONTIGUOUS"]

        mlx_arr = convert_array(non_contiguous)
        assert mlx_arr.shape == non_contiguous.shape
        np.testing.assert_allclose(
            np.array(mlx_arr), non_contiguous, rtol=1e-6, atol=1e-6
        )

    def test_single_element(self):
        """Test single-element array conversion."""
        np_arr = np.array([42.0], dtype=np.float32)
        mlx_arr = convert_array(np_arr)

        assert mlx_arr.shape == (1,)
        assert float(mlx_arr[0]) == 42.0


class TestUnsupportedDtypes:
    """Tests for explicit UnsupportedDtypeError on unsupported dtypes."""

    def test_complex64_raises_error(self):
        """Test that complex64 raises UnsupportedDtypeError explicitly."""
        np_arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)

        with pytest.raises(UnsupportedDtypeError) as exc_info:
            convert_array(np_arr)

        assert "complex64" in str(exc_info.value)
        assert "Supported dtypes" in str(exc_info.value)

    def test_complex128_raises_error(self):
        """Test that complex128 raises UnsupportedDtypeError explicitly."""
        np_arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)

        with pytest.raises(UnsupportedDtypeError) as exc_info:
            convert_array(np_arr)

        assert "complex128" in str(exc_info.value)

    def test_object_dtype_raises_error(self):
        """Test that object dtype raises UnsupportedDtypeError."""
        np_arr = np.array(["a", "b", "c"], dtype=object)

        with pytest.raises(UnsupportedDtypeError) as exc_info:
            convert_array(np_arr)

        assert "object" in str(exc_info.value)


class TestErrorTypes:
    """Tests for proper exception types per API contract."""

    def test_weights_not_found_is_file_not_found(self):
        """Test that WeightsNotFoundError is a subclass of FileNotFoundError."""
        assert issubclass(WeightsNotFoundError, FileNotFoundError)

    def test_platform_error_exists(self):
        """Test that PlatformError is properly defined."""
        assert issubclass(PlatformError, Exception)
