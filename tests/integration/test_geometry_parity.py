"""Integration tests for JAX parity validation.

These tests validate that the MLX geometry modules and neural network layers
produce outputs matching JAX reference outputs within tolerance.

Requirements:
    - Golden outputs must be generated first using:
      PYTHONPATH=src python scripts/generate_geometry_golden.py

This module validates:
    - Vec3Array/Rot3Array parity with JAX
    - Linear/LayerNorm/GLU parity with Haiku/JAX
    - Reduced precision (float16/bfloat16) parity
    -: Numerical accuracy requirements
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx


# Path to golden outputs
GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "geometry_golden"


def skip_if_no_golden():
    """Skip test if golden outputs don't exist."""
    if not GOLDEN_DIR.exists():
        pytest.skip(f"Golden outputs not found at {GOLDEN_DIR}. Run generate_geometry_golden.py first.")


# Tolerances for different dtypes (per)
TOLERANCES = {
    "float32": {"rtol": 1e-5, "atol": 1e-5},
    "float16": {"rtol": 2e-3, "atol": 5e-4},
    "bfloat16": {"rtol": 5e-3, "atol": 5e-3},
}


class TestVec3ArrayJaxParity:
    """Tests for Vec3Array parity with JAX."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load golden outputs."""
        skip_if_no_golden()
        golden_file = GOLDEN_DIR / "vec3array_ops.npz"
        if not golden_file.exists():
            pytest.skip(f"Golden file not found: {golden_file}")
        self.golden = np.load(golden_file)

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((4,), "float32"),
            ((2, 3), "float32"),
            ((1, 5000), "float32"),
            ((4,), "float16"),
            ((2, 3), "float16"),
            ((1, 5000), "float16"),
            ((4,), "bfloat16"),
            ((2, 3), "bfloat16"),
            ((1, 5000), "bfloat16"),
        ],
    )
    def test_vec3_addition(self, shape, dtype):
        """Test Vec3Array addition matches JAX."""
        from alphafold3_mlx.geometry import Vec3Array

        prefix = f"vec3_{shape}_{dtype}"
        tol = TOLERANCES[dtype]

        # Load inputs (stored as float32)
        v1_x = mx.array(self.golden[f"{prefix}_v1_x"])
        v1_y = mx.array(self.golden[f"{prefix}_v1_y"])
        v1_z = mx.array(self.golden[f"{prefix}_v1_z"])
        v2_x = mx.array(self.golden[f"{prefix}_v2_x"])
        v2_y = mx.array(self.golden[f"{prefix}_v2_y"])
        v2_z = mx.array(self.golden[f"{prefix}_v2_z"])

        # Convert to target dtype
        mlx_dtype = getattr(mx, dtype)
        v1 = Vec3Array(x=v1_x.astype(mlx_dtype), y=v1_y.astype(mlx_dtype), z=v1_z.astype(mlx_dtype))
        v2 = Vec3Array(x=v2_x.astype(mlx_dtype), y=v2_y.astype(mlx_dtype), z=v2_z.astype(mlx_dtype))

        # Compute
        result = v1 + v2

        # Compare (convert bfloat16 to float32 for numpy)
        result_x = np.array(result.x.astype(mx.float32))
        result_y = np.array(result.y.astype(mx.float32))
        result_z = np.array(result.z.astype(mx.float32))

        np.testing.assert_allclose(result_x, self.golden[f"{prefix}_add_x"], **tol)
        np.testing.assert_allclose(result_y, self.golden[f"{prefix}_add_y"], **tol)
        np.testing.assert_allclose(result_z, self.golden[f"{prefix}_add_z"], **tol)

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((4,), "float32"),
            ((2, 3), "float32"),
            ((4,), "float16"),
            ((4,), "bfloat16"),
        ],
    )
    def test_vec3_dot_product(self, shape, dtype):
        """Test Vec3Array dot product matches JAX."""
        from alphafold3_mlx.geometry import Vec3Array

        prefix = f"vec3_{shape}_{dtype}"
        tol = TOLERANCES[dtype]

        # Load inputs
        v1_x = mx.array(self.golden[f"{prefix}_v1_x"])
        v1_y = mx.array(self.golden[f"{prefix}_v1_y"])
        v1_z = mx.array(self.golden[f"{prefix}_v1_z"])
        v2_x = mx.array(self.golden[f"{prefix}_v2_x"])
        v2_y = mx.array(self.golden[f"{prefix}_v2_y"])
        v2_z = mx.array(self.golden[f"{prefix}_v2_z"])

        mlx_dtype = getattr(mx, dtype)
        v1 = Vec3Array(x=v1_x.astype(mlx_dtype), y=v1_y.astype(mlx_dtype), z=v1_z.astype(mlx_dtype))
        v2 = Vec3Array(x=v2_x.astype(mlx_dtype), y=v2_y.astype(mlx_dtype), z=v2_z.astype(mlx_dtype))

        # Compute dot product
        result = v1.dot(v2)

        np.testing.assert_allclose(np.array(result.astype(mx.float32)), self.golden[f"{prefix}_dot"], **tol)

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((4,), "float32"),
            ((2, 3), "float32"),
            ((4,), "float16"),
            ((4,), "bfloat16"),
        ],
    )
    def test_vec3_cross_product(self, shape, dtype):
        """Test Vec3Array cross product matches JAX."""
        from alphafold3_mlx.geometry import Vec3Array

        prefix = f"vec3_{shape}_{dtype}"
        tol = TOLERANCES[dtype]

        v1_x = mx.array(self.golden[f"{prefix}_v1_x"])
        v1_y = mx.array(self.golden[f"{prefix}_v1_y"])
        v1_z = mx.array(self.golden[f"{prefix}_v1_z"])
        v2_x = mx.array(self.golden[f"{prefix}_v2_x"])
        v2_y = mx.array(self.golden[f"{prefix}_v2_y"])
        v2_z = mx.array(self.golden[f"{prefix}_v2_z"])

        mlx_dtype = getattr(mx, dtype)
        v1 = Vec3Array(x=v1_x.astype(mlx_dtype), y=v1_y.astype(mlx_dtype), z=v1_z.astype(mlx_dtype))
        v2 = Vec3Array(x=v2_x.astype(mlx_dtype), y=v2_y.astype(mlx_dtype), z=v2_z.astype(mlx_dtype))

        result = v1.cross(v2)

        np.testing.assert_allclose(np.array(result.x.astype(mx.float32)), self.golden[f"{prefix}_cross_x"], **tol)
        np.testing.assert_allclose(np.array(result.y.astype(mx.float32)), self.golden[f"{prefix}_cross_y"], **tol)
        np.testing.assert_allclose(np.array(result.z.astype(mx.float32)), self.golden[f"{prefix}_cross_z"], **tol)

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((4,), "float32"),
            ((2, 3), "float32"),
            ((4,), "float16"),
            ((4,), "bfloat16"),
        ],
    )
    def test_vec3_norm(self, shape, dtype):
        """Test Vec3Array norm matches JAX."""
        from alphafold3_mlx.geometry import Vec3Array

        prefix = f"vec3_{shape}_{dtype}"
        tol = TOLERANCES[dtype]

        v1_x = mx.array(self.golden[f"{prefix}_v1_x"])
        v1_y = mx.array(self.golden[f"{prefix}_v1_y"])
        v1_z = mx.array(self.golden[f"{prefix}_v1_z"])

        mlx_dtype = getattr(mx, dtype)
        v1 = Vec3Array(x=v1_x.astype(mlx_dtype), y=v1_y.astype(mlx_dtype), z=v1_z.astype(mlx_dtype))

        result_norm = v1.norm()
        result_norm2 = v1.norm2()

        np.testing.assert_allclose(np.array(result_norm.astype(mx.float32)), self.golden[f"{prefix}_norm"], **tol)
        np.testing.assert_allclose(np.array(result_norm2.astype(mx.float32)), self.golden[f"{prefix}_norm2"], **tol)

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((4,), "float32"),
            ((2, 3), "float32"),
            ((4,), "float16"),
            ((4,), "bfloat16"),
        ],
    )
    def test_vec3_normalized(self, shape, dtype):
        """Test Vec3Array normalized matches JAX."""
        from alphafold3_mlx.geometry import Vec3Array

        prefix = f"vec3_{shape}_{dtype}"
        tol = TOLERANCES[dtype]

        v1_x = mx.array(self.golden[f"{prefix}_v1_x"])
        v1_y = mx.array(self.golden[f"{prefix}_v1_y"])
        v1_z = mx.array(self.golden[f"{prefix}_v1_z"])

        mlx_dtype = getattr(mx, dtype)
        v1 = Vec3Array(x=v1_x.astype(mlx_dtype), y=v1_y.astype(mlx_dtype), z=v1_z.astype(mlx_dtype))

        result = v1.normalized()

        np.testing.assert_allclose(np.array(result.x.astype(mx.float32)), self.golden[f"{prefix}_normalized_x"], **tol)
        np.testing.assert_allclose(np.array(result.y.astype(mx.float32)), self.golden[f"{prefix}_normalized_y"], **tol)
        np.testing.assert_allclose(np.array(result.z.astype(mx.float32)), self.golden[f"{prefix}_normalized_z"], **tol)


class TestRot3ArrayJaxParity:
    """Tests for Rot3Array parity with JAX."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load golden outputs."""
        skip_if_no_golden()
        golden_file = GOLDEN_DIR / "rot3array_ops.npz"
        if not golden_file.exists():
            pytest.skip(f"Golden file not found: {golden_file}")
        self.golden = np.load(golden_file)

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((4,), "float32"),
            ((2, 3), "float32"),
            ((4,), "float16"),
            ((4,), "bfloat16"),
        ],
    )
    def test_rot3_identity(self, shape, dtype):
        """Test Rot3Array identity matches JAX."""
        from alphafold3_mlx.geometry import Rot3Array

        prefix = f"rot3_{shape}_{dtype}"
        tol = TOLERANCES[dtype]

        mlx_dtype = getattr(mx, dtype)
        identity = Rot3Array.identity(shape, dtype=mlx_dtype)

        np.testing.assert_allclose(np.array(identity.xx.astype(mx.float32)), self.golden[f"{prefix}_identity_xx"], **tol)
        np.testing.assert_allclose(np.array(identity.xy.astype(mx.float32)), self.golden[f"{prefix}_identity_xy"], **tol)
        np.testing.assert_allclose(np.array(identity.xz.astype(mx.float32)), self.golden[f"{prefix}_identity_xz"], **tol)
        np.testing.assert_allclose(np.array(identity.yx.astype(mx.float32)), self.golden[f"{prefix}_identity_yx"], **tol)
        np.testing.assert_allclose(np.array(identity.yy.astype(mx.float32)), self.golden[f"{prefix}_identity_yy"], **tol)
        np.testing.assert_allclose(np.array(identity.yz.astype(mx.float32)), self.golden[f"{prefix}_identity_yz"], **tol)
        np.testing.assert_allclose(np.array(identity.zx.astype(mx.float32)), self.golden[f"{prefix}_identity_zx"], **tol)
        np.testing.assert_allclose(np.array(identity.zy.astype(mx.float32)), self.golden[f"{prefix}_identity_zy"], **tol)
        np.testing.assert_allclose(np.array(identity.zz.astype(mx.float32)), self.golden[f"{prefix}_identity_zz"], **tol)

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((4,), "float32"),
            ((2, 3), "float32"),
            ((4,), "float16"),
            ((4,), "bfloat16"),
        ],
    )
    def test_rot3_from_quaternion_identity(self, shape, dtype):
        """Test Rot3Array from_quaternion with unit quaternion matches JAX."""
        from alphafold3_mlx.geometry import Rot3Array

        prefix = f"rot3_{shape}_{dtype}"
        tol = TOLERANCES[dtype]

        mlx_dtype = getattr(mx, dtype)

        # Unit quaternion (1, 0, 0, 0) should give identity
        w = mx.ones(shape, dtype=mlx_dtype)
        x = mx.zeros(shape, dtype=mlx_dtype)
        y = mx.zeros(shape, dtype=mlx_dtype)
        z = mx.zeros(shape, dtype=mlx_dtype)

        result = Rot3Array.from_quaternion(w, x, y, z, normalize=False)

        np.testing.assert_allclose(np.array(result.xx.astype(mx.float32)), self.golden[f"{prefix}_from_quat_xx"], **tol)
        np.testing.assert_allclose(np.array(result.xy.astype(mx.float32)), self.golden[f"{prefix}_from_quat_xy"], **tol)
        np.testing.assert_allclose(np.array(result.xz.astype(mx.float32)), self.golden[f"{prefix}_from_quat_xz"], **tol)
        np.testing.assert_allclose(np.array(result.yx.astype(mx.float32)), self.golden[f"{prefix}_from_quat_yx"], **tol)
        np.testing.assert_allclose(np.array(result.yy.astype(mx.float32)), self.golden[f"{prefix}_from_quat_yy"], **tol)
        np.testing.assert_allclose(np.array(result.yz.astype(mx.float32)), self.golden[f"{prefix}_from_quat_yz"], **tol)
        np.testing.assert_allclose(np.array(result.zx.astype(mx.float32)), self.golden[f"{prefix}_from_quat_zx"], **tol)
        np.testing.assert_allclose(np.array(result.zy.astype(mx.float32)), self.golden[f"{prefix}_from_quat_zy"], **tol)
        np.testing.assert_allclose(np.array(result.zz.astype(mx.float32)), self.golden[f"{prefix}_from_quat_zz"], **tol)

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((4,), "float32"),
            ((2, 3), "float32"),
            ((4,), "float16"),
            ((4,), "bfloat16"),
        ],
    )
    def test_rot3_apply_to_point(self, shape, dtype):
        """Test Rot3Array.apply_to_point matches JAX."""
        from alphafold3_mlx.geometry import Rot3Array, Vec3Array

        prefix = f"rot3_{shape}_{dtype}"
        tol = TOLERANCES[dtype]
        mlx_dtype = getattr(mx, dtype)

        # Load rotation matrix from golden
        rot = Rot3Array(
            xx=mx.array(self.golden[f"{prefix}_rot_xx"]).astype(mlx_dtype),
            xy=mx.array(self.golden[f"{prefix}_rot_xy"]).astype(mlx_dtype),
            xz=mx.array(self.golden[f"{prefix}_rot_xz"]).astype(mlx_dtype),
            yx=mx.array(self.golden[f"{prefix}_rot_yx"]).astype(mlx_dtype),
            yy=mx.array(self.golden[f"{prefix}_rot_yy"]).astype(mlx_dtype),
            yz=mx.array(self.golden[f"{prefix}_rot_yz"]).astype(mlx_dtype),
            zx=mx.array(self.golden[f"{prefix}_rot_zx"]).astype(mlx_dtype),
            zy=mx.array(self.golden[f"{prefix}_rot_zy"]).astype(mlx_dtype),
            zz=mx.array(self.golden[f"{prefix}_rot_zz"]).astype(mlx_dtype),
        )

        # Load input vector
        v = Vec3Array(
            x=mx.array(self.golden[f"{prefix}_apply_input_x"]).astype(mlx_dtype),
            y=mx.array(self.golden[f"{prefix}_apply_input_y"]).astype(mlx_dtype),
            z=mx.array(self.golden[f"{prefix}_apply_input_z"]).astype(mlx_dtype),
        )

        result = rot.apply_to_point(v)

        np.testing.assert_allclose(np.array(result.x.astype(mx.float32)), self.golden[f"{prefix}_apply_result_x"], **tol)
        np.testing.assert_allclose(np.array(result.y.astype(mx.float32)), self.golden[f"{prefix}_apply_result_y"], **tol)
        np.testing.assert_allclose(np.array(result.z.astype(mx.float32)), self.golden[f"{prefix}_apply_result_z"], **tol)

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((4,), "float32"),
            ((2, 3), "float32"),
            ((4,), "float16"),
            ((4,), "bfloat16"),
        ],
    )
    def test_rot3_inverse(self, shape, dtype):
        """Test Rot3Array.inverse matches JAX (transpose)."""
        from alphafold3_mlx.geometry import Rot3Array

        prefix = f"rot3_{shape}_{dtype}"
        tol = TOLERANCES[dtype]
        mlx_dtype = getattr(mx, dtype)

        # Load rotation matrix from golden
        rot = Rot3Array(
            xx=mx.array(self.golden[f"{prefix}_rot_xx"]).astype(mlx_dtype),
            xy=mx.array(self.golden[f"{prefix}_rot_xy"]).astype(mlx_dtype),
            xz=mx.array(self.golden[f"{prefix}_rot_xz"]).astype(mlx_dtype),
            yx=mx.array(self.golden[f"{prefix}_rot_yx"]).astype(mlx_dtype),
            yy=mx.array(self.golden[f"{prefix}_rot_yy"]).astype(mlx_dtype),
            yz=mx.array(self.golden[f"{prefix}_rot_yz"]).astype(mlx_dtype),
            zx=mx.array(self.golden[f"{prefix}_rot_zx"]).astype(mlx_dtype),
            zy=mx.array(self.golden[f"{prefix}_rot_zy"]).astype(mlx_dtype),
            zz=mx.array(self.golden[f"{prefix}_rot_zz"]).astype(mlx_dtype),
        )

        result = rot.inverse()

        np.testing.assert_allclose(np.array(result.xx.astype(mx.float32)), self.golden[f"{prefix}_inverse_xx"], **tol)
        np.testing.assert_allclose(np.array(result.xy.astype(mx.float32)), self.golden[f"{prefix}_inverse_xy"], **tol)
        np.testing.assert_allclose(np.array(result.xz.astype(mx.float32)), self.golden[f"{prefix}_inverse_xz"], **tol)
        np.testing.assert_allclose(np.array(result.yx.astype(mx.float32)), self.golden[f"{prefix}_inverse_yx"], **tol)
        np.testing.assert_allclose(np.array(result.yy.astype(mx.float32)), self.golden[f"{prefix}_inverse_yy"], **tol)
        np.testing.assert_allclose(np.array(result.yz.astype(mx.float32)), self.golden[f"{prefix}_inverse_yz"], **tol)
        np.testing.assert_allclose(np.array(result.zx.astype(mx.float32)), self.golden[f"{prefix}_inverse_zx"], **tol)
        np.testing.assert_allclose(np.array(result.zy.astype(mx.float32)), self.golden[f"{prefix}_inverse_zy"], **tol)
        np.testing.assert_allclose(np.array(result.zz.astype(mx.float32)), self.golden[f"{prefix}_inverse_zz"], **tol)

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((4,), "float32"),
            ((2, 3), "float32"),
            ((4,), "float16"),
            ((4,), "bfloat16"),
        ],
    )
    def test_rot3_compose(self, shape, dtype):
        """Test Rot3Array composition (R @ R) matches JAX."""
        from alphafold3_mlx.geometry import Rot3Array

        prefix = f"rot3_{shape}_{dtype}"
        tol = TOLERANCES[dtype]
        mlx_dtype = getattr(mx, dtype)

        # Load rotation matrix from golden
        rot = Rot3Array(
            xx=mx.array(self.golden[f"{prefix}_rot_xx"]).astype(mlx_dtype),
            xy=mx.array(self.golden[f"{prefix}_rot_xy"]).astype(mlx_dtype),
            xz=mx.array(self.golden[f"{prefix}_rot_xz"]).astype(mlx_dtype),
            yx=mx.array(self.golden[f"{prefix}_rot_yx"]).astype(mlx_dtype),
            yy=mx.array(self.golden[f"{prefix}_rot_yy"]).astype(mlx_dtype),
            yz=mx.array(self.golden[f"{prefix}_rot_yz"]).astype(mlx_dtype),
            zx=mx.array(self.golden[f"{prefix}_rot_zx"]).astype(mlx_dtype),
            zy=mx.array(self.golden[f"{prefix}_rot_zy"]).astype(mlx_dtype),
            zz=mx.array(self.golden[f"{prefix}_rot_zz"]).astype(mlx_dtype),
        )

        # Compose: R @ R (90-deg + 90-deg = 180-deg)
        result = rot @ rot

        np.testing.assert_allclose(np.array(result.xx.astype(mx.float32)), self.golden[f"{prefix}_compose_xx"], **tol)
        np.testing.assert_allclose(np.array(result.xy.astype(mx.float32)), self.golden[f"{prefix}_compose_xy"], **tol)
        np.testing.assert_allclose(np.array(result.xz.astype(mx.float32)), self.golden[f"{prefix}_compose_xz"], **tol)
        np.testing.assert_allclose(np.array(result.yx.astype(mx.float32)), self.golden[f"{prefix}_compose_yx"], **tol)
        np.testing.assert_allclose(np.array(result.yy.astype(mx.float32)), self.golden[f"{prefix}_compose_yy"], **tol)
        np.testing.assert_allclose(np.array(result.yz.astype(mx.float32)), self.golden[f"{prefix}_compose_yz"], **tol)
        np.testing.assert_allclose(np.array(result.zx.astype(mx.float32)), self.golden[f"{prefix}_compose_zx"], **tol)
        np.testing.assert_allclose(np.array(result.zy.astype(mx.float32)), self.golden[f"{prefix}_compose_zy"], **tol)
        np.testing.assert_allclose(np.array(result.zz.astype(mx.float32)), self.golden[f"{prefix}_compose_zz"], **tol)

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((4,), "float32"),
            ((2, 3), "float32"),
            ((4,), "float16"),
            ((4,), "bfloat16"),
        ],
    )
    def test_rot3_from_svd_properties(self, shape, dtype):
        """Test Rot3Array.from_svd produces valid rotation.

        from_svd projects an arbitrary 3x3 matrix to SO(3). We test that the
        result is a valid rotation matrix (orthogonal, det=1) for all dtypes.
        """
        from alphafold3_mlx.geometry import Rot3Array

        prefix = f"rot3_{shape}_{dtype}"
        tol = TOLERANCES[dtype]
        mlx_dtype = getattr(mx, dtype)

        # Load non-orthogonal input matrix
        svd_input = mx.array(self.golden[f"{prefix}_svd_input"]).astype(mlx_dtype)

        # Apply from_svd
        result = Rot3Array.from_svd(svd_input)

        # Convert to dense matrix for property checks
        R = result.to_array()  # [..., 3, 3]

        # Property 1: Orthogonality (R @ R^T = I)
        R_T = mx.swapaxes(R, -2, -1)
        RRT = mx.matmul(R, R_T)
        I = mx.eye(3, dtype=mlx_dtype)
        I_broadcast = mx.broadcast_to(I, RRT.shape)

        np.testing.assert_allclose(
            np.array(RRT.astype(mx.float32)),
            np.array(I_broadcast.astype(mx.float32)),
            **tol,
            err_msg="from_svd result is not orthogonal"
        )

        # Property 2: Determinant = 1 (proper rotation, not reflection)
        # Compute 3x3 determinant manually: det = a(ei-fh) - b(di-fg) + c(dh-eg)
        # For R[..., 3, 3], indices are R[..., row, col]
        a, b, c = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
        d, e, f = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
        g, h, i = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

        np.testing.assert_allclose(
            np.array(det.astype(mx.float32)),
            1.0,
            **tol,
            err_msg="from_svd result has determinant != 1"
        )

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((4,), "float32"),
            ((2, 3), "float32"),
            ((4,), "float16"),
            ((2, 3), "float16"),
            ((4,), "bfloat16"),
            ((2, 3), "bfloat16"),
        ],
    )
    def test_rot3_from_svd_jax_parity(self, shape, dtype):
        """Test Rot3Array.from_svd matches JAX reference output.

        This test validates numerical parity with the JAX quaternion-based
        from_svd algorithm for all supported dtypes (float32, float16, bfloat16).
        """
        from alphafold3_mlx.geometry import Rot3Array

        prefix = f"rot3_{shape}_{dtype}"
        tol = TOLERANCES[dtype]
        mlx_dtype = getattr(mx, dtype)

        # Load non-orthogonal input matrix
        svd_input = mx.array(self.golden[f"{prefix}_svd_input"]).astype(mlx_dtype)

        # Apply from_svd
        result = Rot3Array.from_svd(svd_input)

        # Compare against JAX golden output
        np.testing.assert_allclose(
            np.array(result.xx.astype(mx.float32)),
            self.golden[f"{prefix}_svd_result_xx"],
            **tol,
            err_msg="from_svd xx component doesn't match JAX"
        )
        np.testing.assert_allclose(
            np.array(result.xy.astype(mx.float32)),
            self.golden[f"{prefix}_svd_result_xy"],
            **tol,
            err_msg="from_svd xy component doesn't match JAX"
        )
        np.testing.assert_allclose(
            np.array(result.xz.astype(mx.float32)),
            self.golden[f"{prefix}_svd_result_xz"],
            **tol,
            err_msg="from_svd xz component doesn't match JAX"
        )
        np.testing.assert_allclose(
            np.array(result.yx.astype(mx.float32)),
            self.golden[f"{prefix}_svd_result_yx"],
            **tol,
            err_msg="from_svd yx component doesn't match JAX"
        )
        np.testing.assert_allclose(
            np.array(result.yy.astype(mx.float32)),
            self.golden[f"{prefix}_svd_result_yy"],
            **tol,
            err_msg="from_svd yy component doesn't match JAX"
        )
        np.testing.assert_allclose(
            np.array(result.yz.astype(mx.float32)),
            self.golden[f"{prefix}_svd_result_yz"],
            **tol,
            err_msg="from_svd yz component doesn't match JAX"
        )
        np.testing.assert_allclose(
            np.array(result.zx.astype(mx.float32)),
            self.golden[f"{prefix}_svd_result_zx"],
            **tol,
            err_msg="from_svd zx component doesn't match JAX"
        )
        np.testing.assert_allclose(
            np.array(result.zy.astype(mx.float32)),
            self.golden[f"{prefix}_svd_result_zy"],
            **tol,
            err_msg="from_svd zy component doesn't match JAX"
        )
        np.testing.assert_allclose(
            np.array(result.zz.astype(mx.float32)),
            self.golden[f"{prefix}_svd_result_zz"],
            **tol,
            err_msg="from_svd zz component doesn't match JAX"
        )


class TestModulesJaxParity:
    """Tests for neural network modules parity with JAX."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load golden outputs."""
        skip_if_no_golden()
        golden_file = GOLDEN_DIR / "modules_outputs.npz"
        if not golden_file.exists():
            pytest.skip(f"Golden file not found: {golden_file}")
        self.golden = np.load(golden_file)

    def test_linear_parity_float32(self):
        """Test Linear forward pass matches JAX."""
        from alphafold3_mlx.modules import Linear

        tol = TOLERANCES["float32"]

        # Load golden data
        x = mx.array(self.golden["linear_input"])
        w = mx.array(self.golden["linear_weight"])
        expected = self.golden["linear_output"]

        # Create layer and set weights
        input_dim = w.shape[0]
        output_dim = w.shape[1]
        layer = Linear(output_dim, input_dims=input_dim, initializer="zeros")
        layer.weight = w

        # Forward pass
        result = layer(x)

        np.testing.assert_allclose(np.array(result), expected, **tol)

    def test_linear_parity_float16(self):
        """Test Linear with float16 matches JAX float16 computation.

        Uses dtype-specific golden outputs for proper reduced-precision parity.
        """
        from alphafold3_mlx.modules import Linear

        # Spec tolerances for float16 (from Phase 0 findings)
        tol = TOLERANCES["float16"]

        # Load dtype-specific golden data
        x = mx.array(self.golden["linear_float16_input"]).astype(mx.float16)
        w = mx.array(self.golden["linear_float16_weight"]).astype(mx.float16)
        expected = self.golden["linear_float16_output"]

        input_dim = w.shape[0]
        output_dim = w.shape[1]
        layer = Linear(output_dim, input_dims=input_dim, precision="highest", initializer="zeros")
        layer.weight = w

        result = layer(x)

        # Compare against dtype-specific golden
        np.testing.assert_allclose(np.array(result.astype(mx.float32)), expected, **tol)

        # Verify no NaN/Inf
        assert not mx.isnan(result).any().item()
        assert not mx.isinf(result).any().item()

    def test_layernorm_parity_float32(self):
        """Test LayerNorm forward pass matches JAX."""
        from alphafold3_mlx.modules import LayerNorm

        tol = TOLERANCES["float32"]

        # Load golden data
        x = mx.array(self.golden["layernorm_input"])
        expected = self.golden["layernorm_output"]

        # Create layer without learnable params (to match golden)
        dims = x.shape[-1]
        layer = LayerNorm(dims, create_scale=False, create_offset=False)

        result = layer(x)

        np.testing.assert_allclose(np.array(result), expected, **tol)

    def test_layernorm_parity_float16(self):
        """Test LayerNorm with float16 matches JAX float16 computation.

        Uses dtype-specific golden outputs for proper reduced-precision parity.
        """
        from alphafold3_mlx.modules import LayerNorm

        tol = TOLERANCES["float16"]

        # Load dtype-specific golden data
        x = mx.array(self.golden["layernorm_float16_input"]).astype(mx.float16)
        expected = self.golden["layernorm_float16_output"]

        dims = x.shape[-1]
        layer = LayerNorm(dims, create_scale=False, create_offset=False, upcast=True)

        result = layer(x)

        # Compare against dtype-specific golden
        np.testing.assert_allclose(np.array(result.astype(mx.float32)), expected, **tol)

        # Verify no NaN/Inf
        assert not mx.isnan(result).any().item()
        assert not mx.isinf(result).any().item()

    def test_layernorm_parity_bfloat16(self):
        """Test LayerNorm with bfloat16 matches JAX bfloat16 computation.

        Uses dtype-specific golden outputs for proper reduced-precision parity.
        """
        from alphafold3_mlx.modules import LayerNorm

        tol = TOLERANCES["bfloat16"]

        # Load dtype-specific golden data
        x = mx.array(self.golden["layernorm_bfloat16_input"]).astype(mx.bfloat16)
        expected = self.golden["layernorm_bfloat16_output"]

        dims = x.shape[-1]
        layer = LayerNorm(dims, create_scale=False, create_offset=False, upcast=True)

        result = layer(x)

        # Compare against dtype-specific golden
        np.testing.assert_allclose(np.array(result.astype(mx.float32)), expected, **tol)

        # Verify no NaN/Inf
        assert not mx.isnan(result).any().item()
        assert not mx.isinf(result).any().item()

    def test_glu_parity_float32(self):
        """Test GatedLinearUnit forward pass matches JAX."""
        from alphafold3_mlx.modules import GatedLinearUnit

        tol = TOLERANCES["float32"]

        # Load golden data
        x = mx.array(self.golden["glu_input"])
        w = mx.array(self.golden["glu_weight"])
        expected = self.golden["glu_output"]

        input_dim = w.shape[0]
        output_dim = w.shape[1] // 2  # GLU projects to 2x output
        glu = GatedLinearUnit(input_dim, output_dim, activation="swish")
        glu.linear.weight = w

        result = glu(x)

        np.testing.assert_allclose(np.array(result), expected, **tol)

    def test_glu_parity_float16(self):
        """Test GatedLinearUnit with float16 matches JAX float16 computation."""
        from alphafold3_mlx.modules import GatedLinearUnit

        tol = TOLERANCES["float16"]

        # Load dtype-specific golden data
        x = mx.array(self.golden["glu_float16_input"]).astype(mx.float16)
        w = mx.array(self.golden["glu_float16_weight"]).astype(mx.float16)
        expected = self.golden["glu_float16_output"]

        input_dim = w.shape[0]
        output_dim = w.shape[1] // 2
        glu = GatedLinearUnit(input_dim, output_dim, activation="swish", precision="highest")
        glu.linear.weight = w

        result = glu(x)

        np.testing.assert_allclose(np.array(result.astype(mx.float32)), expected, **tol)

        # Verify no NaN/Inf
        assert not mx.isnan(result).any().item()
        assert not mx.isinf(result).any().item()

    def test_linear_parity_bfloat16(self):
        """Test Linear with bfloat16 matches JAX bfloat16 computation.

        Uses dtype-specific golden outputs for proper reduced-precision parity.
        """
        from alphafold3_mlx.modules import Linear

        # Spec tolerances for bfloat16 (from Phase 0 findings)
        tol = TOLERANCES["bfloat16"]

        # Load dtype-specific golden data
        x = mx.array(self.golden["linear_bfloat16_input"]).astype(mx.bfloat16)
        w = mx.array(self.golden["linear_bfloat16_weight"]).astype(mx.bfloat16)
        expected = self.golden["linear_bfloat16_output"]

        input_dim = w.shape[0]
        output_dim = w.shape[1]
        layer = Linear(output_dim, input_dims=input_dim, precision="highest", initializer="zeros")
        layer.weight = w

        result = layer(x)

        # Compare against dtype-specific golden
        np.testing.assert_allclose(np.array(result.astype(mx.float32)), expected, **tol)

        # Verify no NaN/Inf
        assert not mx.isnan(result).any().item()
        assert not mx.isinf(result).any().item()

    def test_glu_parity_bfloat16(self):
        """Test GatedLinearUnit with bfloat16 matches JAX bfloat16 computation.

        Uses dtype-specific golden outputs for proper reduced-precision parity.
        GLU has chained operations (einsum → split → sigmoid → multiply) which
        accumulate rounding errors in bfloat16's 7-bit mantissa.
        """
        from alphafold3_mlx.modules import GatedLinearUnit

        # Slightly relaxed tolerance for GLU due to chained operations
        # Base bfloat16 spec: rtol=5e-3, atol=5e-3
        # GLU adds: einsum + split + sigmoid + multiply = ~4x error accumulation
        tol = {"rtol": 2e-2, "atol": 4e-2}

        # Load dtype-specific golden data
        x = mx.array(self.golden["glu_bfloat16_input"]).astype(mx.bfloat16)
        w = mx.array(self.golden["glu_bfloat16_weight"]).astype(mx.bfloat16)
        expected = self.golden["glu_bfloat16_output"]

        input_dim = w.shape[0]
        output_dim = w.shape[1] // 2
        glu = GatedLinearUnit(input_dim, output_dim, activation="swish", precision="highest")
        glu.linear.weight = w

        result = glu(x)

        # Compare against dtype-specific golden
        np.testing.assert_allclose(np.array(result.astype(mx.float32)), expected, **tol)

        # Verify no NaN/Inf
        assert not mx.isnan(result).any().item()
        assert not mx.isinf(result).any().item()


class TestScaleRequirements:
    """Tests for scale requirements (1, 5000 shape)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Check golden outputs exist."""
        skip_if_no_golden()

    def test_vec3_scale_5000(self):
        """Test Vec3Array operations at scale (1, 5000)."""
        from alphafold3_mlx.geometry import Vec3Array

        golden_file = GOLDEN_DIR / "vec3array_ops.npz"
        if not golden_file.exists():
            pytest.skip(f"Golden file not found: {golden_file}")

        golden = np.load(golden_file)
        prefix = "vec3_(1, 5000)_float32"

        # Check the shape is correct
        assert golden[f"{prefix}_v1_x"].shape == (1, 5000)

        # Verify MLX can handle this scale
        v1 = Vec3Array(
            x=mx.array(golden[f"{prefix}_v1_x"]),
            y=mx.array(golden[f"{prefix}_v1_y"]),
            z=mx.array(golden[f"{prefix}_v1_z"]),
        )
        v2 = Vec3Array(
            x=mx.array(golden[f"{prefix}_v2_x"]),
            y=mx.array(golden[f"{prefix}_v2_y"]),
            z=mx.array(golden[f"{prefix}_v2_z"]),
        )

        # Run operations
        _ = v1 + v2
        _ = v1.dot(v2)
        _ = v1.cross(v2)
        _ = v1.norm()
        _ = v1.normalized()

        # If we got here without OOM or error, test passes
        assert True
