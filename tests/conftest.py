"""Shared pytest fixtures for alphafold3_mlx tests."""

import pytest
import numpy as np


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "arch_parity: tests requiring MLX architecture to match JAX AF3"
    )
    config.addinivalue_line(
        "markers", "synthetic: tests using synthetic baselines (NOT compliant)"
    )


# AF3 representative shapes for validation
AF3_SHAPES = [
    {"batch": 1, "heads": 4, "seq": 256, "head_dim": 64},
    {"batch": 1, "heads": 4, "seq": 512, "head_dim": 64},
    {"batch": 1, "heads": 4, "seq": 1024, "head_dim": 64},
]

# Edge case shapes
EDGE_CASE_SHAPES = [
    {"batch": 1, "heads": 4, "seq": 1, "head_dim": 64, "name": "seq_1"},
    {"batch": 1, "heads": 4, "seq_q": 256, "seq_k": 128, "head_dim": 64, "name": "cross_attention"},
    {"batch": 1, "heads": 4, "seq": 256, "head_dim": 48, "name": "non_power2_head_dim"},
]

# Tolerance thresholds by precision
TOLERANCES = {
    "float32": {"rtol": 1e-4, "atol": 1e-5},
    "float16": {"rtol": 1e-3, "atol": 1e-4},
    "bfloat16": {"rtol": 1e-3, "atol": 1e-4},
}


@pytest.fixture
def af3_shapes():
    """Return AF3 representative shapes."""
    return AF3_SHAPES.copy()


@pytest.fixture
def edge_case_shapes():
    """Return edge case shapes for testing."""
    return EDGE_CASE_SHAPES.copy()


@pytest.fixture
def tolerances():
    """Return tolerance thresholds by dtype."""
    return TOLERANCES.copy()


@pytest.fixture
def default_seed():
    """Return default random seed for reproducibility."""
    return 42


@pytest.fixture
def rng(default_seed):
    """Return numpy random generator with fixed seed."""
    return np.random.default_rng(default_seed)


@pytest.fixture
def make_attention_inputs(rng):
    """Factory fixture for creating random attention inputs."""
    def _make_inputs(
        batch: int = 1,
        heads: int = 4,
        seq_q: int = 256,
        seq_k: int | None = None,
        head_dim: int = 64,
        dtype: str = "float32",
        use_mask: bool = False,
        use_bias: bool = False,
        mask_ratio: float = 0.1,
    ) -> dict:
        if seq_k is None:
            seq_k = seq_q

        np_dtype = getattr(np, dtype)

        q = rng.standard_normal((batch, heads, seq_q, head_dim)).astype(np_dtype)
        k = rng.standard_normal((batch, heads, seq_k, head_dim)).astype(np_dtype)
        v = rng.standard_normal((batch, heads, seq_k, head_dim)).astype(np_dtype)

        boolean_mask = None
        if use_mask:
            # Create mask with some positions masked (True = attend, False = mask)
            boolean_mask = rng.random((batch, seq_k)) > mask_ratio

        additive_bias = None
        if use_bias:
            additive_bias = rng.standard_normal((batch, heads, seq_q, seq_k)).astype(np_dtype) * 0.1

        return {
            "q": q,
            "k": k,
            "v": v,
            "boolean_mask": boolean_mask,
            "additive_bias": additive_bias,
        }

    return _make_inputs


@pytest.fixture
def golden_outputs_path(tmp_path):
    """Return path to golden outputs directory."""
    return tmp_path / "golden"
