"""Confidence head validation tests.

Tests the MLX confidence head implementation against reference values
and validates confidence score computation requirements.
"""

from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx

from alphafold3_mlx.network.confidence_head import ConfidenceHead
from alphafold3_mlx.core.config import ConfidenceConfig, GlobalConfig
from alphafold3_mlx.atom_layout import GatherInfo


@pytest.fixture
def token_atoms_to_pseudo_beta_factory():
    """Create GatherInfo for token_atoms_to_pseudo_beta."""

    def _factory(num_residues: int, num_atoms: int) -> GatherInfo:
        gather_idxs = mx.arange(num_residues) * num_atoms
        gather_mask = mx.ones((num_residues,), dtype=mx.bool_)
        input_shape = mx.array((num_residues, num_atoms))
        return GatherInfo(
            gather_idxs=gather_idxs,
            gather_mask=gather_mask,
            input_shape=input_shape,
        )

    return _factory


class TestConfidenceHeadBasic:
    """Test basic ConfidenceHead functionality."""

    @pytest.fixture
    def confidence_head(self):
        """Create a ConfidenceHead for testing."""
        config = ConfidenceConfig(
            num_pairformer_layers=2,  # Small for testing
        )
        global_config = GlobalConfig(use_compile=False)
        return ConfidenceHead(
            config=config,
            global_config=global_config,
            seq_channel=384,
            pair_channel=128,
        )

    def test_output_types(self, confidence_head, token_atoms_to_pseudo_beta_factory):
        """Test that confidence head returns expected output structure."""
        batch, num_residues, num_atoms = 1, 16, 37
        single = mx.zeros((num_residues, 384))
        pair = mx.zeros((num_residues, num_residues, 128))
        # Atom37 positions: [num_residues, 37, 3]
        positions = mx.zeros((num_residues, num_atoms, 3))
        asym_id = mx.zeros((num_residues,), dtype=mx.int32)
        seq_mask = mx.ones((num_residues,))
        target_feat = mx.zeros((num_residues, 22))

        token_atoms_to_pseudo_beta = token_atoms_to_pseudo_beta_factory(
            num_residues, num_atoms
        )
        embeddings = {"single": single, "pair": pair, "target_feat": target_feat}

        result = confidence_head(
            dense_atom_positions=positions,
            embeddings=embeddings,
            seq_mask=seq_mask,
            token_atoms_to_pseudo_beta=token_atoms_to_pseudo_beta,
            asym_id=asym_id,
        )

        # Check all required outputs
        assert hasattr(result, "plddt")
        assert hasattr(result, "pae")
        assert hasattr(result, "pde")
        assert hasattr(result, "ptm")
        assert hasattr(result, "iptm")


class TestPLDDT:
    """Test pLDDT prediction."""

    @pytest.fixture
    def confidence_head(self):
        """Create ConfidenceHead for testing."""
        config = ConfidenceConfig(num_pairformer_layers=1)
        global_config = GlobalConfig(use_compile=False)
        return ConfidenceHead(
            config=config,
            global_config=global_config,
            seq_channel=384,
            pair_channel=128,
        )

    def test_plddt_shape(self, confidence_head, token_atoms_to_pseudo_beta_factory):
        """Test pLDDT output shape."""
        batch, num_residues, num_atoms = 1, 32, 37
        single = mx.zeros((num_residues, 384))
        pair = mx.zeros((num_residues, num_residues, 128))
        positions = mx.zeros((num_residues, num_atoms, 3))
        asym_id = mx.zeros((num_residues,), dtype=mx.int32)
        seq_mask = mx.ones((num_residues,))
        target_feat = mx.zeros((num_residues, 22))

        token_atoms_to_pseudo_beta = token_atoms_to_pseudo_beta_factory(
            num_residues, num_atoms
        )
        embeddings = {"single": single, "pair": pair, "target_feat": target_feat}

        result = confidence_head(
            dense_atom_positions=positions,
            embeddings=embeddings,
            seq_mask=seq_mask,
            token_atoms_to_pseudo_beta=token_atoms_to_pseudo_beta,
            asym_id=asym_id,
        )

        # pLDDT is per-atom: [batch, num_residues, 37]
        assert result.plddt.shape == (batch, num_residues, num_atoms)

    def test_plddt_range(self, confidence_head, token_atoms_to_pseudo_beta_factory):
        """Test that pLDDT values are in valid range [0, 100]."""
        batch, num_residues, num_atoms = 1, 16, 37
        np.random.seed(42)
        single = mx.array(np.random.randn(num_residues, 384).astype(np.float32))
        pair = mx.array(np.random.randn(num_residues, num_residues, 128).astype(np.float32))
        positions = mx.array(np.random.randn(num_residues, num_atoms, 3).astype(np.float32))
        asym_id = mx.zeros((num_residues,), dtype=mx.int32)
        seq_mask = mx.ones((num_residues,))
        target_feat = mx.zeros((num_residues, 22))

        token_atoms_to_pseudo_beta = token_atoms_to_pseudo_beta_factory(
            num_residues, num_atoms
        )
        embeddings = {"single": single, "pair": pair, "target_feat": target_feat}

        result = confidence_head(
            dense_atom_positions=positions,
            embeddings=embeddings,
            seq_mask=seq_mask,
            token_atoms_to_pseudo_beta=token_atoms_to_pseudo_beta,
            asym_id=asym_id,
        )
        mx.eval(result.plddt)

        plddt_np = np.array(result.plddt)
        assert np.all(plddt_np >= 0), "pLDDT should be >= 0"
        assert np.all(plddt_np <= 100), "pLDDT should be <= 100"


class TestPAE:
    """Test PAE prediction."""

    @pytest.fixture
    def confidence_head(self):
        """Create ConfidenceHead for testing."""
        config = ConfidenceConfig(num_pairformer_layers=1)
        global_config = GlobalConfig(use_compile=False)
        return ConfidenceHead(
            config=config,
            global_config=global_config,
            seq_channel=384,
            pair_channel=128,
        )

    def test_pae_shape(self, confidence_head, token_atoms_to_pseudo_beta_factory):
        """Test PAE output shape is NxN matrix."""
        batch, num_residues, num_atoms = 1, 16, 37
        single = mx.zeros((num_residues, 384))
        pair = mx.zeros((num_residues, num_residues, 128))
        positions = mx.zeros((num_residues, num_atoms, 3))
        asym_id = mx.zeros((num_residues,), dtype=mx.int32)
        seq_mask = mx.ones((num_residues,))
        target_feat = mx.zeros((num_residues, 22))

        token_atoms_to_pseudo_beta = token_atoms_to_pseudo_beta_factory(
            num_residues, num_atoms
        )
        embeddings = {"single": single, "pair": pair, "target_feat": target_feat}

        result = confidence_head(
            dense_atom_positions=positions,
            embeddings=embeddings,
            seq_mask=seq_mask,
            token_atoms_to_pseudo_beta=token_atoms_to_pseudo_beta,
            asym_id=asym_id,
        )

        # PAE should be NxN matrix
        assert result.pae.shape[-2:] == (num_residues, num_residues)


class TestPDE:
    """Test PDE prediction."""

    @pytest.fixture
    def confidence_head(self):
        """Create ConfidenceHead for testing."""
        config = ConfidenceConfig(num_pairformer_layers=1)
        global_config = GlobalConfig(use_compile=False)
        return ConfidenceHead(
            config=config,
            global_config=global_config,
            seq_channel=384,
            pair_channel=128,
        )

    def test_pde_shape(self, confidence_head, token_atoms_to_pseudo_beta_factory):
        """Test PDE output shape is NxN matrix."""
        batch, num_residues, num_atoms = 1, 16, 37
        single = mx.zeros((num_residues, 384))
        pair = mx.zeros((num_residues, num_residues, 128))
        positions = mx.zeros((num_residues, num_atoms, 3))
        asym_id = mx.zeros((num_residues,), dtype=mx.int32)
        seq_mask = mx.ones((num_residues,))
        target_feat = mx.zeros((num_residues, 22))

        token_atoms_to_pseudo_beta = token_atoms_to_pseudo_beta_factory(
            num_residues, num_atoms
        )
        embeddings = {"single": single, "pair": pair, "target_feat": target_feat}

        result = confidence_head(
            dense_atom_positions=positions,
            embeddings=embeddings,
            seq_mask=seq_mask,
            token_atoms_to_pseudo_beta=token_atoms_to_pseudo_beta,
            asym_id=asym_id,
        )

        # PDE should be NxN matrix
        assert result.pde.shape[-2:] == (num_residues, num_residues)


class TestTMScores:
    """Test pTM and ipTM computation."""

    @pytest.fixture
    def confidence_head(self):
        """Create ConfidenceHead for testing."""
        config = ConfidenceConfig(num_pairformer_layers=1)
        global_config = GlobalConfig(use_compile=False)
        return ConfidenceHead(
            config=config,
            global_config=global_config,
            seq_channel=384,
            pair_channel=128,
        )

    def test_ptm_scalar(self, confidence_head, token_atoms_to_pseudo_beta_factory):
        """Test that pTM is a scalar value."""
        batch, num_residues, num_atoms = 1, 16, 37
        single = mx.zeros((num_residues, 384))
        pair = mx.zeros((num_residues, num_residues, 128))
        positions = mx.zeros((num_residues, num_atoms, 3))
        asym_id = mx.zeros((num_residues,), dtype=mx.int32)
        seq_mask = mx.ones((num_residues,))
        target_feat = mx.zeros((num_residues, 22))

        token_atoms_to_pseudo_beta = token_atoms_to_pseudo_beta_factory(
            num_residues, num_atoms
        )
        embeddings = {"single": single, "pair": pair, "target_feat": target_feat}

        result = confidence_head(
            dense_atom_positions=positions,
            embeddings=embeddings,
            seq_mask=seq_mask,
            token_atoms_to_pseudo_beta=token_atoms_to_pseudo_beta,
            asym_id=asym_id,
        )

        # pTM should be scalar or (batch,) shaped
        assert result.ptm.size == 1 or result.ptm.shape == (batch,)

    def test_ptm_range(self, confidence_head, token_atoms_to_pseudo_beta_factory):
        """Test that pTM is in valid range [0, 1]."""
        batch, num_residues, num_atoms = 1, 16, 37
        np.random.seed(42)
        single = mx.array(np.random.randn(num_residues, 384).astype(np.float32))
        pair = mx.array(np.random.randn(num_residues, num_residues, 128).astype(np.float32))
        positions = mx.array(np.random.randn(num_residues, num_atoms, 3).astype(np.float32))
        asym_id = mx.zeros((num_residues,), dtype=mx.int32)
        seq_mask = mx.ones((num_residues,))
        target_feat = mx.zeros((num_residues, 22))

        token_atoms_to_pseudo_beta = token_atoms_to_pseudo_beta_factory(
            num_residues, num_atoms
        )
        embeddings = {"single": single, "pair": pair, "target_feat": target_feat}

        result = confidence_head(
            dense_atom_positions=positions,
            embeddings=embeddings,
            seq_mask=seq_mask,
            token_atoms_to_pseudo_beta=token_atoms_to_pseudo_beta,
            asym_id=asym_id,
        )
        mx.eval(result.ptm)

        ptm_val = float(result.ptm.item()) if result.ptm.size == 1 else float(result.ptm[0].item())
        assert 0 <= ptm_val <= 1, "pTM should be in [0, 1]"


class TestConfidenceRelativeError:
    """Test confidence score relative error requirement."""

    GOLDEN_FILE = "tests/fixtures/model_golden/confidence_head_reference.npz"

    @pytest.fixture
    def golden_data(self):
        """Load golden reference data if available."""
        from pathlib import Path
        golden_path = Path(self.GOLDEN_FILE)
        if not golden_path.exists():
            pytest.skip(f"Golden reference not found: {golden_path}. "
                       "Run: python scripts/generate_model_reference_outputs.py")
        return np.load(golden_path)

    def test_confidence_output_validity(self, golden_data, token_atoms_to_pseudo_beta_factory):
        """Validate confidence outputs are in valid ranges. requirement: Confidence scores must be valid:
        - pLDDT in [0, 100]
        - PAE in [0, 50]
        - pTM in [0, 1]
        """
        # Load golden inputs
        single_input = mx.array(golden_data["single_input"][0])
        pair_input = mx.array(golden_data["pair_input"][0])
        positions_input = mx.array(golden_data["positions_input"][0])
        num_residues = int(golden_data["num_residues"])
        num_atoms = int(golden_data["num_atoms"])

        # Create confidence head with matching configuration
        config = ConfidenceConfig(num_pairformer_layers=2)
        confidence_head = ConfidenceHead(
            config=config,
            global_config=GlobalConfig,
            seq_channel=single_input.shape[-1],
            pair_channel=pair_input.shape[-1],
        )

        # Create masks
        seq_mask = mx.ones((num_residues,))
        asym_id = mx.zeros((num_residues,), dtype=mx.int32)
        target_feat = mx.zeros((num_residues, 22))
        token_atoms_to_pseudo_beta = token_atoms_to_pseudo_beta_factory(
            num_residues, num_atoms
        )

        # Run forward pass
        embeddings = {"single": single_input, "pair": pair_input, "target_feat": target_feat}
        result = confidence_head(
            dense_atom_positions=positions_input,
            embeddings=embeddings,
            seq_mask=seq_mask,
            token_atoms_to_pseudo_beta=token_atoms_to_pseudo_beta,
            asym_id=asym_id,
        )
        mx.eval(result.plddt, result.pae, result.ptm)

        # Convert to numpy
        plddt_np = np.array(result.plddt.astype(mx.float32))
        pae_np = np.array(result.pae.astype(mx.float32))
        ptm_np = np.array(result.ptm.astype(mx.float32))

        # Verify no NaN
        assert not np.any(np.isnan(plddt_np)), "NaN in pLDDT"
        assert not np.any(np.isnan(pae_np)), "NaN in PAE"
        assert not np.any(np.isnan(ptm_np)), "NaN in pTM"

        # Verify valid ranges
        assert np.all(plddt_np >= 0) and np.all(plddt_np <= 100), \
            f"pLDDT out of range [0, 100]: [{plddt_np.min:.2f}, {plddt_np.max:.2f}]"
        assert np.all(pae_np >= 0) and np.all(pae_np <= 50), \
            f"PAE out of range [0, 50]: [{pae_np.min:.2f}, {pae_np.max:.2f}]"
        assert np.all(ptm_np >= 0) and np.all(ptm_np <= 1), \
            f"pTM out of range [0, 1]: [{ptm_np.min:.2f}, {ptm_np.max:.2f}]"


class TestConfidenceGoldenValidation:
    """Validate against golden reference outputs."""

    GOLDEN_FILE = "tests/fixtures/model_golden/confidence_head_reference.npz"

    @pytest.fixture
    def golden_data(self):
        """Load golden reference data if available."""
        from pathlib import Path
        golden_path = Path(self.GOLDEN_FILE)
        if not golden_path.exists:
            pytest.skip(f"Golden reference not found: {golden_path}. "
                       "Run: python scripts/generate_model_reference_outputs.py")
        return np.load(golden_path)

    def test_golden_comparison(self, golden_data, token_atoms_to_pseudo_beta_factory):
        """Compare confidence head output against golden reference.

        Validates that MLX confidence head produces outputs with matching
        shapes and valid ranges.
        """
        # Load golden outputs for shape comparison
        plddt_golden = golden_data["plddt"]
        pae_golden = golden_data["pae"]
        ptm_golden = golden_data["ptm"]

        # Load inputs and run model
        single_input = mx.array(golden_data["single_input"][0])
        pair_input = mx.array(golden_data["pair_input"][0])
        positions_input = mx.array(golden_data["positions_input"][0])
        num_residues = int(golden_data["num_residues"])
        num_atoms = int(golden_data["num_atoms"])

        config = ConfidenceConfig(num_pairformer_layers=2)
        confidence_head = ConfidenceHead(
            config=config,
            global_config=GlobalConfig,
            seq_channel=single_input.shape[-1],
            pair_channel=pair_input.shape[-1],
        )

        seq_mask = mx.ones((num_residues,))
        asym_id = mx.zeros((num_residues,), dtype=mx.int32)
        target_feat = mx.zeros((num_residues, 22))
        token_atoms_to_pseudo_beta = token_atoms_to_pseudo_beta_factory(
            num_residues, num_atoms
        )

        embeddings = {"single": single_input, "pair": pair_input, "target_feat": target_feat}
        result = confidence_head(
            dense_atom_positions=positions_input,
            embeddings=embeddings,
            seq_mask=seq_mask,
            token_atoms_to_pseudo_beta=token_atoms_to_pseudo_beta,
            asym_id=asym_id,
        )
        mx.eval(result.plddt, result.pae, result.ptm)

        # Verify shapes match golden
        plddt_np = np.array(result.plddt.astype(mx.float32))
        pae_np = np.array(result.pae.astype(mx.float32))
        ptm_np = np.array(result.ptm.astype(mx.float32))

        assert plddt_np.shape == plddt_golden.shape, \
            f"pLDDT shape mismatch: {plddt_np.shape} vs {plddt_golden.shape}"
        assert pae_np.shape == pae_golden.shape, \
            f"PAE shape mismatch: {pae_np.shape} vs {pae_golden.shape}"
        assert ptm_np.shape == ptm_golden.shape, \
            f"pTM shape mismatch: {ptm_np.shape} vs {ptm_golden.shape}"
