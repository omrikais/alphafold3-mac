"""Tests for Model.compile functionality.

Verifies that:
- compile() can be called without errors
- Compiled model can run inference
- Compilation is optional (controlled by GlobalConfig.use_compile)
"""

from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx

from alphafold3_mlx.model import Model
from alphafold3_mlx.core import ModelConfig, FeatureBatch
from alphafold3_mlx.core.config import EvoformerConfig, DiffusionConfig, GlobalConfig


def create_test_batch(num_residues: int = 8, seed: int = 42) -> FeatureBatch:
    """Create minimal feature batch for testing."""
    np.random.seed(seed)

    feature_dict = {
        "aatype": np.random.randint(0, 20, size=num_residues).astype(np.int32),
        "token_mask": np.ones(num_residues, dtype=np.float32),
        "residue_index": np.arange(num_residues, dtype=np.int32),
        "asym_id": np.zeros(num_residues, dtype=np.int32),
        "entity_id": np.zeros(num_residues, dtype=np.int32),
        "sym_id": np.zeros(num_residues, dtype=np.int32),
    }

    return FeatureBatch.from_numpy(feature_dict)


def create_small_model(use_compile: bool = True) -> Model:
    """Create a small model for testing."""
    config = ModelConfig(
        evoformer=EvoformerConfig(
            num_pairformer_layers=2,
            use_msa_stack=False,
        ),
        diffusion=DiffusionConfig(
            num_steps=3,
            num_samples=1,
            num_transformer_blocks=4,
        ),
        global_config=GlobalConfig(
            precision="float32",
            use_compile=use_compile,
        ),
        num_recycles=1,
    )
    return Model(config)


class TestModelCompile:
    """Test Model.compile() method."""

    def test_compile_no_crash(self):
        """Test that compile() can be called without errors."""
        model = create_small_model(use_compile=True)

        # Should not raise
        model.compile()

        # Verify state
        assert model._compiled is True
        assert hasattr(model, '_compiled_evoformer')
        # DiffusionHead's internal transformer should also be compiled
        assert model.diffusion_head._compiled is True
        assert model.diffusion_head._compiled_denoise_step is not None
        # Note: ConfidenceHead is not compiled because it takes GatherInfo
        # which mx.compile cannot handle

    def test_compile_disabled(self):
        """Test that compile() respects use_compile=False."""
        model = create_small_model(use_compile=False)

        model.compile()

        # Should be marked compiled but without actual compiled functions
        assert model._compiled is True
        assert not hasattr(model, '_compiled_evoformer')
        # DiffusionHead should also respect use_compile=False
        assert model.diffusion_head._compiled is True  # Flag set
        assert model.diffusion_head._compiled_denoise_step is None  # No actual compilation

    def test_compile_idempotent(self):
        """Test that compile() can be called multiple times safely."""
        model = create_small_model(use_compile=True)

        model.compile()
        model.compile()  # Second call should be no-op

        assert model._compiled is True

    def test_compiled_model_inference(self):
        """Test that compiled model can run inference."""
        model = create_small_model(use_compile=True)
        model.compile()

        batch = create_test_batch(num_residues=6)
        key = mx.random.key(42)

        # Run inference - should not crash
        result = model(batch, key)
        mx.eval(result.atom_positions.positions)

        # Basic sanity check
        coords = np.array(result.atom_positions.positions)
        assert coords.shape[0] == 1  # num_samples
        assert coords.shape[1] == 6  # num_residues
        assert not np.any(np.isnan(coords)), "Compiled model produced NaN"

    def test_uncompiled_model_inference(self):
        """Test that uncompiled model works identically."""
        model = create_small_model(use_compile=False)
        model.compile()  # No-op since use_compile=False

        batch = create_test_batch(num_residues=6)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.atom_positions.positions)

        coords = np.array(result.atom_positions.positions)
        assert coords.shape[0] == 1
        assert coords.shape[1] == 6
        assert not np.any(np.isnan(coords))


class TestDiffusionHeadCompile:
    """Test DiffusionHead.compile method."""

    def test_diffusion_head_compile_standalone(self):
        """Test that DiffusionHead can be compiled independently."""
        from alphafold3_mlx.network.diffusion_head import DiffusionHead
        from alphafold3_mlx.core.config import DiffusionConfig, GlobalConfig

        # num_transformer_blocks must be divisible by super_block_size (4)
        config = DiffusionConfig(num_transformer_blocks=4)
        global_config = GlobalConfig(use_compile=True)

        head = DiffusionHead(config=config, global_config=global_config)

        # Should not raise
        head.compile()

        assert head._compiled is True
        assert head._compiled_denoise_step is not None

    def test_diffusion_head_compile_disabled(self):
        """Test that DiffusionHead respects use_compile=False."""
        from alphafold3_mlx.network.diffusion_head import DiffusionHead
        from alphafold3_mlx.core.config import DiffusionConfig, GlobalConfig

        # num_transformer_blocks must be divisible by super_block_size (4)
        config = DiffusionConfig(num_transformer_blocks=4)
        global_config = GlobalConfig(use_compile=False)

        head = DiffusionHead(config=config, global_config=global_config)
        head.compile()

        assert head._compiled is True
        assert head._compiled_denoise_step is None

    def test_diffusion_head_compile_idempotent(self):
        """Test that DiffusionHead.compile() is idempotent."""
        from alphafold3_mlx.network.diffusion_head import DiffusionHead
        from alphafold3_mlx.core.config import DiffusionConfig, GlobalConfig

        # num_transformer_blocks must be divisible by super_block_size (4)
        config = DiffusionConfig(num_transformer_blocks=4)
        global_config = GlobalConfig(use_compile=True)

        head = DiffusionHead(config=config, global_config=global_config)

        head.compile()
        first_transformer = head._compiled_denoise_step

        head.compile()  # Second call should be no-op

        assert head._compiled_denoise_step is first_transformer


class TestConfidenceHeadCompile:
    """Test ConfidenceHead.compile method."""

    def test_confidence_head_compile_standalone(self):
        """Test that ConfidenceHead.compile() can be called without errors."""
        from alphafold3_mlx.network.confidence_head import ConfidenceHead
        from alphafold3_mlx.core.config import ConfidenceConfig, GlobalConfig

        config = ConfidenceConfig()
        global_config = GlobalConfig(use_compile=True)

        head = ConfidenceHead(config=config, global_config=global_config)

        # Should not raise
        head.compile()

        # Verify _compiled flag is set (API consistency)
        assert head._compiled is True
        # Verify no _compiled_denoise_step (GatherInfo limitation documented)
        assert not hasattr(head, '_compiled_denoise_step')

    def test_confidence_head_compile_disabled(self):
        """Test that ConfidenceHead respects compile being disabled."""
        from alphafold3_mlx.network.confidence_head import ConfidenceHead
        from alphafold3_mlx.core.config import ConfidenceConfig, GlobalConfig

        config = ConfidenceConfig()
        global_config = GlobalConfig(use_compile=False)

        head = ConfidenceHead(config=config, global_config=global_config)
        head.compile()

        # Flag still gets set for API consistency
        assert head._compiled is True

    def test_confidence_head_compile_idempotent(self):
        """Test that ConfidenceHead.compile() is idempotent."""
        from alphafold3_mlx.network.confidence_head import ConfidenceHead
        from alphafold3_mlx.core.config import ConfidenceConfig, GlobalConfig

        config = ConfidenceConfig()
        global_config = GlobalConfig(use_compile=True)

        head = ConfidenceHead(config=config, global_config=global_config)

        head.compile()
        head.compile()  # Second call should be no-op

        assert head._compiled is True

    def test_confidence_head_gatherinfo_separation_documented(self):
        """Verify ConfidenceHead documents GatherInfo separation for compilation."""
        from alphafold3_mlx.network.confidence_head import ConfidenceHead

        # Check class docstring mentions GatherInfo-dependent separation
        assert "GatherInfo" in ConfidenceHead.__doc__
        assert "can be compiled" in ConfidenceHead.__doc__.lower()

        # Check compile method docstring documents the array-only approach
        assert "array-only" in ConfidenceHead.compile.__doc__.lower()
        assert "layout_convert" in ConfidenceHead.compile.__doc__


class TestCompileCoverage:
    """Test compile coverage documentation."""

    def test_compile_coverage_documentation(self):
        """Verify compile coverage matches documentation."""
        model = create_small_model(use_compile=True)
        model.compile()

        # Evoformer: Fully compiled
        assert hasattr(model, '_compiled_evoformer')

        # DiffusionHead: Compiled denoise step
        assert model.diffusion_head._compiled_denoise_step is not None

        # ConfidenceHead: Compiled confidence forward
        assert model.confidence_head._compiled is True
        assert model.confidence_head._compiled_confidence_forward is not None

    def test_compile_coverage_all_modules_called(self):
        """Verify Model.compile() calls compile on all submodules."""
        model = create_small_model(use_compile=True)

        # Before compile
        assert model._compiled is False
        assert model.diffusion_head._compiled is False
        assert model.confidence_head._compiled is False

        model.compile()

        # After compile - all modules should have _compiled=True
        assert model._compiled is True
        assert model.diffusion_head._compiled is True
        assert model.confidence_head._compiled is True

    def test_compile_coverage_disabled_all_modules(self):
        """Verify compile with use_compile=False still sets flags."""
        model = create_small_model(use_compile=False)
        model.compile()

        # All _compiled flags should be True for API consistency
        assert model._compiled is True
        assert model.diffusion_head._compiled is True
        assert model.confidence_head._compiled is True

        # But no actual compiled functions
        assert not hasattr(model, '_compiled_evoformer')
        assert model.diffusion_head._compiled_denoise_step is None
        assert model.confidence_head._compiled_confidence_forward is None
