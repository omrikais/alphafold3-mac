"""Unit tests for output_handler module.

These tests verify disk space validation
and other output handling functionality.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest


class TestDiskSpaceCheck:
    """Tests for disk space validation."""

    def test_disk_space_check_sufficient_space(self) -> None:
        """Verify check_disk_space returns True when sufficient space available."""
        from alphafold3_mlx.pipeline.output_handler import check_disk_space

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Should have sufficient space for 1GB (test directories have space)
            result = check_disk_space(output_dir, required_gb=0.001)  # 1MB
            assert result is True

    def test_disk_space_check_insufficient_space(self) -> None:
        """Verify check_disk_space returns False when insufficient space."""
        from alphafold3_mlx.pipeline.output_handler import check_disk_space

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Request impossibly large amount
            result = check_disk_space(output_dir, required_gb=99999999)
            assert result is False

    def test_get_available_disk_space_gb_returns_positive(self) -> None:
        """Verify get_available_disk_space_gb returns positive value."""
        from alphafold3_mlx.pipeline.output_handler import get_available_disk_space_gb

        with tempfile.TemporaryDirectory() as tmpdir:
            available = get_available_disk_space_gb(Path(tmpdir))

            assert isinstance(available, float)
            assert available > 0

    def test_create_output_directory_checks_disk_space(self) -> None:
        """Verify create_output_directory checks disk space."""
        from alphafold3_mlx.pipeline.output_handler import create_output_directory
        from alphafold3_mlx.pipeline.errors import ResourceError

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"

            # Should raise ResourceError for impossibly large requirement
            with pytest.raises(ResourceError, match="Insufficient disk space"):
                create_output_directory(output_dir, required_space_gb=99999999)


class TestWriteMMCIFFile:
    """Tests for mmCIF file writing."""

    def test_write_mmcif_creates_file(self) -> None:
        """Verify write_mmcif_file creates a file."""
        import numpy as np
        from alphafold3_mlx.pipeline.output_handler import write_mmcif_file

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "structure_rank_1.cif"

            # Structure data with proper shapes for 2 residues
            num_residues = 2
            num_atoms = 37  # atom37 format
            structure_data = {
                "coords": np.zeros((num_residues, num_atoms, 3), dtype=np.float32),
                "atom_mask": np.ones((num_residues, num_atoms), dtype=np.float32),
                "plddt": np.full((num_residues, num_atoms), 85.0, dtype=np.float32),
                "pae": np.ones((num_residues, num_residues), dtype=np.float32),
                "ptm": 0.9,
                "iptm": 0.0,
                "aatype": np.zeros(num_residues, dtype=np.int32),
                "residue_index": np.arange(num_residues, dtype=np.int32),
                "asym_id": np.zeros(num_residues, dtype=np.int32),
            }

            write_mmcif_file(structure_data, output_path, rank=1)

            assert output_path.exists()
            content = output_path.read_text()
            assert "data_" in content  # mmCIF data block marker

    def test_write_mmcif_fallback_minimal(self) -> None:
        """Verify write_mmcif_file falls back to minimal format for invalid data."""
        from alphafold3_mlx.pipeline.output_handler import write_mmcif_file

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "structure_rank_1.cif"

            # Minimal structure data that will trigger fallback
            structure_data = {
                "coords": None,  # Missing coords triggers fallback
                "plddt": [85.0],
            }

            write_mmcif_file(structure_data, output_path, rank=1)

            assert output_path.exists()
            content = output_path.read_text()
            assert "data_" in content


class TestWriteConfidenceScores:
    """Tests for confidence scores JSON writing."""

    def test_write_confidence_scores_creates_valid_json(self) -> None:
        """Verify write_confidence_scores creates valid JSON.

        Phase 4+ spec contract: plddt is per-residue (not per-atom).
        The per-residue pLDDT is computed as mean over atoms using atom_mask.
        """
        from alphafold3_mlx.pipeline.output_handler import write_confidence_scores

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "confidence_scores.json"

            # Per-residue pLDDT values (one value per residue, not per atom)
            confidence_data = {
                "num_samples": 1,
                "ranking_metric": "pTM",
                "is_complex": False,
                "samples": {
                    "0": {
                        "plddt": [85.0, 90.0],  # Per-residue (2 residues)
                        "pae": [[0.5, 1.0], [1.0, 0.5]],  # [num_residues, num_residues]
                        "ptm": 0.9,
                        "iptm": 0.0,
                        "mean_plddt": 87.5,
                    }
                },
                "best_sample_index": 0,
            }

            write_confidence_scores(confidence_data, output_path)

            assert output_path.exists()

            # Verify it's valid JSON
            with open(output_path) as f:
                loaded = json.load(f)

            assert loaded == confidence_data

    def test_plddt_is_per_residue_not_per_atom(self) -> None:
        """Verify plddt in confidence_scores.json is per-residue (alignment).

        Phase 4+ spec: plddt must be per-residue, computed as masked mean over atoms.
        This is important for compatibility with visualization tools that expect
        one pLDDT value per residue (not per atom).
        """
        from alphafold3_mlx.pipeline.output_handler import write_confidence_scores

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "confidence_scores.json"

            # Simulate a 5-residue protein with per-residue pLDDT
            num_residues = 5
            confidence_data = {
                "num_samples": 1,
                "ranking_metric": "pTM",
                "is_complex": False,
                "samples": {
                    "0": {
                        "plddt": [85.0, 90.0, 87.5, 92.0, 88.0],  # One per residue
                        "pae": [[0.0] * num_residues for _ in range(num_residues)],
                        "ptm": 0.9,
                        "iptm": 0.0,
                        "mean_plddt": 88.5,  # Mean of per-residue values
                    }
                },
                "best_sample_index": 0,
            }

            write_confidence_scores(confidence_data, output_path)

            with open(output_path) as f:
                loaded = json.load(f)

            # Verify plddt length matches num_residues (not num_residues * 37 atoms)
            plddt = loaded["samples"]["0"]["plddt"]
            assert len(plddt) == num_residues, (
                f"plddt should be per-residue (length {num_residues}), "
                f"not per-atom (length {num_residues * 37}). Got length {len(plddt)}"
            )

            # Verify mean_plddt is consistent with per-residue values
            import statistics
            expected_mean = statistics.mean(plddt)
            assert abs(loaded["samples"]["0"]["mean_plddt"] - expected_mean) < 0.1


class TestWriteTiming:
    """Tests for timing JSON writing."""

    def test_write_timing_creates_valid_json(self) -> None:
        """Verify write_timing creates valid JSON.

        Phase 4+ timing.json stage names (spec-aligned):
        - weight_loading: Model weight loading
        - feature_preparation: Input featurisation
        - recycling: Evoformer/recycling iterations (from InferenceStats)
        - diffusion: Diffusion denoising steps (from InferenceStats)
        - confidence: Confidence head computation (from InferenceStats)
        - output_writing: Output file writing
        """
        from alphafold3_mlx.pipeline.output_handler import write_timing

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "timing.json"

            # Spec-aligned timing data structure
            timing_data = {
                "total_seconds": 120.5,
                "stages": {
                    "weight_loading": 5.2,
                    "feature_preparation": 2.1,
                    "recycling": 30.5,
                    "diffusion": 75.2,
                    "confidence": 5.0,
                    "output_writing": 2.5,
                },
            }

            write_timing(timing_data, output_path)

            assert output_path.exists()

            with open(output_path) as f:
                loaded = json.load(f)

            assert loaded == timing_data

    def test_write_timing_spec_aligned_stages(self) -> None:
        """Verify timing.json stage names match spec contract."""
        from alphafold3_mlx.pipeline.output_handler import write_timing

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "timing.json"

            # Verify all spec-aligned stage names are accepted
            timing_data = {
                "total_seconds": 100.0,
                "stages": {
                    "weight_loading": 5.0,      # Runner stage (ProgressReporter)
                    "feature_preparation": 2.0, # Runner stage (ProgressReporter)
                    "recycling": 30.0,          # From InferenceStats.evoformer_duration_seconds
                    "diffusion": 50.0,          # From InferenceStats.diffusion_duration_seconds
                    "confidence": 8.0,          # From InferenceStats.confidence_duration_seconds
                    "output_writing": 5.0,      # Runner stage (ProgressReporter)
                },
            }

            write_timing(timing_data, output_path)

            with open(output_path) as f:
                loaded = json.load(f)

            # Verify all spec-required stages are present
            required_stages = ["weight_loading", "feature_preparation", "recycling",
                               "diffusion", "confidence", "output_writing"]
            for stage in required_stages:
                assert stage in loaded["stages"], f"Missing required stage: {stage}"


class TestWriteRankingDebug:
    """Tests for ranking debug JSON writing."""

    def test_write_ranking_debug_creates_valid_json(self) -> None:
        """Verify write_ranking_debug creates valid JSON."""
        from alphafold3_mlx.pipeline.output_handler import write_ranking_debug

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ranking_debug.json"

            ranking_data = {
                "ranking_metric": "pTM",
                "is_complex": False,
                "num_samples": 2,
                "samples": [
                    {"index": 0, "rank": 1, "ptm": 0.9, "iptm": 0.0, "mean_plddt": 87.5},
                    {"index": 1, "rank": 2, "ptm": 0.8, "iptm": 0.0, "mean_plddt": 82.0},
                ],
                "aggregate_metrics": {
                    "best_ptm": 0.9,
                    "mean_plddt_all_samples": 84.75,
                    "plddt_variance": 15.0,
                },
            }

            write_ranking_debug(ranking_data, output_path)

            assert output_path.exists()

            with open(output_path) as f:
                loaded = json.load(f)

            assert loaded == ranking_data


class TestWriteFailureLog:
    """Tests for failure log JSON writing."""

    def test_write_failure_log_creates_valid_json(self) -> None:
        """Verify write_failure_log creates valid JSON."""
        from alphafold3_mlx.pipeline.output_handler import write_failure_log

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            failure_data = {
                "error_type": "InputError",
                "error_message": "Invalid input file",
                "stage_reached": "startup",
                "timing_snapshot": {"weight_loading": 2.5},
                "traceback": None,
            }

            result_path = write_failure_log(failure_data, output_dir)

            assert result_path == output_dir / "failure_log.json"
            assert result_path.exists()

            with open(result_path) as f:
                loaded = json.load(f)

            assert loaded == failure_data


class TestOutputBundle:
    """Tests for OutputBundle dataclass."""

    def test_output_bundle_initialization(self) -> None:
        """Verify OutputBundle initializes correctly."""
        from alphafold3_mlx.pipeline.output_handler import OutputBundle

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            bundle = OutputBundle(output_dir=output_dir)

            assert bundle.output_dir == output_dir
            assert bundle.confidence_scores_file == output_dir / "confidence_scores.json"
            assert bundle.timing_file == output_dir / "timing.json"
            assert bundle.ranking_debug_file == output_dir / "ranking_debug.json"
            assert bundle.failure_log_file is None

    def test_output_bundle_structure_path(self) -> None:
        """Verify structure_path method returns correct paths."""
        from alphafold3_mlx.pipeline.output_handler import OutputBundle

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            bundle = OutputBundle(output_dir=output_dir)

            assert bundle.structure_path(1) == output_dir / "structure_rank_1.cif"
            assert bundle.structure_path(2) == output_dir / "structure_rank_2.cif"
            assert bundle.structure_path(5) == output_dir / "structure_rank_5.cif"

    def test_output_bundle_initialize_structure_files(self) -> None:
        """Verify initialize_structure_files creates paths for all samples."""
        from alphafold3_mlx.pipeline.output_handler import OutputBundle

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            bundle = OutputBundle(output_dir=output_dir)
            bundle.initialize_structure_files(num_samples=3)

            assert len(bundle.structure_files) == 3
            assert bundle.structure_files[0] == output_dir / "structure_rank_1.cif"
            assert bundle.structure_files[1] == output_dir / "structure_rank_2.cif"
            assert bundle.structure_files[2] == output_dir / "structure_rank_3.cif"

    def test_output_bundle_all_files(self) -> None:
        """Verify all_files returns all output file paths."""
        from alphafold3_mlx.pipeline.output_handler import OutputBundle

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            bundle = OutputBundle(output_dir=output_dir)
            bundle.initialize_structure_files(num_samples=2)

            all_files = bundle.all_files()

            # Should include 2 structure files + 3 JSON files
            assert len(all_files) == 5
            assert output_dir / "structure_rank_1.cif" in all_files
            assert output_dir / "structure_rank_2.cif" in all_files
            assert output_dir / "confidence_scores.json" in all_files
            assert output_dir / "timing.json" in all_files
            assert output_dir / "ranking_debug.json" in all_files


class TestWriteRankedOutputs:
    """Tests for write_ranked_outputs function."""

    def test_write_ranked_outputs_creates_all_files(self) -> None:
        """Verify write_ranked_outputs creates all output files."""
        import numpy as np
        from alphafold3_mlx.pipeline.output_handler import (
            OutputBundle,
            write_ranked_outputs,
        )
        from alphafold3_mlx.pipeline.ranking import rank_samples

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_bundle = OutputBundle(output_dir=output_dir)
            output_bundle.initialize_structure_files(num_samples=3)

            # Create mock model result
            num_samples = 3
            num_residues = 5
            num_atoms = 37

            class MockResult:
                def __init__(self):
                    self._data = {
                        "atom_positions": np.random.randn(num_samples, num_residues, num_atoms, 3).astype(np.float32),
                        "atom_mask": np.ones((num_samples, num_residues, num_atoms), dtype=np.float32),
                        "plddt": np.full((num_samples, num_residues, num_atoms), 85.0, dtype=np.float32),
                        "pae": np.ones((num_samples, num_residues, num_residues), dtype=np.float32),
                        "ptm": np.array([0.75, 0.92, 0.80]),
                        "iptm": np.array([0.0, 0.0, 0.0]),
                    }
                    self.num_samples = num_samples

                def to_numpy(self):
                    return self._data

            result = MockResult()

            # Create ranking
            ranking = rank_samples(
                ptm_scores=[0.75, 0.92, 0.80],
                iptm_scores=[0.0, 0.0, 0.0],
                plddt_scores=[[85.0, 86.0] for _ in range(3)],
                is_complex=False,
            )

            # Write outputs
            timing_data = {"total_seconds": 100.0, "stages": {"test": 100.0}}
            write_ranked_outputs(result, output_bundle, ranking, timing_data)

            # Verify all files created
            assert (output_dir / "structure_rank_1.cif").exists()
            assert (output_dir / "structure_rank_2.cif").exists()
            assert (output_dir / "structure_rank_3.cif").exists()
            assert (output_dir / "confidence_scores.json").exists()
            assert (output_dir / "ranking_debug.json").exists()
            assert (output_dir / "timing.json").exists()

    def test_write_ranked_outputs_with_token_metadata(self) -> None:
        """ : Verify token_metadata is used for mmCIF chain labeling.

        Phase 2 alignment: token_metadata (aatype, residue_index, asym_id) from
        the featurisation batch should be passed through to mmCIF generation
        to ensure deterministic chain labeling for multi-chain inputs.
        """
        import numpy as np
        from alphafold3_mlx.pipeline.output_handler import (
            OutputBundle,
            write_ranked_outputs,
        )
        from alphafold3_mlx.pipeline.ranking import rank_samples

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_bundle = OutputBundle(output_dir=output_dir)
            output_bundle.initialize_structure_files(num_samples=1)

            # Create mock model result for 2-chain complex
            num_samples = 1
            num_residues = 4  # 2 in chain A, 2 in chain B
            num_atoms = 37

            class MockResult:
                def __init__(self):
                    self._data = {
                        "atom_positions": np.random.randn(num_samples, num_residues, num_atoms, 3).astype(np.float32),
                        "atom_mask": np.ones((num_samples, num_residues, num_atoms), dtype=np.float32),
                        "plddt": np.full((num_samples, num_residues, num_atoms), 85.0, dtype=np.float32),
                        "pae": np.ones((num_samples, num_residues, num_residues), dtype=np.float32),
                        "ptm": np.array([0.85]),
                        "iptm": np.array([0.75]),  # Non-zero for complex
                    }
                    self.num_samples = num_samples

                def to_numpy(self):
                    return self._data

            result = MockResult()

            ranking = rank_samples(
                ptm_scores=[0.85],
                iptm_scores=[0.75],
                plddt_scores=[[85.0, 86.0, 87.0, 88.0]],
                is_complex=True,
            )

            # token_metadata from featurisation batch - determines chain labeling
            token_metadata = {
                "aatype": np.zeros(num_residues, dtype=np.int32),  # All ALA
                "residue_index": np.array([0, 1, 0, 1], dtype=np.int32),  # Reset per chain
                "asym_id": np.array([0, 0, 1, 1], dtype=np.int32),  # Chain assignment
            }

            # Write outputs with token_metadata
            write_ranked_outputs(
                result, output_bundle, ranking,
                timing_data={"total_seconds": 10.0, "stages": {}},
                token_metadata=token_metadata,
            )

            # Verify structure file was created
            assert (output_dir / "structure_rank_1.cif").exists()

            # Verify mmCIF is valid
            content = (output_dir / "structure_rank_1.cif").read_text()
            assert "data_" in content, "Missing mmCIF data block"

    def test_write_ranked_outputs_orders_by_ranking(self) -> None:
        """Verify structures are written in ranked order."""
        import numpy as np
        from alphafold3_mlx.pipeline.output_handler import (
            OutputBundle,
            write_ranked_outputs,
        )
        from alphafold3_mlx.pipeline.ranking import rank_samples

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_bundle = OutputBundle(output_dir=output_dir)
            output_bundle.initialize_structure_files(num_samples=3)

            # Create mock result with distinguishable ptm scores
            num_samples = 3
            num_residues = 2
            num_atoms = 37

            # Sample 1 has highest pTM (0.92), sample 2 second (0.80), sample 0 third (0.75)
            ptm_scores = [0.75, 0.92, 0.80]

            class MockResult:
                def __init__(self):
                    self._data = {
                        "atom_positions": np.random.randn(num_samples, num_residues, num_atoms, 3).astype(np.float32),
                        "atom_mask": np.ones((num_samples, num_residues, num_atoms), dtype=np.float32),
                        "plddt": np.full((num_samples, num_residues, num_atoms), 85.0, dtype=np.float32),
                        "pae": np.ones((num_samples, num_residues, num_residues), dtype=np.float32),
                        "ptm": np.array(ptm_scores),
                        "iptm": np.array([0.0, 0.0, 0.0]),
                    }
                    self.num_samples = num_samples

                def to_numpy(self):
                    return self._data

            result = MockResult()

            # Create ranking - expected order: sample 1, 2, 0
            ranking = rank_samples(
                ptm_scores=ptm_scores,
                iptm_scores=[0.0, 0.0, 0.0],
                plddt_scores=[[85.0] for _ in range(3)],
                is_complex=False,
            )

            write_ranked_outputs(result, output_bundle, ranking, timing_data=None)

            # Verify ranking_debug.json shows correct order
            with open(output_dir / "ranking_debug.json") as f:
                ranking_debug = json.load(f)

            # Sample 1 (pTM=0.92) should be rank 1
            assert ranking_debug["samples"][0]["index"] == 1
            assert ranking_debug["samples"][0]["rank"] == 1

            # Sample 2 (pTM=0.80) should be rank 2
            assert ranking_debug["samples"][1]["index"] == 2
            assert ranking_debug["samples"][1]["rank"] == 2

            # Sample 0 (pTM=0.75) should be rank 3
            assert ranking_debug["samples"][2]["index"] == 0
            assert ranking_debug["samples"][2]["rank"] == 3

    def test_write_ranked_outputs_confidence_scores_format(self) -> None:
        """Verify confidence_scores.json format is correct."""
        import numpy as np
        from alphafold3_mlx.pipeline.output_handler import (
            OutputBundle,
            write_ranked_outputs,
        )
        from alphafold3_mlx.pipeline.ranking import rank_samples

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_bundle = OutputBundle(output_dir=output_dir)
            output_bundle.initialize_structure_files(num_samples=2)

            num_samples = 2
            num_residues = 3
            num_atoms = 37

            class MockResult:
                def __init__(self):
                    self._data = {
                        "atom_positions": np.zeros((num_samples, num_residues, num_atoms, 3), dtype=np.float32),
                        "atom_mask": np.ones((num_samples, num_residues, num_atoms), dtype=np.float32),
                        "plddt": np.full((num_samples, num_residues, num_atoms), 85.0, dtype=np.float32),
                        "pae": np.ones((num_samples, num_residues, num_residues), dtype=np.float32),
                        "ptm": np.array([0.80, 0.90]),
                        "iptm": np.array([0.0, 0.0]),
                    }
                    self.num_samples = num_samples

                def to_numpy(self):
                    return self._data

            result = MockResult()

            ranking = rank_samples(
                ptm_scores=[0.80, 0.90],
                iptm_scores=[0.0, 0.0],
                plddt_scores=[[85.0, 86.0] for _ in range(2)],
                is_complex=False,
            )

            write_ranked_outputs(result, output_bundle, ranking, timing_data=None)

            # Verify confidence_scores.json format
            with open(output_dir / "confidence_scores.json") as f:
                confidence = json.load(f)

            assert confidence["num_samples"] == 2
            assert confidence["ranking_metric"] == "pTM"
            assert confidence["is_complex"] is False
            assert confidence["best_sample_index"] == 1  # Sample 1 has highest pTM

            # Check sample data
            assert "0" in confidence["samples"]
            assert "1" in confidence["samples"]
            assert "ptm" in confidence["samples"]["0"]
            assert "iptm" in confidence["samples"]["0"]
            assert "plddt" in confidence["samples"]["0"]
            assert "pae" in confidence["samples"]["0"]
            assert "mean_plddt" in confidence["samples"]["0"]
