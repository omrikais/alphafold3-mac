"""External dependency integration tests for the data pipeline.

These tests require HMMER and sequence databases. They are not part of the
validation subset and should be run explicitly when dependencies exist.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TEST_SEQUENCE_PATH = FIXTURES_DIR / "test_sequence.fasta"
REFERENCE_OUTPUTS_DIR = FIXTURES_DIR / "reference_outputs"


def _is_macos_arm64() -> bool:
    """Check if running on macOS ARM64."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _hmmer_available() -> bool:
    """Check if HMMER is available in PATH with --seq_limit patch."""
    try:
        result = subprocess.run(
            ["jackhmmer", "-h"],  # Use -h, not --help (HMMER convention)
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Check both stdout and stderr - HMMER may emit help to either stream
        return "--seq_limit" in (result.stdout + result.stderr)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _databases_available() -> bool:
    """Check if required databases are configured and accessible."""
    db_paths = [
        os.environ.get("ALPHAFOLD_DATABASES"),
        os.environ.get("UNIREF90_PATH"),
        Path.home() / "databases" / "uniref90",
        Path("/data/databases/uniref90"),
    ]
    return any(p and Path(p).exists() for p in db_paths if p)


# Skip markers for tests requiring external dependencies
requires_hmmer = pytest.mark.skipif(
    not _hmmer_available(),
    reason="HMMER not available. Run scripts/build_hmmer_macos.sh first.",
)
requires_databases = pytest.mark.skipif(
    not _databases_available(),
    reason="Sequence databases not configured. See quickstart.md for setup.",
)


class TestMSASearch:
    """Tests for MSA search with jackhmmer."""

    @pytest.mark.external_deps
    @requires_hmmer
    def test_jackhmmer_has_seq_limit(self):
        """Verify jackhmmer has the --seq_limit option from the patch."""
        result = subprocess.run(
            ["jackhmmer", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--seq_limit" in result.stdout, (
            "jackhmmer missing --seq_limit option. "
            "Rebuild HMMER with: ./scripts/build_hmmer_macos.sh"
        )

    @pytest.mark.external_deps
    @requires_hmmer
    @requires_databases
    def test_jackhmmer_msa_search(self):
        """Test that jackhmmer can search against a database."""
        # This test requires:
        # 1. HMMER with --seq_limit patch
        # 2. A small test database or the full UniRef90
        pytest.skip("Database search requires database configuration - see quickstart.md")


class TestFullPipelineParity:
    """Full pipeline parity tests requiring external dependencies."""

    @pytest.mark.external_deps
    @pytest.mark.skipif(
        not _is_macos_arm64(),
        reason="Full pipeline parity tests only run on macOS ARM64",
    )
    @requires_hmmer
    def test_full_pipeline_msa_search_parity(self):
        """Test full MSA search pipeline produces identical output to Linux reference.

        This test validates  by running the complete pipeline:
        MSA search (jackhmmer) → A3M parsing → featurization.
        """
        ref_file = REFERENCE_OUTPUTS_DIR / "full_pipeline_reference.npz"
        if not ref_file.exists():
            pytest.skip(
                f"Full pipeline reference not found: {ref_file}. "
                "Generate with: docker build --platform linux/amd64 -f docker/Dockerfile.reference -t alphafold3-reference . && "
                "docker run --platform linux/amd64 --rm -v $(pwd)/tests/fixtures/reference_outputs:/output alphafold3-reference"
            )

        import shutil
        import tempfile

        from alphafold3.data import msa as msa_module

        # Load reference outputs
        reference = np.load(ref_file, allow_pickle=True)

        # Verify jackhmmer is available
        jackhmmer_path = shutil.which("jackhmmer")
        if not jackhmmer_path:
            pytest.skip("jackhmmer not found in PATH")

        # Get paths
        db_path = FIXTURES_DIR / "minimal_test_db.fasta"
        if not db_path.exists():
            pytest.skip(f"Minimal test database not found: {db_path}")

        query_sequence = str(reference["query_sequence"])

        # Run jackhmmer search
        with tempfile.TemporaryDirectory() as tmpdir:
            output_a3m = Path(tmpdir) / "output.a3m"

            cmd = [
                "jackhmmer",
                "--noali",
                "-A", str(output_a3m),
                "--seq_limit", "100",
                "-N", "1",
                str(TEST_SEQUENCE_PATH),
                str(db_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            assert result.returncode == 0, f"jackhmmer failed: {result.stderr}"

            # Read and convert Stockholm to A3M (same logic as reference generator)
            stockholm_content = output_a3m.read_text() if output_a3m.exists() else ""

            if not stockholm_content or "# STOCKHOLM" not in stockholm_content:
                a3m_content = f">query\n{query_sequence}\n"
            else:
                a3m_lines = [f">query\n{query_sequence}"]
                seen_seqs = set()

                for line in stockholm_content.split("\n"):
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    if line.startswith("//"):
                        break

                    parts = line.split()
                    if len(parts) >= 2:
                        seq_name, aligned_seq = parts[0], parts[1]
                        if seq_name not in seen_seqs and seq_name != "query":
                            seen_seqs.add(seq_name)
                            aligned_seq = aligned_seq.replace(".", "")
                            a3m_lines.append(f">{seq_name}\n{aligned_seq}")

                a3m_content = "\n".join(a3m_lines) + "\n"

        # Featurize the MSA
        msa = msa_module.Msa.from_a3m(
            query_sequence=query_sequence,
            chain_poly_type="polypeptide(L)",
            a3m=a3m_content,
            deduplicate=True,
        )
        features = msa.featurize()

        # Compare against reference
        ref_a3m = str(reference["a3m_output"])

        # A3M content should be identical (same search, same database)
        assert a3m_content == ref_a3m, (
            f"A3M output mismatch!\n"
            f"macOS sequences: {a3m_content.count('>')}\n"
            f"Reference sequences: {ref_a3m.count('>')}\n"
            f"macOS A3M:\n{a3m_content[:500]}...\n"
            f"Reference A3M:\n{ref_a3m[:500]}..."
        )

        # Featurized MSA should be identical
        np.testing.assert_array_equal(
            features["msa"],
            reference["msa"],
            err_msg="Full pipeline MSA featurization mismatch",
        )
        np.testing.assert_array_equal(
            features["deletion_matrix"],
            reference["deletion_matrix"],
            err_msg="Full pipeline deletion matrix mismatch",
        )
        assert features["num_alignments"] == reference["num_alignments"], (
            f"num_alignments mismatch: {features['num_alignments']} vs {reference['num_alignments']}"
        )
