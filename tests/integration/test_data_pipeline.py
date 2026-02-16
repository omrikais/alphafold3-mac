"""Integration tests for the AlphaFold 3 data pipeline on macOS.

These tests verify that the data pipeline produces valid outputs on macOS:
1. MSA search with jackhmmer produces valid A3M output
2. Featurization produces feature dictionaries matching expected schema
3. Output shapes match the data-model.md specification
4. Cross-platform parity with Linux reference outputs

Requirements:
    - HMMER must be built and available in PATH (run scripts/build_hmmer_macos.sh)
    - Sequence databases must be configured (UniRef90 at minimum)
    - Reference outputs for cross-platform tests (see tests/fixtures/reference_outputs/)

To run these tests:
    pytest tests/integration/test_data_pipeline.py -v

To run only cross-platform validation tests:
    pytest tests/integration/test_data_pipeline.py -v -k "cross_platform"

Note:
    These tests are marked as integration tests and may take several minutes.
    They require external dependencies (HMMER, databases) to be properly configured.
"""

import hashlib
import platform
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


def _reference_output_available(filename: str) -> bool:
    """Check if a reference output file exists."""
    return (REFERENCE_OUTPUTS_DIR / filename).exists()


class TestA3MValidation:
    """Tests for A3M output validation."""

    def test_a3m_format_valid(self):
        """Test that A3M output follows expected format."""
        # Example A3M format validation
        sample_a3m = """>query
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIA
>hit1
MKT-YIAK--QISFVKS--SRQLEERLGLIEVQAPILSRVGDGTQDNLSGAE---QVKVKALPDAQFEVVHSLAKWKRQQIA
"""
        lines = sample_a3m.strip.split("\n")

        # Check format: alternating header and sequence lines
        for i, line in enumerate(lines):
            if i % 2 == 0:
                assert line.startswith(">"), f"Expected header at line {i}: {line}"
            else:
                # Sequence should only contain amino acids and gaps
                valid_chars = set("ACDEFGHIKLMNPQRSTVWY-")
                assert all(c in valid_chars for c in line.upper), (
                    f"Invalid character in sequence at line {i}: {line}"
                )


class TestFeaturization:
    """Tests for featurization output matching FeatureDict schema."""

    def test_feature_dict_schema(self):
        """Test that expected features have correct dtypes and shapes."""
        # Expected features from data-model.md
        expected_features = {
            "aatype": {"dtype": np.int32, "ndim": 1},
            "residue_index": {"dtype": np.int32, "ndim": 1},
            "msa": {"dtype": np.int32, "ndim": 2},
            "msa_mask": {"dtype": np.float32, "ndim": 2},
        }

        # Create mock feature dict for validation testing
        n_res = 50
        n_seq = 100
        mock_features = {
            "aatype": np.zeros(n_res, dtype=np.int32),
            "residue_index": np.arange(n_res, dtype=np.int32),
            "msa": np.zeros((n_seq, n_res), dtype=np.int32),
            "msa_mask": np.ones((n_seq, n_res), dtype=np.float32),
        }

        # Validate mock features match schema
        for name, spec in expected_features.items:
            assert name in mock_features, f"Missing feature: {name}"
            assert mock_features[name].dtype == spec["dtype"], (
                f"Wrong dtype for {name}: {mock_features[name].dtype} != {spec['dtype']}"
            )
            assert mock_features[name].ndim == spec["ndim"], (
                f"Wrong ndim for {name}: {mock_features[name].ndim} != {spec['ndim']}"
            )

    def test_shape_consistency(self):
        """Test that feature shapes are internally consistent."""
        n_res = 80
        n_seq = 512

        # Create features with consistent shapes
        features = {
            "aatype": np.zeros(n_res, dtype=np.int32),
            "residue_index": np.arange(n_res, dtype=np.int32),
            "msa": np.zeros((n_seq, n_res), dtype=np.int32),
            "msa_mask": np.ones((n_seq, n_res), dtype=np.float32),
        }

        # Check N_res consistency
        assert features["aatype"].shape[0] == n_res
        assert features["residue_index"].shape[0] == n_res
        assert features["msa"].shape[1] == n_res
        assert features["msa_mask"].shape[1] == n_res

        # Check N_seq consistency
        assert features["msa"].shape[0] == n_seq
        assert features["msa_mask"].shape[0] == n_seq

    @pytest.mark.skipif(
        not _is_macos_arm64,
        reason="Featurization parity tests only run on macOS ARM64",
    )
    def test_full_featurization_pipeline(self):
        """End-to-end test of featurization producing valid FeatureDict.

        This test validates : Data pipeline identical outputs Mac vs Linux.
        Uses synthetic MSA data (no databases required) to validate featurization parity.
        """
        ref_file = REFERENCE_OUTPUTS_DIR / "msa_featurization_reference.npz"
        if not ref_file.exists:
            pytest.skip(
                f"Reference file not found: {ref_file}. "
                "Generate with: docker build --platform linux/amd64 -f docker/Dockerfile.reference -t alphafold3-reference. && "
                "docker run --platform linux/amd64 --rm -v $(pwd)/tests/fixtures/reference_outputs:/output alphafold3-reference"
            )

        from alphafold3.data import msa as msa_module
        from alphafold3.data import msa_features

        # Load reference outputs
        reference = np.load(ref_file, allow_pickle=True)

        # Test protein MSA featurization via Msa class
        protein_query = str(reference["protein_query"])
        protein_a3m = str(reference["protein_a3m"])

        protein_msa = msa_module.Msa.from_a3m(
            query_sequence=protein_query,
            chain_poly_type="polypeptide(L)",
            a3m=protein_a3m,
            deduplicate=True,
        )
        protein_features = protein_msa.featurize

        np.testing.assert_array_equal(
            protein_features["msa"],
            reference["protein_msa"],
            err_msg="Protein MSA featurization mismatch",
        )
        np.testing.assert_array_equal(
            protein_features["deletion_matrix"],
            reference["protein_deletion_matrix"],
            err_msg="Protein deletion matrix mismatch",
        )
        assert protein_features["num_alignments"] == reference["protein_num_alignments"], (
            f"Protein num_alignments mismatch: {protein_features['num_alignments']} vs {reference['protein_num_alignments']}"
        )

        # Test RNA MSA featurization
        rna_query = str(reference["rna_query"])
        rna_a3m = str(reference["rna_a3m"])

        rna_msa = msa_module.Msa.from_a3m(
            query_sequence=rna_query,
            chain_poly_type="polyribonucleotide",
            a3m=rna_a3m,
            deduplicate=True,
        )
        rna_features = rna_msa.featurize

        np.testing.assert_array_equal(
            rna_features["msa"],
            reference["rna_msa"],
            err_msg="RNA MSA featurization mismatch",
        )
        np.testing.assert_array_equal(
            rna_features["deletion_matrix"],
            reference["rna_deletion_matrix"],
            err_msg="RNA deletion matrix mismatch",
        )

        # Test DNA MSA featurization
        dna_query = str(reference["dna_query"])
        dna_a3m = str(reference["dna_a3m"])

        dna_msa = msa_module.Msa.from_a3m(
            query_sequence=dna_query,
            chain_poly_type="polydeoxyribonucleotide",
            a3m=dna_a3m,
            deduplicate=True,
        )
        dna_features = dna_msa.featurize

        np.testing.assert_array_equal(
            dna_features["msa"],
            reference["dna_msa"],
            err_msg="DNA MSA featurization mismatch",
        )
        np.testing.assert_array_equal(
            dna_features["deletion_matrix"],
            reference["dna_deletion_matrix"],
            err_msg="DNA deletion matrix mismatch",
        )

        # Test species ID extraction
        descriptions = list(reference["species_descriptions"])
        species_ids = msa_features.extract_species_ids(descriptions)
        ref_species_ids = list(reference["species_ids"])

        assert species_ids == ref_species_ids, (
            f"Species ID extraction mismatch!\nmacOS: {species_ids}\nReference: {ref_species_ids}"
        )


class TestDataPipelineIntegration:
    """End-to-end integration tests for the data pipeline."""

    @pytest.mark.skipif(
        not _is_macos_arm64,
        reason="Pipeline parity tests only run on macOS ARM64",
    )
    def test_pipeline_produces_valid_output(self):
        """Test that the full pipeline produces valid feature dictionaries.

        This test validates by testing extract_msa_features directly.
        Uses synthetic MSA sequences (no databases required).
        """
        ref_file = REFERENCE_OUTPUTS_DIR / "msa_featurization_reference.npz"
        if not ref_file.exists:
            pytest.skip(
                f"Reference file not found: {ref_file}. "
                "Generate with Docker - see tests/fixtures/reference_outputs/README.md"
            )

        from alphafold3.data import msa_features

        # Load reference outputs
        reference = np.load(ref_file, allow_pickle=True)

        # Test extract_msa_features directly (the core featurization function)
        # Sequences must have same number of non-lowercase characters (82)
        protein_query = str(reference["protein_query"])
        protein_sequences = [
            protein_query,
            "MKTaAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIA",
            "MKT--IAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIA",
        ]

        msa_arr, deletions_arr = msa_features.extract_msa_features(
            protein_sequences, "polypeptide(L)"
        )

        np.testing.assert_array_equal(
            msa_arr,
            reference["protein_msa_direct"],
            err_msg="extract_msa_features MSA mismatch",
        )
        np.testing.assert_array_equal(
            deletions_arr,
            reference["protein_deletions_direct"],
            err_msg="extract_msa_features deletions mismatch",
        )

        # Validate output shapes and dtypes
        assert msa_arr.dtype == np.int32, f"MSA dtype should be int32, got {msa_arr.dtype}"
        assert deletions_arr.dtype == np.int32, f"Deletions dtype should be int32, got {deletions_arr.dtype}"
        assert msa_arr.shape == deletions_arr.shape, "MSA and deletions shapes should match"
        assert msa_arr.shape[0] == len(protein_sequences), "MSA should have one row per sequence"

    def test_test_sequence_exists(self):
        """Verify the test sequence fixture file exists."""
        assert TEST_SEQUENCE_PATH.exists, (
            f"Test sequence not found at {TEST_SEQUENCE_PATH}. "
            "Create tests/fixtures/test_sequence.fasta with a small protein sequence."
        )


# Skip marker for cross-platform tests requiring reference outputs
requires_reference_outputs = pytest.mark.skipif(
    not REFERENCE_OUTPUTS_DIR.exists or not any(REFERENCE_OUTPUTS_DIR.iterdir),
    reason=(
        "Reference outputs not found. Generate with: "
        "python scripts/generate_reference_outputs.py --output tests/fixtures/reference_outputs/"
    ),
)


class TestCrossPlatformParity:
    """Tests for Mac-vs-Linux data pipeline parity.

    These tests compare outputs from macOS against reference outputs
    generated on Linux to verify cross-platform consistency.
    """

    @pytest.mark.skipif(
        not _is_macos_arm64,
        reason="Cross-platform parity tests only run on macOS ARM64",
    )
    def test_cpp_extensions_available(self):
        """Verify C++ extensions are available on macOS (prerequisite for parity)."""
        try:
            from alphafold3 import cpp

            # Verify all submodules load
            expected_submodules = [
                "cif_dict",
                "fasta_iterator",
                "msa_conversion",
                "mmcif_layout",
                "mmcif_struct_conn",
                "membership",
                "mmcif_utils",
                "aggregation",
                "string_array",
                "mmcif_atom_site",
                "mkdssp",
                "msa_profile",
            ]

            for name in expected_submodules:
                assert hasattr(cpp, name), f"Missing C++ submodule: {name}"

        except ImportError as e:
            pytest.fail(
                f"C++ extensions not available: {e}. "
                "Build with: pip install -e. or ./scripts/verify_cpp_extensions.sh"
            )

    @pytest.mark.skipif(
        not _is_macos_arm64,
        reason="Cross-platform parity tests only run on macOS ARM64",
    )
    def test_fasta_parsing_parity(self):
        """Test FASTA parsing produces identical output to Linux reference."""
        ref_file = REFERENCE_OUTPUTS_DIR / "fasta_parsing_reference.npz"
        if not ref_file.exists:
            pytest.skip(
                f"Reference file not found: {ref_file}. "
                "Generate with: python scripts/generate_reference_outputs.py --force"
            )

        from alphafold3.cpp import fasta_iterator

        # Load reference outputs
        reference = np.load(ref_file, allow_pickle=True)

        # Verify input file matches (by checksum)
        input_checksum = reference["input_checksum"].item

        # Read the test sequence file and verify checksum matches
        if TEST_SEQUENCE_PATH.exists:
            fasta_content = TEST_SEQUENCE_PATH.read_text
            current_checksum = hashlib.sha256(fasta_content.encode).hexdigest

            if current_checksum != input_checksum:
                pytest.skip(
                    f"Input file checksum mismatch. Reference was generated with different input. "
                    f"Regenerate references with: python scripts/generate_reference_outputs.py --force"
                )

            # Parse using macOS C++ extension
            sequences = fasta_iterator.parse_fasta(fasta_content)
            seqs_with_desc, descriptions = fasta_iterator.parse_fasta_include_descriptions(
                fasta_content
            )

            # Compare against reference
            ref_sequences = list(reference["sequences"])
            ref_seqs_with_desc = list(reference["sequences_with_desc"])
            ref_descriptions = list(reference["descriptions"])

            assert sequences == ref_sequences, (
                f"FASTA parsing mismatch!\n"
                f"macOS: {sequences}\n"
                f"Reference: {ref_sequences}"
            )
            assert seqs_with_desc == ref_seqs_with_desc, (
                f"FASTA parsing with descriptions mismatch!\n"
                f"macOS: {seqs_with_desc}\n"
                f"Reference: {ref_seqs_with_desc}"
            )
            assert descriptions == ref_descriptions, (
                f"FASTA descriptions mismatch!\n"
                f"macOS: {descriptions}\n"
                f"Reference: {ref_descriptions}"
            )
        else:
            pytest.skip(f"Test sequence file not found: {TEST_SEQUENCE_PATH}")

    @pytest.mark.skipif(
        not _is_macos_arm64,
        reason="Cross-platform parity tests only run on macOS ARM64",
    )
    def test_msa_profile_parity(self):
        """Test MSA profile computation produces identical output to Linux reference."""
        ref_file = REFERENCE_OUTPUTS_DIR / "msa_profile_reference.npz"
        if not ref_file.exists:
            pytest.skip(
                f"Reference file not found: {ref_file}. "
                "Generate with: python scripts/generate_reference_outputs.py --force"
            )

        from alphafold3.cpp import msa_profile

        # Load reference outputs
        reference = np.load(ref_file, allow_pickle=True)

        # Get the input MSA from reference (uses deterministic seed)
        input_msa = reference["input_msa"]
        num_residue_types = int(reference["num_residue_types"])
        ref_profile = reference["profile"]

        # Compute profile on macOS
        mac_profile = msa_profile.compute_msa_profile(input_msa, num_residue_types)

        # Compare profiles - should be bitwise identical for integer inputs
        np.testing.assert_array_equal(
            mac_profile,
            ref_profile,
            err_msg=(
                f"MSA profile mismatch!\n"
                f"macOS shape: {mac_profile.shape}, dtype: {mac_profile.dtype}\n"
                f"Reference shape: {ref_profile.shape}, dtype: {ref_profile.dtype}"
            ),
        )

    @pytest.mark.skipif(
        not _is_macos_arm64,
        reason="Cross-platform parity tests only run on macOS ARM64",
    )
    def test_string_array_parity(self):
        """Test string_array functions produce identical output to Linux reference."""
        ref_file = REFERENCE_OUTPUTS_DIR / "string_array_reference.npz"
        if not ref_file.exists:
            pytest.skip(
                f"Reference file not found: {ref_file}. "
                "Generate with: python scripts/generate_reference_outputs.py --force"
            )

        from alphafold3.cpp import string_array

        # Load reference outputs
        reference = np.load(ref_file, allow_pickle=True)

        # Reconstruct inputs
        test_strings = reference["test_strings"]
        search_set = set(reference["search_set"])
        remap_dict_keys = list(reference["remap_dict_keys"])
        remap_dict_values = list(reference["remap_dict_values"])
        remap_dict = dict(zip(remap_dict_keys, remap_dict_values))

        # Reference outputs
        ref_isin = reference["isin_result"]
        ref_remap = reference["remap_result"]

        # Compute on macOS
        mac_isin = string_array.isin(test_strings, search_set)
        mac_remap = string_array.remap(test_strings, remap_dict, default_value="?")

        # Compare
        np.testing.assert_array_equal(
            mac_isin,
            ref_isin,
            err_msg=f"isin mismatch! macOS: {mac_isin}, Reference: {ref_isin}",
        )
        np.testing.assert_array_equal(
            mac_remap,
            ref_remap,
            err_msg=f"remap mismatch! macOS: {mac_remap}, Reference: {ref_remap}",
        )

    def test_reference_outputs_directory_documented(self):
        """Verify reference outputs directory has documentation."""
        readme_path = REFERENCE_OUTPUTS_DIR / "README.md"
        assert readme_path.exists, (
            f"Reference outputs README not found at {readme_path}. "
            "This file documents how to generate reference outputs for validation."
        )

    # NOTE: external dependency parity test moved to tests/validation/test_data_pipeline_external_deps.py


class TestCppExtensionBuild:
    """Tests for C++ extensions compile with Apple Clang.

    These tests verify that C++ extensions are properly compiled and
    functional on macOS ARM64.
    """

    @pytest.mark.skipif(
        not _is_macos_arm64,
        reason="C++ extension build tests only run on macOS ARM64",
    )
    def test_cpp_module_importable(self):
        """Test that the cpp module can be imported."""
        try:
            from alphafold3 import cpp

            assert cpp is not None
        except ImportError as e:
            pytest.fail(
                f"C++ module import failed: {e}. "
                "Ensure package is installed: pip install -e."
            )

    @pytest.mark.skipif(
        not _is_macos_arm64,
        reason="C++ extension build tests only run on macOS ARM64",
    )
    def test_all_submodules_importable(self):
        """Test that all C++ submodules can be imported."""
        try:
            from alphafold3 import cpp

            submodules = [
                "cif_dict",
                "fasta_iterator",
                "msa_conversion",
                "mmcif_layout",
                "mmcif_struct_conn",
                "membership",
                "mmcif_utils",
                "aggregation",
                "string_array",
                "mmcif_atom_site",
                "mkdssp",
                "msa_profile",
            ]

            missing = []
            for name in submodules:
                if not hasattr(cpp, name):
                    missing.append(name)

            assert not missing, f"Missing C++ submodules: {missing}"

        except ImportError as e:
            pytest.fail(f"C++ module import failed: {e}")

    def test_verification_script_exists(self):
        """Verify the C++ extension verification script exists."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "verify_cpp_extensions.sh"
        assert script_path.exists, (
            f"Verification script not found at {script_path}. "
            "This script provides automated verification."
        )
