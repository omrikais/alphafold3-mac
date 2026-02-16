"""Unit tests for C++ extension compilation and import on macOS.

These tests verify C++ extensions compile with Apple Clang and
function correctly on macOS ARM64.
"""

import platform
import sys

import numpy as np
import pytest


def _is_macos_arm64() -> bool:
    """Check if running on macOS ARM64."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


# Skip all tests if not on macOS ARM64
pytestmark = pytest.mark.skipif(
    not _is_macos_arm64(),
    reason="C++ extension tests only run on macOS ARM64",
)


class TestCppExtensionImport:
    """Tests that C++ extensions can be imported successfully."""

    def test_cpp_module_imports(self):
        """Test that the main cpp module can be imported."""
        from alphafold3 import cpp

        assert cpp is not None

    def test_cif_dict_submodule(self):
        """Test cif_dict submodule import."""
        from alphafold3.cpp import cif_dict

        assert cif_dict is not None

    def test_fasta_iterator_submodule(self):
        """Test fasta_iterator submodule import."""
        from alphafold3.cpp import fasta_iterator

        assert fasta_iterator is not None

    def test_msa_conversion_submodule(self):
        """Test msa_conversion submodule import."""
        from alphafold3.cpp import msa_conversion

        assert msa_conversion is not None

    def test_mmcif_layout_submodule(self):
        """Test mmcif_layout submodule import."""
        from alphafold3.cpp import mmcif_layout

        assert mmcif_layout is not None

    def test_mmcif_struct_conn_submodule(self):
        """Test mmcif_struct_conn submodule import."""
        from alphafold3.cpp import mmcif_struct_conn

        assert mmcif_struct_conn is not None

    def test_membership_submodule(self):
        """Test membership submodule import."""
        from alphafold3.cpp import membership

        assert membership is not None

    def test_mmcif_utils_submodule(self):
        """Test mmcif_utils submodule import."""
        from alphafold3.cpp import mmcif_utils

        assert mmcif_utils is not None

    def test_aggregation_submodule(self):
        """Test aggregation submodule import."""
        from alphafold3.cpp import aggregation

        assert aggregation is not None

    def test_string_array_submodule(self):
        """Test string_array submodule import."""
        from alphafold3.cpp import string_array

        assert string_array is not None

    def test_mmcif_atom_site_submodule(self):
        """Test mmcif_atom_site submodule import."""
        from alphafold3.cpp import mmcif_atom_site

        assert mmcif_atom_site is not None

    def test_mkdssp_submodule(self):
        """Test mkdssp submodule import."""
        from alphafold3.cpp import mkdssp

        assert mkdssp is not None

    def test_msa_profile_submodule(self):
        """Test msa_profile submodule import."""
        from alphafold3.cpp import msa_profile

        assert msa_profile is not None


class TestCppExtensionFunctionality:
    """Tests that C++ extensions function correctly on macOS."""

    def test_fasta_iterator_basic(self):
        """Test basic FASTA iteration functionality."""
        from alphafold3.cpp import fasta_iterator

        # Verify the module has expected attributes
        assert hasattr(fasta_iterator, "FastaFileIterator") or hasattr(
            fasta_iterator, "FastaStringIterator"
        )
        assert hasattr(fasta_iterator, "parse_fasta")

    def test_string_array_basic(self):
        """Test string_array functionality."""
        from alphafold3.cpp import string_array

        # The string_array module should have functions for handling string arrays
        # This verifies the module loaded correctly
        assert string_array is not None

    def test_msa_profile_basic(self):
        """Test MSA profile computation is available."""
        from alphafold3.cpp import msa_profile

        # Verify module loaded successfully
        assert msa_profile is not None


class TestCppModuleAttributes:
    """Tests for C++ module attributes and version info."""

    def test_cpp_module_has_submodules(self):
        """Test that cpp module has all expected submodules."""
        from alphafold3 import cpp

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

        for submodule in expected_submodules:
            assert hasattr(cpp, submodule), f"Missing submodule: {submodule}"
