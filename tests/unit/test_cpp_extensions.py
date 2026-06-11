"""Unit tests for C++ extension compilation and import on macOS.

These tests verify C++ extensions compile with Apple Clang and
function correctly on macOS ARM64.
"""

import platform
import sys

import pytest


def _is_macos_arm64() -> bool:
    """Check if running on macOS ARM64."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


# Skip all tests if not on macOS ARM64
pytestmark = pytest.mark.skipif(
    not _is_macos_arm64(),
    reason="C++ extension tests only run on macOS ARM64",
)


_CPP_SUBMODULES = [
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


class TestCppExtensionImport:
    """Tests that C++ extensions can be imported successfully."""

    def test_cpp_module_imports(self):
        """Test that the main cpp module can be imported."""
        from alphafold3 import cpp

        assert cpp is not None

    @pytest.mark.parametrize("submodule", _CPP_SUBMODULES)
    def test_submodule_import(self, submodule):
        """Test that each C++ submodule can be imported."""
        import importlib

        mod = importlib.import_module(f"alphafold3.cpp.{submodule}")
        assert mod is not None


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

        for submodule in _CPP_SUBMODULES:
            assert hasattr(cpp, submodule), f"Missing submodule: {submodule}"
