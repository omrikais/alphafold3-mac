"""Unit tests for structure_parser module."""

from __future__ import annotations

import socket
import urllib.error
from unittest import mock

import pytest

try:
    import gemmi  # noqa: F401

    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False


# ---------------------------------------------------------------------------
# Fixtures: PDB strings for testing
# ---------------------------------------------------------------------------

# Single protein chain (3 residues)
SINGLE_CHAIN_PDB = """\
HEADER    TEST PROTEIN                            01-JAN-00   TEST
ATOM      1  N   MET A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  MET A   1       2.000   2.000   3.000  1.00  0.00           C
ATOM      3  C   MET A   1       3.000   2.000   3.000  1.00  0.00           C
ATOM      4  O   MET A   1       3.500   3.000   3.000  1.00  0.00           O
ATOM      5  N   ALA A   2       4.000   2.000   3.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       5.000   2.000   3.000  1.00  0.00           C
ATOM      7  C   ALA A   2       6.000   2.000   3.000  1.00  0.00           C
ATOM      8  O   ALA A   2       6.500   3.000   3.000  1.00  0.00           O
ATOM      9  N   GLY A   3       7.000   2.000   3.000  1.00  0.00           N
ATOM     10  CA  GLY A   3       8.000   2.000   3.000  1.00  0.00           C
ATOM     11  C   GLY A   3       9.000   2.000   3.000  1.00  0.00           C
ATOM     12  O   GLY A   3       9.500   3.000   3.000  1.00  0.00           O
END
"""

# Two identical protein chains (homodimer)
HOMODIMER_PDB = """\
HEADER    TEST HOMODIMER                          01-JAN-00   TEST
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  CA  ALA B   1      11.000  12.000  13.000  1.00  0.00           C
ATOM      4  CA  GLY B   2      14.000  15.000  16.000  1.00  0.00           C
END
"""

# Two different protein chains
HETERODIMER_PDB = """\
HEADER    TEST HETERODIMER                        01-JAN-00   TEST
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  CA  MET B   1      11.000  12.000  13.000  1.00  0.00           C
ATOM      4  CA  LEU B   2      14.000  15.000  16.000  1.00  0.00           C
ATOM      5  CA  LYS B   3      17.000  18.000  19.000  1.00  0.00           C
END
"""

# Protein chain with a ligand
PROTEIN_WITH_LIGAND_PDB = """\
HEADER    TEST WITH LIGAND                        01-JAN-00   TEST
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C
HETATM    3 MG    MG A 101      10.000  10.000  10.000  1.00  0.00          MG
HETATM    4  C1  ATP A 102      15.000  15.000  15.000  1.00  0.00           C
END
"""


# ---------------------------------------------------------------------------
# Tests: parse_structure (gemmi-based)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_GEMMI, reason="gemmi not installed")
class TestParseStructure:
    """Tests for the gemmi-based parse_structure function."""

    def test_single_protein_chain(self):
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        result = parse_structure(SINGLE_CHAIN_PDB, "pdb")

        assert len(result["sequences"]) == 1
        assert "proteinChain" in result["sequences"][0]
        seq = result["sequences"][0]["proteinChain"]["sequence"]
        assert seq == "MAG"
        assert result["sequences"][0]["proteinChain"]["count"] == 1
        assert result["warnings"] == []
        assert result["num_residues"] == 3

    def test_homodimer_deduplication(self):
        """Two identical protein chains → single entry with count=2."""
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        result = parse_structure(HOMODIMER_PDB, "pdb")

        assert len(result["sequences"]) == 1
        assert result["sequences"][0]["proteinChain"]["count"] == 2
        assert result["num_residues"] == 4  # 2 + 2

    def test_heterodimer_separate_entries(self):
        """Two different protein chains → two separate entries."""
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        result = parse_structure(HETERODIMER_PDB, "pdb")

        # Should have at least 2 distinct chain entries
        protein_seqs = [
            s["proteinChain"]["sequence"]
            for s in result["sequences"]
            if "proteinChain" in s
        ]
        assert len(protein_seqs) >= 2
        # Verify both chains' residues are present
        all_residues = "".join(protein_seqs)
        assert "A" in all_residues and "G" in all_residues  # chain A: ALA, GLY
        assert "M" in all_residues and "L" in all_residues  # chain B: MET, LEU

    def test_ligand_extraction(self):
        """Ligands extracted from HETATM records."""
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        result = parse_structure(PROTEIN_WITH_LIGAND_PDB, "pdb")

        # Should have protein + at least one ligand/ion
        types = set()
        for seq in result["sequences"]:
            types.update(seq.keys())
        assert "proteinChain" in types
        # MG and ATP should appear as ligand or ion
        assert len(result["sequences"]) >= 2

    def test_ion_classification(self):
        """MG should be classified as ion if in IONS set."""
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        with mock.patch(
            "alphafold3.constants.chemical_component_sets.IONS",
            new=frozenset({"MG", "CA", "ZN"}),
        ):
            result = parse_structure(PROTEIN_WITH_LIGAND_PDB, "pdb")

        # Find the MG entity
        ion_entries = [s for s in result["sequences"] if "ion" in s]
        assert len(ion_entries) >= 1
        assert ion_entries[0]["ion"]["ion"] == "MG"

    def test_result_structure(self):
        """Verify result dict has expected keys and no modelSeeds."""
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        result = parse_structure(SINGLE_CHAIN_PDB, "pdb")

        assert result["source"] == "upload"
        assert result["dialect"] == "alphafoldserver"
        assert result["version"] == 1
        assert result["pdb_id"] is None
        assert "modelSeeds" not in result
        assert "model_seeds" not in result
        assert "rng_seeds" not in result
        assert isinstance(result["warnings"], list)
        assert result["num_chains"] >= 1
        assert result["num_residues"] >= 1

    def test_no_model_seeds_in_entities(self):
        """No entity should contain modelSeeds."""
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        result = parse_structure(SINGLE_CHAIN_PDB, "pdb")

        for seq in result["sequences"]:
            for entity_data in seq.values():
                assert "modelSeeds" not in entity_data

    def test_empty_structure_raises(self):
        """Structure with no parseable entities raises ValueError."""
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        empty_pdb = "HEADER    EMPTY\nEND\n"
        with pytest.raises(ValueError, match="No supported entities"):
            parse_structure(empty_pdb, "pdb")

    def test_mmcif_format(self):
        """Parse a mmCIF file fetched from RCSB (mocked)."""
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        # Use a simple PDB converted via gemmi to get valid mmCIF
        s = gemmi.read_pdb_string(SINGLE_CHAIN_PDB)
        s.setup_entities()
        doc = s.make_mmcif_document()
        mmcif_str = doc.as_string()

        result = parse_structure(mmcif_str, "cif")
        assert len(result["sequences"]) >= 1
        assert result["source"] == "upload"

    def test_water_excluded(self):
        """Water molecules (HOH) should not appear in sequences."""
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        pdb_with_water = """\
HEADER    TEST WITH WATER                         01-JAN-00   TEST
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C
HETATM    3  O   HOH A 301      20.000  20.000  20.000  1.00  0.00           O
HETATM    4  O   HOH A 302      25.000  25.000  25.000  1.00  0.00           O
END
"""
        result = parse_structure(pdb_with_water, "pdb")

        # Should only have protein, no water
        for seq in result["sequences"]:
            for key, val in seq.items():
                if key in ("ligand", "ion"):
                    # Water should not appear as ligand
                    assert val.get("ligand") != "CCD_HOH"
                    assert val.get("ion") != "HOH"

    def test_multi_component_ligand_skipped_with_warning(self):
        """Multi-residue non-polymer entity with distinct CCDs → skipped."""
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        # mmCIF with MG and ATP sharing label_asym_id B (same subchain),
        # forcing them into a single multi-component entity.
        multi_cif = """\
data_test
_entry.id test
_cell.entry_id test
_cell.length_a 50
_cell.length_b 50
_cell.length_c 50
_cell.angle_alpha 90
_cell.angle_beta 90
_cell.angle_gamma 90
_symmetry.entry_id test
_symmetry.space_group_name_H-M 'P 1'

loop_
_entity.id
_entity.type
1 polymer
2 non-polymer

loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_strand_id
_entity_poly.pdbx_seq_one_letter_code
1 polypeptide(L) A AG

loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2

loop_
_chem_comp.id
_chem_comp.type
ALA .
GLY .
MG .
ATP .

loop_
_atom_type.symbol
C
MG

loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA A 1 1 ? 1 2 3 1 0 ? 1 A 1
ATOM 2 C CA . GLY A 1 2 ? 4 5 6 1 0 ? 2 A 1
HETATM 3 MG MG . MG B 2 . ? 10 10 10 1 0 ? 1 B 1
HETATM 4 C C1 . ATP B 2 . ? 15 15 15 1 0 ? 2 B 1
"""
        result = parse_structure(multi_cif, "cif")

        # Only protein should survive; MG+ATP entity skipped
        assert len(result["sequences"]) == 1
        assert "proteinChain" in result["sequences"][0]

        # Warning must reference the skipped chain and both CCDs
        assert len(result["warnings"]) == 1
        assert "multi-component" in result["warnings"][0]
        assert "ATP" in result["warnings"][0]
        assert "MG" in result["warnings"][0]

    def test_multi_residue_same_ccd_skipped_with_warning(self):
        """Multi-residue non-polymer entity with same CCD → also skipped."""
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        # mmCIF with two ATP residues sharing label_asym_id B.
        # Even though both are ATP, len(entity_residues) > 1 → skip.
        same_ccd_cif = """\
data_test
_entry.id test
_cell.entry_id test
_cell.length_a 50
_cell.length_b 50
_cell.length_c 50
_cell.angle_alpha 90
_cell.angle_beta 90
_cell.angle_gamma 90
_symmetry.entry_id test
_symmetry.space_group_name_H-M 'P 1'

loop_
_entity.id
_entity.type
1 polymer
2 non-polymer

loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_strand_id
_entity_poly.pdbx_seq_one_letter_code
1 polypeptide(L) A AG

loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2

loop_
_chem_comp.id
_chem_comp.type
ALA .
GLY .
ATP .

loop_
_atom_type.symbol
C

loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA A 1 1 ? 1 2 3 1 0 ? 1 A 1
ATOM 2 C CA . GLY A 1 2 ? 4 5 6 1 0 ? 2 A 1
HETATM 3 C C1 . ATP B 2 . ? 10 10 10 1 0 ? 1 B 1
HETATM 4 C C1 . ATP B 2 . ? 15 15 15 1 0 ? 2 B 1
"""
        result = parse_structure(same_ccd_cif, "cif")

        # Only protein should survive; ATP×2 entity skipped
        assert len(result["sequences"]) == 1
        assert "proteinChain" in result["sequences"][0]

        # Warning must reference multi-residue and the CCD name
        assert len(result["warnings"]) == 1
        assert "multi-residue" in result["warnings"][0]
        assert "ATP" in result["warnings"][0]
        assert "x2" in result["warnings"][0]

    def test_modification_aware_dedup_same_seq_different_mods(self):
        """Same sequence but different modifications → separate entries."""
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        # Chain A has standard SER at position 2; Chain B has SEP at position 2
        # (phosphoserine). Both have one-letter sequence "AS" but differ in
        # modifications, so they must not be merged.
        pdb_mods = """\
HEADER    TEST MODS                               01-JAN-00   TEST
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  SER A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  CA  ALA B   1      11.000  12.000  13.000  1.00  0.00           C
ATOM      4  CA  SEP B   2      14.000  15.000  16.000  1.00  0.00           C
END
"""
        result = parse_structure(pdb_mods, "pdb")

        protein_entries = [
            s for s in result["sequences"] if "proteinChain" in s
        ]
        # Chain A (ALA,SER) and chain B (ALA,SEP) have the same one-letter seq
        # but different modifications → two separate entries
        assert len(protein_entries) == 2

        # The entry with SEP should carry modifications in the payload
        modified_entry = [
            e for e in protein_entries
            if "modifications" in e["proteinChain"]
        ]
        assert len(modified_entry) == 1
        mods = modified_entry[0]["proteinChain"]["modifications"]
        assert len(mods) == 1
        assert mods[0]["ptmType"] == "CCD_SEP"
        assert mods[0]["ptmPosition"] == 2

        # The unmodified entry should have no modifications key
        unmodified_entry = [
            e for e in protein_entries
            if "modifications" not in e["proteinChain"]
        ]
        assert len(unmodified_entry) == 1

    def test_modification_aware_dedup_identical_mods_merged(self):
        """Same sequence with same modifications → merged (count=2)."""
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        # Both chains have SEP at position 2 → same mod fingerprint → merged
        pdb_same_mods = """\
HEADER    TEST SAME MODS                          01-JAN-00   TEST
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  SEP A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  CA  ALA B   1      11.000  12.000  13.000  1.00  0.00           C
ATOM      4  CA  SEP B   2      14.000  15.000  16.000  1.00  0.00           C
END
"""
        result = parse_structure(pdb_same_mods, "pdb")

        protein_entries = [
            s for s in result["sequences"] if "proteinChain" in s
        ]
        assert len(protein_entries) == 1
        assert protein_entries[0]["proteinChain"]["count"] == 2
        # Merged entry should carry the SEP modification
        mods = protein_entries[0]["proteinChain"].get("modifications", [])
        assert len(mods) == 1
        assert mods[0]["ptmType"] == "CCD_SEP"
        assert mods[0]["ptmPosition"] == 2


# ---------------------------------------------------------------------------
# Tests: fetch_pdb
# ---------------------------------------------------------------------------


class TestFetchPdb:
    """Tests for RCSB fetch functionality."""

    def test_successful_fetch(self):
        from alphafold3_mlx.pipeline.structure_parser import fetch_pdb

        mock_response = mock.MagicMock()
        mock_response.read.return_value = b"data_1UBQ\n"
        mock_response.__enter__ = mock.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            result = fetch_pdb("1UBQ")

        assert result == "data_1UBQ\n"

    def test_not_found(self):
        from alphafold3_mlx.pipeline.structure_parser import (
            PdbNotFoundError,
            fetch_pdb,
        )

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="", code=404, msg="Not Found", hdrs=None, fp=None  # type: ignore[arg-type]
            ),
        ):
            with pytest.raises(PdbNotFoundError):
                fetch_pdb("ZZZZ")

    def test_timeout(self):
        from alphafold3_mlx.pipeline.structure_parser import (
            RcsbTimeoutError,
            fetch_pdb,
        )

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(reason=socket.timeout("timed out")),
        ):
            with pytest.raises(RcsbTimeoutError):
                fetch_pdb("1UBQ")

    def test_connection_refused_not_classified_as_timeout(self):
        """ConnectionRefusedError (an OSError) should be RcsbUnavailableError."""
        from alphafold3_mlx.pipeline.structure_parser import (
            RcsbUnavailableError,
            fetch_pdb,
        )

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(
                reason=ConnectionRefusedError("Connection refused")
            ),
        ):
            with pytest.raises(RcsbUnavailableError):
                fetch_pdb("1UBQ")

    def test_timeout_error_classified_correctly(self):
        """TimeoutError (not socket.timeout) should still be RcsbTimeoutError."""
        from alphafold3_mlx.pipeline.structure_parser import (
            RcsbTimeoutError,
            fetch_pdb,
        )

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(reason=TimeoutError("timed out")),
        ):
            with pytest.raises(RcsbTimeoutError):
                fetch_pdb("1UBQ")


# ---------------------------------------------------------------------------
# Tests: fetch_and_parse
# ---------------------------------------------------------------------------


class TestFetchAndParse:
    """Tests for the combined fetch + parse flow."""

    def test_sets_source_and_pdb_id(self):
        """fetch_and_parse should set source='rcsb' and pdb_id."""
        from alphafold3_mlx.pipeline.structure_parser import fetch_and_parse

        mock_result = {
            "name": "test",
            "sequences": [{"proteinChain": {"sequence": "MAG", "count": 1}}],
            "dialect": "alphafoldserver",
            "version": 1,
            "source": "upload",
            "pdb_id": None,
            "num_chains": 1,
            "num_residues": 3,
            "warnings": [],
        }

        with mock.patch(
            "alphafold3_mlx.pipeline.structure_parser.fetch_pdb",
            return_value="data_1UBQ\n",
        ), mock.patch(
            "alphafold3_mlx.pipeline.structure_parser.parse_structure",
            return_value=mock_result,
        ):
            result = fetch_and_parse("1ubq")

        assert result["source"] == "rcsb"
        assert result["pdb_id"] == "1UBQ"
        assert result["name"] == "1UBQ"
