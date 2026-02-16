"""Unit tests for the structure API routes."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("fastapi not installed", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_PARSE_RESULT = {
    "name": "test",
    "sequences": [{"proteinChain": {"sequence": "ACGT", "count": 1}}],
    "dialect": "alphafoldserver",
    "version": 1,
    "source": "upload",
    "pdb_id": None,
    "num_chains": 1,
    "num_residues": 4,
    "warnings": [],
}

MOCK_FETCH_RESULT = {
    "name": "1UBQ",
    "sequences": [{"proteinChain": {"sequence": "MQIFVKTLTG", "count": 1}}],
    "dialect": "alphafoldserver",
    "version": 1,
    "source": "rcsb",
    "pdb_id": "1UBQ",
    "num_chains": 1,
    "num_residues": 10,
    "warnings": [],
}


@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    from alphafold3_mlx.api.app import create_app
    from alphafold3_mlx.api.config import APIConfig

    config = APIConfig(model_dir=Path("nonexistent"))

    # Patch lifespan to avoid model loading
    with mock.patch("alphafold3_mlx.api.app.lifespan") as mock_lifespan:
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _noop_lifespan(app):
            yield

        mock_lifespan.side_effect = _noop_lifespan
        app = create_app(config)

    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/structure/parse
# ---------------------------------------------------------------------------


class TestParseEndpoint:
    """Tests for POST /api/structure/parse."""

    def test_valid_pdb_file(self, client):
        """Upload a valid PDB file → 200 with sequences."""
        with mock.patch(
            "alphafold3_mlx.pipeline.structure_parser.parse_structure",
            return_value=MOCK_PARSE_RESULT,
        ):
            response = client.post(
                "/api/structure/parse",
                files={"file": ("test.pdb", b"ATOM ...", "chemical/x-pdb")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "upload"
        assert len(data["sequences"]) == 1
        assert "modelSeeds" not in data

    def test_valid_cif_file(self, client):
        """Upload a valid CIF file → 200."""
        with mock.patch(
            "alphafold3_mlx.pipeline.structure_parser.parse_structure",
            return_value=MOCK_PARSE_RESULT,
        ):
            response = client.post(
                "/api/structure/parse",
                files={"file": ("test.cif", b"data_test", "text/plain")},
            )

        assert response.status_code == 200

    def test_unsupported_extension(self, client):
        """Upload a .txt file → 422."""
        response = client.post(
            "/api/structure/parse",
            files={"file": ("test.txt", b"some content", "text/plain")},
        )

        assert response.status_code == 422
        assert "Unsupported file type" in response.json()["detail"]

    def test_corrupt_content(self, client):
        """Upload corrupt PDB content → 422."""
        with mock.patch(
            "alphafold3_mlx.pipeline.structure_parser.parse_structure",
            side_effect=ValueError("Failed to parse"),
        ):
            response = client.post(
                "/api/structure/parse",
                files={"file": ("test.pdb", b"garbage", "chemical/x-pdb")},
            )

        assert response.status_code == 422

    def test_multi_component_ligand_warning(self, client):
        """Upload structure with multi-component ligand → 200 with warnings."""
        result_with_warnings = dict(MOCK_PARSE_RESULT)
        result_with_warnings["warnings"] = [
            "Skipped chain B: multi-component ligand (ALA, GLY) is not supported"
        ]

        with mock.patch(
            "alphafold3_mlx.pipeline.structure_parser.parse_structure",
            return_value=result_with_warnings,
        ):
            response = client.post(
                "/api/structure/parse",
                files={"file": ("test.pdb", b"ATOM ...", "chemical/x-pdb")},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["warnings"]) == 1
        assert "multi-component" in data["warnings"][0]

    def test_all_unsupported_structure(self, client):
        """Upload structure where all entities are unsupported → 422."""
        with mock.patch(
            "alphafold3_mlx.pipeline.structure_parser.parse_structure",
            side_effect=ValueError("No supported entities could be imported"),
        ):
            response = client.post(
                "/api/structure/parse",
                files={"file": ("test.pdb", b"ATOM ...", "chemical/x-pdb")},
            )

        assert response.status_code == 422
        assert "No supported entities" in response.json()["detail"]

    def test_no_model_seeds_in_response(self, client):
        """Response should not contain modelSeeds field."""
        with mock.patch(
            "alphafold3_mlx.pipeline.structure_parser.parse_structure",
            return_value=MOCK_PARSE_RESULT,
        ):
            response = client.post(
                "/api/structure/parse",
                files={"file": ("test.pdb", b"ATOM ...", "chemical/x-pdb")},
            )

        data = response.json()
        assert "modelSeeds" not in data


# ---------------------------------------------------------------------------
# GET /api/structure/fetch/{pdb_id}
# ---------------------------------------------------------------------------


class TestFetchEndpoint:
    """Tests for GET /api/structure/fetch/{pdb_id}."""

    def test_valid_pdb_id(self, client):
        """Fetch 1UBQ → 200 with protein chain."""
        with mock.patch(
            "alphafold3_mlx.pipeline.structure_parser.fetch_and_parse",
            return_value=MOCK_FETCH_RESULT,
        ):
            response = client.get("/api/structure/fetch/1UBQ")

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "rcsb"
        assert data["pdb_id"] == "1UBQ"
        assert len(data["sequences"]) == 1

    def test_not_found(self, client):
        """Fetch ZZZZ (non-existent) → 404."""
        from alphafold3_mlx.pipeline.structure_parser import PdbNotFoundError

        with mock.patch(
            "alphafold3_mlx.pipeline.structure_parser.fetch_and_parse",
            side_effect=PdbNotFoundError("ZZZZ"),
        ):
            response = client.get("/api/structure/fetch/ZZZZ")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_invalid_pdb_id_too_short(self, client):
        """3-character PDB ID → 422."""
        response = client.get("/api/structure/fetch/abc")

        assert response.status_code == 422
        assert "4 alphanumeric" in response.json()["detail"]

    def test_invalid_pdb_id_special_chars(self, client):
        """PDB ID with special chars → 422."""
        response = client.get("/api/structure/fetch/ab!d")

        assert response.status_code == 422
        assert "4 alphanumeric" in response.json()["detail"]

    def test_timeout(self, client):
        """RCSB timeout → 504."""
        from alphafold3_mlx.pipeline.structure_parser import RcsbTimeoutError

        with mock.patch(
            "alphafold3_mlx.pipeline.structure_parser.fetch_and_parse",
            side_effect=RcsbTimeoutError("1UBQ"),
        ):
            response = client.get("/api/structure/fetch/1UBQ")

        assert response.status_code == 504
        assert "timed out" in response.json()["detail"]

    def test_no_model_seeds_in_response(self, client):
        """Response should not contain modelSeeds field."""
        with mock.patch(
            "alphafold3_mlx.pipeline.structure_parser.fetch_and_parse",
            return_value=MOCK_FETCH_RESULT,
        ):
            response = client.get("/api/structure/fetch/1UBQ")

        data = response.json()
        assert "modelSeeds" not in data
