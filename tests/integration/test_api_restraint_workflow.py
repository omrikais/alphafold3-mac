"""Integration tests for restraint-guided docking API workflow.

Tests the API validation and submission endpoints with actual HTTP requests
via FastAPI TestClient. Validates:
1. /api/validate catches invalid restraint references
2. /api/validate accepts valid restraint payloads
3. Job submission with restraints creates a job (mock inference)
4. Results endpoint returns restraint satisfaction data

These tests run without model weights by using TestClient and mocking
the inference path where needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

try:
    from fastapi.testclient import TestClient
    from alphafold3_mlx.api.app import create_app
    from alphafold3_mlx.api.config import APIConfig

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

requires_fastapi = pytest.mark.skipif(
    not _HAS_FASTAPI, reason="FastAPI not installed"
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def api_config(tmp_path: Path) -> "APIConfig":
    """Minimal API config with temp data dir."""
    return APIConfig(
        data_dir=tmp_path / "jobs",
        model_dir=Path("weights/model"),
    )


@pytest.fixture
def client(api_config):
    """FastAPI TestClient that skips lifespan (no model loading needed).

    The validation endpoint doesn't need model_manager, job_store, etc.
    Results tests inject their own job_store via app.state.
    """
    if not _HAS_FASTAPI:
        pytest.skip("FastAPI not installed")

    # Create app but replace lifespan with a no-op to skip model loading
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    app = create_app(api_config)
    app.router.lifespan_context = _noop_lifespan
    with TestClient(app) as c:
        yield c


@pytest.fixture
def ubiquitin_sequences() -> list[dict[str, Any]]:
    """Two ubiquitin-like chains (20 residues each)."""
    seq = "MQIFVKTLTGKTITLEVEPS"
    return [
        {"proteinChain": {"sequence": seq, "count": 2}},
    ]


# ── Validation Endpoint Tests ─────────────────────────────────────────────


@requires_fastapi
class TestValidateEndpoint:
    """Tests for POST /api/validate with restraints."""

    def test_validate_valid_protein_restraints(self, client, ubiquitin_sequences):
        """Valid protein-protein distance restraint passes validation."""
        payload = {
            "sequences": ubiquitin_sequences,
            "restraints": {
                "distance": [{
                    "chain_i": "A", "residue_i": 7, "atom_i": "CA",
                    "chain_j": "B", "residue_j": 7, "atom_j": "CA",
                    "target_distance": 5.0,
                }],
            },
        }

        response = client.post("/api/validate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["errors"] == []

    def test_validate_catches_nonexistent_chain(self, client, ubiquitin_sequences):
        """Nonexistent chain 'Z' in restraints is caught."""
        payload = {
            "sequences": ubiquitin_sequences,
            "restraints": {
                "distance": [{
                    "chain_i": "Z", "residue_i": 1,
                    "chain_j": "A", "residue_j": 1,
                    "target_distance": 5.0,
                }],
            },
        }

        response = client.post("/api/validate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any("chain" in e.lower() and "Z" in e for e in data["errors"])

    def test_validate_catches_out_of_range_residue(self, client, ubiquitin_sequences):
        """Residue 9999 exceeds chain length."""
        payload = {
            "sequences": ubiquitin_sequences,
            "restraints": {
                "distance": [{
                    "chain_i": "A", "residue_i": 9999,
                    "chain_j": "B", "residue_j": 1,
                    "target_distance": 5.0,
                }],
            },
        }

        response = client.post("/api/validate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any("9999" in e for e in data["errors"])

    def test_validate_catches_non_protein_chain_reference(self, client):
        """Restraint referencing RNA chain gives non-protein error."""
        payload = {
            "sequences": [
                {"proteinChain": {"sequence": "MQIFVKTLTGKTITLEVEPS", "count": 1}},
                {"rnaSequence": {"sequence": "GCGCGCGC", "count": 1}},
            ],
            "restraints": {
                "distance": [{
                    "chain_i": "A", "residue_i": 1,
                    "chain_j": "B", "residue_j": 1,
                    "target_distance": 5.0,
                }],
            },
        }

        response = client.post("/api/validate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        error_text = " ".join(data["errors"]).lower()
        assert "non-protein" in error_text

    def test_validate_no_restraints_passes(self, client, ubiquitin_sequences):
        """Input without restraints passes validation."""
        payload = {"sequences": ubiquitin_sequences}

        response = client.post("/api/validate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True

    def test_validate_returns_residue_count(self, client, ubiquitin_sequences):
        """Validation response includes num_residues."""
        payload = {"sequences": ubiquitin_sequences}

        response = client.post("/api/validate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["num_residues"] is not None
        assert data["num_residues"] > 0

    def test_validate_contact_restraint(self, client, ubiquitin_sequences):
        """Contact restraint with candidates passes validation."""
        payload = {
            "sequences": ubiquitin_sequences,
            "restraints": {
                "contact": [{
                    "chain_i": "A", "residue_i": 7,
                    "candidates": [
                        {"chain_j": "B", "residue_j": 10},
                        {"chain_j": "B", "residue_j": 11},
                    ],
                    "threshold": 8.0, "weight": 1.0,
                }],
            },
        }

        response = client.post("/api/validate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True

    def test_validate_repulsive_restraint(self, client, ubiquitin_sequences):
        """Repulsive restraint passes validation."""
        payload = {
            "sequences": ubiquitin_sequences,
            "restraints": {
                "repulsive": [{
                    "chain_i": "A", "residue_i": 7,
                    "chain_j": "B", "residue_j": 15,
                    "min_distance": 15.0, "weight": 1.0,
                }],
            },
        }

        response = client.post("/api/validate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True


# ── Results Endpoint Tests ────────────────────────────────────────────────


@requires_fastapi
class TestResultsEndpoint:
    """Tests for restraint satisfaction data in results endpoints."""

    def test_confidence_endpoint_returns_restraint_satisfaction(self, client, tmp_path):
        """GET /api/jobs/{id}/results/confidence/{sample} includes satisfaction."""
        from alphafold3_mlx.api.services.job_store import JobStore
        from alphafold3_mlx.api.models import JobStatus

        store = JobStore(tmp_path / "store")
        job_id = store.create_job(name="test", input_json={"sequences": []})
        store.update_status(job_id, JobStatus.COMPLETED)

        output_dir = store.job_output_dir(job_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        confidence_data = {
            "num_samples": 1,
            "best_sample_index": 0,
            "is_complex": True,
            "samples": {
                "0": {
                    "ptm": 0.85, "iptm": 0.83, "mean_plddt": 90.5,
                    "plddt": [85.0, 90.0],
                    "pae": [[0.5, 1.0], [1.0, 0.5]],
                    "rank": 1,
                    "restraint_satisfaction": {
                        "distance": [{
                            "chain_i": "A", "residue_i": 48, "atom_i": "NZ",
                            "chain_j": "B", "residue_j": 76, "atom_j": "C",
                            "target_distance": 1.5, "actual_distance": 1.72,
                            "satisfied": True,
                        }],
                    },
                },
            },
        }
        (output_dir / "confidence_scores.json").write_text(json.dumps(confidence_data))

        # Inject store into app
        client.app.state.job_store = store

        response = client.get(f"/api/jobs/{job_id}/results/confidence/0")
        assert response.status_code == 200
        data = response.json()
        assert "restraint_satisfaction" in data
        assert data["restraint_satisfaction"]["distance"][0]["satisfied"] is True

    def test_confidence_without_satisfaction_returns_none(self, client, tmp_path):
        """Endpoint returns None when no restraint_satisfaction in data."""
        from alphafold3_mlx.api.services.job_store import JobStore
        from alphafold3_mlx.api.models import JobStatus

        store = JobStore(tmp_path / "store2")
        job_id = store.create_job(name="no-restraints", input_json={"sequences": []})
        store.update_status(job_id, JobStatus.COMPLETED)

        output_dir = store.job_output_dir(job_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        confidence_data = {
            "num_samples": 1,
            "best_sample_index": 0,
            "samples": {
                "0": {
                    "ptm": 0.80, "mean_plddt": 85.0,
                    "plddt": [85.0], "pae": [[0.5]], "rank": 1,
                },
            },
        }
        (output_dir / "confidence_scores.json").write_text(json.dumps(confidence_data))

        client.app.state.job_store = store

        response = client.get(f"/api/jobs/{job_id}/results/confidence/0")
        assert response.status_code == 200
        data = response.json()
        assert data.get("restraint_satisfaction") is None


# ── Diffusion Steps Validation Tests ─────────────────────────────────────


@requires_fastapi
class TestDiffusionStepsValidation:
    """Tests for diffusionSteps-aware guidance step range validation."""

    def test_end_step_exceeds_custom_diffusion_steps(self, client, ubiquitin_sequences):
        """end_step=150 with diffusionSteps=100 is rejected."""
        payload = {
            "sequences": ubiquitin_sequences,
            "restraints": {
                "distance": [{
                    "chain_i": "A", "residue_i": 1, "atom_i": "CA",
                    "chain_j": "B", "residue_j": 1, "atom_j": "CA",
                    "target_distance": 5.0,
                }],
            },
            "guidance": {"start_step": 0, "end_step": 150},
            "diffusionSteps": 100,
        }

        response = client.post("/api/validate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any("end_step" in e and "150" in e for e in data["errors"])

    def test_end_step_within_custom_diffusion_steps(self, client, ubiquitin_sequences):
        """end_step=150 with diffusionSteps=200 is accepted."""
        payload = {
            "sequences": ubiquitin_sequences,
            "restraints": {
                "distance": [{
                    "chain_i": "A", "residue_i": 1, "atom_i": "CA",
                    "chain_j": "B", "residue_j": 1, "atom_j": "CA",
                    "target_distance": 5.0,
                }],
            },
            "guidance": {"start_step": 0, "end_step": 150},
            "diffusionSteps": 200,
        }

        response = client.post("/api/validate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True

    def test_default_diffusion_steps_200(self, client, ubiquitin_sequences):
        """Without diffusionSteps, default 200 is used (end_step=150 valid)."""
        payload = {
            "sequences": ubiquitin_sequences,
            "restraints": {
                "distance": [{
                    "chain_i": "A", "residue_i": 1, "atom_i": "CA",
                    "chain_j": "B", "residue_j": 1, "atom_j": "CA",
                    "target_distance": 5.0,
                }],
            },
            "guidance": {"start_step": 0, "end_step": 150},
        }

        response = client.post("/api/validate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True


# ── Unknown Key Rejection API Tests ──────────────────────────────────────


@requires_fastapi
class TestUnknownKeyRejectionAPI:
    """Tests that unknown keys in restraints/guidance produce validation errors."""

    def test_unknown_restraint_key_rejected(self, client, ubiquitin_sequences):
        """Unknown key 'bogus' in restraints dict is caught."""
        payload = {
            "sequences": ubiquitin_sequences,
            "restraints": {"distance": [], "bogus": True},
        }

        response = client.post("/api/validate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any("bogus" in e.lower() for e in data["errors"])

    def test_unknown_guidance_key_rejected(self, client, ubiquitin_sequences):
        """Unknown key 'warmup' in guidance dict is caught."""
        payload = {
            "sequences": ubiquitin_sequences,
            "restraints": {
                "distance": [{
                    "chain_i": "A", "residue_i": 1,
                    "chain_j": "B", "residue_j": 1,
                    "target_distance": 5.0,
                }],
            },
            "guidance": {"scale": 1.0, "warmup": True},
        }

        response = client.post("/api/validate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert any("warmup" in e.lower() for e in data["errors"])
