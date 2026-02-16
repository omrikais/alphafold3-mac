"""Tests for input_handler module.

Tests the fix for AlphaFold Server top-level list input format.
"""

import json
import tempfile
import warnings
from pathlib import Path

import pytest

from alphafold3_mlx.pipeline.input_handler import (
    InputError,
    load_fold_inputs,
    load_restraints_file,
    parse_input_json,
)


class TestAlphaFoldServerListFormat:
    """Tests for AlphaFold Server list format handling."""

    def test_single_fold_job_in_list(self, tmp_path: Path) -> None:
        """Single fold job in list format should parse successfully."""
        fold_job = {
            "name": "test_job",
            "sequences": [
                {"proteinChain": {"sequence": "MKTAYIAKQRQ", "count": 1}}
            ],
            "modelSeeds": [42],
        }
        json_file = tmp_path / "single_job.json"
        json_file.write_text(json.dumps([fold_job]))

        result = parse_input_json(json_file)

        assert result.name == "test_job"
        assert len(result.sequences) == 1
        assert result.sequences[0].sequence == "MKTAYIAKQRQ"

    def test_multiple_fold_jobs_warns_and_uses_first(self, tmp_path: Path) -> None:
        """Multiple fold jobs should warn and process only the first."""
        fold_jobs = [
            {
                "name": "job_1",
                "sequences": [
                    {"proteinChain": {"sequence": "MKTAYIAKQRQ", "count": 1}}
                ],
                "modelSeeds": [1],
            },
            {
                "name": "job_2",
                "sequences": [
                    {"proteinChain": {"sequence": "GGGGGGGGGG", "count": 1}}
                ],
                "modelSeeds": [2],
            },
        ]
        json_file = tmp_path / "multi_job.json"
        json_file.write_text(json.dumps(fold_jobs))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = parse_input_json(json_file)

            # Should warn about multiple jobs
            assert len(w) == 1
            assert "2 fold jobs" in str(w[0].message)
            assert "processing only the first one" in str(w[0].message)

        # Should use first job
        assert result.name == "job_1"
        assert result.sequences[0].sequence == "MKTAYIAKQRQ"

    def test_empty_list_raises_error(self, tmp_path: Path) -> None:
        """Empty list should raise InputError."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("[]")

        with pytest.raises(InputError, match="Empty AlphaFold Server fold job list"):
            parse_input_json(json_file)

    @pytest.mark.parametrize("content", ['"just a string"', "42", "true", "null"])
    def test_non_object_top_level_raises_input_error(
        self, tmp_path: Path, content: str
    ) -> None:
        """Non-object top-level JSON should raise InputError, not AttributeError."""
        json_file = tmp_path / "bad.json"
        json_file.write_text(content)

        with pytest.raises(InputError, match="Expected a JSON object"):
            parse_input_json(json_file)

    def test_load_fold_inputs_yields_all_jobs(self, tmp_path: Path) -> None:
        """load_fold_inputs should yield all jobs from a list."""
        fold_jobs = [
            {
                "name": "job_1",
                "sequences": [
                    {"proteinChain": {"sequence": "MKTAYIAKQRQ", "count": 1}}
                ],
                "modelSeeds": [1],
            },
            {
                "name": "job_2",
                "sequences": [
                    {"proteinChain": {"sequence": "GGGGGGGGGG", "count": 1}}
                ],
                "modelSeeds": [2],
            },
            {
                "name": "job_3",
                "sequences": [
                    {"proteinChain": {"sequence": "AAAAAAAAAA", "count": 1}}
                ],
                "modelSeeds": [3],
            },
        ]
        json_file = tmp_path / "batch_jobs.json"
        json_file.write_text(json.dumps(fold_jobs))

        results = list(load_fold_inputs(json_file))

        assert len(results) == 3
        assert results[0].name == "job_1"
        assert results[1].name == "job_2"
        assert results[2].name == "job_3"

    def test_load_fold_inputs_single_dict(self, tmp_path: Path) -> None:
        """load_fold_inputs should handle single dict format."""
        fold_job = {
            "name": "single_job",
            "sequences": [
                {"protein": {"id": "A", "sequence": "MKTAYIAKQRQ"}}
            ],
            "modelSeeds": [42],
        }
        json_file = tmp_path / "single.json"
        json_file.write_text(json.dumps(fold_job))

        results = list(load_fold_inputs(json_file))

        assert len(results) == 1
        assert results[0].name == "single_job"

    def test_load_fold_inputs_empty_list_raises(self, tmp_path: Path) -> None:
        """load_fold_inputs should raise for empty list."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("[]")

        with pytest.raises(InputError, match="Empty AlphaFold Server fold job list"):
            list(load_fold_inputs(json_file))


class TestDictFormatUnchanged:
    """Ensure dict format still works after the fix."""

    def test_alphafold3_format_still_works(self, tmp_path: Path) -> None:
        """Official AlphaFold 3 dict format should still parse."""
        af3_input = {
            "dialect": "alphafold3",
            "version": 1,
            "name": "test",
            "sequences": [
                {"protein": {"id": "A", "sequence": "MKTAYIAKQRQ"}}
            ],
            "modelSeeds": [42],
        }
        json_file = tmp_path / "af3.json"
        json_file.write_text(json.dumps(af3_input))

        result = parse_input_json(json_file)

        assert result.name == "test"
        assert len(result.sequences) == 1

    def test_simple_format_still_works(self, tmp_path: Path) -> None:
        """Simple dict format should still parse."""
        simple_input = {
            "name": "simple_test",
            "sequences": [
                {"protein": {"id": "A", "sequence": "MKTAYIAKQRQ"}}
            ],
            "modelSeeds": [42],
        }
        json_file = tmp_path / "simple.json"
        json_file.write_text(json.dumps(simple_input))

        result = parse_input_json(json_file)

        assert result.name == "simple_test"


class TestRestraintsAndGuidanceSupport:
    """Test restraints and guidance support in both parse_input_json and load_fold_inputs."""

    def test_parse_input_json_with_restraints(self, tmp_path: Path) -> None:
        """parse_input_json should extract restraints from inline JSON."""
        fold_job = {
            "name": "test_restraints",
            "sequences": [
                {"protein": {"id": "A", "sequence": "MKTAYIAKQRQ"}},
                {"protein": {"id": "B", "sequence": "GAVLIMPFW"}},
            ],
            "modelSeeds": [42],
            "restraints": {
                "distance": [
                    {
                        "chain_i": "A",
                        "residue_i": 1,
                        "chain_j": "B",
                        "residue_j": 1,
                        "target_distance": 5.0,
                        "atom_i": "CA",
                        "atom_j": "CA",
                        "sigma": 1.0,
                        "weight": 1.0,
                    }
                ]
            },
        }
        json_file = tmp_path / "with_restraints.json"
        json_file.write_text(json.dumps(fold_job))

        result = parse_input_json(json_file)

        assert result.name == "test_restraints"
        assert result.restraints is not None
        assert len(result.restraints.distance) == 1
        assert result.restraints.distance[0].chain_i == "A"
        assert result.restraints.distance[0].target_distance == 5.0

    def test_parse_input_json_with_guidance(self, tmp_path: Path) -> None:
        """parse_input_json should extract guidance from inline JSON."""
        fold_job = {
            "name": "test_guidance",
            "sequences": [
                {"protein": {"id": "A", "sequence": "MKTAYIAKQRQ"}}
            ],
            "modelSeeds": [42],
            "guidance": {
                "scale": 2.0,
                "annealing": "cosine",
                "start_step": 10,
                "end_step": 190,
            },
        }
        json_file = tmp_path / "with_guidance.json"
        json_file.write_text(json.dumps(fold_job))

        result = parse_input_json(json_file)

        assert result.name == "test_guidance"
        assert result.guidance is not None
        assert result.guidance.scale == 2.0
        assert result.guidance.annealing == "cosine"
        assert result.guidance.start_step == 10
        assert result.guidance.end_step == 190

    def test_parse_input_json_with_both(self, tmp_path: Path) -> None:
        """parse_input_json should extract both restraints and guidance."""
        fold_job = {
            "name": "test_both",
            "sequences": [
                {"protein": {"id": "A", "sequence": "MKTAYIAKQRQ"}},
                {"protein": {"id": "B", "sequence": "GAVLIMPFW"}},
            ],
            "modelSeeds": [42],
            "restraints": {
                "contact": [
                    {
                        "chain_i": "A",
                        "residue_i": 5,
                        "candidates": [
                            {"chain_j": "B", "residue_j": 3},
                            {"chain_j": "B", "residue_j": 4},
                        ],
                        "threshold": 8.0,
                        "weight": 1.0,
                    }
                ]
            },
            "guidance": {
                "scale": 1.5,
            },
        }
        json_file = tmp_path / "with_both.json"
        json_file.write_text(json.dumps(fold_job))

        result = parse_input_json(json_file)

        assert result.name == "test_both"
        assert result.restraints is not None
        assert len(result.restraints.contact) == 1
        assert result.restraints.contact[0].chain_i == "A"
        assert len(result.restraints.contact[0].candidates) == 2
        assert result.guidance is not None
        assert result.guidance.scale == 1.5

    def test_load_fold_inputs_batch_with_restraints(self, tmp_path: Path) -> None:
        """load_fold_inputs should extract restraints from each job in a batch."""
        fold_jobs = [
            {
                "name": "job_1",
                "sequences": [
                    {"proteinChain": {"sequence": "MKTAYIAKQRQ", "count": 1}},
                    {"proteinChain": {"sequence": "GAVLIMPFW", "count": 1}},
                ],
                "modelSeeds": [1],
                "restraints": {
                    "distance": [
                        {
                            "chain_i": "A",
                            "residue_i": 1,
                            "chain_j": "B",
                            "residue_j": 1,
                            "target_distance": 5.0,
                        }
                    ]
                },
            },
            {
                "name": "job_2",
                "sequences": [
                    {"proteinChain": {"sequence": "GGGGGGGGGG", "count": 1}}
                ],
                "modelSeeds": [2],
                "restraints": {
                    "repulsive": [
                        {
                            "chain_i": "A",
                            "residue_i": 1,
                            "chain_j": "A",
                            "residue_j": 5,
                            "min_distance": 10.0,
                            "weight": 2.0,
                        }
                    ]
                },
            },
        ]
        json_file = tmp_path / "batch_restraints.json"
        json_file.write_text(json.dumps(fold_jobs))

        results = list(load_fold_inputs(json_file))

        assert len(results) == 2

        # Job 1
        assert results[0].name == "job_1"
        assert results[0].restraints is not None
        assert len(results[0].restraints.distance) == 1
        assert results[0].restraints.distance[0].target_distance == 5.0

        # Job 2
        assert results[1].name == "job_2"
        assert results[1].restraints is not None
        assert len(results[1].restraints.repulsive) == 1
        assert results[1].restraints.repulsive[0].min_distance == 10.0
        assert results[1].restraints.repulsive[0].weight == 2.0

    def test_load_fold_inputs_batch_with_guidance(self, tmp_path: Path) -> None:
        """load_fold_inputs should extract guidance from each job in a batch."""
        fold_jobs = [
            {
                "name": "job_1",
                "sequences": [
                    {"proteinChain": {"sequence": "MKTAYIAKQRQ", "count": 1}}
                ],
                "modelSeeds": [1],
                "guidance": {
                    "scale": 1.0,
                    "annealing": "linear",
                },
            },
            {
                "name": "job_2",
                "sequences": [
                    {"proteinChain": {"sequence": "GGGGGGGGGG", "count": 1}}
                ],
                "modelSeeds": [2],
                "guidance": {
                    "scale": 2.0,
                    "annealing": "cosine",
                    "start_step": 20,
                    "end_step": 180,
                },
            },
        ]
        json_file = tmp_path / "batch_guidance.json"
        json_file.write_text(json.dumps(fold_jobs))

        results = list(load_fold_inputs(json_file))

        assert len(results) == 2

        # Job 1
        assert results[0].name == "job_1"
        assert results[0].guidance is not None
        assert results[0].guidance.scale == 1.0
        assert results[0].guidance.annealing == "linear"

        # Job 2
        assert results[1].name == "job_2"
        assert results[1].guidance is not None
        assert results[1].guidance.scale == 2.0
        assert results[1].guidance.annealing == "cosine"
        assert results[1].guidance.start_step == 20
        assert results[1].guidance.end_step == 180

    def test_load_fold_inputs_batch_with_both(self, tmp_path: Path) -> None:
        """load_fold_inputs should extract both restraints and guidance from batch jobs."""
        fold_jobs = [
            {
                "name": "job_1",
                "sequences": [
                    {"proteinChain": {"sequence": "MKTAYIAKQRQ", "count": 1}},
                    {"proteinChain": {"sequence": "GAVLIMPFW", "count": 1}},
                ],
                "modelSeeds": [1],
                "restraints": {
                    "distance": [
                        {
                            "chain_i": "A",
                            "residue_i": 1,
                            "chain_j": "B",
                            "residue_j": 1,
                            "target_distance": 5.0,
                        }
                    ]
                },
                "guidance": {
                    "scale": 1.5,
                    "annealing": "linear",
                },
            },
        ]
        json_file = tmp_path / "batch_both.json"
        json_file.write_text(json.dumps(fold_jobs))

        results = list(load_fold_inputs(json_file))

        assert len(results) == 1
        assert results[0].name == "job_1"
        assert results[0].restraints is not None
        assert len(results[0].restraints.distance) == 1
        assert results[0].guidance is not None
        assert results[0].guidance.scale == 1.5

    def test_load_fold_inputs_single_dict_with_restraints(self, tmp_path: Path) -> None:
        """load_fold_inputs should handle single dict format with restraints."""
        fold_job = {
            "name": "single_job",
            "sequences": [
                {"protein": {"id": "A", "sequence": "MKTAYIAKQRQ"}},
                {"protein": {"id": "B", "sequence": "GAVLIMPFW"}},
            ],
            "modelSeeds": [42],
            "restraints": {
                "distance": [
                    {
                        "chain_i": "A",
                        "residue_i": 1,
                        "chain_j": "B",
                        "residue_j": 1,
                        "target_distance": 5.0,
                    }
                ]
            },
        }
        json_file = tmp_path / "single_restraints.json"
        json_file.write_text(json.dumps(fold_job))

        results = list(load_fold_inputs(json_file))

        assert len(results) == 1
        assert results[0].name == "single_job"
        assert results[0].restraints is not None
        assert len(results[0].restraints.distance) == 1

    def test_backward_compatibility_no_restraints(self, tmp_path: Path) -> None:
        """Inputs without restraints/guidance should work unchanged."""
        fold_job = {
            "name": "no_restraints",
            "sequences": [
                {"proteinChain": {"sequence": "MKTAYIAKQRQ", "count": 1}}
            ],
            "modelSeeds": [42],
        }
        json_file = tmp_path / "no_restraints.json"
        json_file.write_text(json.dumps(fold_job))

        # Test parse_input_json
        result1 = parse_input_json(json_file)
        assert result1.name == "no_restraints"
        assert result1.restraints is None
        assert result1.guidance is None

        # Test load_fold_inputs
        results2 = list(load_fold_inputs(json_file))
        assert len(results2) == 1
        assert results2[0].name == "no_restraints"
        assert results2[0].restraints is None
        assert results2[0].guidance is None

    def test_batch_mixed_restraints_presence(self, tmp_path: Path) -> None:
        """Batch jobs can have different restraints/guidance configurations."""
        fold_jobs = [
            {
                "name": "job_with_restraints",
                "sequences": [
                    {"proteinChain": {"sequence": "MKTAYIAKQRQ", "count": 1}}
                ],
                "modelSeeds": [1],
                "restraints": {
                    "distance": [
                        {
                            "chain_i": "A",
                            "residue_i": 1,
                            "chain_j": "A",
                            "residue_j": 5,
                            "target_distance": 10.0,
                        }
                    ]
                },
            },
            {
                "name": "job_without_restraints",
                "sequences": [
                    {"proteinChain": {"sequence": "GGGGGGGGGG", "count": 1}}
                ],
                "modelSeeds": [2],
            },
            {
                "name": "job_with_guidance_only",
                "sequences": [
                    {"proteinChain": {"sequence": "AAAAAAAAAA", "count": 1}}
                ],
                "modelSeeds": [3],
                "guidance": {
                    "scale": 0.5,
                },
            },
        ]
        json_file = tmp_path / "batch_mixed.json"
        json_file.write_text(json.dumps(fold_jobs))

        results = list(load_fold_inputs(json_file))

        assert len(results) == 3

        # Job 1: has restraints, no guidance
        assert results[0].restraints is not None
        assert results[0].guidance is None

        # Job 2: no restraints or guidance
        assert results[1].restraints is None
        assert results[1].guidance is None

        # Job 3: has guidance, no restraints
        assert results[2].restraints is None
        assert results[2].guidance is not None


class TestMalformedRestraintTypes:
    """Test that non-dict restraints/guidance produce clear InputError messages."""

    def _base_job(self) -> dict:
        return {
            "name": "test",
            "sequences": [
                {"protein": {"id": "A", "sequence": "MKTAYIAKQRQ"}},
                {"protein": {"id": "B", "sequence": "GAVLIMPFW"}},
            ],
            "modelSeeds": [42],
        }

    # -- parse_input_json -------------------------------------------------------

    @pytest.mark.parametrize("bad_restraints", [
        ["distance"],
        "not a dict",
        42,
        True,
    ])
    def test_parse_input_json_restraints_wrong_type(
        self, tmp_path: Path, bad_restraints
    ) -> None:
        job = self._base_job()
        job["restraints"] = bad_restraints
        json_file = tmp_path / "bad_restraints.json"
        json_file.write_text(json.dumps(job))

        with pytest.raises(InputError, match="Invalid restraints.*must be a JSON object"):
            parse_input_json(json_file)

    @pytest.mark.parametrize("bad_guidance", [
        ["scale"],
        "not a dict",
        42,
        True,
    ])
    def test_parse_input_json_guidance_wrong_type(
        self, tmp_path: Path, bad_guidance
    ) -> None:
        job = self._base_job()
        job["guidance"] = bad_guidance
        json_file = tmp_path / "bad_guidance.json"
        json_file.write_text(json.dumps(job))

        with pytest.raises(InputError, match="Invalid guidance.*must be a JSON object"):
            parse_input_json(json_file)

    # -- load_fold_inputs (single dict) -----------------------------------------

    @pytest.mark.parametrize("bad_restraints", [["a list"], "a string", 99])
    def test_load_fold_inputs_dict_restraints_wrong_type(
        self, tmp_path: Path, bad_restraints
    ) -> None:
        job = self._base_job()
        job["restraints"] = bad_restraints
        json_file = tmp_path / "bad.json"
        json_file.write_text(json.dumps(job))

        with pytest.raises(InputError, match="Invalid restraints.*must be a JSON object"):
            list(load_fold_inputs(json_file))

    @pytest.mark.parametrize("bad_guidance", [["a list"], "a string", 99])
    def test_load_fold_inputs_dict_guidance_wrong_type(
        self, tmp_path: Path, bad_guidance
    ) -> None:
        job = self._base_job()
        job["guidance"] = bad_guidance
        json_file = tmp_path / "bad.json"
        json_file.write_text(json.dumps(job))

        with pytest.raises(InputError, match="Invalid guidance.*must be a JSON object"):
            list(load_fold_inputs(json_file))

    # -- load_fold_inputs (batch list) ------------------------------------------

    def test_load_fold_inputs_batch_restraints_wrong_type(
        self, tmp_path: Path
    ) -> None:
        jobs = [
            {
                "name": "job_0",
                "sequences": [
                    {"proteinChain": {"sequence": "MKTAYIAKQRQ", "count": 1}}
                ],
                "modelSeeds": [1],
                "restraints": "not a dict",
            }
        ]
        json_file = tmp_path / "bad_batch.json"
        json_file.write_text(json.dumps(jobs))

        with pytest.raises(InputError, match="Invalid restraints.*must be a JSON object"):
            list(load_fold_inputs(json_file))

    def test_load_fold_inputs_batch_guidance_wrong_type(
        self, tmp_path: Path
    ) -> None:
        jobs = [
            {
                "name": "job_0",
                "sequences": [
                    {"proteinChain": {"sequence": "MKTAYIAKQRQ", "count": 1}}
                ],
                "modelSeeds": [1],
                "guidance": [1, 2, 3],
            }
        ]
        json_file = tmp_path / "bad_batch.json"
        json_file.write_text(json.dumps(jobs))

        with pytest.raises(InputError, match="Invalid guidance.*must be a JSON object"):
            list(load_fold_inputs(json_file))

    # -- load_restraints_file ---------------------------------------------------

    @pytest.mark.parametrize("bad_restraints", [["a list"], "a string", 99])
    def test_load_restraints_file_restraints_wrong_type(
        self, tmp_path: Path, bad_restraints
    ) -> None:
        data = {"restraints": bad_restraints}
        json_file = tmp_path / "bad_restraints_file.json"
        json_file.write_text(json.dumps(data))

        with pytest.raises(InputError, match="Invalid restraints.*must be a JSON object"):
            load_restraints_file(json_file)

    @pytest.mark.parametrize("bad_guidance", [["a list"], "a string", 99])
    def test_load_restraints_file_guidance_wrong_type(
        self, tmp_path: Path, bad_guidance
    ) -> None:
        data = {"guidance": bad_guidance}
        json_file = tmp_path / "bad_guidance_file.json"
        json_file.write_text(json.dumps(data))

        with pytest.raises(InputError, match="Invalid guidance.*must be a JSON object"):
            load_restraints_file(json_file)
