"""Integration tests for sample ranking (User Story 4).

These tests verify ranking functionality according to:
- Sample ranking correctly orders structures by confidence metric
- Rank samples by pTM (monomers) or ipTM (complexes)
- Output structures in ranked order
- Record ranking metrics in ranking_debug.json
- Compute aggregate metrics

Tests:
- test_ptm_ordering
- test_ranking_debug_content
- test_aggregate_metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestPTMOrdering:
    """Tests for pTM-based sample ranking."""

    def test_ptm_ordering_monomer(self) -> None:
        """Verify samples are ranked by pTM for monomers."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        # Create test data with known pTM scores
        ptm_scores = [0.75, 0.92, 0.80, 0.88, 0.70]
        iptm_scores = [0.0, 0.0, 0.0, 0.0, 0.0]  # Not used for monomers
        plddt_scores = [
            [85.0, 86.0, 87.0],  # Sample 0
            [90.0, 91.0, 92.0],  # Sample 1
            [80.0, 81.0, 82.0],  # Sample 2
            [88.0, 89.0, 90.0],  # Sample 3
            [70.0, 71.0, 72.0],  # Sample 4
        ]

        ranking = rank_samples(
            ptm_scores=ptm_scores,
            iptm_scores=iptm_scores,
            plddt_scores=plddt_scores,
            is_complex=False,
        )

        # For monomers, should rank by pTM
        assert ranking.ranking_metric == "pTM"

        # Expected order by pTM: sample 1 (0.92), 3 (0.88), 2 (0.80), 0 (0.75), 4 (0.70)
        expected_order = [1, 3, 2, 0, 4]
        assert ranking.ranked_indices == expected_order

        # Verify rank 1 has highest pTM
        assert ranking.best_index == 1
        assert ranking.best_score == 0.92

    def test_iptm_ordering_complex(self) -> None:
        """Verify samples are ranked by ipTM for complexes."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        # Create test data with known ipTM scores
        ptm_scores = [0.85, 0.80, 0.90, 0.75, 0.82]
        iptm_scores = [0.78, 0.92, 0.70, 0.88, 0.65]  # Used for complexes
        plddt_scores = [
            [85.0, 86.0],  # Sample 0
            [90.0, 91.0],  # Sample 1
            [80.0, 81.0],  # Sample 2
            [88.0, 89.0],  # Sample 3
            [70.0, 71.0],  # Sample 4
        ]

        ranking = rank_samples(
            ptm_scores=ptm_scores,
            iptm_scores=iptm_scores,
            plddt_scores=plddt_scores,
            is_complex=True,
        )

        # For complexes, should rank by ipTM
        assert ranking.ranking_metric == "ipTM"

        # Expected order by ipTM: sample 1 (0.92), 3 (0.88), 0 (0.78), 2 (0.70), 4 (0.65)
        expected_order = [1, 3, 0, 2, 4]
        assert ranking.ranked_indices == expected_order

        # Verify rank 1 has highest ipTM
        assert ranking.best_index == 1
        assert ranking.best_score == 0.92

    def test_ranking_preserves_all_samples(self) -> None:
        """Verify all samples are included in ranking."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        num_samples = 7
        ptm_scores = [0.5 + 0.05 * i for i in range(num_samples)]
        iptm_scores = [0.0] * num_samples
        plddt_scores = [[80.0, 81.0, 82.0] for _ in range(num_samples)]

        ranking = rank_samples(
            ptm_scores=ptm_scores,
            iptm_scores=iptm_scores,
            plddt_scores=plddt_scores,
            is_complex=False,
        )

        # All samples should be in ranked_indices
        assert len(ranking.ranked_indices) == num_samples
        assert set(ranking.ranked_indices) == set(range(num_samples))

        # All samples should have scores
        assert len(ranking.scores) == num_samples

    def test_ranking_handles_tied_scores(self) -> None:
        """Verify ranking handles tied pTM scores gracefully."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        # Two samples with identical pTM
        ptm_scores = [0.85, 0.85, 0.80]
        iptm_scores = [0.0, 0.0, 0.0]
        plddt_scores = [[80.0], [81.0], [75.0]]

        ranking = rank_samples(
            ptm_scores=ptm_scores,
            iptm_scores=iptm_scores,
            plddt_scores=plddt_scores,
            is_complex=False,
        )

        # All samples should be ranked
        assert len(ranking.ranked_indices) == 3
        # Sample 2 (0.80) should be last
        assert ranking.ranked_indices[-1] == 2
        # Tied samples (0, 1) should be in top 2 positions
        assert set(ranking.ranked_indices[:2]) == {0, 1}

    def test_ranking_single_sample(self) -> None:
        """Verify ranking works with single sample."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        ranking = rank_samples(
            ptm_scores=[0.85],
            iptm_scores=[0.0],
            plddt_scores=[[90.0, 91.0, 92.0]],
            is_complex=False,
        )

        assert ranking.ranked_indices == [0]
        assert ranking.best_index == 0
        assert ranking.num_samples == 1


class TestRankingDebugContent:
    """Tests for ranking_debug.json content."""

    def test_ranking_debug_content(self) -> None:
        """Verify ranking_debug.json contains required fields."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        ptm_scores = [0.75, 0.92, 0.80]
        iptm_scores = [0.0, 0.0, 0.0]
        plddt_scores = [
            [85.0, 86.0, 87.0],
            [90.0, 91.0, 92.0],
            [80.0, 81.0, 82.0],
        ]

        ranking = rank_samples(
            ptm_scores=ptm_scores,
            iptm_scores=iptm_scores,
            plddt_scores=plddt_scores,
            is_complex=False,
        )

        debug_dict = ranking.to_ranking_debug_dict()

        # Required top-level fields
        assert "ranking_metric" in debug_dict
        assert "is_complex" in debug_dict
        assert "num_samples" in debug_dict
        assert "samples" in debug_dict
        assert "aggregate_metrics" in debug_dict

        # Verify ranking_metric
        assert debug_dict["ranking_metric"] == "pTM"

        # Verify is_complex
        assert debug_dict["is_complex"] is False

        # Verify num_samples
        assert debug_dict["num_samples"] == 3

    def test_ranking_debug_samples_format(self) -> None:
        """Verify samples array format in ranking_debug.json."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        ptm_scores = [0.75, 0.92, 0.80]
        iptm_scores = [0.60, 0.70, 0.65]
        plddt_scores = [
            [85.0, 86.0, 87.0],
            [90.0, 91.0, 92.0],
            [80.0, 81.0, 82.0],
        ]

        ranking = rank_samples(
            ptm_scores=ptm_scores,
            iptm_scores=iptm_scores,
            plddt_scores=plddt_scores,
            is_complex=False,
        )

        debug_dict = ranking.to_ranking_debug_dict()
        samples = debug_dict["samples"]

        # Should have 3 sample entries
        assert len(samples) == 3

        # Verify each sample has required fields
        for sample in samples:
            assert "index" in sample
            assert "rank" in sample
            assert "ptm" in sample
            assert "iptm" in sample
            assert "mean_plddt" in sample

        # Verify samples are in rank order
        for i, sample in enumerate(samples):
            assert sample["rank"] == i + 1

        # Verify best sample is first (sample 1 with pTM 0.92)
        assert samples[0]["index"] == 1
        assert samples[0]["rank"] == 1
        assert samples[0]["ptm"] == 0.92

    def test_ranking_debug_preserves_original_indices(self) -> None:
        """Verify original sample indices are preserved in ranking_debug."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        ptm_scores = [0.50, 0.90, 0.70, 0.80, 0.60]
        iptm_scores = [0.0] * 5
        plddt_scores = [[80.0, 81.0] for _ in range(5)]

        ranking = rank_samples(
            ptm_scores=ptm_scores,
            iptm_scores=iptm_scores,
            plddt_scores=plddt_scores,
            is_complex=False,
        )

        debug_dict = ranking.to_ranking_debug_dict()
        samples = debug_dict["samples"]

        # Expected order by pTM: 1 (0.90), 3 (0.80), 2 (0.70), 4 (0.60), 0 (0.50)
        expected_indices = [1, 3, 2, 4, 0]
        actual_indices = [s["index"] for s in samples]

        assert actual_indices == expected_indices

    def test_ranking_debug_json_serializable(self) -> None:
        """Verify ranking_debug dict is JSON serializable."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        ranking = rank_samples(
            ptm_scores=[0.75, 0.92],
            iptm_scores=[0.0, 0.0],
            plddt_scores=[[85.0, 86.0], [90.0, 91.0]],
            is_complex=False,
        )

        debug_dict = ranking.to_ranking_debug_dict()

        # Should not raise
        json_str = json.dumps(debug_dict)
        assert isinstance(json_str, str)

        # Should round-trip correctly
        parsed = json.loads(json_str)
        assert parsed["ranking_metric"] == "pTM"
        assert parsed["num_samples"] == 2

    def test_ranking_debug_for_complex(self) -> None:
        """Verify ranking_debug reflects complex status."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        ranking = rank_samples(
            ptm_scores=[0.80, 0.85],
            iptm_scores=[0.90, 0.75],  # Used for complex ranking
            plddt_scores=[[85.0], [80.0]],
            is_complex=True,
        )

        debug_dict = ranking.to_ranking_debug_dict()

        assert debug_dict["ranking_metric"] == "ipTM"
        assert debug_dict["is_complex"] is True

        # Sample 0 should be rank 1 (ipTM 0.90 > 0.75)
        assert debug_dict["samples"][0]["index"] == 0
        assert debug_dict["samples"][0]["rank"] == 1


class TestAggregateMetrics:
    """Tests for aggregate metrics computation."""

    def test_aggregate_metrics_basic(self) -> None:
        """Verify aggregate metrics are computed correctly."""
        from alphafold3_mlx.pipeline.ranking import rank_samples, compute_aggregate_metrics

        ptm_scores = [0.75, 0.92, 0.80]
        iptm_scores = [0.0, 0.0, 0.0]
        plddt_scores = [
            [85.0, 86.0, 87.0],  # mean = 86.0
            [90.0, 91.0, 92.0],  # mean = 91.0
            [80.0, 81.0, 82.0],  # mean = 81.0
        ]

        ranking = rank_samples(
            ptm_scores=ptm_scores,
            iptm_scores=iptm_scores,
            plddt_scores=plddt_scores,
            is_complex=False,
        )

        aggregate = compute_aggregate_metrics(ranking)

        # Required fields
        assert "best_ptm" in aggregate
        assert "mean_plddt_all_samples" in aggregate
        assert "plddt_variance" in aggregate

        # Best pTM is from sample 1 (rank 1)
        assert aggregate["best_ptm"] == 0.92

        # Mean pLDDT across samples: (86.0 + 91.0 + 81.0) / 3 = 86.0
        assert abs(aggregate["mean_plddt_all_samples"] - 86.0) < 0.01

    def test_aggregate_metrics_variance(self) -> None:
        """Verify pLDDT variance is computed correctly."""
        from alphafold3_mlx.pipeline.ranking import rank_samples, compute_aggregate_metrics

        # Samples with known variance
        plddt_scores = [
            [80.0],  # mean = 80.0
            [90.0],  # mean = 90.0
            [85.0],  # mean = 85.0
        ]
        # Overall mean = 85.0
        # Variance = ((80-85)^2 + (90-85)^2 + (85-85)^2) / 3 = (25 + 25 + 0) / 3 = 16.67

        ranking = rank_samples(
            ptm_scores=[0.8, 0.9, 0.85],
            iptm_scores=[0.0, 0.0, 0.0],
            plddt_scores=plddt_scores,
            is_complex=False,
        )

        aggregate = compute_aggregate_metrics(ranking)

        expected_variance = 50.0 / 3.0  # ~16.67
        assert abs(aggregate["plddt_variance"] - expected_variance) < 0.01

    def test_aggregate_metrics_single_sample(self) -> None:
        """Verify aggregate metrics work with single sample."""
        from alphafold3_mlx.pipeline.ranking import rank_samples, compute_aggregate_metrics

        ranking = rank_samples(
            ptm_scores=[0.88],
            iptm_scores=[0.0],
            plddt_scores=[[85.0, 86.0, 87.0]],  # mean = 86.0
            is_complex=False,
        )

        aggregate = compute_aggregate_metrics(ranking)

        assert aggregate["best_ptm"] == 0.88
        assert abs(aggregate["mean_plddt_all_samples"] - 86.0) < 0.01
        # Variance with single sample should be 0
        assert aggregate["plddt_variance"] == 0.0

    def test_aggregate_metrics_in_debug_output(self) -> None:
        """Verify aggregate metrics appear in ranking_debug dict."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        ranking = rank_samples(
            ptm_scores=[0.75, 0.92, 0.80],
            iptm_scores=[0.0, 0.0, 0.0],
            plddt_scores=[
                [85.0, 86.0],
                [90.0, 91.0],
                [80.0, 81.0],
            ],
            is_complex=False,
        )

        debug_dict = ranking.to_ranking_debug_dict()
        aggregate = debug_dict["aggregate_metrics"]

        # All required fields present
        assert "best_ptm" in aggregate
        assert "mean_plddt_all_samples" in aggregate
        assert "plddt_variance" in aggregate

        # Best pTM from best sample
        assert aggregate["best_ptm"] == 0.92

    def test_aggregate_metrics_empty_ranking(self) -> None:
        """Verify aggregate metrics handle edge case of empty ranking."""
        from alphafold3_mlx.pipeline.ranking import SampleRanking, compute_aggregate_metrics

        # Create empty ranking (edge case)
        empty_ranking = SampleRanking(
            ranked_indices=[],
            scores={},
            ranking_metric="pTM",
            is_complex=False,
        )

        aggregate = compute_aggregate_metrics(empty_ranking)

        # Should return zeros for empty ranking
        assert aggregate["best_ptm"] == 0.0
        assert aggregate["mean_plddt_all_samples"] == 0.0
        assert aggregate["plddt_variance"] == 0.0


class TestAutoDetectComplex:
    """Tests for auto-detection of complex (multi-chain) inputs."""

    def test_auto_detect_complex_single_chain(self) -> None:
        """Verify single chain is not detected as complex."""
        from alphafold3_mlx.pipeline.ranking import auto_detect_complex

        assert auto_detect_complex(["A"]) is False
        assert auto_detect_complex(["A", "A", "A"]) is False  # Same chain ID

    def test_auto_detect_complex_multiple_chains(self) -> None:
        """Verify multiple chains are detected as complex."""
        from alphafold3_mlx.pipeline.ranking import auto_detect_complex

        assert auto_detect_complex(["A", "B"]) is True
        assert auto_detect_complex(["A", "B", "C"]) is True
        assert auto_detect_complex(["A", "A", "B"]) is True  # Mixed

    def test_auto_detect_complex_empty_chains(self) -> None:
        """Verify empty chain list is not detected as complex."""
        from alphafold3_mlx.pipeline.ranking import auto_detect_complex

        assert auto_detect_complex([]) is False


class TestRankingScores:
    """Tests for RankingScores dataclass."""

    def test_ranking_scores_to_dict(self) -> None:
        """Verify RankingScores.to_dict() returns correct format."""
        from alphafold3_mlx.pipeline.ranking import RankingScores

        scores = RankingScores(
            ptm=0.85,
            iptm=0.72,
            mean_plddt=87.5,
            plddt_variance=12.3,
        )

        d = scores.to_dict()

        assert d["ptm"] == 0.85
        assert d["iptm"] == 0.72
        assert d["mean_plddt"] == 87.5
        assert d["plddt_variance"] == 12.3

    def test_ranking_scores_json_serializable(self) -> None:
        """Verify RankingScores.to_dict() is JSON serializable."""
        from alphafold3_mlx.pipeline.ranking import RankingScores

        scores = RankingScores(
            ptm=0.85,
            iptm=0.72,
            mean_plddt=87.5,
            plddt_variance=12.3,
        )

        # Should not raise
        json_str = json.dumps(scores.to_dict())
        assert isinstance(json_str, str)


class TestSampleRankingProperties:
    """Tests for SampleRanking property accessors."""

    def test_best_index_property(self) -> None:
        """Verify best_index returns first ranked sample."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        ranking = rank_samples(
            ptm_scores=[0.50, 0.90, 0.70],
            iptm_scores=[0.0, 0.0, 0.0],
            plddt_scores=[[80.0], [90.0], [85.0]],
            is_complex=False,
        )

        # Sample 1 has highest pTM (0.90)
        assert ranking.best_index == 1

    def test_best_score_property_monomer(self) -> None:
        """Verify best_score returns pTM for monomers."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        ranking = rank_samples(
            ptm_scores=[0.50, 0.90, 0.70],
            iptm_scores=[0.60, 0.80, 0.75],
            plddt_scores=[[80.0], [90.0], [85.0]],
            is_complex=False,
        )

        # Best pTM is 0.90
        assert ranking.best_score == 0.90

    def test_best_score_property_complex(self) -> None:
        """Verify best_score returns ipTM for complexes."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        ranking = rank_samples(
            ptm_scores=[0.90, 0.50, 0.70],
            iptm_scores=[0.60, 0.80, 0.75],
            plddt_scores=[[80.0], [90.0], [85.0]],
            is_complex=True,
        )

        # Best ipTM is 0.80 (sample 1)
        assert ranking.best_score == 0.80

    def test_num_samples_property(self) -> None:
        """Verify num_samples returns correct count."""
        from alphafold3_mlx.pipeline.ranking import rank_samples

        ranking = rank_samples(
            ptm_scores=[0.5, 0.6, 0.7, 0.8, 0.9],
            iptm_scores=[0.0] * 5,
            plddt_scores=[[80.0] for _ in range(5)],
            is_complex=False,
        )

        assert ranking.num_samples == 5
