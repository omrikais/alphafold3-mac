"""Sample ranking for AlphaFold 3 MLX pipeline.

This module provides sample ranking by confidence metrics .

Example:
    ranking = rank_samples(confidence_scores, is_complex=False)
    best_sample = ranking.best_index
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class RankingScores:
    """Confidence scores for a single sample.

    Attributes:
        ptm: Predicted TM-score [0-1].
        iptm: Interface pTM for complexes [0-1].
        mean_plddt: Mean per-atom confidence [0-100].
        plddt_variance: Variance of per-atom confidence.
    """

    ptm: float
    iptm: float
    mean_plddt: float
    plddt_variance: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to JSON-serializable dict."""
        return {
            "ptm": self.ptm,
            "iptm": self.iptm,
            "mean_plddt": self.mean_plddt,
            "plddt_variance": self.plddt_variance,
        }


@dataclass
class SampleRanking:
    """Ranking information for structure samples.

    Attributes:
        ranked_indices: Sample indices in ranked order (best first).
        scores: Per-sample metrics keyed by sample index.
        ranking_metric: Metric used for ranking ("pTM" or "ipTM").
        is_complex: Whether input is a multi-chain complex.
    """

    ranked_indices: list[int]
    scores: dict[int, RankingScores]
    ranking_metric: Literal["pTM", "ipTM"]
    is_complex: bool

    @property
    def best_index(self) -> int:
        """Index of the best-ranked sample."""
        return self.ranked_indices[0]

    @property
    def best_score(self) -> float:
        """Score of the best-ranked sample."""
        metric_name = "ptm" if self.ranking_metric == "pTM" else "iptm"
        return getattr(self.scores[self.best_index], metric_name)

    @property
    def num_samples(self) -> int:
        """Number of samples."""
        return len(self.ranked_indices)

    def to_ranking_debug_dict(self) -> dict[str, Any]:
        """Convert to ranking_debug.json format.

        Returns:
            Dictionary formatted for ranking_debug.json output.
        """
        samples = []
        for rank, idx in enumerate(self.ranked_indices, start=1):
            score = self.scores[idx]
            samples.append({
                "index": idx,
                "rank": rank,
                "ptm": score.ptm,
                "iptm": score.iptm,
                "mean_plddt": score.mean_plddt,
            })

        aggregate = compute_aggregate_metrics(self)

        return {
            "ranking_metric": self.ranking_metric,
            "is_complex": self.is_complex,
            "num_samples": self.num_samples,
            "samples": samples,
            "aggregate_metrics": aggregate,
        }


def rank_samples(
    ptm_scores: list[float],
    iptm_scores: list[float],
    plddt_scores: list[list[float]],
    is_complex: bool,
) -> SampleRanking:
    """Rank samples by confidence metric.

    Uses pTM for monomers and ipTM for complexes, matching official
    AlphaFold 3 behavior.

    Args:
        ptm_scores: pTM scores for each sample.
        iptm_scores: ipTM scores for each sample.
        plddt_scores: Per-residue pLDDT scores for each sample.
        is_complex: Whether this is a multi-chain complex.

    Returns:
        SampleRanking with ranked indices and per-sample scores.
    """
    num_samples = len(ptm_scores)

    # Choose ranking metric
    if is_complex:
        ranking_metric: Literal["pTM", "ipTM"] = "ipTM"
        metric_values = iptm_scores
    else:
        ranking_metric = "pTM"
        metric_values = ptm_scores

    # Sort by metric in descending order
    ranked_indices = sorted(
        range(num_samples),
        key=lambda i: metric_values[i],
        reverse=True,
    )

    # Build per-sample scores
    scores = {}
    for i in range(num_samples):
        plddt_array = plddt_scores[i]
        mean_plddt = sum(plddt_array) / len(plddt_array) if plddt_array else 0.0

        # Compute variance
        if len(plddt_array) > 1:
            variance = sum((x - mean_plddt) ** 2 for x in plddt_array) / len(plddt_array)
        else:
            variance = 0.0

        scores[i] = RankingScores(
            ptm=ptm_scores[i],
            iptm=iptm_scores[i],
            mean_plddt=mean_plddt,
            plddt_variance=variance,
        )

    return SampleRanking(
        ranked_indices=ranked_indices,
        scores=scores,
        ranking_metric=ranking_metric,
        is_complex=is_complex,
    )


def auto_detect_complex(chain_ids: list[str]) -> bool:
    """Auto-detect if input is a complex based on chain IDs.

    Args:
        chain_ids: List of chain identifiers.

    Returns:
        True if multiple unique chain IDs, False otherwise.
    """
    return len(set(chain_ids)) > 1


def compute_aggregate_metrics(ranking: SampleRanking) -> dict[str, float]:
    """Compute aggregate metrics across all samples.

    Args:
        ranking: Sample ranking with per-sample scores.

    Returns:
        Dictionary of aggregate metrics.
    """
    if not ranking.scores:
        return {
            "best_ptm": 0.0,
            "mean_plddt_all_samples": 0.0,
            "plddt_variance": 0.0,
        }

    # Best pTM (from best-ranked sample)
    best_score = ranking.scores[ranking.best_index]
    best_ptm = best_score.ptm

    # Mean pLDDT across all samples
    mean_plddts = [s.mean_plddt for s in ranking.scores.values]
    mean_plddt_all = sum(mean_plddts) / len(mean_plddts)

    # Variance of mean pLDDT across samples
    if len(mean_plddts) > 1:
        plddt_variance = sum((x - mean_plddt_all) ** 2 for x in mean_plddts) / len(mean_plddts)
    else:
        plddt_variance = 0.0

    return {
        "best_ptm": best_ptm,
        "mean_plddt_all_samples": mean_plddt_all,
        "plddt_variance": plddt_variance,
    }
