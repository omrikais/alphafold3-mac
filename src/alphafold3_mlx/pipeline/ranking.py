"""Sample ranking for AlphaFold 3 MLX pipeline.

This module provides sample ranking by confidence metrics.

Example:
    ranking = rank_samples(confidence_scores, is_complex=False)
    best_sample = ranking.best_index
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


_IPTM_WEIGHT = 0.8
_FRACTION_DISORDERED_WEIGHT = 0.5
_CLASH_PENALTY = 100.0


def compute_structure_quality_metrics(
    *,
    atom_positions: Any,
    atom_mask: Any,
    atom_names: Any,
    element_symbols: Any,
    comp_ids: Any,
    chain_ids: Any,
    residue_indices: Any,
    chain_types: Any,
) -> tuple[list[float], list[bool]]:
    """Compute the structure-dependent terms in the official AF3 ranking."""
    import numpy as np

    from alphafold3 import structure
    from alphafold3.model import confidences

    positions = np.asarray(atom_positions)
    masks = np.asarray(atom_mask, dtype=bool)
    names = np.asarray(atom_names, dtype=object)
    elements = np.asarray(element_symbols, dtype=object)
    residues = np.asarray(comp_ids, dtype=object)
    chains = np.asarray(chain_ids, dtype=object)
    residue_ids = np.asarray(residue_indices, dtype=np.int32)
    types = np.asarray(chain_types, dtype=object)

    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError("atom_positions must have shape [samples, residues, atoms, 3]")
    if masks.shape != positions.shape[:-1]:
        raise ValueError("atom_mask must match atom_positions sample/residue/atom axes")
    if names.shape != positions.shape[1:3] or elements.shape != names.shape:
        raise ValueError("atom metadata must match the residue/atom layout")
    for values, field_name in (
        (residues, "comp_ids"),
        (chains, "chain_ids"),
        (residue_ids, "residue_indices"),
        (types, "chain_types"),
    ):
        if values.shape != (positions.shape[1],):
            raise ValueError(f"{field_name} must match the residue axis")

    fraction_disordered_scores = []
    has_clash_scores = []
    for sample_positions, sample_mask in zip(positions, masks, strict=True):
        predicted_structure = structure.from_res_arrays(
            atom_mask=sample_mask,
            atom_x=sample_positions[..., 0],
            atom_y=sample_positions[..., 1],
            atom_z=sample_positions[..., 2],
            atom_name=names,
            atom_element=elements,
            chain_id=chains,
            chain_type=types,
            res_id=residue_ids,
            res_name=residues,
        )
        fraction_disordered_scores.append(
            float(confidences.fraction_disordered(predicted_structure))
        )
        has_clash_scores.append(bool(confidences.has_clash(predicted_structure)))

    return fraction_disordered_scores, has_clash_scores


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
    fraction_disordered: float = 0.0
    has_clash: bool = False
    ranking_score: float | None = None

    def to_dict(self) -> dict[str, float]:
        """Convert to JSON-serializable dict."""
        return {
            "ptm": self.ptm,
            "iptm": self.iptm,
            "mean_plddt": self.mean_plddt,
            "plddt_variance": self.plddt_variance,
            "fraction_disordered": self.fraction_disordered,
            "has_clash": self.has_clash,
            "ranking_score": self.ranking_score,
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
    ranking_metric: Literal["pTM", "ipTM", "ranking_score"]
    is_complex: bool

    @property
    def best_index(self) -> int:
        """Index of the best-ranked sample."""
        return self.ranked_indices[0]

    @property
    def best_score(self) -> float:
        """Score of the best-ranked sample."""
        if self.ranking_metric == "ranking_score":
            score = self.scores[self.best_index].ranking_score
            if score is None:
                raise ValueError("ranking_score metric selected without scores")
            return score
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
                "ranking_score": score.ranking_score,
                "fraction_disordered": score.fraction_disordered,
                "has_clash": score.has_clash,
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
    fraction_disordered_scores: list[float] | None = None,
    has_clash_scores: list[bool] | None = None,
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

    use_official_score = (
        fraction_disordered_scores is not None or has_clash_scores is not None
    )
    if fraction_disordered_scores is None:
        fraction_disordered_scores = [0.0] * num_samples
    if has_clash_scores is None:
        has_clash_scores = [False] * num_samples
    if len(fraction_disordered_scores) != num_samples:
        raise ValueError("fraction_disordered_scores must match sample count")
    if len(has_clash_scores) != num_samples:
        raise ValueError("has_clash_scores must match sample count")

    if use_official_score:
        ranking_metric: Literal["pTM", "ipTM", "ranking_score"] = "ranking_score"
        metric_values = [
            (
                _IPTM_WEIGHT * iptm_scores[i]
                + (1.0 - _IPTM_WEIGHT) * ptm_scores[i]
                if is_complex
                else ptm_scores[i]
            )
            + _FRACTION_DISORDERED_WEIGHT * fraction_disordered_scores[i]
            - _CLASH_PENALTY * has_clash_scores[i]
            for i in range(num_samples)
        ]
    elif is_complex:
        ranking_metric = "ipTM"
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
            fraction_disordered=fraction_disordered_scores[i],
            has_clash=has_clash_scores[i],
            ranking_score=metric_values[i] if use_official_score else None,
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
    mean_plddts = [s.mean_plddt for s in ranking.scores.values()]
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
