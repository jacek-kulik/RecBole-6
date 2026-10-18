"""Jacek's independent beyond-accuracy metrics, reviewed by Caio."""

from collections.abc import Mapping
from itertools import combinations
from math import log, log2

from ..data import (
    POPULARITY_GROUPS,
    TrainingStatistics,
    genre_distribution,
    popularity_distribution,
)
from ._validation import validate_rankings
from .types import (
    BeyondAccuracyResults,
    GenderFairnessResults,
    ItemGroupExposure,
    MetricAvailability,
    QualityGap,
    RankingInput,
)

METRIC_DEFINITIONS = {
    "ILD": "Mean unordered-pair Jaccard distance between movie genre sets; undefined for K=1",
    "MeanSelfInformation": "Mean -log2((training_count + 1) / (training_interactions + catalogue_size))",
    "GenreMiscalibration": "KL(p || 0.99*q + 0.01*p), natural log; equal movie weights and fractional genres",
    "GRU": "Absolute F/M mean accuracy gap, with both means and signed F-minus-M difference",
    "CatalogueCoverage": "Distinct recommended movies / checkpoint catalogue size",
    "ExposureGini": "Gini over all catalogue movies, including zeros; exposure = 1/log2(rank + 1)",
    "UPD": "Base-2 Jensen-Shannon divergence of unweighted training/recommendation head/middle/tail shares",
    "ItemGroupExposure": "Discounted exposure total, share, and per-movie exposure in each popularity group"
}


def gender_fairness(
        per_user: Mapping[str, Mapping[str, float]],
        genders: Mapping[str, str],
        k: int
) -> GenderFairnessResults:
    """Compare recorded F/M users within the supplied evaluation cohort."""
    names = [f"{name}@{k}" for name in ("Precision", "Recall", "F1", "MRR", "NDCG", "MAP")]
    members = {gender: [user for user in per_user if genders.get(user) == gender] for gender in ("F", "M")}
    counts = {gender: len(users) for gender, users in members.items()}
    counts["unknown"] = len(per_user) - counts["F"] - counts["M"]
    means = {
        gender: {name: sum(per_user[user][name] for user in users) / len(users) for name in names}
        if users else None
        for gender, users in members.items()
    }
    female, male = means["F"], means["M"]
    gaps: dict[str, QualityGap] = {}
    for name in names:
        difference = female[name] - male[name] if female is not None and male is not None else None
        gaps[name] = {"absolute": abs(difference) if difference is not None else None,
                      "signed_f_minus_m": difference}
    return {
        "counts": counts, "means": means, "gaps": gaps,
        "n_users": counts["F"] + counts["M"],
        "reason": None if female is not None and male is not None else "Both recorded F and M users are needed",
    }


def jensen_shannon(
        p: Mapping[str, float],
        q: Mapping[str, float]
) -> float:
    """Base-2 divergence (not its square root); zero probability terms vanish."""
    result = 0.0
    for category in p.keys() | q.keys():
        left, right = p.get(category, 0.0), q.get(category, 0.0)
        middle = (left + right) / 2
        if left:
            result += .5 * left * log2(left / middle)
        if right:
            result += .5 * right * log2(right / middle)
    return result


def exposure_gini(
        exposures: Mapping[str, float]
) -> float:
    ordered = sorted(exposures.values())
    total = sum(ordered)
    weighted = sum((2 * rank - len(ordered) - 1) * value for rank, value in enumerate(ordered, 1))
    return weighted / (len(ordered) * total)


def evaluate_beyond_accuracy(
        rankings: RankingInput,
        statistics: TrainingStatistics,
        per_user_accuracy: Mapping[str, Mapping[str, float]],
        k: int = 10,
) -> BeyondAccuracyResults:
    """Evaluate one cohort. Empty cohorts return explicit unavailable values.

    The user profile and item popularity never include held-out interactions.
    Coverage, Gini and item-group exposure are calculated over this cohort's
    lists, rather than averaged from another cohort's measurements."""
    rows = validate_rankings(rankings, k, set(statistics.catalogue)) if rankings else {}
    if rows.keys() != per_user_accuracy.keys():
        raise ValueError("Accuracy and beyond-accuracy cohorts must have the same users")

    names = [f"{name}@{k}" for name in ("ILD", "MeanSelfInformation", "GenreMiscalibration", "UPD")]
    per_user: dict[str, dict[str, float | None]] = {}
    exposure = dict.fromkeys(statistics.catalogue, 0.0)
    novelty_denominator = statistics.n_interactions + len(statistics.catalogue)
    metadata_reason = statistics.pending.get("metadata", "Prepared genre metadata or training profiles are unavailable")
    recommended: set[str] = set()
    for user, items in rows.items():
        distances = []
        has_genres = all(statistics.item_features.get(item) for item in items)
        if has_genres:
            for left, right in combinations(items, 2):
                a, b = set(statistics.item_features[left]), set(statistics.item_features[right])
                distances.append(1 - len(a & b) / len(a | b))
        novelty = sum(-log2((statistics.item_counts[item] + 1) / novelty_denominator) for item in items) / k
        profile = statistics.genre_profiles.get(user)
        miscalibration = None
        if profile and has_genres:
            distribution = genre_distribution(items, statistics.item_features)
            miscalibration = sum(
                probability * log(probability / (.99 * distribution.get(genre, 0) + .01 * probability))
                for genre, probability in profile.items()
            )
        popularity = statistics.popularity_profiles.get(user)
        deviation = jensen_shannon(popularity,
                                   popularity_distribution(items, statistics.item_groups)) if popularity else None
        per_user[user] = dict(zip(names, (
            sum(distances) / len(distances) if distances else None, novelty, miscalibration, deviation,
        )))
        for rank, item in enumerate(items, 1):
            exposure[item] += 1 / log2(rank + 1)
        recommended.update(items)

    aggregate: dict[str, float | None] = {}
    availability: dict[str, MetricAvailability] = {}
    for name in names:
        values: list[float] = []
        for row in per_user.values():
            value = row[name]
            if value is not None:
                values.append(value)
        aggregate[name] = sum(values) / len(values) if values else None
        reason = None
        if not rows:
            reason = "No evaluation users in this cohort"
        elif name == f"ILD@{k}" and k == 1:
            reason = "ILD needs at least two recommendations"
        elif len(values) < len(rows):
            reason = metadata_reason if name in (f"ILD@{k}",
                                                 f"GenreMiscalibration@{k}") else f"{len(rows) - len(values)} users have no training profile"
        availability[name] = {"n_users": len(values), "reason": reason}

    aggregate[f"CatalogueCoverage@{k}"] = len(recommended) / len(statistics.catalogue) if rows else None
    aggregate[f"ExposureGini@{k}"] = exposure_gini(exposure) if rows else None
    for name in (f"CatalogueCoverage@{k}", f"ExposureGini@{k}"):
        availability[name] = {"n_users": len(rows), "reason": None if rows else "No evaluation users in this cohort"}

    total_exposure = sum(exposure.values())
    item_exposure: dict[str, ItemGroupExposure] = {}
    for group in POPULARITY_GROUPS:
        members = [item for item in statistics.catalogue if statistics.item_groups[item] == group]
        amount = sum(exposure[item] for item in members)
        item_exposure[group] = {
            "n_items": len(members), "total": amount if rows else None,
            "share": amount / total_exposure if rows else None,
            "per_item": amount / len(members) if rows and members else None,
            "reason": "No evaluation users in this cohort" if not rows else "No movies in this group" if not members else None,
        }
    fairness = gender_fairness(per_user_accuracy, statistics.gender_groups, k)
    if statistics.pending.get("metadata") and not statistics.gender_groups:
        fairness["reason"] = metadata_reason
    return {
        "per_user": per_user, "aggregate": aggregate, "availability": availability,
        "gender_fairness": fairness,
        "item_group_exposure": item_exposure,
    }
