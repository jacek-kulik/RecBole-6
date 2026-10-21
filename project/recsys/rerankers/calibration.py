"""Jacek's greedy genre/popularity calibration; Victor reviews."""

from collections import Counter
from typing import Literal

from ..contracts import Rankings, RerankContext, ScoreTable
from ..data import genre_distribution
from ..metrics.beyond_accuracy import jensen_shannon, smoothed_kl
from ._common import MissingMetadataError, normalized_scores, validate_profile


def rerank(scores: ScoreTable, context: RerankContext, k: int, strength: float,
           *, attribute: Literal["genre", "popularity"] = "genre") -> Rankings:
    """Choose K from the supplied pool. Profiles must come only from training.

    Genre calibration follows the lecture's smoothed KL objective. Popularity
    calibration adapts it using the same Jensen-Shannon divergence as UPD.
    The runner truncates full candidate scores before calling this function.
    """
    relevance = normalized_scores(scores, k, strength)
    if attribute not in ("genre", "popularity"):
        raise ValueError("Calibration attribute must be genre or popularity")
    if strength == 0:
        return scores.top_k(k)
    profiles = context.user_profiles if attribute == "genre" else context.popularity_profiles
    distributions = {}
    for item in {item for row in scores.scores.values() for item in row}:
        if attribute == "genre":
            if not context.item_features.get(item):
                raise MissingMetadataError(f"No genre metadata for item {item}")
            distributions[item] = genre_distribution([item], context.item_features)
        else:
            if item not in context.item_groups:
                raise MissingMetadataError(f"No popularity group for item {item}")
            distributions[item] = {context.item_groups[item]: 1.0}

    divergence = smoothed_kl if attribute == "genre" else jensen_shannon
    rankings = {}
    for user, row in sorted(scores.scores.items()):
        profile = profiles.get(user, {})
        validate_profile(profile, user)
        chosen: list[str] = []
        counts: Counter[str] = Counter()
        remaining = set(row)
        for length in range(1, k + 1):
            choices = []
            for item in remaining:
                addition = distributions[item]
                q = {category: (counts[category] + addition.get(category, 0)) / length
                     for category in counts.keys() | addition.keys()}
                value = (1 - strength) * relevance[user][item] - strength * divergence(profile, q)
                choices.append((-value, -row[item], item))

            item = min(choices)[2]
            chosen.append(item)
            counts.update(distributions[item])
            remaining.remove(item)
        rankings[user] = chosen
    return rankings
