"""Gabriel's independent genre, long-tail discovery, and coverage rerankers."""

from math import log2
from typing import Literal

from ..contracts import Rankings, RerankContext, ScoreTable
from ._common import MissingMetadataError, normalized_scores


def rerank(scores: ScoreTable, context: RerankContext, k: int, strength: float,
           *, attribute: Literal["genre", "popularity", "coverage"] = "genre") -> Rankings:
    """Select K eligible items using exactly one diversification objective.

    Genre maximizes mean Jaccard distance to the selected prefix. Popularity
    promotes training-only self-information, min-max normalized within each
    candidate pool. Coverage rewards movies not yet recommended in this call;
    rank rounds visit users in sorted external-ID order. No held-out labels or
    persistent exposure counters are used. Greedy choices do not guarantee a
    globally optimal target metric or an improvement on test data.
    """
    relevance = normalized_scores(scores, k, strength)
    if attribute not in ("genre", "popularity", "coverage"):
        raise ValueError("Diversification attribute must be genre, popularity or coverage")
    if strength == 0:
        return scores.top_k(k)

    candidates = {item for row in scores.scores.values() for item in row}
    features = {}
    novelty = {}
    if attribute == "genre":
        for item in sorted(candidates):
            if not context.item_features.get(item):
                raise MissingMetadataError(f"No genre metadata for item {item}")
            features[item] = set(context.item_features[item])
    elif attribute == "popularity":
        if not context.item_counts:
            raise MissingMetadataError("No training item counts for popularity diversification")
        if any(not isinstance(item, str) or not item or type(count) is not int or count < 0
               for item, count in context.item_counts.items()):
            raise ValueError("Training item counts must map external IDs to nonnegative integers")
        for item in sorted(candidates):
            if item not in context.item_counts:
                raise MissingMetadataError(f"No training count for item {item}")
        denominator = sum(context.item_counts.values()) + len(context.item_counts)
        information = {item: -log2((context.item_counts[item] + 1) / denominator)
                       for item in candidates}
        for user, row in scores.scores.items():
            low, high = min(information[i] for i in row), max(information[i] for i in row)
            novelty[user] = {item: (information[item] - low) / (high - low) if high > low else 0.
                             for item in row}

    rankings: Rankings = {user: [] for user in sorted(scores.scores)}
    remaining = {user: set(row) for user, row in scores.scores.items()}
    distances = {user: dict.fromkeys(row, 0.) for user, row in scores.scores.items()} if attribute == "genre" else {}
    covered: set[str] = set()
    for rank in range(k):
        for user, chosen in rankings.items():
            row = scores.scores[user]

            choices = []
            for item in remaining[user]:
                if attribute == "genre":
                    bonus = distances[user][item] / rank if rank else 0.
                elif attribute == "popularity":
                    bonus = novelty[user][item]
                else:
                    bonus = float(item not in covered)
                value = (1 - strength) * relevance[user][item] + strength * bonus
                choices.append((-value, -row[item], item))

            selected = min(choices)[2]
            chosen.append(selected)
            remaining[user].remove(selected)
            covered.add(selected)
            if attribute == "genre":
                a = features[selected]
                for item in remaining[user]:
                    b = features[item]
                    distances[user][item] += 1 - len(a & b) / len(a | b)
    return rankings
