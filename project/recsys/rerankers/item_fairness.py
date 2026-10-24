"""Jacek's cohort-wide exposure heuristic; Gabriel reviews."""

from collections import defaultdict
from math import log2

from ..contracts import Rankings, RerankContext, ScoreTable
from ._common import normalized_scores


def rerank(scores: ScoreTable, context: RerankContext, k: int, strength: float) -> Rankings:
    """Balance relevance with inverse accumulated discounted item exposure.

    Visit users in external-ID order for each rank. This deterministic greedy
    heuristic does not guarantee minimum Gini or equal exposure. Its state is
    local to this cohort/run; it never consumes evaluation relevance labels.
    """
    relevance = normalized_scores(scores, k, strength)
    if strength == 0:
        return scores.top_k(k)
    rankings: Rankings = {user: [] for user in sorted(scores.scores)}
    remaining = {user: set(row) for user, row in scores.scores.items()}
    exposure: dict[str, float] = defaultdict(float)
    for rank in range(1, k + 1):
        for user in rankings:
            choices = []
            for item in remaining[user]:
                value = (1 - strength) * relevance[user][item] + strength / (1 + exposure[item])
                choices.append((-value, -scores.scores[user][item], item))

            item = min(choices)[2]
            rankings[user].append(item)
            remaining[user].remove(item)
            exposure[item] += 1 / log2(rank + 1)
    return rankings
