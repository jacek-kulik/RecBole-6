"""Matched reranker orders for a frozen score-weighted hybrid.

Inputs are already standardized on full eligible candidates, then restricted
to one common candidate pool. Never standardize again after reranking.
"""

import numpy as np

from ..contracts import ScoreTable, require_aligned


def score_preserving_permutation(scores, ranked):
    """Promote the reranked prefix; retain the exact original score multiset.

    Append unselected items in original score order. Assign original descending
    scores to the resulting positions. This explicit rank-to-score convention
    conveys reranking to the fixed Ridge combiner without inventing score units.
    """
    scores.validate()
    if ranked.keys() != scores.scores.keys():
        raise ValueError('Reranked and scored users differ')
    result = {}
    for user, row in scores.scores.items():
        prefix = ranked[user]
        if not prefix or len(prefix) != len(set(prefix)) or not set(prefix) <= row.keys():
            raise ValueError('Invalid reranked prefix')
        original = sorted(row, key=lambda item: (-row[item], item))
        selected = set(prefix)
        order = list(prefix) + [item for item in original if item not in selected]
        result[user] = dict(zip(order, (row[item] for item in original)))
    return ScoreTable(scores.model, scores.split_id, scores.partition, result)


def combine_standardized_scores(tables, fitted):
    """Apply frozen coefficients directly to already standardized input scores."""
    require_aligned(tables)
    if [table.model for table in tables] != fitted['components']:
        raise ValueError('Combiner component ordering differs')
    weights = np.asarray(fitted['coefficients'], dtype=float)
    if weights.shape != (len(tables),) or not np.isfinite(weights).all():
        raise ValueError('Invalid combiner weights')
    scores = {}
    for user in sorted(tables[0].scores):
        items = sorted(tables[0].scores[user])
        x = np.array([[table.scores[user][i] for table in tables] for i in items], dtype=float)
        values = x @ weights + fitted['intercept']
        scores[user] = dict(zip(items, map(float, values)))
    first = tables[0]
    result = ScoreTable('ridge-hybrid', first.split_id, first.partition, scores)
    result.validate()
    return result


def rerank_then_combine(component_scores, reranker, hybrid, context, settings):
    require_aligned(component_scores)
    transformed = [score_preserving_permutation(table, reranker(
        table, context, settings['k'], settings['strength'])) for table in component_scores]
    return combine_standardized_scores(transformed, hybrid).top_k(settings['k'])


def combine_then_rerank(component_scores, reranker, hybrid, context, settings):
    return reranker(combine_standardized_scores(component_scores, hybrid), context,
                   settings['k'], settings['strength'])


def run():
    # Jacek integrates four rerankers with contributions from Gabriel and Victor.
    # Bogdan owns the two order comparisons above; Victor examines groups.
    # Gabriel reviews integration (3.1); Jacek reviews the orders (3.3).
    # Caio reviews trade-offs (3.2) and group effects (3.4), with closure.
    # A small experiment script can call these directly if no shared CLI helps.
    # TODO: 3.1 integrate the four methods; 3.2 evaluate strength sweeps on selected
    # individual models; 3.3 call BOTH orders above; 3.4 compare group outcomes.
    # Select strengths on development data, freeze them, then report test effects.
    raise NotImplementedError("Task 3 orchestration awaits rerankers and pipeline orders.")
