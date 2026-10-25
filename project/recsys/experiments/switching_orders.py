"""Frozen user routing with local and cohort-wide rerankers."""

from ..contracts import ScoreTable, require_aligned


def switch_scores(component_scores, routes):
    """Select complete score rows; routing is fixed independently of evaluation."""
    require_aligned(component_scores)
    tables = {table.model: table for table in component_scores}
    if (len(tables) != len(component_scores) or routes.keys() != component_scores[0].scores.keys()
            or not set(routes.values()) <= tables.keys()):
        raise ValueError('Routing users or components differ')
    first = component_scores[0]
    return ScoreTable('switch', first.split_id, first.partition,
                      {u: dict(tables[name].scores[u]) for u, name in routes.items()})


def rerank_then_switch(component_scores, reranker, routes, context, k, strength):
    """Each component reranks the entire cohort with a fresh exposure counter."""
    switch_scores(component_scores, routes)  # Verify alignment before any reranking.
    ranked = {table.model: reranker(table, context, k, strength) for table in component_scores}
    if any(result.keys() != routes.keys() for result in ranked.values()):
        raise ValueError('Reranked users differ from the routing cohort')
    return {u: ranked[name][u] for u, name in routes.items()}


def switch_then_rerank(component_scores, reranker, routes, context, k, strength):
    return reranker(switch_scores(component_scores, routes), context, k, strength)
