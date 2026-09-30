"""Jacek implements; Victor reviews."""

from ..contracts import Rankings, RerankContext, ScoreTable


def rerank(scores: ScoreTable, context: RerankContext, k: int, strength: float) -> Rankings:
    # TODO Task 3.1: match the agreed attribute distribution in the user's training
    # history. Specify divergence, smoothing, and empty-profile behavior. Do not
    # build target profiles from the held-out interactions being evaluated.
    raise NotImplementedError("Jacek: implement calibration reranking.")
