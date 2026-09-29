"""PERSON-2 implements; PERSON-4 reviews."""

from ..contracts import Rankings, RerankContext, ScoreTable


def rerank(scores: ScoreTable, context: RerankContext, k: int, strength: float) -> Rankings:
    # TODO Task 3.1: choose the lecture diversification objective and similarity
    # representation. Balance relevance against redundancy, with deterministic
    # tie handling. Use item_features from context and the common candidates.
    raise NotImplementedError("PERSON-2: implement diversification reranking.")
