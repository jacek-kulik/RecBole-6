"""PERSON-5 implements; PERSON-4 reviews."""

from ..contracts import Rankings, RerankContext, ScoreTable


def rerank(scores: ScoreTable, context: RerankContext, k: int, strength: float) -> Rankings:
    # TODO Task 3.1: define the user-group objective and develop any group-specific
    # parameters using development data. Test relevance labels are unavailable
    # at reranking time. Return rankings for all users, including small groups.
    raise NotImplementedError("PERSON-5: implement user-side fairness reranking.")
