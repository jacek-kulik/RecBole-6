"""PERSON-4 implements; PERSON-2 reviews."""

from ..contracts import Rankings, RerankContext, ScoreTable


def rerank(scores: ScoreTable, context: RerankContext, k: int, strength: float) -> Rankings:
    # TODO Task 3.1: define item/provider groups, exposure accounting, and the
    # fairness objective from lectures. Decide whether allocation is per user or
    # across all users; this interface supplies the entire evaluation cohort.
    raise NotImplementedError("PERSON-4: implement item-side fairness reranking.")
