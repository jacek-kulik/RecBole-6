"""Task 2.1: Jacek implements, Caio reviews; Recall is a working example."""

from collections.abc import Mapping
from math import log2
from typing import Dict

from .types import RankingInput, RelevanceInput, AccuracyResults
from ._validation import external_id, validate_rankings

def evaluate_accuracy(
        rankings: RankingInput, relevant: RelevanceInput, k: int=10) -> AccuracyResults:
    """ Return all six per-user metrics and their equally weighted user means
    """
    rows = validate_rankings(rankings, k)
    if not isinstance(relevant, Mapping) or rows.keys() != relevant.keys():
        raise ValueError("Relevance data does not have the evaluation users")
    per_user: Dict[str, Dict[str, float]] = {}

    for user, items in rows.items():
        positives = relevant[user]
        if not isinstance(positives, (set, frozenset, list, tuple)) or not positives:
            raise ValueError("Relevance sets bad for users")
        if any(not external_id(item) for item in positives):
            raise ValueError("Relevance data does not hold external IDs")

        positives = set(positives)
        hits = 0
        reciprocal_rank = 0.0
        dcg = 0.0
        precision_sum = 0.0

        for rank, item in enumerate(items, 1):
            if item in positives:
                hits += 1
                if not reciprocal_rank:
                    reciprocal_rank = 1 / rank
                dcg += 1 / log2(rank + 1)
                precision_sum += hits / rank
        precision = hits / k
        recall = hits / len(positives)
        ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(positives)) + 1))

        # Not sure about saving the information like Precision@number of just Precision@K. This makes it harder
        # to actually retrieve them, but it is technically more informative
        per_user[user] = {
            f"Precision@{k}": precision,
            f"Recall@{k}": recall,
            f"F1@{k}": 2 * precision * recall / (precision + recall) if hits else 0.0,
            f"MRR@{k}": reciprocal_rank,
            f"NDCG@{k}": dcg / ideal_dcg,
            f"MAP@{k}": precision_sum / len(positives)
        }

    aggregate = {
        metric: sum(row[metric] for row in per_user.values()) / len(per_user)
        for metric in next(iter(per_user.values()))
    }
    return {"per_user": per_user, "aggregate": aggregate}

def recall_at_k(rankings: RankingInput, relevant: RelevanceInput, k: int) -> float:
    """Macro recall, with one equally weighted contribution per evaluation user.

    This example requires nonempty relevance sets and complete user coverage.
    Decide the real cohort/relevance rule before calling it on MovieLens outputs.
    """
    return evaluate_accuracy(rankings, relevant, k)["aggregate"][f"Recall@{k}"]

def get_per_user_accuracy_metrics(rankings: RankingInput, relevant: RelevanceInput, k: int) -> Dict[str, Dict[str, float]]:
    return evaluate_accuracy(rankings, relevant, k)["per_user"]

def get_aggregate_mrr(rankings: RankingInput, relevant: RelevanceInput, k: int) -> float:
    return evaluate_accuracy(rankings, relevant, k)["aggregate"][f"MRR@{k}"]

def get_aggregate_accuracy_metrics(rankings: RankingInput, relevant: RelevanceInput, k: int) -> Dict[str, float]:
    return evaluate_accuracy(rankings, relevant, k)["aggregate"]
