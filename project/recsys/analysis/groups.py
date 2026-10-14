"""Reusable cohort reports for Victor's Tasks 2.5 and 3.4, reviewed by Caio."""

from typing import Iterable, TypedDict

from ..data import ACTIVITY_GROUPS, POPULARITY_GROUPS, TrainingStatistics
from ..metrics.accuracy import evaluate_accuracy
from ..metrics.beyond_accuracy import evaluate_beyond_accuracy
from ..metrics.types import BeyondAccuracyResults, RankingInput, RelevanceInput


class CohortResults(TypedDict):
    n_users: int
    accuracy: dict[str, float] | None
    beyond_accuracy: BeyondAccuracyResults
    reason: str | None


def compare_groups(
    rankings: RankingInput, relevant: RelevanceInput, statistics: TrainingStatistics, k: int = 10
) -> dict[str, CohortResults]:
    """Report all metrics overall and within each fixed activity cohort.

    Gender unfairness is calculated from F/M users inside each cohort. Activity
    membership selects the reporting cohort, rather than the fairness groups.
    """
    results: dict[str, CohortResults] = {}
    for group in ("overall", *ACTIVITY_GROUPS):
        rows = {user: items for user, items in rankings.items()
                if group == "overall" or statistics.activity_groups.get(user, "low") == group}
        labels = {user: relevant[user] for user in rows}
        accuracy = evaluate_accuracy(rows, labels, k) if rows else None
        results[group] = {
            "n_users": len(rows), "accuracy": accuracy["aggregate"] if accuracy else None,
            "beyond_accuracy": evaluate_beyond_accuracy(rows, statistics, accuracy["per_user"] if accuracy else {}, k),
            "reason": None if rows else "No evaluation users in this cohort",
        }
    return results


def group_definitions(statistics: TrainingStatistics, evaluation_users: Iterable[str]) -> dict[str, object]:
    """Save fixed membership alongside training and evaluated group counts."""
    evaluated = set(evaluation_users)
    activity = dict(statistics.activity_groups)
    for user in evaluated:
        activity.setdefault(user, "low")
    activity_counts = {
        group: {
            "n_training_users": sum(statistics.activity_groups[user] == group for user in statistics.histories),
            "n_evaluated_users": sum(activity[user] == group for user in evaluated),
        } for group in ACTIVITY_GROUPS
    }
    item_counts = {}
    for group in POPULARITY_GROUPS:
        members = [item for item in statistics.catalogue if statistics.item_groups[item] == group]
        count = sum(statistics.item_counts[item] for item in members)
        item_counts[group] = {
            "n_items": len(members), "training_interactions": count,
            "training_share": count / statistics.n_interactions,
        }
    return {
        "activity": {
            "definition": "Training-count terciles: low <= q1, q1 < medium <= q2, high > q2; zero-history users join low",
            "boundaries": list(statistics.activity_boundaries), "members": activity,
            "training_counts": {user: len(statistics.histories.get(user, [])) for user in activity},
            "counts": activity_counts,
        },
        "gender": {
            "definition": "Recorded F/M metadata; unavailable gender is excluded only from gender comparisons",
            "members": {user: statistics.gender_groups.get(user) for user in activity},
            "evaluated_counts": {
                gender: sum(statistics.gender_groups.get(user) == gender for user in evaluated) for gender in ("F", "M")
            },
            "n_unavailable": sum(user not in statistics.gender_groups for user in evaluated),
        },
        "item_popularity": {
            "definition": "Descending training-count buckets cross cumulative interaction shares 0.2 and 0.8; ties stay together",
            "members": statistics.item_groups, "training_counts": statistics.item_counts, "counts": item_counts,
        },
        "unknown_genre_items": [item for item in statistics.catalogue if statistics.item_features[item] == ("unknown",)],
    }
