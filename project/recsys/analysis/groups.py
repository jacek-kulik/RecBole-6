"""Runnable placeholders for Victor's Tasks 2.5 and 3.4, reviewed by Caio."""

from typing import Iterable, TypedDict

from ..data import TrainingStatistics
from ..metrics.types import BeyondAccuracyResults


class PendingGroupAnalysis(TypedDict):
    status: str
    owner: str
    reason: str


def compare_groups(
        per_user_results: dict[str, dict[str, float]],
        item_results: BeyondAccuracyResults,
        context: TrainingStatistics,
) -> PendingGroupAnalysis:
    """Keep evaluation runnable without presenting Task 2.5/3.4 as completed."""
    # TODO Victor: compare metrics within fixed user/item groups, report sizes
    # and compare reranker effects. Jacek's metric functions are ready to call.
    return {
        "status": "pending", "owner": "Victor",
        "reason": "User/item group comparisons and reranker group analysis are not implemented",
    }


def group_definitions(
        statistics: TrainingStatistics, evaluation_users: Iterable[str]
) -> dict[str, object]:
    """Expose Jacek's existing memberships while group reporting is unfinished."""
    return {
        "status": "pending", "owner": "Victor",
        "reason": "Group summaries and metadata-based definitions still need preparation/review",
        "activity": {
            "boundaries": list(statistics.activity_boundaries),
            "members": {user: statistics.activity_groups.get(user, "low") for user in evaluation_users},
        },
        "item_popularity": {"members": statistics.item_groups},
        "pending_inputs": statistics.pending,
    }
