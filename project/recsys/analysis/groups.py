"""Reusable cohort reports for Victor's Tasks 2.5 and 3.4, reviewed by Caio."""

from typing import Iterable, TypedDict

from ..data import ACTIVITY_GROUPS, POPULARITY_GROUPS, TrainingStatistics

def compare_groups(per_user_results, item_results, context):
    # TODO: use fixed groups from training data, report counts and the same
    # accuracy/beyond-accuracy metrics before and after reranking. Keep user-side
    # and item-side aggregation explicit and expose small/empty groups.
    raise NotImplementedError("Victor: implement user/item group comparisons.")
