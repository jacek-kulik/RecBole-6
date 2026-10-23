"""Victor's Task 2.5 group comparisons (Task 3.4 reuses them), reviewed by Caio."""

import math

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import TypedDict, Dict, Tuple

from ..contracts import Rankings
from ..data import (
    ACTIVITY_GROUPS,
    GENDER_GROUPS,
    POPULARITY_GROUPS,
    TrainingStatistics,
)
from ..metrics._validation import validate_rankings
from ..metrics.accuracy import evaluate_accuracy
from ..metrics.beyond_accuracy import evaluate_beyond_accuracy
from ..metrics.types import (
    ItemGroupExposure,
    MetricAvailability,
    RankingInput,
    RelevanceInput,
)

GENRE_TOPK_GROUPS = ["Action","Adventure","Animation","Children's","Comedy","Crime",
                     "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
                     "Mystery","Romance","Sci-Fi","Thriller","War","Western","unknown"]

UNKNOWN_GENDER = "unknown"
ACCURACY_METRICS = ("Precision", "Recall", "F1", "MRR", "NDCG", "MAP")
BEYOND_ACCURACY_METRICS = ("ILD", "MeanSelfInformation", "GenreMiscalibration", "UPD")


class CohortReport(TypedDict):
    """Accuracy and beyond-accuracy metrics for one user cohort (a group or overall)."""

    n_users: int
    meets_minimum_size: bool
    accuracy: dict[str, float | None]
    beyond_accuracy: dict[str, float | None]
    availability: dict[str, MetricAvailability]
    item_group_exposure: dict[str, ItemGroupExposure]
    reason: str | None


class ItemGroupReport(TypedDict):
    """Item-side aggregation for one training-popularity group."""

    n_items: int
    n_recommended: int
    n_hits: int
    n_relevant: int
    recommended_share: float | None
    hit_share: float | None
    relevant_share: float | None
    precision: float | None
    recall: float | None
    exposure: ItemGroupExposure | None
    audience_accuracy: dict[str, float | None]
    reason: str | None


class GroupComparison(TypedDict):
    status: str
    owner: str
    k: int
    minimum_group_size: int
    overall: CohortReport
    user_groups: dict[str, dict[str, CohortReport]]
    item_groups: dict[str, dict[str, ItemGroupReport]]
    pending: dict[str, str]


def _activity_group(statistics: TrainingStatistics, user: str) -> str:
    # Zero-history users join low, matching the manifest's group construction note.
    return statistics.activity_groups.get(user, ACTIVITY_GROUPS[0])

def _item_popularity_groups(statistics: TrainingStatistics, item: str) -> Tuple[str, ...]:
    return (statistics.item_groups[item],)

def _item_genre_groups(statistics: TrainingStatistics, item: str) -> Tuple[str, ...]:
    return tuple(statistics.item_features.get(item, ("unknown",)))


def _gender_group(statistics: TrainingStatistics, user: str) -> str:
    return statistics.gender_groups.get(user, UNKNOWN_GENDER)

def _user_genre_group(statistic: TrainingStatistics, user: str) -> list:
    genre_dict = statistic.genre_profiles.get(user, {})
    genres = sorted(genre_dict, key=genre_dict.get, reverse=True)
    return genres[:3]#[max(genre_dict, key=genre_dict.get)]


def _cohort_report(
        rows: Rankings,
        relevant: RelevanceInput,
        statistics: TrainingStatistics,
        k: int,
        minimum_group_size: int,
) -> CohortReport:
    """Score one cohort with the shared metric implementations; empty cohorts stay explicit."""
    if rows:
        accuracy = evaluate_accuracy(rows, relevant, k)
        beyond = evaluate_beyond_accuracy(rows, statistics, accuracy["per_user"], k)
        accuracy_values: dict[str, float | None] = dict(accuracy["aggregate"])
        reason = None
    else:
        beyond = evaluate_beyond_accuracy({}, statistics, {}, k)
        accuracy_values = {f"{name}@{k}": None for name in ACCURACY_METRICS}
        reason = "No evaluation users in this group"
    return {
        "n_users": len(rows),
        "meets_minimum_size": len(rows) >= minimum_group_size,
        "accuracy": accuracy_values,
        "beyond_accuracy": dict(beyond["aggregate"]),
        "availability": dict(beyond["availability"]),
        "item_group_exposure": dict(beyond["item_group_exposure"]),
        "reason": reason,
        }

def _item_group_accuracy_from_users(rows, per_user_accuracy, membership, k, groups, discounted=False) -> dict[str, dict[str, float | None]]:
    sums = defaultdict(lambda: defaultdict(float))
    weights = defaultdict(float)
    for user, items in rows.items():
        value = per_user_accuracy[user]
        for rank, item in enumerate(items):
            w = 1 / math.log2(rank + 2) if discounted else 1.0
            for group in membership(item):        # tuple of groups per item
                for metric in ACCURACY_METRICS:
                    sums[group][metric] += value[f"{metric}@{k}"] * w
                weights[group] += w
    return {
        group: {metric :
            sums[group][metric] / weights[group] if  weights[group] else None
            for metric in ACCURACY_METRICS}
        for group in groups
        
    }

def _item_group_accuracy_beyond_from_users(rows, per_user_accuracy, membership, k, groups, discounted=False) -> dict[str, dict[str, float | None]]:
    sums = defaultdict(lambda: defaultdict(float))
    weights = defaultdict(lambda: defaultdict(float))
    for user, items in rows.items():
        value = per_user_accuracy[user]
        for rank, item in enumerate(items):
            w = 1 / math.log2(rank + 2) if discounted else 1.0
            for group in membership(item):        # tuple of groups per item
                for metric in BEYOND_ACCURACY_METRICS:
                    v = value[f"{metric}@{k}"]
                    if v is not None:
                        sums[group][metric] += v * w
                        weights[group][metric] += w
    return {
        group: {metric :
            sums[group][metric] / weights[group][metric] if  weights[group][metric] else None
            for metric in BEYOND_ACCURACY_METRICS}
        for group in groups

    }

def _item_group_reports(
        rows: Rankings,
        relevant: RelevanceInput,
        statistics: TrainingStatistics,
        exposure: Mapping[str, ItemGroupExposure],
        groups: Iterable[str],
        membership: Callable[[str], Iterable[str]],
        audience: Mapping[str, Mapping[str, float | None]],
        audience_beyond: Mapping[str, Mapping[str, float | None]],
) -> dict[str, ItemGroupReport]:
    """Aggregate recommendation slots, hits, and held-out items per item group.

    ``membership`` maps an item to every group it belongs to (one popularity
    band, or several genres), so multi-label groupings count an item in each
    of their groups. ``audience`` carries the audience-accuracy values per
    group; ``exposure`` may be empty when discounted exposure is only defined
    per popularity band, leaving that field None.
    """
    totals = {group: {"recommended": 0, "hits": 0, "relevant": 0} for group in groups}
    for user, items in rows.items():
        positives = relevant[user]
        for item in items:
            for group in membership(item):
                bucket = totals[group]
                bucket["recommended"] += 1
                if item in positives:
                    bucket["hits"] += 1
        for item in positives:
            if item not in statistics.item_groups:
                raise ValueError("Relevance data contains items outside the catalogue")
            for group in membership(item):
                totals[group]["relevant"] += 1

    total_recommended = sum(bucket["recommended"] for bucket in totals.values())
    total_hits = sum(bucket["hits"] for bucket in totals.values())
    total_relevant = sum(bucket["relevant"] for bucket in totals.values())
    report: dict[str, ItemGroupReport] = {}
    for group in groups:
        bucket = totals[group]
        n_items = sum(1 for item in statistics.catalogue if group in membership(item))
        report[group] = {
            "n_items": n_items,
            "n_recommended": bucket["recommended"],
            "n_hits": bucket["hits"],
            "n_relevant": bucket["relevant"],
            "recommended_share": bucket["recommended"] / total_recommended if total_recommended else None,
            "hit_share": bucket["hits"] / total_hits if total_hits else None,
            "relevant_share": bucket["relevant"] / total_relevant if total_relevant else None,
            "precision": bucket["hits"] / bucket["recommended"] if bucket["recommended"] else None,
            "recall": bucket["hits"] / bucket["relevant"] if bucket["relevant"] else None,
            "exposure": exposure.get(group),
            "audience_accuracy": dict(audience.get(group, {})),
            "audience_beyond_accuracy": dict(audience_beyond.get(group, {})),
            "reason": None if n_items else "No movies in this group",
        }
    return report

def compare_groups(
        rankings: RankingInput,
        relevant: RelevanceInput,
        statistics: TrainingStatistics,
        k: int,
        minimum_group_size: int = 30,
) -> GroupComparison:
    """Compare accuracy and beyond-accuracy metrics across fixed training-derived groups.

    User groups are training-count activity which Seperates users based on how 
    many items it has ranked in the training data. High for items > 92 
    Medium for 34 < items < 92 and low for items < 34 
    and recorded gender(F/M). Item groups are training popularity bands (head/middle/tail). Every
    cohort is scored with the shared metric implementations, so group values use the
    same definitions as the overall report. Groups are never hidden: empty cohorts
    report null metrics with an explicit reason, and cohorts below
    ``minimum_group_size`` are flagged. Groups come from training data and metadata
    only; held-out labels never enter group construction.

    Call once per ranking set (Task 3.4 compares before and after reranking); the
    report layout is identical so the two outputs diff cleanly.
    """
    if type(minimum_group_size) is not int or minimum_group_size < 1:
        raise ValueError("minimum_group_size must be a positive integer")
    rows = validate_rankings(rankings, k, set(statistics.catalogue))
    if not isinstance(relevant, Mapping) or rows.keys() != relevant.keys():
        raise ValueError("Relevance data does not have the evaluation users")

    overall = _cohort_report(rows, relevant, statistics, k, minimum_group_size)
    per_user = evaluate_accuracy(rows, relevant, k)["per_user"]
    per_user_beyond = evaluate_beyond_accuracy(rows, statistics, per_user, k)["per_user"]

    groupings = {
        "activity": (ACTIVITY_GROUPS, _activity_group),
        "gender": ((*GENDER_GROUPS, UNKNOWN_GENDER), _gender_group),
        "genre" : ((GENRE_TOPK_GROUPS),_user_genre_group)
    }

    user_groups: dict[str, dict[str, CohortReport]] = {}
    pending: dict[str, str] = {}

    for name, (groups, membership) in groupings.items():
        if name == "gender" and not statistics.gender_groups:
            pending["gender"] = statistics.pending.get(
                "metadata", "Recorded gender metadata is unavailable")
        cohorts: dict[str, CohortReport] = {}
        for group in groups:
            members = [user for user in rows if group in membership(statistics, user)]# == group
            cohorts[group] = _cohort_report(
                {user: rows[user] for user in members},
                {user: relevant[user] for user in members},
                statistics, k, minimum_group_size)
        user_groups[name] = cohorts

    item_groupings = {
        "Popularity": (POPULARITY_GROUPS, _item_popularity_groups),
        "Genre": (GENRE_TOPK_GROUPS, _item_genre_groups),
    }
    item_groups: dict[str, dict[str, ItemGroupReport]] = {}
    for name, (groups, membership) in item_groupings.items():
        audience = _item_group_accuracy_from_users(
            rows, per_user,
            lambda item, rule=membership: rule(statistics, item),
            k, groups)
        audience_beyond = _item_group_accuracy_beyond_from_users(
            rows, per_user_beyond,
            lambda item, rule=membership: rule(statistics, item),
            k, groups)
        exposure = overall["item_group_exposure"] if name == "Popularity" else {}
        item_groups[name] = _item_group_reports(
            rows, relevant, statistics, exposure, groups,
            lambda item, rule=membership: rule(statistics, item),
            audience, audience_beyond)
    return {
        "status": "complete",
        "owner": "Victor",
        "k": k,
        "minimum_group_size": minimum_group_size,
        "overall": overall,
        "user_groups": user_groups,
        "item_groups": item_groups,
        "pending": pending,
    }

def group_definitions(
        statistics: TrainingStatistics,
        evaluation_users: Iterable[str],
) -> dict[str, object]:
    """Record the fixed group definitions and each evaluation user's membership."""
    users = list(evaluation_users)
    return {
        "status": "complete",
        "owner": "Victor",
        "activity": {
            "definition": "Seperates users based on how many items it has ranked in the training data. High for items > 92 Medium for 34 < items < 92 and low for items < 34",
            "boundaries": list(statistics.activity_boundaries),
            "members": {user: _activity_group(statistics, user) for user in users},
        },
        "gender": {
            "definition": "Recorded F/M from the dataset user file; any other value counts as unknown",
            "members": {user: _gender_group(statistics, user) for user in users},
        },
        "user genre": {
            "Defintion": "top-k genre per user",
             "members": {user: _user_genre_group(statistics, user) for user in users},
        },
        #"user_genre_complete": {
            #"definition": "all genres per user",
            #"members": statistics.genre_profiles,
        #},
        "item_popularity": {
            "definition": "Training interaction shares 20/60/20",
            "members": statistics.item_groups,
        },
        "item_genre": {
            "definition": "top 1 classification of item genre",
            "members": statistics.item_features,
        },
        "pending_inputs": statistics.catalogue,
    }
