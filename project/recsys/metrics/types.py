"""Input aliases and data structures with JSON-compatibility for metric usage"""

from collections.abc import Mapping, Sequence
from typing import TypedDict

RankingInput = Mapping[str, Sequence[str]]
RelevantItems = set[str] | frozenset[str] | list[str] | tuple[str, ...]
RelevanceInput = Mapping[str, RelevantItems]


class AccuracyResults(TypedDict):
    per_user: dict[str, dict[str, float]]
    aggregate: dict[str, float]


class MetricAvailability(TypedDict):
    n_users: int
    reason: str | None


class QualityGap(TypedDict):
    absolute: float | None
    signed_f_minus_m: float | None


class GenderFairnessResults(TypedDict):
    counts: dict[str, int]
    means: dict[str, dict[str, float] | None]
    gaps: dict[str, QualityGap]
    n_users: int
    reason: str | None


class ItemGroupExposure(TypedDict):
    n_items: int
    total: float | None
    share: float | None
    per_item: float | None
    reason: str | None


class BeyondAccuracyResults(TypedDict):
    per_user: dict[str, dict[str, float | None]]
    aggregate: dict[str, float | None]
    availability: dict[str, MetricAvailability]
    gender_fairness: GenderFairnessResults
    item_group_exposure: dict[str, ItemGroupExposure]