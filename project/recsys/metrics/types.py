"""Input aliases and data structures with JSON-compatibility for metric usage"""

from typing import Dict, FrozenSet, List, Mapping, Sequence, Set, Tuple, TypedDict, Union

RankingInput = Mapping[str, Sequence[str]]
RelevantItems = Union[Set[str], FrozenSet[str], List[str], Tuple[str, ...]]
RelevanceInput = Mapping[str, RelevantItems]


class AccuracyResults(TypedDict):
    per_user: Dict[str, Dict[str, float]]
    aggregate: Dict[str, float]


class MetricAvailability(TypedDict):
    n_users: int
    reason: str | None


class QualityGap(TypedDict):
    absolute: float | None
    signed_f_minus_m: float | None


class GenderFairnessResults(TypedDict):
    counts: Dict[str, int]
    means: Dict[str, Dict[str, float] | None]
    gaps: Dict[str, QualityGap]
    n_users: int
    reason: str | None


class ItemGroupExposure(TypedDict):
    n_items: int
    total: float | None
    share: float | None
    per_item: float | None
    reason: str | None


class BeyondAccuracyResults(TypedDict):
    per_user: Dict[str, Dict[str, float | None]]
    aggregate: Dict[str, float | None]
    availability: Dict[str, MetricAvailability]
    gender_fairness: GenderFairnessResults
    item_group_exposure: Dict[str, ItemGroupExposure]
