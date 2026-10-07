"""Input aliases and data structures with JSON-compatibility for metric usage"""

from typing import AbstractSet, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple, TypedDict, Union

RankingInput = Mapping[str, Sequence[str]]
RelevantItems = Union[Set[str], FrozenSet[str], List[str], Tuple[str, ...]]
RelevanceInput = Mapping[str, RelevantItems]


class AccuracyResults(TypedDict):
    per_user: Dict[str, Dict[str, float]]
    aggregate: Dict[str, float]