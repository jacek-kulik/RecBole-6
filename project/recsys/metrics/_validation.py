"""Shared input checks for metric evaluation"""

from typing import AbstractSet, Optional
from collections.abc import Mapping, Sequence

from ..contracts import Rankings
from .types import RankingInput


def external_id(value: object) -> bool:
    return isinstance(value, str) and bool(value)

def validate_rankings(
        rankings: RankingInput,
        k: int,
        catalogue: Optional[AbstractSet[str]] = None
) -> Rankings:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if not isinstance(rankings, Mapping) or not rankings:
        raise ValueError("Rankings must be a map of users")

    rows = {}
    for user, ranking in rankings.items():
        if not external_id(user) or not isinstance(rankings, Sequence) or isinstance(ranking, str):
            raise ValueError("External ids not valid or not in a sequence")
        items = ranking[:k]
        if len(items) != k or any(not external_id(item) for item in items):
            raise ValueError("all users do not have k ids")
        if len(set(items)) != k:
            raise ValueError("there are not k distinct recommended items")
        if catalogue is not None and not set(items) <= catalogue:
            raise ValueError("recommendations must belong to the evaluation catalogue")
        rows[user] = list(items)
    return rows