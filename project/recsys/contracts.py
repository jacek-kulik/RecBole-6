"""Caio and Bogdan: shared external-ID contract for models, hybrids, and rerankers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite

Rankings = dict[str, list[str]]


@dataclass
class ScoreTable:
    """Scores for every eligible candidate, not just each model's top K.

    IDs are original dataset tokens as strings, never RecBole internal IDs.
    split_id identifies the actual exported split. Candidates may vary by user,
    but every model in a comparison must score the same candidates for that user.
    """

    model: str
    split_id: str
    partition: str
    scores: dict[str, dict[str, float]]

    def validate(self):
        if not self.model or not self.split_id or self.partition not in {"valid", "test"}:
            raise ValueError("Supply a model, split ID, and valid/test partition.")
        if not self.scores:
            raise ValueError("A score table must contain users.")
        for user, items in self.scores.items():
            if not isinstance(user, str) or not user or not items:
                raise ValueError("Use nonempty external user IDs and candidate sets.")
            for item, score in items.items():
                if not isinstance(item, str) or not item or not isfinite(score):
                    raise ValueError("Use external item IDs and finite scores.")

    def top_k(self, k: int) -> Rankings:
        self.validate()
        if k <= 0 or any(len(items) < k for items in self.scores.values()):
            raise ValueError("K must be positive and no larger than each candidate set.")
        # Resolve ties by external token so runs do not depend on dictionary order.
        return {
            user: sorted(items, key=lambda item: (-items[item], item))[:k]
            for user, items in sorted(self.scores.items())
        }


def require_aligned(tables: Sequence[ScoreTable]):
    """Call before fusion; matching array shapes alone does not establish alignment."""
    if not tables:
        raise ValueError("At least one score table is required.")
    first = tables[0]
    for table in tables:
        table.validate()
        if (table.split_id, table.partition) != (first.split_id, first.partition):
            raise ValueError("Models must use the same split and evaluation partition.")
        if table.scores.keys() != first.scores.keys() or any(
            table.scores[user].keys() != first.scores[user].keys()
            for user in first.scores
        ):
            raise ValueError("Models must score identical users and candidate items.")


@dataclass
class RerankContext:
    """Training-only profiles and prepared metadata; never held-out outcomes.

    user_profiles contains fractional genre distributions. Popularity profiles
    use the training-derived head/middle/tail item groups.
    """

    item_features: Mapping[str, Sequence[str]] = field(default_factory=dict)
    user_profiles: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    user_groups: Mapping[str, str] = field(default_factory=dict)
    item_groups: Mapping[str, str] = field(default_factory=dict)
    popularity_profiles: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    # Full catalogue counts from training, including explicit zero-count items.
    item_counts: Mapping[str, int] = field(default_factory=dict)
