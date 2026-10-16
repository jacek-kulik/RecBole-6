"""Shared metadata and training statistics for evaluation and later rerankers."""

from collections import Counter
import csv
from dataclasses import dataclass
from itertools import groupby
from os import PathLike
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ACTIVITY_GROUPS = ("low", "medium", "high")
POPULARITY_GROUPS = ("head", "middle", "tail")

@dataclass(frozen=True)
class Interaction:
    user: str
    item: str


def candidate_items(
        train: Iterable[Interaction],
        users: Iterable[str],
        catalog: Iterable[str]
) -> dict[str, list[str]]:
    """Reference train-seen filtering used by the toy demo.

    For real experiments, derive candidates from the approved protocol and check
    they match RecBole's validation/test masking. Test masking may also exclude
    validation history; this helper deliberately accepts only training history.
    """
    seen: dict[str, set[str]] = {}
    for row in train:
        seen.setdefault(row.user, set()).add(row.item)
    catalog = set(catalog)
    return {user: sorted(catalog - seen.get(user, set())) for user in sorted(set(users))}


@dataclass
class TrainingStatistics:
    catalogue: tuple[str, ...]
    histories: dict[str, list[str]]
    item_counts: dict[str, int]
    item_features: dict[str, tuple[str, ...]]
    genre_profiles: dict[str, dict[str, float]]
    popularity_profiles: dict[str, dict[str, float]]
    activity_boundaries: tuple[float, float]
    activity_groups: dict[str, str]
    gender_groups: dict[str, str]
    item_groups: dict[str, str]

    @property
    def n_interactions(self) -> int:
        return sum(self.item_counts.values())


def read_training_interactions(path: str | PathLike[str]) -> list[Interaction]:
    """Read only train.csv. Ratings and timestamps remain in the source export."""
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"user_id", "item_id"} <= set(reader.fieldnames or []):
            raise ValueError("Training CSV needs user_id and item_id columns")
        return [Interaction(row["user_id"], row["item_id"]) for row in reader]


def activity_terciles(counts: Mapping[str, int]
) -> tuple[tuple[float, float], dict[str, str]]:
    """Linear count quantiles, with equal countrs kept in the same group."""
    ordered = sorted(counts.values)

    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    boundaries = (quantile(1/3), quantile(2 / 3))
    groups = {
        user: ACTIVITY_GROUPS[0] if count <= boundaries[0]
        else ACTIVITY_GROUPS[1] if count <= boundaries[1]
        else ACTIVITY_GROUPS[2]
        for user, count in counts.items()
    }
    return boundaries, groups


def item_popularity_groups(counts: Mapping[str, int]) -> dict[str, str]:
    """Keep count ties together while crossing 20% and 80% interaction shares."""
    total = sum(counts.values())
    cumulative = 0
    groups: dict[str, str] = {}
    ordered = sorted(counts, key=lambda x: (-counts[x], x))

    for count, items in groupby(ordered, key=lambda x: counts[x]):
        bucket = list(items)
        group = POPULARITY_GROUPS[0] if cumulative < .2 * total \
        else POPULARITY_GROUPS[1] if cumulative < .8 * total \
        else POPULARITY_GROUPS[2]

        groups.update({item: group for item in bucket})
        cumulative += count * len(bucket)
    return groups


def genre_distribution(
        items: Sequence[str],
        features: Mapping[str, Sequence[str]]
) -> dict[str, float]:
    """Each movie contributes one genre unit, divided equally among its genres."""
    counts: Counter[str] = Counter()
    for item in items:
        genres = features[item]
        for genre in genres:
            counts[genre] += 1 / len(genres)
    return {genre: count / len(items) for genre, count in counts.items()} \
        if items else {}


def popularity_distribution(
        items: Sequence[str],
        groups: Mapping[str, str]
) -> dict[str, float]:
    counts = Counter(groups[item] for item in items)
    return {group: counts[group] / len(items) for group in POPULARITY_GROUPS} \
        if items else {}


def prepare_metadata(train_rows, item_file, user_file=None):
    """TODO Caio and Victor: load genre/features and training-derived profiles/groups.

    Preserve original IDs. Define relevance, cold-item policy, and available
    attributes in PROTOCOL.md. Do not infer demographic groups from model scores.
    Keep ratings/timestamps when reading the exported split CSVs.
    """
    raise NotImplementedError("Caio and Victor: implement approved metadata preparation.")
