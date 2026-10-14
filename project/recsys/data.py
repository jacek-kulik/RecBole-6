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
    train: Iterable[Interaction], users: Iterable[str], catalog: Iterable[str]
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


def activity_terciles(counts: Mapping[str, int]) -> tuple[tuple[float, float], dict[str, str]]:
    """Linear count quantiles, with equal counts kept in the same group."""
    if not counts:
        raise ValueError("Activity groups need training users")
    ordered = sorted(counts.values())

    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    boundaries = (quantile(1 / 3), quantile(2 / 3))
    groups = {
        user: "low" if count <= boundaries[0] else "medium" if count <= boundaries[1] else "high"
        for user, count in counts.items()
    }
    return boundaries, groups


def item_popularity_groups(counts: Mapping[str, int]) -> dict[str, str]:
    """Keep count ties together while crossing 20% and 80% interaction shares."""
    total = sum(counts.values())
    cumulative = 0
    groups: dict[str, str] = {}
    ordered = sorted(counts, key=lambda item: (-counts[item], item))
    for count, items in groupby(ordered, key=lambda item: counts[item]):
        bucket = list(items)
        group = "head" if cumulative < .2 * total else "middle" if cumulative < .8 * total else "tail"
        groups.update({item: group for item in bucket})
        cumulative += count * len(bucket)
    return groups


def genre_distribution(
    items: Sequence[str], features: Mapping[str, Sequence[str]]
) -> dict[str, float]:
    """Each movie contributes one unit, divided equally among its genres."""
    counts: dict[str, float] = {}
    for item in items:
        genres = features[item]
        for genre in genres:
            counts[genre] = counts.get(genre, 0.0) + 1 / len(genres)
    return {genre: count / len(items) for genre, count in counts.items()} if items else {}


def popularity_distribution(items: Sequence[str], groups: Mapping[str, str]) -> dict[str, float]:
    counts = Counter(groups[item] for item in items)
    return {group: counts[group] / len(items) for group in POPULARITY_GROUPS} if items else {}


def prepare_metadata(
    train_rows: Iterable[Interaction],
    item_file: str | PathLike[str],
    user_file: str | PathLike[str] | None = None,
    catalogue: Iterable[str] | None = None,
) -> TrainingStatistics:
    """Build profiles and activity/popularity groups from training alone.

    Gender membership comes directly from static metadata. Original IDs are
    retained, and unavailable gender values never remove a user from evaluation.
    The optional catalogue is the checkpoint dataset's actual movie catalogue.
    """
    with Path(item_file).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"item_id:token", "class:token_seq"} <= set(reader.fieldnames or []):
            raise ValueError("Item metadata needs item_id:token and class:token_seq")
        features = {
            row["item_id:token"]: tuple(sorted(set(row["class:token_seq"].split()))) or ("unknown",)
            for row in reader
        }
    items = tuple(sorted(features if catalogue is None else set(catalogue)))
    if not items:
        raise ValueError("The evaluation catalogue is empty")
    features = {item: features.get(item, ("unknown",)) for item in items}
    histories: dict[str, list[str]] = {}
    counts = dict.fromkeys(items, 0)
    for row in train_rows:
        if not row.user or row.item not in counts:
            raise ValueError("Training interactions need external users and catalogue items")
        histories.setdefault(row.user, []).append(row.item)
        counts[row.item] += 1
    boundaries, activity = activity_terciles({user: len(history) for user, history in histories.items()})
    genders: dict[str, str] = {}
    if user_file is not None:
        with Path(user_file).open(newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not {"user_id:token", "gender:token"} <= set(reader.fieldnames or []):
                raise ValueError("User metadata needs user_id:token and gender:token")
            for row in reader:
                user = row["user_id:token"]
                if row["gender:token"] in {"F", "M"}:
                    genders[user] = row["gender:token"]
                activity.setdefault(user, "low")
    item_groups = item_popularity_groups(counts)
    return TrainingStatistics(
        catalogue=items, histories=histories, item_counts=counts, item_features=features,
        genre_profiles={user: genre_distribution(history, features) for user, history in histories.items()},
        popularity_profiles={user: popularity_distribution(history, item_groups) for user, history in histories.items()},
        activity_boundaries=boundaries, activity_groups=activity, gender_groups=genders,
        item_groups=item_groups,
    )
