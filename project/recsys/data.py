"""Shared metadata and training statistics for evaluation and later rerankers."""

import csv
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import groupby
from os import PathLike
from pathlib import Path

ACTIVITY_GROUPS = ("low", "medium", "high")
POPULARITY_GROUPS = ("head", "middle", "tail")
GENDER_GROUPS = ("F","M")

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
    pending: dict[str, str] = field(default_factory=dict)

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
    ordered = sorted(counts.values())

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

def column_read(path: Path, id_column: str, goal_column: str) -> dict[str, str] | None:
    """Using the specified path to a Recbole dataset file, read a specific "goal_column" keyed by "id_column"."""
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = [name.split(":", 1)[0] for name in next(reader, [])]

        if id_column not in header:
            raise ValueError(f"{path} has no {id_column} column")
        if goal_column not in header:
            return None

        key = header.index(id_column)
        value = header.index(goal_column)

        result = {}
        for row in reader:
            if len(row) <= key or not row[key]:
                continue

            if value < len(row):
                result[row[key]] = row[value]
            else:
                result[row[key]] = ""
        return result

def load_item_genres(item_path: Path) -> dict[str, tuple[str, ...]]:
    """Retrieve the list of genres from each item without duplicates"""
    genre_columns = column_read(item_path, "item_id", "class")

    if genre_columns is None:
        raise ValueError(f"{item_path} has no class (genre) column")

    result = {}
    for item, value in genre_columns.items():
        genres = value.split()

        if not genres:
            result[item] = ("?",)
            continue

        unique_genres = []
        for genre in genres:
            if genre not in unique_genres:
                unique_genres.append(genre)

        result[item] = tuple(unique_genres)
    return result

def load_user_genders(user_path: Path) -> dict[str, str]:
    """Record if user is female or male (F/M). Any other value is exculded"""
    gender_columns = column_read(user_path, "user_id", "gender")

    if gender_columns is None:
        raise ValueError(f"{user_path} has no gender column")

    result = {}
    for user, gender in gender_columns.items():
        gender = gender.strip().upper()

        if gender in GENDER_GROUPS:
            result[user] = gender
    return result


def prepare_metadata(
        train_rows: Iterable[Interaction],
        item_file: str | PathLike[str],
        user_file: str | PathLike[str] | None = None,
        catalogue: Iterable[str] | None = None
) -> TrainingStatistics:
    """Function Loads genre and training derived groups. Original IDs are preserved and data is based of training set"""
    rows = list(train_rows)
    items = tuple(sorted(set(catalogue) \
                         if catalogue is not None else {row.item for row in rows}))
    histories: dict[str, list[str]] = {}
    counts = dict.fromkeys(items, 0)
    for row in rows:
        histories.setdefault(row.user, []).append(row.item)
        counts[row.item] += 1

    boundaries, activity = activity_terciles({user: len(history) \
                                              for user, history in histories.items()})
    item_groups = item_popularity_groups(counts)

    pending = {}

    item_path = Path(item_file)
    item_features = {}
    if item_path.is_file():
        genres = load_item_genres(item_path)
        missing_items = []

        for item in items:
            if item in genres:
                item_features[item] = genres[item]
            else:
                missing_items.append(item)

        if missing_items:
            pending["item_genres"] = (f"{len(missing_items)} catalogue items (Items {missing_items}) have no genres in {item_path.name}")
    else:
        pending["metadata"] = f"Item file not found: {item_path}"

    genre_profiles = {}
    unprofiled = 0
    for user, history in histories.items():
        if not history:
            continue
        all_have_genres = True

        for item in history:
            if item not in item_features:
                all_have_genres = False
                break

        if not all_have_genres:
            unprofiled += 1
            continue
        genre_profiles[user] = genre_distribution(history, item_features)

    if item_features and unprofiled:
        pending["genre_profiles"] = f"{unprofiled} users have training items without genres"

    gender_groups = {}
    if user_file is None:
        pending.setdefault("metadata", "No user file supplied")
    else:
        user_path = Path(user_file)

        if user_path.is_file():
            gender_groups = load_user_genders(user_path)
        else:
           pending.setdefault("metadata", f"User file not found: {user_path}")

    return TrainingStatistics(
        catalogue=items,
        histories=histories,
        item_counts=counts,
        item_features=item_features,
        genre_profiles=genre_profiles,
        gender_groups=gender_groups,
        popularity_profiles={user: popularity_distribution(history, item_groups)
                             for user, history in histories.items()},
        activity_boundaries=boundaries,
        activity_groups=activity,
        item_groups=item_groups,
        pending=pending
    )