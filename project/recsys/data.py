"""PERSON-1: data preparation shared by all tasks."""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Interaction:
    user: str
    item: str


def candidate_items(train: Iterable[Interaction], users, catalog):
    """Reference train-seen filtering used by the toy demo.

    For real experiments, derive candidates from the approved protocol and check
    they match RecBole's validation/test masking. Test masking may also exclude
    validation history; this helper deliberately accepts only training history.
    """
    seen = {}
    for row in train:
        seen.setdefault(row.user, set()).add(row.item)
    catalog = set(catalog)
    return {user: sorted(catalog - seen.get(user, set())) for user in sorted(set(users))}


def prepare_metadata(train_rows, item_file, user_file=None):
    """TODO PERSON-1/5: load genre/features and training-derived profiles/groups.

    Preserve original IDs. Define relevance, cold-item policy, and available
    attributes in PROTOCOL.md. Do not infer demographic groups from model scores.
    Keep ratings/timestamps when reading the exported split CSVs.
    """
    raise NotImplementedError("PERSON-1/5: implement approved metadata preparation.")
