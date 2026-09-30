"""Caio, Task 2.2: working reference baselines for the shared score contract."""

from collections import Counter
import random

from .contracts import ScoreTable


def popularity(train, candidates, split_id, partition="valid"):
    # Counts come only from training interactions. Candidate items with no
    # training interactions get zero; whether they are eligible is a protocol choice.
    counts = Counter(row.item for row in train)
    table = ScoreTable("popularity", split_id, partition, {
        user: {item: float(counts[item]) for item in items}
        for user, items in candidates.items()
    })
    table.validate()
    return table


def random_scores(candidates, split_id, seed, partition="valid"):
    rng = random.Random(seed)
    table = ScoreTable("random", split_id, partition, {
        user: {item: rng.random() for item in sorted(candidates[user])}
        for user in sorted(candidates)
    })
    table.validate()
    return table
