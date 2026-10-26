import csv
from collections import Counter
from pathlib import Path
from statistics import correlation, fmean
from typing import Iterable, Mapping, Sequence

from ..data import ACTIVITY_GROUPS, activity_terciles, genre_distribution


def training_history_counts(train_csv: Path) -> Counter[str]:
    """Per-user number of training interactions (the sparsity proxy)."""
    counts: Counter[str] = Counter()
    with open(train_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            counts[row["user_id"]] += 1
    return counts


def sparsity_accuracy_curve(
        history_counts: Mapping[str, int],
        per_user_metrics: Mapping[str, Mapping[str, float]],
        metric: str,
        n_bins: int = 5,
) -> dict:
    """Equal-count history-length bins with mean accuracy per bin.

    Users are split into ``n_bins`` equal-size bins by training history length
    (low to high).
    """
    users = sorted(per_user_metrics, key=lambda user: (history_counts[user], user))
    edges = [round(i * len(users) / n_bins) for i in range(n_bins + 1)]
    bins = []
    for index in range(n_bins):
        members = users[edges[index]:edges[index + 1]]
        lengths = [history_counts[user] for user in members]
        values = [per_user_metrics[user][metric] for user in members]
        bins.append({
            "n_users": len(members),
            "mean_history": fmean(lengths),
            "mean_accuracy": fmean(values),
        })
    return {
        "metric": metric,
        "n_bins": n_bins,
        "bins": bins,
        "pearson_correlation": correlation(
            [history_counts[user] for user in per_user_metrics],
            [per_user_metrics[user][metric] for user in per_user_metrics]),
    }


def pairwise_disagreement(
        rankings_a: Mapping[str, Sequence[str]],
        rankings_b: Mapping[str, Sequence[str]],
        relevant: Mapping[str, Sequence[str]],
) -> dict:
    """Top-k set disagreement between two models and what it means for hits.

    For every shared user: Jaccard distance of the two top-k lists (1 minus
    overlap share), plus the hit rate on items only one model recommends.
    """
    distances, a_only_hits, b_only_hits = [], [], []
    a_only_total = b_only_total = 0
    for user, items_a in rankings_a.items():
        items_b = rankings_b[user]
        positives = set(relevant[user])
        set_a, set_b = set(items_a), set(items_b)
        union = set_a | set_b
        distances.append(1 - len(set_a & set_b) / len(union))
        only_a = set_a - set_b
        only_b = set_b - set_a
        a_only_hits.extend(item in positives for item in only_a)
        b_only_hits.extend(item in positives for item in only_b)
        a_only_total += len(only_a)
        b_only_total += len(only_b)
    return {
        "mean_jaccard_distance": fmean(distances),
        "a_only_precision": fmean(a_only_hits) if a_only_hits else None,
        "b_only_precision": fmean(b_only_hits) if b_only_hits else None,
        "n_a_only_items": a_only_total,
        "n_b_only_items": b_only_total,
    }


def analyze(
        run_records: Mapping[str, Mapping[str, Path]],
        metric: str,
) -> dict:
    """Explain model behavior across one evaluation partition.
    """
    evidence: dict = {"metric": metric, "sparsity": {}, "disagreement": {}, "group_facts": {}}
    loaded: dict[str, dict] = {}
    for model, paths in run_records.items():
        per_user = _load_json(Path(paths["evaluation"]) / "per_user_metrics.json")
        rankings = _load_json(Path(paths["evaluation"]) / "rankings.json")
        relevant = _load_json(Path(paths["evaluation"]) / "relevant.json")
        history = training_history_counts(Path(paths["train"]) / "splits" / "train.csv")
        loaded[model] = {"rankings": rankings, "relevant": relevant}
        evidence["sparsity"][model] = sparsity_accuracy_curve(history, per_user, metric)
        group_metrics = _load_json(Path(paths["evaluation"]) / "group_metrics.json")
        gender = group_metrics["user_groups"]["gender"]
        evidence["group_facts"][model] = {
            "gender_ndcg_gap": _relative_gap(gender, metric),
            "head_recommended_share": _head_share(group_metrics),
        }
    models = sorted(loaded)
    for index, left in enumerate(models):
        for right in models[index + 1:]:
            evidence["disagreement"][f"{left}|{right}"] = pairwise_disagreement(
                loaded[left]["rankings"], loaded[right]["rankings"],
                loaded[left]["relevant"])
    return evidence


def gender_activity_table(
        history_counts: Mapping[str, int],
        gender_groups: Mapping[str, str],
        per_user_metrics: Mapping[str, Mapping[str, float]],
        metric: str,
) -> dict:
    """Per-user accuracy by activity tercile and recorded gender.
    """
    boundaries, activity = activity_terciles(history_counts)
    table = {group: {gender: [] for gender in ("F", "M")} for group in ACTIVITY_GROUPS}
    lengths = {group: {gender: [] for gender in ("F", "M")} for group in ACTIVITY_GROUPS}
    for user, values in per_user_metrics.items():
        gender = gender_groups.get(user)
        group = activity[user]
        if gender in table[group]:
            table[group][gender].append(values[metric])
            lengths[group][gender].append(history_counts[user])
    rows = []
    for group in ACTIVITY_GROUPS:
        female, male = table[group]["F"], table[group]["M"]
        row = {gender: {"n_users": len(values),
                        "mean_history": fmean(lengths[group][gender]) if lengths[group][gender] else None,
                        "mean_accuracy": fmean(values) if values else None}
               for gender, values in (("F", female), ("M", male))}
        if row["F"]["mean_accuracy"] and row["M"]["mean_accuracy"]:
            row["gap_within"] = (row["M"]["mean_accuracy"] - row["F"]["mean_accuracy"]) / row["F"]["mean_accuracy"]
        else:
            row["gap_within"] = None
        rows.append(row)
    return {"metric": metric, "boundaries": boundaries, "terciles": dict(zip(ACTIVITY_GROUPS, rows))}


def gender_genre_profiles(
        histories: Mapping[str, Sequence[str]],
        item_features: Mapping[str, Sequence[str]],
        gender_groups: Mapping[str, str],
        users: Iterable[str],
) -> dict:
    """Per-gender genre shares of the training histories of ``users``.

    Each history item contributes one genre unit split equally among its
    genres (the same weighting as ``genre_distribution``), so shares per
    gender sum to one and the two profiles are directly comparable.
    """
    mass: dict[str, Counter] = {"F": Counter(), "M": Counter()}
    totals = {"F": 0.0, "M": 0.0}
    for user in users:
        gender = gender_groups.get(user)
        if gender not in mass or user not in histories:
            continue
        for genre, share in genre_distribution(histories[user], item_features).items():
            mass[gender][genre] += share
        totals[gender] += 1.0
    return {gender: {genre: value / totals[gender] for genre, value in counter.items()}
            for gender, counter in mass.items() if totals[gender]}


def _load_json(path: Path):
    import json
    return json.loads(path.read_text())


def _relative_gap(gender: Mapping, metric: str) -> float | None:
    name = metric.replace("NDCG@10", "NDCG@10")  # metric names already carry @k
    female, male = gender["F"]["accuracy"].get(name), gender["M"]["accuracy"].get(name)
    if female is None or male is None:
        return None
    return (male - female) / female


def _head_share(group_metrics: dict) -> float | None:
    item_groups = group_metrics["item_groups"]
    head = item_groups.get("Popularity", item_groups).get("head", {})
    return head.get("recommended_share")
