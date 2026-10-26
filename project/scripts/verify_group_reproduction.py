"""Audit fresh baseline runs and regenerate PR 26's group explanations."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean

from project.recsys.analysis.explanations import (
    analyze,
    gender_activity_table,
    gender_genre_profiles,
)
from project.recsys.artifacts import REPO_ROOT, new_run, provenance, write_json
from project.recsys.data import prepare_metadata, read_training_interactions
from project.scripts.reproduce_baseline_comparison import MODELS


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    if expected is None:
        require(actual is None, f"Expected unavailable value, got {actual}")
    else:
        require(actual is not None and math.isclose(actual, expected, abs_tol=1e-12,
                                                    rel_tol=1e-12),
                f"Metric mismatch: {actual} != {expected}")


def accuracy(items, positives, k):
    """Known binary definitions, independent of the production metric functions."""
    hits = [item in positives for item in items]
    count = sum(hits)
    precision, recall = count / k, count / len(positives)
    cumulative = 0
    ap = dcg = 0.0
    for rank, hit in enumerate(hits, 1):
        cumulative += hit
        if hit:
            ap += cumulative / rank
            dcg += 1 / math.log2(rank + 1)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(positives)) + 1))
    return {f"{name}@{k}": value for name, value in {
        "Precision": precision, "Recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if count else 0.0,
        "MRR": next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0),
        "NDCG": dcg / ideal, "MAP": ap / len(positives),
    }.items()}


def list_metrics(rankings, users, catalogue, k):
    if not users:
        return {f"CatalogueCoverage@{k}": None, f"ExposureGini@{k}": None}
    exposure = Counter({item: 0.0 for item in catalogue})
    recommended = set()
    for user in users:
        for rank, item in enumerate(rankings[user], 1):
            exposure[item] += 1 / math.log2(rank + 1)
            recommended.add(item)
    values = sorted(exposure.values())
    gini = sum((2 * rank - len(values) - 1) * value for rank, value in enumerate(values, 1)) \
        / (len(values) * sum(values))
    return {f"CatalogueCoverage@{k}": len(recommended) / len(catalogue), f"ExposureGini@{k}": gini}


def audit_evaluation(run, evaluation, partition):
    manifest = read_json(evaluation / "manifest.json")
    rankings = read_json(evaluation / "rankings.json")
    relevant = read_json(evaluation / "relevant.json")
    groups = read_json(evaluation / "group_metrics.json")
    definitions = read_json(evaluation / "groups.json")
    beyond = read_json(evaluation / "beyond_accuracy.json")
    per_user = read_json(evaluation / "per_user_metrics.json")
    k = manifest["k"]
    splits = {name: read_training_interactions(run / f"splits/{name}.csv")
              for name in ("train", "valid", "test")}
    catalogue = {row.item for rows in splits.values() for row in rows}
    metadata = REPO_ROOT / "dataset/ml-100k"
    statistics = prepare_metadata(splits["train"], metadata / "ml-100k.item",
                                  metadata / "ml-100k.user", sorted(catalogue))
    truth, seen = {}, {}
    for row in splits[partition]:
        truth.setdefault(row.user, set()).add(row.item)
    for name in (("train",) if partition == "valid" else ("train", "valid")):
        for row in splits[name]:
            seen.setdefault(row.user, set()).add(row.item)
    require(set(rankings) == set(truth) == set(relevant), "Evaluation user mismatch")
    require(all(set(relevant[u]) == truth[u] for u in truth), "Relevance differs from saved split")
    independent = {}
    for user, items in rankings.items():
        require(len(items) == len(set(items)) == k, "Nonunique or short top-k list")
        require(set(items) <= catalogue and not set(items) & seen.get(user, set()),
                "Recommendations violate the catalogue or history mask")
        independent[user] = accuracy(items, truth[user], k)
        for metric, value in independent[user].items():
            close(per_user[user][metric], value)
    metrics = read_json(evaluation / "metrics.json")
    for metric, value in metrics.items():
        close(value, fmean(row[metric] for row in independent.values()))
        close(groups["overall"]["accuracy"][metric], value)
    for metric, value in beyond["aggregate"].items():
        if metric in groups["overall"]["beyond_accuracy"]:
            close(groups["overall"]["beyond_accuracy"][metric], value)

    memberships = {
        "activity": {u: [statistics.activity_groups.get(u, "low")] for u in rankings},
        "gender": {u: [statistics.gender_groups.get(u, "unknown")] for u in rankings},
        "genre": {u: sorted(statistics.genre_profiles.get(u, {}),
                             key=statistics.genre_profiles.get(u, {}).get, reverse=True)[:3]
                  for u in rankings},
    }
    require(definitions["activity"]["members"] ==
            {u: memberships["activity"][u][0] for u in rankings}, "Activity memberships differ")
    require(definitions["gender"]["members"] ==
            {u: memberships["gender"][u][0] for u in rankings}, "Gender memberships differ")
    require(definitions["user genre"]["members"] == memberships["genre"], "Genre memberships differ")
    for family, cohorts in groups["user_groups"].items():
        for name, cohort in cohorts.items():
            users = [u for u in rankings if name in memberships[family][u]]
            require(cohort["n_users"] == len(users), "Cohort size mismatch")
            require(cohort["meets_minimum_size"] == (len(users) >= groups["minimum_group_size"]),
                    "Minimum cohort size flag mismatch")
            for metric, value in cohort["accuracy"].items():
                close(value, fmean(independent[u][metric] for u in users) if users else None)
            aggregate_lists = list_metrics(rankings, users, catalogue, k)
            for metric, value in cohort["beyond_accuracy"].items():
                if metric in aggregate_lists:
                    close(value, aggregate_lists[metric])
                    continue
                values = [beyond["per_user"][u][metric] for u in users
                          if beyond["per_user"][u][metric] is not None]
                close(value, fmean(values) if values else None)

    def item_memberships(family, item):
        return [statistics.item_groups[item]] if family == "Popularity" else \
            statistics.item_features.get(item, ("unknown",))

    for family, reports in groups["item_groups"].items():
        slots, hits, positives, weights = Counter(), Counter(), Counter(), Counter()
        audience = {}
        for user, items in rankings.items():
            for rank, item in enumerate(items, 1):
                for name in item_memberships(family, item):
                    slots[name] += 1
                    audience.setdefault(name, []).append(user)
                    hits[name] += item in truth[user]
                    weights[name] += 1 / math.log2(rank + 1)
            for item in truth[user]:
                positives.update(item_memberships(family, item))
        for name, report in reports.items():
            require(report["n_items"] == sum(name in item_memberships(family, item)
                                             for item in catalogue), "Item group size mismatch")
            for field, counts in (("n_recommended", slots), ("n_hits", hits), ("n_relevant", positives)):
                require(report[field] == counts[name], f"Item group {field} mismatch")
            for field, counts in (("recommended_share", slots), ("hit_share", hits),
                                  ("relevant_share", positives)):
                close(report[field], counts[name] / counts.total() if counts.total() else None)
            close(report["precision"], hits[name] / slots[name] if slots[name] else None)
            close(report["recall"], hits[name] / positives[name] if positives[name] else None)
            users = audience.get(name, [])
            for metric, value in report["audience_accuracy"].items():
                close(value, fmean(independent[u][f"{metric}@{k}"] for u in users) if users else None)
            for metric, value in report["audience_beyond_accuracy"].items():
                values = [beyond["per_user"][u][f"{metric}@{k}"] for u in users
                          if beyond["per_user"][u][f"{metric}@{k}"] is not None]
                close(value, fmean(values) if values else None)
            if family == "Popularity":
                close(report["exposure"]["total"], weights[name])
                close(report["exposure"]["share"], weights[name] / weights.total())
        if family == "Popularity":
            require(slots.total() == k * len(rankings), "Popularity slots do not conserve top-k lists")
            require(positives.total() == sum(map(len, truth.values())), "Popularity relevance not conserved")

    require(groups["status"] == "complete" and not groups["pending"], "Incomplete group analysis")
    require(manifest["pending_work"] == {}, "Unexpected pending metadata or groups")
    return statistics, {
        "n_users": len(rankings), "n_items": len(catalogue), "k": k,
        "n_training_interactions": len(splits["train"]),
        "activity_boundaries": list(statistics.activity_boundaries),
        "cohort_counts": {family: {name: cohort["n_users"] for name, cohort in cohorts.items()}
                          for family, cohorts in groups["user_groups"].items()},
        "split_id": manifest["split_id"], "history_masks_verified": True,
        "independent_accuracy_verified": True, "cohort_metrics_verified": True,
        "item_counts_and_exposure_verified": True,
        "item_audience_metrics_verified": True,
        "sha256": {name: digest(evaluation / name) for name in
                   ("rankings.json", "relevant.json", "metrics.json", "group_metrics.json", "groups.json")},
    }


def verify(runs, output):
    fresh = read_json(runs / "verification.json")
    require(fresh["status"] == "passed" and fresh["fresh_fits"] == len(MODELS), "Fresh training incomplete")
    output = new_run(output)
    audits, summaries = {}, {}
    for partition in ("valid", "test"):
        records, activity, profiles = {}, {}, {}
        for label, name in MODELS:
            run, evaluation = runs / "training" / name, runs / partition / name
            statistics, audit = audit_evaluation(run, evaluation, partition)
            audits[f"{partition}/{name}"] = audit
            records[label] = {"train": run, "evaluation": evaluation}
            per_user = read_json(evaluation / "per_user_metrics.json")
            counts = Counter({user: len(history) for user, history in statistics.histories.items()})
            activity[label] = gender_activity_table(counts, statistics.gender_groups, per_user, "NDCG@10")
            profiles[label] = gender_genre_profiles(statistics.histories, statistics.item_features,
                                                  statistics.gender_groups, per_user)
        explanations = analyze(records, "NDCG@10")
        explanations.update(gender_activity=activity, gender_genre_profiles=profiles)
        write_json(output / f"explanations-{partition}.json", explanations)
        summaries[partition] = {label: {
            "NDCG@10": read_json(paths["evaluation"] / "metrics.json")["NDCG@10"],
            **explanations["group_facts"][label],
            "history_accuracy_correlation": explanations["sparsity"][label]["pearson_correlation"],
            "gender_activity": activity[label],
            **{f"{family}_ndcg": {name: cohort["accuracy"]["NDCG@10"]
                                   for name, cohort in read_json(
                                       paths["evaluation"] / "group_metrics.json")["user_groups"][family].items()}
               for family in ("activity", "gender")},
        } for label, paths in records.items()}
    for _, name in MODELS[:2]:
        for filename in ("groups.json", "group_metrics.json"):
            require((runs / "test" / name / filename).read_bytes() ==
                    (runs / "replay" / name / filename).read_bytes(), "Batch size changed group results")
    saved = {row["label"]: row for row in read_json(
        REPO_ROOT / "project/reports/taks-2.2_data/compare-final/comparison.json")}
    current = read_json(runs / "comparison/comparison.json")
    deltas = {row["label"]: {key: value - saved[row["label"]][key]
                            for key, value in row.items()
                            if type(value) is float and type(saved[row["label"]].get(key)) in (int, float)}
              for row in current}
    require({audit["split_id"] for audit in audits.values()} == {fresh["split_id"]}, "Split mismatch")
    write_json(output / "summary.json", summaries)
    write_json(output / "verification.json", {
        **provenance(), "status": "passed", "fresh_fits": len(MODELS),
        "evaluations_checked": len(audits), "split_id": fresh["split_id"],
        "source_runs": str(runs.resolve()), "training_verification_sha256": digest(runs / "verification.json"),
        "training_git_commits": sorted({read_json(runs / "training" / name / "manifest.json")["git_commit"]
                                        for _, name in MODELS}),
        "training_git_dirty": {label: read_json(runs / "training" / name / "manifest.json")["git_dirty"]
                               for label, name in MODELS},
        "source_sha256": {name: digest(REPO_ROOT / name) for name in (
            "project/scripts/verify_group_reproduction.py",
            "project/recsys/analysis/groups.py", "project/recsys/analysis/explanations.py",
            "project/recsys/evaluation.py", "project/recsys/rerank.py",
            "project/recsys/metrics/accuracy.py", "project/recsys/metrics/beyond_accuracy.py",
            "project/configs/protocol.yaml")},
        "evaluation_audits": audits, "baseline_metric_deltas": deltas,
        "max_absolute_baseline_metric_delta": max(abs(v) for row in deltas.values() for v in row.values()),
        "baseline_group_batch_replay_verified": True,
        "limitations": ["No saved PR 26 group results exist for a historical numeric comparison.",
                        "Fresh fits use the committed model YAMLs; no new hyperparameter tuning.",
                        "Single-seed descriptive group differences are not causal or significance claims."],
    })
    print(f"Verified {len(audits)} evaluations from {len(MODELS)} fresh fits: {output}")
    return output


def plot_summary(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = read_json(output / "summary.json")["test"]
    labels = [label for label, _ in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained", sharey=True)
    families = (("gender", ("F", "M"), ("#845EC2", "#008E9B"), "Recorded gender"),
                ("activity", ("low", "medium", "high"),
                 ("#81B29A", "#E9C46A", "#E07A5F"), "Training activity"))
    for ax, (family, groups, colors, title) in zip(axes, families):
        width = .8 / len(groups)
        for index, (group, color) in enumerate(zip(groups, colors)):
            positions = [i + (index - (len(groups) - 1) / 2) * width for i in range(len(labels))]
            ax.bar(positions, [summary[label][f"{family}_ndcg"][group] for label in labels],
                   width=width, label=group, color=color)
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, .65)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncols=len(groups))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test NDCG@10")
    fig.suptitle("Fresh MovieLens-100K fits: descriptive cohort results (seed 42)")
    fig.savefig(output / "group-summary.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="Output from reproduce_baseline_comparison")
    parser.add_argument("--output", type=Path, required=True, help="New compact evidence directory")
    parser.add_argument("--plot", action="store_true", help="Also render a summary PNG; requires matplotlib")
    args = parser.parse_args()
    verify(args.runs, args.output)
    if args.plot:
        plot_summary(args.output)
