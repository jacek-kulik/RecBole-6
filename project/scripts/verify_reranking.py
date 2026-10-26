"""Audit saved rankings, accuracy, exposure, selections and comparison tables."""

import argparse
import csv
import json
from collections import defaultdict
from math import isclose, log2
from pathlib import Path

from project.recsys.artifacts import write_json
from project.recsys.data import read_training_interactions
from project.recsys.recbole_adapter import load_scores
from project.recsys.rerank import file_hash, source_identity


def read(path: Path):
    return json.loads(path.read_text())


def close(actual, expected, label):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{label}: availability differs")
    elif not isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{label}: {actual} differs from {expected}")


def accuracy(items, relevant):
    """Independent binary ranking calculation, including all-relevant AP."""
    k, count = len(items), len(relevant)
    hits = [int(item in relevant) for item in items]
    precision, recall = sum(hits) / k, sum(hits) / count if count else 0
    cumulative = 0
    ap = 0.
    for rank, hit in enumerate(hits, 1):
        cumulative += hit
        ap += hit * cumulative / rank
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(k, count) + 1))
    return dict(zip((f"{name}@{k}" for name in ("Precision", "Recall", "F1", "MRR", "NDCG", "MAP")), (
        precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0,
        next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0),
        sum(hit / log2(rank + 1) for rank, hit in enumerate(hits, 1)) / ideal if ideal else 0,
        ap / count if count else 0,
    )))


def verify(run: Path) -> dict:
    root = read(run / "manifest.json")
    if root["status"] != "complete":
        raise ValueError("The six-model run is incomplete")
    expected_comparison = {}
    checked_lists = 0
    evaluation_count = 0
    selections = 0
    test_budget_exceeded = []
    for model in root["models"]:
        source = Path(root["source_root"]) / f"{model}-{root['run_suffix']}"
        training = read(source / "manifest.json")
        identity = source_identity(source, training)
        selection = read(run / model / "valid/selection.json")
        if selection["identity"] != identity:
            raise ValueError(f"{model}: source identity changed")
        valid_rows = read(run / model / "valid/summary.json")
        baseline_ndcg = valid_rows[0]["ndcg"]
        for method, settings in selection["methods"].items():
            if settings["status"] != "complete":
                continue
            rows = [row for row in valid_rows if row["method"] == method
                    and row["ndcg"] >= .95 * baseline_ndcg and row["target"] is not None]
            sign = 1 if settings["direction"] == "minimize" else -1
            best = min(rows, key=lambda row: (sign * row["target"], -row["ndcg"], row["strength"]))
            if settings["selected"]["strength"] != best["strength"]:
                raise ValueError(f"{model}/{method}: validation selection is incorrect")
            selections += 1

        for partition in ("valid", "test"):
            folder = run / model / partition
            manifest = read(folder / "manifest.json")
            if manifest["identity"] != identity or manifest["status"] != "complete":
                raise ValueError(f"{model}/{partition}: identity or status differs")
            if partition == "test" and manifest["selection"]["sha256"] != file_hash(run / model / "valid/selection.json"):
                raise ValueError(f"{model}: frozen selection hash differs")
            table = load_scores(source / training["score_files"][partition])
            history = defaultdict(set)
            for name in (("train",) if partition == "valid" else ("train", "valid")):
                for row in read_training_interactions(source / f"splits/{name}.csv"):
                    history[row.user].add(row.item)
            catalogue = set().union(*(set(row) for row in table.scores.values()))
            if len(catalogue) != read(folder / "original/manifest.json")["n_catalogue_items"]:
                raise ValueError("Catalogue union does not cover the recorded catalogue")
            rows = read(folder / "summary.json")
            for row in rows:
                evaluation = folder / row["evaluation"]
                rankings, relevant = read(evaluation / "rankings.json"), read(evaluation / "relevant.json")
                saved, aggregate = read(evaluation / "per_user_metrics.json"), read(evaluation / "metrics.json")
                exposures = dict.fromkeys(catalogue, 0.)
                computed = []
                for user, items in rankings.items():
                    pool = sorted(table.scores[user], key=lambda item: (-table.scores[user][item], item))[:50]
                    if len(items) != 10 or len(set(items)) != 10 or not set(items) <= set(pool) or set(items) & history[user]:
                        raise ValueError(f"{model}/{partition}: ineligible or repeated recommendation")
                    if row["strength"] == 0 and items != pool[:10]:
                        raise ValueError("Strength zero differs from the original ranking")
                    values = accuracy(items, set(relevant[user]))
                    for metric, value in values.items():
                        close(saved[user][metric], value, metric)
                    computed.append(values)
                    for rank, item in enumerate(items, 1):
                        exposures[item] += 1 / log2(rank + 1)
                for metric, value in aggregate.items():
                    close(value, sum(user[metric] for user in computed) / len(computed), metric)
                    close(row[metric], value, metric)
                beyond = read(evaluation / "beyond_accuracy.json")["aggregate"]
                cumulative, area = 0., 0.
                for exposure in sorted(exposures.values()):
                    area += cumulative + exposure / 2
                    cumulative += exposure
                close(beyond["ExposureGini@10"], 1 - 2 * area / (len(exposures) * cumulative), "Gini")
                close(beyond["CatalogueCoverage@10"], sum(value > 0 for value in exposures.values()) / len(exposures), "Coverage")
                if partition == "test":
                    if row["method"] != "original" and row["strength"] != selection["methods"][row["method"]]["selected"]["strength"]:
                        raise ValueError("Test changed a frozen validation strength")
                    expected_comparison[f"{model}-{row['method']}"] = {**aggregate, **beyond}
                    if row["exceeds_5_percent_loss"]:
                        test_budget_exceeded.append(f"{model}/{row['method']}")
                evaluation_count += 1
                checked_lists += len(rankings)
    comparison = read(run / "comparison/comparison.json")
    if {row["label"] for row in comparison} != expected_comparison.keys():
        raise ValueError("Comparison labels differ from final test evaluations")
    for row in comparison:
        for metric, expected in expected_comparison[row["label"]].items():
            close(row[metric], expected, metric)
    with (run / "comparison/per_user.csv").open() as stream:
        per_user_rows = sum(1 for _ in csv.DictReader(stream))
    return {"status": "passed", "models": len(root["models"]), "evaluations": evaluation_count,
            "checked_ranking_lists": checked_lists, "validation_selections": selections,
            "test_comparison_rows": len(comparison), "comparison_per_user_rows": per_user_rows,
            "test_cases_exceeding_validation_loss_budget": test_budget_exceeded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    if args.output is not None:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
