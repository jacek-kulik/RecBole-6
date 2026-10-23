"""Jacek's accuracy validation runner for a saved training run's data"""

from __future__ import annotations

import hashlib
import json
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TypedDict

from .analysis.groups import GroupComparison, compare_groups, group_definitions
from .artifacts import REPO_ROOT, new_run, provenance, write_json
from .data import prepare_metadata, read_training_interactions
from .metrics.accuracy import evaluate_accuracy
from .metrics.beyond_accuracy import METRIC_DEFINITIONS, evaluate_beyond_accuracy
from .metrics.types import AccuracyResults, BeyondAccuracyResults
from .recbole_adapter import partition_predictions


class MetricComparison(TypedDict):
    independent: float
    recbole: float
    difference: float
    matches_reported_precision: bool


def print_summary(
        accuracy: AccuracyResults,
        beyond: BeyondAccuracyResults,
        group_report: GroupComparison,
        k: int
) -> None:
    print(f"Users: {len(accuracy['per_user'])}; MRR@{k}: {accuracy['aggregate'][f'MRR@{k}']:.5f}; "
          f"NDCG@{k}: {accuracy['aggregate'][f'NDCG@{k}']:.5f}")
    if beyond["gender_fairness"]["reason"]:
        print(f"Gender fairness unavailable: {beyond['gender_fairness']['reason']}")
    for section, reason in sorted(group_report["pending"].items()):
        print(f"Group analysis pending for {section}: {reason}")
    n_user_groups = sum(len(cohorts) for cohorts in group_report["user_groups"].values())
    print(f"Group analysis {group_report['status']}: {n_user_groups} user cohorts and "
          f"{len(group_report['item_groups'])} item groups at k={group_report['k']}")


def run_evaluation(
        run_dir: str | PathLike[str],
        output: str | PathLike[str],
        partition: Literal["valid", "test"],
        k: int = 10,
        batch_size: int = 1024) -> Path:
    """Run Jacek's metrics; identify unfinished teammate inputs and reports."""
    if type(k) is not int or k <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("k and batch_size must be positive integers")
    if partition not in {"valid", "test"}:
        raise ValueError("Evaluation partition must be valid or test")
    run_dir = Path(run_dir).resolve()
    source: dict[str, Any] = json.loads((run_dir / "manifest.json").read_text())
    if source.get("kind") != "recbole-training" or source.get("status") != "complete":
        raise ValueError("Source must be a completed RecBole training run")
    checkpoint = run_dir / source["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    diagnostics = json.loads((run_dir / "validation.json").read_text()) if partition == "valid" else None
    output = new_run(output)
    history = "training history" if partition == "valid" else "training and validation history"
    manifest: dict[str, Any] = {
        **provenance(), "status": "running",
        "kind": "independent-validation" if partition == "valid" else "independent-test-evaluation",
        "source_run": str(run_dir), "source_git_commit": source["git_commit"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "split_id": source["split_id"], "model": source["model"], "dataset": source["dataset"],
        "seed": source["seed"], "partition": partition, "k": k, "batch_size": batch_size,
        "test_evaluated": partition == "test", "relevance": f"all held-out {partition} interactions",
        "candidate_policy": f"RecBole full-sort catalogue excluding padding and {history}",
        "metric_definitions": METRIC_DEFINITIONS,
        "aggregation": "Equal user means for per-user metrics; cohort-wide coverage and exposure",
        "smoothing": {"novelty_pseudocount": 1, "genre_profile_mixture": .01},
        "group_construction": {
            "activity": "Linear training-count terciles; ties stay together; zero-history users join low",
            "gender": "Recorded F/M from the dataset user file; any other value counts as unknown",
            "genres": "class column of the item file",
            "item_popularity": "Training interaction shares 20/60/20, with complete count buckets",
        },
        "profile_weights": "Equal interactions; each movie contributes equal total genre mass",
    }
    write_json(output / "manifest.json", manifest)
    try:
        predictions = partition_predictions(checkpoint, run_dir / "splits", source["split_id"],
                                            k, batch_size, partition=partition)
        config, rankings, relevant = predictions.config, predictions.rankings, predictions.relevant
        if any(config[field] != source[field] for field in ("model", "dataset", "seed")):
            raise ValueError("Checkpoint metadata does not match the training manifest")
        data_dir = REPO_ROOT / "dataset" / source["dataset"]
        item_file = data_dir / f"{source['dataset']}.item"
        user_file = data_dir / f"{source['dataset']}.user"
        manifest["metadata_sources"] = {
            "items": {"path": str(item_file), "sha256": hashlib.sha256(item_file.read_bytes()).hexdigest() if item_file.is_file() else None},
            "users": {"path": str(user_file), "sha256": hashlib.sha256(user_file.read_bytes()).hexdigest() if user_file.is_file() else None},
        }
        statistics = prepare_metadata(read_training_interactions(run_dir / "splits/train.csv"),
                                      item_file, user_file, predictions.catalogue)
        accuracy = evaluate_accuracy(rankings, relevant, k)
        beyond = evaluate_beyond_accuracy(rankings, statistics, accuracy["per_user"], k)
        group_report = compare_groups(rankings, relevant, statistics, k)
        comparison: dict[str, MetricComparison] = {}
        if diagnostics is not None:
            reference = {name.lower(): value for name, value in diagnostics["metrics"].items()}
            for name, value in accuracy["aggregate"].items():
                if name.lower() in reference:
                    baseline = reference[name.lower()]
                    comparison[name] = {
                        "independent": value, "recbole": baseline, "difference": value - baseline,
                        "matches_reported_precision": round(value, config["metric_decimal_place"]) == baseline,
                    }
        user = next((user for user in sorted(rankings) if accuracy["per_user"][user][f"Recall@{k}"] > 0), min(rankings))
        example = {
            "user": user, "relevant": sorted(relevant[user]), "k": k,
            "ranking": [{"rank": rank, "item": item, "hit": item in relevant[user]}
                        for rank, item in enumerate(rankings[user][:k], 1)],
            "metrics": accuracy["per_user"][user],
            "n_hits": sum(item in relevant[user] for item in rankings[user][:k]),
            "n_relevant": len(relevant[user]),
        }
        write_json(output / "rankings.json", rankings)
        write_json(output / "relevant.json", {user: sorted(items) for user, items in relevant.items()})
        write_json(output / "metrics.json", accuracy["aggregate"])
        write_json(output / "per_user_metrics.json", accuracy["per_user"])
        if partition == "valid":
            write_json(output / "comparison.json", comparison)
        write_json(output / "example.json", example)
        write_json(output / "beyond_accuracy.json", beyond)
        #write_json(output / "statistics.json", statistics)
        write_json(output / "groups.json", group_definitions(statistics, rankings))
        write_json(output / "group_metrics.json", group_report)
        manifest.update(status="complete", n_users=len(rankings), splits_verified=True,
                        activity_boundaries=list(statistics.activity_boundaries),
                        n_training_interactions=statistics.n_interactions,
                        n_catalogue_items=len(statistics.catalogue),
                        group_minimum_size=group_report["minimum_group_size"],
                        project_evaluation_complete=False)
        print_summary(accuracy, beyond, group_report, k)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return output