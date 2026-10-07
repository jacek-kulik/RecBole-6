"""Jacek's accuracy validation runner for a saved training run's data"""

import hashlib
import json
from os import PathLike
from pathlib import Path
from typing import Any, Dict, TypedDict
from .artifacts import new_run, provenance, write_json
from .metrics.accuracy import evaluate_accuracy
from .recbole_adapter import validation_predictions


class MetricComparison(TypedDict):
    independent: float
    recbole: float
    difference: float
    matches_reported_precision: bool

def validate_training_run(
        run_dir: str | PathLike[str],
        output: str | PathLike[str],
        k: int = 10,
        batch_size: int = 1024) -> Path:
    """Save rankings, accuracy, a worked example, and diagnostic comparisons.
    """

    if type(k) is not int or k <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("k and batch_size must be positive integers")

    run_dir = Path(run_dir).resolve()
    source: Dict[str, Any] = json.loads((run_dir / "manifest.json").read_text())

    if source.get("kind") != "recbole-training" or source.get("status") != "complete":
        raise ValueError("Source must be a completed RecBole training run")

    checkpoint = run_dir / source["checkpoint"]

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    diagnostics: Dict[str, Any] = json.loads((run_dir / "validation.json").read_text())
    output = new_run(output)

    manifest: Dict[str, Any] = {
        **provenance(), "status": "running", "kind": "independent-validation",
        "source_run": str(run_dir), "source_git_commit": source["git_commit"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "split_id": source["split_id"], "model": source["model"],
        "dataset": source["dataset"], "seed": source["seed"],
        "partition": "valid", "k": k, "batch_size": batch_size,
        "test_evaluated": False,
        "relevance": "all held-out validation interactions",
        "candidate_policy": "RecBole full-sort catalogue excluding padding and training history"
    }
    write_json(output / "manifest.json", manifest)

    try:
        config, rankings, relevant = validation_predictions(
            checkpoint, run_dir / "splits", source["split_id"], k, batch_size
        )

        if any(config[field] != source[field] for field in ("model", "dataset", "seed")):
            raise ValueError("Checkpoint metadata does not match the training manifest")

        results = evaluate_accuracy(rankings, relevant, k)
        aggregate = results["aggregate"]
        comparison = {}
        reference = {name.lower(): value for name, value in diagnostics["metrics"].items()}
        decimals = config["metric_decimal_place"]

        # For the metrics that recbole also tracks, are we close? Might also not be close because
        # our metrics may compute things differently
        for name, value in aggregate.items():
            if name.lower() in reference:
                baseline = reference[name.lower()]
                comparison[name] = {
                    "independent": value, "recbole": baseline, "difference": value - baseline,
                    "matches_reported_precision": round(value, decimals) == baseline
                }

        # User for the worked example with some hits
        user = next((user for user in sorted(rankings) if results["per_user"][user][f"Recall@{k}"] > 0), min(rankings))
        example: Dict[str, Any] = {
            "user": user, "relevant": sorted(relevant[user]), "k": k,
            "ranking": [{"rank": rank, "item": item, "hit": item in relevant[user]}
                        for rank, item in enumerate(rankings[user][:k], 1)],
            "metrics": results["per_user"][user],
            "n_hits": sum(item in relevant[user] for item in rankings[user][:k]),
            "n_relevant": len(relevant[user]),
        }

        write_json(output / "rankings.json", rankings)
        write_json(output / "relevant.json", {user: sorted(items) for user, items in relevant.items()})
        write_json(output / "metrics.json", aggregate)
        write_json(output / "per_user_metrics.json", results["per_user"])
        write_json(output / "comparison.json", comparison)
        write_json(output / "example.json", example)
        manifest.update(status="complete", n_users=len(rankings), splits_verified=True)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise error
    finally:
        write_json(output / "manifest.json", manifest)
    return output