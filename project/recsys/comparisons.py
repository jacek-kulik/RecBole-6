"""
This file compares selected models and baselines.

The script simply reads evaluation results.
Thus, the models must be first trained and evaluated beforehand:

    python -m project.recsys train    --model-config <yaml> --output <train-dir>
    python -m project.recsys evaluate --run <train-dir> --output <test-dir>
 
Then the test folders are passed straight to this command:
    python -m project.recsys compare --output project/artifacts/compare-001 \\
        project/artifacts/random-001-test project/artifacts/pop-001-test ...
 
A folder can be given a table label with Label=folder. 
The script checks that every folder is comparable and writes:

    comparison.csv / comparison.json   one row per model
    per_user.csv                       per-user metrics of every row (for Victor)
    manifest.json                      the command and each row's provenance
"""

from __future__ import annotations
 
import csv
import json
from os import PathLike
from pathlib import Path
from typing import Any
 
from .artifacts import new_run, provenance, write_json
 
BASELINE_MODELS = ("pop", "random")
KIND_ORDER = ("baseline", "individual", "hybrid")

SHARED_FIELDS = (
    "split_id", "dataset", "partition", "k", "relevance", "candidate_policy",
    "metric_definitions", "aggregation", "smoothing", "group_construction",
    "n_catalogue_items", "n_training_interactions",
)

def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation file: {path}")
    return json.loads(path.read_text())

def resolve(folder: str) -> Path:
    path = Path(folder)

    if path.is_absolute():
        return path
    else:
        return (Path.cwd() / path).resolve()

def kind_of(model: str | None) -> str:
    """baseline for Pop/Random, hybrid if the model name says so, otherwise individual."""
    if model and model.lower() in BASELINE_MODELS:
        return "baseline"
    if model and "hybrid" in model.lower():
        return "hybrid"
    return "individual"

def read_evaluation(argument: str) -> dict[str, Any]:
    """Load a 'folder' or 'Label=folder' argument written by the evaluate function"""
    
    if "=" in argument:
        label, folder_text = argument.split("=", 1)
    else:
        label, folder_text = None, argument
 
    folder = resolve(folder_text)
    if not folder.is_dir():
        raise FileNotFoundError(f"Evaluation folder not found: {folder}")
 
    manifest = read_json(folder / "manifest.json")
    model = manifest.get("model")
    return {
        "label": label or model or folder.name,
        "kind": kind_of(model),
        "folder": folder,
        "manifest": manifest,
        "accuracy": read_json(folder / "metrics.json"),
        "per_user_accuracy": read_json(folder / "per_user_metrics.json"),
        "beyond": read_json(folder / "beyond_accuracy.json"),
    }


def check_comparable(evaluations: list[dict[str, Any]], k: int) -> None:
    """Raise ValueError listing every problem; Return nothing if all runs are comparable"""
    problems = []

    labels = []
    for evaluation in evaluations:
        labels.append(evaluation["label"])

    for label in sorted(set(labels)):
        if labels.count(label) > 1:
            problems.append(f"label '{label}' is used {labels.count(label)} times")

    reference = evaluations[0]
    expected_users = set(reference["per_user_accuracy"])
    for evaluation in evaluations:
        label = evaluation["label"] 
        manifest = evaluation["manifest"]

        if manifest.get("partition") != "test" or manifest.get("kind") != "independent-test-evaluation":
            problems.append(f"{label}: not a test evaluation (partition '{manifest.get('partition')}')")

        if manifest.get("status") != "complete":
            problems.append(f"{label}: evaluation status is '{manifest.get('status')}', not 'complete'")
        if manifest.get("k") != k:
            problems.append(f"{label}: evaluated at k={manifest.get('k')}, but this comparison uses k={k}")

        for field in SHARED_FIELDS:
            if manifest.get(field) != reference["manifest"].get(field):
                problems.append(f"{label}: '{field}' differs from {reference['label']}")


        users = set(evaluation["per_user_accuracy"])
        if users != expected_users:
            problems.append(f"{label}: evaluated on different users than {reference['label']}")
        if set(evaluation["beyond"]["per_user"]) != users:
            problems.append(f"{label}: accuracy and beyond-accuracy cover different users")
 
    if problems:
        raise ValueError("Runs are not comparable:\n  - " + "\n  - ".join(problems))
 
def table_row(evaluation: dict[str, Any]) -> dict[str, Any]:
    """One table row: accuracy, beyond-accuracy, gender gaps and exposure shares."""
    beyond = evaluation["beyond"]
    row: dict[str, Any] = {
        "label": evaluation["label"],
        "kind": evaluation["kind"],
        "model": evaluation["manifest"].get("model"),
        "n_users": len(evaluation["per_user_accuracy"]),
    }
    row.update(evaluation["accuracy"])
    row.update(beyond["aggregate"])
 
    gaps = beyond.get("gender_fairness", {}).get("gaps", {})
    for metric, gap in gaps.items():
        row[f"Gender_Absolute_{metric}"] = gap.get("absolute")
        row[f"Gender_Signed_F_Minus_M_{metric}"] = gap.get("signed_f_minus_m")

    item_groups = beyond.get("item_group_exposure", {})
    for group, exposure in item_groups.items():
        row[f"Group_{group}_share"] = exposure.get("share")

    return row
 
 
def build_table(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def kind_position(row: dict[str, Any]) -> int:
        """Where a row's kind comes in the table: 0 = baseline, 1 = individual, 2 = hybrid."""
        return KIND_ORDER.index(row["kind"])

    rows = []
    for evaluation in evaluations:
        rows.append(table_row(evaluation))

    rows = sorted(rows, key=kind_position)
    return rows
 
 
def per_user_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for evaluation in evaluations:
        beyond = evaluation["beyond"]["per_user"]
        for user in sorted(evaluation["per_user_accuracy"]):
            row = {"label": evaluation["label"], "user": user}
            row.update(evaluation["per_user_accuracy"][user])
            row.update(beyond[user])
            rows.append(row)
    return rows
 
 
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" if row.get(column) is None else row.get(column) for column in columns})
 
def compare_runs(evaluation_folders: list[str], output: str | PathLike[str], k: int = 10) -> Path:
    if not evaluation_folders:
        raise ValueError("Give at least one evaluation folder to compare")
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")

    evaluations = []
    for argument in evaluation_folders:
        evaluations.append(read_evaluation(argument))
    check_comparable(evaluations, k)
 
    output = new_run(output)
    rows = build_table(evaluations)
    write_csv(output / "comparison.csv", rows)
    write_json(output / "comparison.json", rows)
    write_csv(output / "per_user.csv", per_user_rows(evaluations))
 
    reference = evaluations[0]["manifest"]
    write_json(output / "manifest.json", {
        **provenance(),
        "kind": "model-comparison",
        "status": "complete",
        "k": k,
        "partition": "test",
        "split_id": reference.get("split_id"),
        "dataset": reference.get("dataset"),
        "pending_work": reference.get("pending_work"),
        "runs": [{
            "label": evaluation["label"],
            "kind": evaluation["kind"],
            "model": evaluation["manifest"].get("model"),
            "evaluation_folder": str(evaluation["folder"]),
            "source_run": evaluation["manifest"].get("source_run"),
            "source_git_commit": evaluation["manifest"].get("source_git_commit"),
            "evaluation_git_commit": evaluation["manifest"].get("git_commit"),
            "checkpoint_sha256": evaluation["manifest"].get("checkpoint_sha256"),
            "seed": evaluation["manifest"].get("seed"),
        } for evaluation in evaluations],
    })
    return output