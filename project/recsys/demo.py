"""Executable example joining shared data, baselines, metrics, and artifacts."""

from dataclasses import asdict
import json
from idlelib.debugger_r import restart_subprocess_debugger
from os import PathLike
from pathlib import Path
from typing import Union

from .artifacts import REPO_ROOT, fingerprint, new_run, provenance, write_json
from .baselines import popularity, random_scores
from .contracts import require_aligned
from .data import Interaction, candidate_items
from .metrics.accuracy import evaluate_accuracy


def run(
        output: Union[str, PathLike[str]],
        seed: int = 42, k: int = 2) -> Path:
    fixture = json.loads((REPO_ROOT / "project/examples/toy.json").read_text())
    train = [Interaction(*row) for row in fixture["train"]]
    relevant: dict[str, set[str]] = {}
    for user, item in fixture["valid"]:
        relevant.setdefault(user, set()).add(item)
    candidates = candidate_items(train, relevant, fixture["catalog"])
    split_id = fingerprint(fixture)
    tables = [popularity(train, candidates, split_id), random_scores(candidates, split_id, seed)]
    require_aligned(tables)
    rankings = {table.model: table.top_k(k) for table in tables}
    evaluations = {model: evaluate_accuracy(rows, relevant, k) for model, rows in rankings.items()}
    metrics = {model: result["aggregate"] for model, result in evaluations.items()}
    output = new_run(output)
    write_json(output / "split.json", fixture)
    write_json(output / "scores.json", [asdict(table) for table in tables])
    write_json(output / "rankings.json", rankings)
    write_json(output / "metrics.json", metrics)
    write_json(output / "per_user_metrics.json", {
        model: result["per_user"] for model, result in evaluations.items()
    })
    write_json(output / "manifest.json", {
        **provenance(), "status": "complete", "kind": "synthetic-demo",
        "split_id": split_id, "seed": seed, "k": k,
        "note": "Synthetic demonstration only; not project experiment results.",
    })
    return output
