"""Validation strength sweeps and frozen final-test evaluation for rerankers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from math import isfinite
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from .artifacts import REPO_ROOT, fingerprint, new_run, provenance, write_json
from .comparisons import compare_runs, write_csv
from .contracts import Rankings, RerankContext, ScoreTable
from .data import TrainingStatistics, prepare_metadata, read_training_interactions
from .evaluation import evaluation_manifest, write_ranking_results
from .recbole_adapter import load_scores, partition_predictions
from .rerankers import calibration, diversification, item_fairness, user_fairness
from .rerankers._common import MissingMetadataError

DEFAULT_STRENGTHS = (0., .25, .5, .75, 1.)
MAX_RELATIVE_LOSS = .05
# Owner, callable, target metric, direction. Teammate placeholders stay explicit.
METHODS = {
    "genre_calibration": ("Jacek", calibration.rerank, "GenreMiscalibration", "minimize"),
    "popularity_calibration": ("Jacek", partial(calibration.rerank, attribute="popularity"), "UPD", "minimize"),
    "item_fairness": ("Jacek", item_fairness.rerank, "ExposureGini", "minimize"),
    "diversification": ("Gabriel", diversification.rerank, "ILD", "maximize"),
    "popularity_diversification": ("Gabriel", partial(diversification.rerank, attribute="popularity"),
                                   "MeanSelfInformation", "maximize"),
    "coverage_diversification": ("Gabriel", partial(diversification.rerank, attribute="coverage"),
                                 "CatalogueCoverage", "maximize"),
    "user_fairness": ("Victor", user_fairness.rerank, "GRU", "minimize"),
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(run_dir: Path, source: dict[str, Any]) -> dict[str, Any]:
    """Bind the selection to the checkpoint, both score exports, data and code."""
    digest = hashlib.sha256()
    for partition in ("train", "valid", "test"):
        digest.update(partition.encode())
        digest.update((run_dir / f"splits/{partition}.csv").read_bytes())
    if digest.hexdigest() != source["split_id"]:
        raise ValueError("Saved splits do not match the training split ID")
    data_dir = REPO_ROOT / "dataset" / source["dataset"]
    code_root = Path(__file__).resolve().parent
    code_files = ["rerank.py", "contracts.py", "data.py", "evaluation.py", "recbole_adapter.py",
                  "metrics/accuracy.py", "metrics/beyond_accuracy.py", "metrics/_validation.py"]
    code_files += [str(path.relative_to(code_root)) for path in sorted((code_root / "rerankers").glob("*.py"))]
    return {
        "model": source["model"], "label": source.get("label", source["model"]),
        "dataset": source["dataset"], "seed": source["seed"], "split_id": source["split_id"],
        "checkpoint_sha256": file_hash(run_dir / source["checkpoint"]),
        "scores_sha256": {part: file_hash(run_dir / source["score_files"][part]) for part in ("valid", "test")},
        "data_sha256": {suffix: file_hash(path) if path.is_file() else None for suffix in ("inter", "item", "user")
                        for path in [data_dir / f"{source['dataset']}.{suffix}"]},
        "code_sha256": fingerprint({name: file_hash(code_root / name) for name in code_files}),
    }


def candidate_pool(table: ScoreTable, candidate_count: int, k: int) -> ScoreTable:
    table.validate()
    if type(k) is not int or k <= 0 or type(candidate_count) is not int or candidate_count < k:
        raise ValueError("K must be positive and candidate_count must be an integer at least K")
    if any(len(row) < k for row in table.scores.values()):
        raise ValueError("Every candidate pool must contain at least K items")
    return ScoreTable(table.model, table.split_id, table.partition, {
        user: {item: row[item] for item in sorted(row, key=lambda item: (-row[item], item))[:candidate_count]}
        for user, row in sorted(table.scores.items())
    })


def verify_candidates(table: ScoreTable, run_dir: Path, source: dict[str, Any],
                      catalogue: tuple[str, ...], relevant: dict[str, set[str]], partition: str) -> None:
    if (table.split_id, table.partition, table.model) != (
            source["split_id"], partition, source.get("label", source["model"])):
        raise ValueError("Score table identity does not match the source run and partition")
    if table.scores.keys() != relevant.keys():
        raise ValueError("Scores and evaluation relevance must contain identical users")
    history: dict[str, set[str]] = defaultdict(set)
    for name in (("train",) if partition == "valid" else ("train", "valid")):
        for row in read_training_interactions(run_dir / f"splits/{name}.csv"):
            history[row.user].add(row.item)
    items = set(catalogue)
    for user, row in table.scores.items():
        if set(row) != items - history[user]:
            raise ValueError(f"Score candidates for user {user} do not match the partition history mask")


def select_strength(rows: Sequence[dict[str, Any]], baseline_ndcg: float,
                    direction: str = "minimize") -> dict[str, Any]:
    """The 5% constraint applies to validation, never to test-based selection."""
    if direction not in ("minimize", "maximize"):
        raise ValueError("Selection direction must be minimize or maximize")
    eligible = [row for row in rows if row["target"] is not None and isfinite(row["target"])
                and row["ndcg"] >= (1 - MAX_RELATIVE_LOSS) * baseline_ndcg]
    if not eligible:
        raise MissingMetadataError("The target metric is unavailable for strength selection")
    sign = 1 if direction == "minimize" else -1
    return min(eligible, key=lambda row: (sign * row["target"], -row["ndcg"], row["strength"]))


def _target(beyond: dict[str, Any], name: str, k: int) -> float | None:
    if name == "GRU":
        return beyond["gender_fairness"]["gaps"][f"NDCG@{k}"]["absolute"]
    return beyond["aggregate"][f"{name}@{k}"]


def _save_evaluation(output: Path, rankings: Rankings, relevant: dict[str, set[str]],
                     statistics: TrainingStatistics, common: dict[str, Any], method: str,
                     strength: float, candidate_count: int) -> dict[str, Any]:
    output = new_run(output)
    manifest = {**common, "status": "running", "reranker": method, "strength": strength,
                "reranking_candidate_count": candidate_count,
                "score_normalization": "Per-user min-max over the candidate pool; constant scores map to zero",
                "reranker_ties": "Objective, original score, external item ID",
                "exposure_order": "Rank rounds, sorted external user IDs; reset each experiment"}
    if method in METHODS:
        _, _, target, direction = METHODS[method]
        manifest.update(target_metric=f"{target}@{common['k']}" if target != "GRU" else "GRU",
                        target_direction=direction)
    try:
        accuracy, beyond, groups = write_ranking_results(output, rankings, relevant, statistics, common["k"])
        manifest.update(status="complete", n_users=len(rankings), splits_verified=True,
                        n_catalogue_items=len(statistics.catalogue), n_training_interactions=statistics.n_interactions,
                        activity_boundaries=list(statistics.activity_boundaries), project_evaluation_complete=False,
                        pending_work={**statistics.pending, "group_analysis": groups["reason"]})
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return {"method": method, "strength": strength, "evaluation": output.name,
            "ndcg": accuracy["aggregate"][f"NDCG@{common['k']}"],
            **accuracy["aggregate"], **beyond["aggregate"],
            "GRU": _target(beyond, "GRU", common["k"])}


def run_reranking(run_dir: str | PathLike[str], output: str | PathLike[str],
                  partition: Literal["valid", "test"] = "valid", k: int = 10,
                  candidate_count: int = 50, strengths: Sequence[float] | None = None,
                  selection: str | PathLike[str] | None = None, batch_size: int = 256) -> Path:
    """Sweep validation or evaluate its frozen choices on test, using saved scores."""
    if partition not in ("valid", "test"):
        raise ValueError("Partition must be valid or test")
    if type(k) is not int or k <= 0 or type(candidate_count) is not int or candidate_count < k:
        raise ValueError("K must be positive and candidate_count must be an integer at least K")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if partition == "test" and (selection is None or strengths is not None):
        raise ValueError("Test evaluation requires a validation selection; strength grids are validation-only")
    if partition == "valid" and selection is not None:
        raise ValueError("A validation sweep cannot consume a previous selection")
    values = DEFAULT_STRENGTHS if strengths is None else strengths
    if not values or any(isinstance(value, bool) or not isinstance(value, (int, float))
                         or not isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Strengths must be finite values between zero and one")
    grid = sorted({0., *values})
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    source = json.loads((run_dir / "manifest.json").read_text())
    if source.get("kind") != "recbole-training" or source.get("status") != "complete":
        raise ValueError("Source must be a completed RecBole training run")
    if not all(part in source.get("score_files", {}) for part in ("valid", "test")):
        raise ValueError("Source has no full score exports; train it using the current adapter")
    identity = source_identity(run_dir, source)
    settings = {"k": k, "candidate_count": candidate_count, "max_relative_ndcg_loss": MAX_RELATIVE_LOSS}
    frozen = None
    if selection is not None:
        frozen = json.loads(Path(selection).read_text())
        if (frozen.get("kind"), frozen.get("status"), frozen.get("partition")) != (
                "reranker-selection", "complete", "valid"):
            raise ValueError("Selection must come from a completed validation sweep")
        if frozen.get("identity") != identity or frozen.get("settings") != settings:
            raise ValueError("Validation selection identity/settings do not match this run")

    output = new_run(output)
    manifest = {**provenance(), "kind": "reranker-sweep" if partition == "valid" else "reranker-test",
                "status": "running", "partition": partition, "source_run": str(run_dir),
                "identity": identity, "settings": settings, "strengths": grid if frozen is None else frozen["strengths"],
                "selection": {"path": str(Path(selection).resolve()), "sha256": file_hash(Path(selection))}
                if selection is not None else None}
    write_json(output / "manifest.json", manifest)
    try:
        predictions = partition_predictions(run_dir / source["checkpoint"], run_dir / "splits",
                                            source["split_id"], k, batch_size, partition=partition)
        if any(predictions.config[field] != source[field] for field in ("model", "dataset", "seed")):
            raise ValueError("Checkpoint metadata does not match the training manifest")
        table = load_scores(run_dir / source["score_files"][partition])
        verify_candidates(table, run_dir, source, predictions.catalogue, predictions.relevant, partition)
        pool = candidate_pool(table, candidate_count, k)
        data_dir = REPO_ROOT / "dataset" / source["dataset"]
        statistics = prepare_metadata(read_training_interactions(run_dir / "splits/train.csv"),
                                      data_dir / f"{source['dataset']}.item", data_dir / f"{source['dataset']}.user",
                                      predictions.catalogue)
        context = RerankContext(item_features=statistics.item_features, user_profiles=statistics.genre_profiles,
                                popularity_profiles=statistics.popularity_profiles,
                                user_groups=statistics.gender_groups, item_groups=statistics.item_groups,
                                item_counts=statistics.item_counts)
        common = {**evaluation_manifest(source, run_dir, partition, k, batch_size),
                  "label": source.get("label", source["model"]), "input_identity": identity}
        baseline = _save_evaluation(output / "original", pool.top_k(k), predictions.relevant, statistics,
                                    common, "original", 0., candidate_count)
        records = [baseline]
        methods = {}
        for name, (owner, reranker, metric, direction) in METHODS.items():
            target_key = "GRU" if metric == "GRU" else f"{metric}@{k}"
            method = {"owner": owner, "target_metric": target_key, "direction": direction}
            if frozen is not None and name not in frozen["methods"]:
                methods[name] = {**method, "status": "unavailable",
                                 "reason": "Method was not included in the frozen validation selection"}
                continue
            if frozen is not None and frozen["methods"][name]["status"] != "complete":
                methods[name] = frozen["methods"][name]
                continue
            choices = grid if frozen is None else [frozen["methods"][name]["selected"]["strength"]]
            rows = []
            try:
                for strength in choices:
                    print(f"{common['label']} {partition}: {name} strength={strength:.17g}", flush=True)
                    rankings = reranker(pool, context, k, strength)
                    folder = f"{name}-{strength:.17g}"
                    row = _save_evaluation(output / folder, rankings, predictions.relevant, statistics,
                                            common, name, strength, candidate_count)
                    row["target"] = row[target_key]
                    rows.append(row)
                if frozen is None:
                    chosen = select_strength(rows, baseline["ndcg"], direction)
                    selected = {key: chosen[key] for key in ("strength", "ndcg", "target", "evaluation")}
                else:
                    selected = frozen["methods"][name]["selected"]
                methods[name] = {**method, "status": "complete", "selected": selected}
            except (MissingMetadataError, NotImplementedError) as error:
                methods[name] = {**method, "status": "pending" if isinstance(error, NotImplementedError) else "unavailable",
                                 "reason": str(error)}
                if frozen is not None:
                    methods[name]["selected"] = frozen["methods"][name]["selected"]
                print(f"{name}: {methods[name]['status']} ({owner}): {error}", flush=True)
            records.extend(rows)

        for row in records:
            row["relative_ndcg_loss"] = 1 - row["ndcg"] / baseline["ndcg"] if baseline["ndcg"] else None
            row["exceeds_5_percent_loss"] = row["ndcg"] < .95 * baseline["ndcg"]
        write_csv(output / "summary.csv", records)
        write_json(output / "summary.json", records)
        if frozen is None:
            write_json(output / "selection.json", {"kind": "reranker-selection", "status": "complete", "partition": "valid",
                       "identity": identity, "settings": settings, "strengths": grid,
                       "baseline_ndcg": baseline["ndcg"], "methods": methods})
        else:
            labels = [f"{common['label']}-{row['method']}={output / row['evaluation']}" for row in records]
            compare_runs(labels, output / "comparison", k)
        manifest.update(status="complete", methods=methods, project_evaluation_complete=False,
                        pending_work={name: method["reason"] for name, method in methods.items()
                                      if method["status"] != "complete"})
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return output
