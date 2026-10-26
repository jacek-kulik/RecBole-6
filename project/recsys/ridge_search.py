"""Expand Ridge penalties on cached OOF/validation data; never access test data."""

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .component_cache import atomic_json, file_hash, run_lock
from .contracts import ScoreTable
from .hybrids import candidate_features, fit_oof_regression
from .metrics.accuracy import evaluate_accuracy
from .recbole_adapter import _write_scores, load_scores

DEFAULT_ALPHAS = [100., 300., 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6,
                  3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11, 3e11, 1e12]


def candidate_blocks(pairs):
    """Preserve the external-ID ordering used to break ScoreTable ties."""
    blocks = []
    start = 0
    while start < len(pairs):
        user = pairs[start][0]
        end = start + 1
        while end < len(pairs) and pairs[end][0] == user:
            end += 1
        items = [item for _, item in pairs[start:end]]
        if items != sorted(items) or len(set(items)) != len(items):
            raise ValueError("Candidate features must have sorted, unique item IDs")
        blocks.append((user, start, end, items))
        start = end
    return blocks


def rank_predictions(blocks, scores, k):
    rankings = {}
    for user, start, end, items in blocks:
        if end - start < k:
            raise ValueError("Too few candidates")
        order = np.argsort(-scores[start:end], kind="stable")[:k]
        rankings[user] = [items[index] for index in order]
    return rankings


def search(run, output, alphas, k=10):
    run, output = Path(run).resolve(), Path(output).resolve()
    if k != 10:
        raise ValueError("This extension compares the existing NDCG@10 experiment")
    alphas = sorted(set(map(float, alphas)))
    if not alphas or any(not np.isfinite(a) or a < 0 for a in alphas):
        raise ValueError("Ridge penalties must be finite and nonnegative")
    with run_lock(run), threadpool_limits(limits=2):
        prior = json.loads((run / "hybrid.json").read_text())
        prior_selection = json.loads((run / "validation.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["status"] != "complete" or manifest["smoke_only"]:
            raise ValueError("Complete a full OOF run first")
        alphas = sorted(set(alphas + [prior["alpha"]]))
        with np.load(run / "oof-progress.npz", allow_pickle=False) as saved:
            oof, target = saved["features"].copy(), saved["targets"].copy()
        if (oof.shape != (len(target), len(prior["components"]))
                or not np.isfinite(oof).all() or not np.isfinite(target).all()
                or target.min() < 1 or target.max() > 5 or target.max() <= 1):
            raise ValueError("Expected complete OOF features and restored raw 1-5 targets")
        proof = {"source_hybrid_sha256": file_hash(run / "hybrid.json"),
                 "source_validation_sha256": file_hash(run / "validation.json"),
                 "source_oof_sha256": file_hash(run / "oof-progress.npz"),
                 "validation_pairs_sha256": file_hash(run / "splits/valid.csv"),
                 "search_source_sha256": file_hash(__file__),
                 "split_id": manifest["split_id"], "test_evaluated": False,
                 "component_retraining": False, "component_score_sha256": {}}
        tables = []
        for name in prior["components"]:
            pointer = json.loads((run / f"cache-{name}.json").read_text())
            directory = Path(pointer["path"])
            path = directory / "scores-valid.json.gz"
            digest = file_hash(path)
            if digest != json.loads((directory / "full.complete.json").read_text())["sha256"]:
                raise ValueError(f"Corrupt validation cache: {name}")
            table = load_scores(path)
            if (table.model, table.partition, table.split_id) != (name, "valid", manifest["split_id"]):
                raise ValueError("Invalid validation score metadata")
            tables.append(table)
            proof["component_score_sha256"][name] = digest
        pairs, design = candidate_features(tables)
        del table, tables
        gc.collect()
        blocks = candidate_blocks(pairs)
        relevant = {}
        with (run / "splits/valid.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                relevant.setdefault(row["user_id"], set()).add(row["item_id"])
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "inputs.json", proof)
        trials = []
        best = None
        for alpha in alphas:
            fitted = fit_oof_regression(oof, target, prior["components"], alpha)
            predictions = design @ np.asarray(fitted["coefficients"]) + fitted["intercept"]
            aggregate = evaluate_accuracy(rank_predictions(blocks, predictions, k), relevant, k)["aggregate"]
            trial = {"alpha": alpha, "metrics": aggregate, "fitted": fitted}
            trials.append(trial)
            if best is None or aggregate["NDCG@10"] > best["metrics"]["NDCG@10"]:
                best = trial
            atomic_json(output / "trials.json", trials)
            print(f"alpha={alpha:g}: validation NDCG@10={aggregate['NDCG@10']:.6f}; MRR@10={aggregate['MRR@10']:.6f}", flush=True)
        baseline = next(t for t in trials if t["alpha"] == prior["alpha"])
        if not np.isclose(baseline["metrics"]["NDCG@10"], prior_selection["selected_score"], atol=1e-12, rtol=0):
            raise ValueError("The baseline does not reproduce the previous validation result")
        atomic_json(output / "hybrid.json", best["fitted"])
        atomic_json(output / "selection.json", {
            "metric": "validation NDCG@10", "best": best, "baseline": baseline,
            "selected_highest_penalty": best["alpha"] == max(alphas),
            "provisional_fism": True, "test_evaluated": False,
            "component_validation_ndcg": prior_selection["components"],
        })
        predictions = design @ np.asarray(best["fitted"]["coefficients"]) + best["fitted"]["intercept"]
        scores = {}
        for (user, item), value in zip(pairs, predictions):
            scores.setdefault(user, {})[item] = float(value)
        _write_scores(ScoreTable("ridge-hybrid", manifest["split_id"], "valid", scores), output / "scores-valid-hybrid.json.gz")
        if file_hash(run / "hybrid.json") != proof["source_hybrid_sha256"]:
            raise ValueError("The original fitted hybrid changed during the search")
        lines = ["# Extended Ridge penalty search", "",
                 "> Provisional: current FISM features retain the target self-contribution issue recorded in the spec audit.", "",
                 "Validation-only search on the existing 60/20/20 OOF experiment. No component retraining and no test evaluation. The original fitted model and reported test results are preserved.", "",
                 "| Ridge alpha | Validation NDCG@10 | MRR@10 | Recall@10 | Precision@10 |",
                 "|---:|---:|---:|---:|---:|"]
        for trial in trials:
            m = trial["metrics"]
            lines.append(f"| {trial['alpha']:g} | {m['NDCG@10']:.6f} | {m['MRR@10']:.6f} | {m['Recall@10']:.6f} | {m['Precision@10']:.6f} |")
        lines += ["", f"Selected alpha: {best['alpha']:g}. Validation NDCG@10: {best['metrics']['NDCG@10']:.6f}; previous alpha {baseline['alpha']:g}: {baseline['metrics']['NDCG@10']:.6f}.", "",
                  f"Highest searched penalty selected: {best['alpha'] == max(alphas)}. SLIMElastic validation NDCG@10: {prior_selection['components']['slimelastic']:.6f}.", "",
                  "This is a validation-selection result. Earlier test results belong to the original alpha-100 model; no test metric is claimed for the newly selected model."]
        (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"Best alpha: {best['alpha']:g}; validation NDCG@10={best['metrics']['NDCG@10']:.6f}", flush=True)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    args = parser.parse_args()
    print(search(args.run, args.output, args.alphas))


if __name__ == "__main__":
    main()
