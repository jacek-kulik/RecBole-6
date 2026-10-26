"""Refit a validation-selected Ridge model and report frozen test metrics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .component_cache import atomic_json, file_hash, run_lock
from .contracts import require_aligned
from .hybrid_report import read_pairs, test_candidates
from .hybrids import fit_oof_regression, predict_weighted_hybrid
from .metrics.accuracy import evaluate_accuracy
from .recbole_adapter import _write_scores, load_scores


def report(run, selected):
    run, selected = Path(run).resolve(), Path(selected).resolve()
    with run_lock(run), threadpool_limits(limits=2):
        inputs = json.loads((selected / "inputs.json").read_text())
        selection = json.loads((selected / "selection.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        for filename, key in (("hybrid.json", "source_hybrid_sha256"),
                              ("validation.json", "source_validation_sha256"),
                              ("oof-progress.npz", "source_oof_sha256"),
                              ("splits/valid.csv", "validation_pairs_sha256")):
            if file_hash(run / filename) != inputs[key]:
                raise ValueError(f"Search input changed: {filename}")
        trials = json.loads((selected / "trials.json").read_text())
        best = max(trials, key=lambda row: row["metrics"]["NDCG@10"])
        saved = json.loads((selected / "hybrid.json").read_text())
        if best != selection["best"] or best["fitted"] != saved:
            raise ValueError("Saved fit does not match validation selection")
        with np.load(run / "oof-progress.npz", allow_pickle=False) as oof:
            fitted = fit_oof_regression(oof["features"], oof["targets"],
                                        saved["components"], saved["alpha"])
        if not np.allclose(fitted["coefficients"], saved["coefficients"], rtol=0, atol=1e-12):
            raise ValueError("Refit differs from selected coefficients")
        if not np.isclose(fitted["intercept"], saved["intercept"], rtol=0, atol=1e-12):
            raise ValueError("Refit intercept differs")
        results = selected / "evaluation"
        results.mkdir(exist_ok=True)
        atomic_json(results / "hybrid.json", fitted)
        training, validation, testing = [read_pairs(run / "splits" / f"{p}.csv")
                                         for p in ("train", "valid", "test")]
        catalogue = set().union(*training.values(), *validation.values(), *testing.values())
        metrics = {"valid": {}, "test": {}}
        proof = {"selected_fit_sha256": file_hash(selected / "hybrid.json"),
                 "selection_sha256": file_hash(selected / "selection.json"),
                 "oof_sha256": inputs["source_oof_sha256"],
                 "split_id": manifest["split_id"], "test_selection_performed": False,
                 "provisional_fism": True, "component_scores_sha256": {}}
        tables = []
        for name in fitted["components"]:
            pointer = json.loads((run / f"cache-{name}.json").read_text())
            source = Path(pointer["path"]) / "scores-valid.json.gz"
            if file_hash(source) != inputs["component_score_sha256"][name]:
                raise ValueError(f"Component changed after selection: {name}")
            table = load_scores(source)
            if (table.model, table.split_id) != (name, manifest["split_id"]):
                raise ValueError("Component metadata differs")
            metrics["valid"][name] = evaluate_accuracy(table.top_k(10), validation, 10)
            test = test_candidates(table, training, validation, testing, catalogue)
            metrics["test"][name] = evaluate_accuracy(test.top_k(10), testing, 10)
            tables.append(test)
            proof["component_scores_sha256"][name] = inputs["component_score_sha256"][name]
            print(f"{name}: frozen component evaluated", flush=True)
        require_aligned(tables)
        valid = load_scores(selected / "scores-valid-hybrid.json.gz")
        metrics["valid"]["regressor"] = evaluate_accuracy(valid.top_k(10), validation, 10)
        for metric, value in best["metrics"].items():
            if not np.isclose(metrics["valid"]["regressor"]["aggregate"][metric], value, atol=1e-12, rtol=0):
                raise ValueError("Validation score table does not reproduce selection")
        test = predict_weighted_hybrid(tables, fitted)
        metrics["test"]["regressor"] = evaluate_accuracy(test.top_k(10), testing, 10)
        _write_scores(test, results / "scores-test-hybrid.json.gz")
        atomic_json(results / "metrics.json", metrics)
        atomic_json(results / "frozen-selection.json", proof)
        columns = [f"{m}@10" for m in ("Precision", "Recall", "F1", "MRR", "NDCG", "MAP")]
        lines = ["# Ridge 100k: updated seven-model comparison", "",
                 "> Provisional: FISM training still includes the candidate in its own history; FISM and the hybrid require correction and rerunning.", "",
                 f"MovieLens-100K; 60/20/20 per-user split, seed 42; {len(testing)} users. Ridge alpha {fitted['alpha']:g} selected using validation NDCG@10 and refitted on {fitted['training_rows']} OOF raw-rating targets. Component models and histories are unchanged. Test candidates exclude train and validation interactions; per-user score normalization is recomputed for that candidate pool. Original alpha-100 results are preserved."]
        old = json.loads((run / "evaluation/metrics.json").read_text())
        for partition in ("valid", "test"):
            lines += ["", f"## {partition.title()}", "", "| Model | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
            with (results / f"metrics-{partition}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["Model", *columns])
                writer.writeheader()
                for name in (*fitted["components"], "regressor"):
                    values = metrics[partition][name]["aggregate"]
                    if name != "regressor" and values != old[partition][name]["aggregate"]:
                        raise ValueError(f"Component baseline changed: {name}")
                    writer.writerow({"Model": name, **values})
                    lines.append("| " + name + " | " + " | ".join(f"{values[c]:.4f}" for c in columns) + " |")
        lines += ["", "## Coefficients", "", "| Component | Alpha 100 | Alpha 100000 |", "|---|---:|---:|"]
        prior = json.loads((run / "hybrid.json").read_text())
        lines += [f"| {name} | {a:.8f} | {b:.8f} |" for name, a, b in zip(fitted["components"], prior["coefficients"], fitted["coefficients"])]
        lines += ["", f"Intercept: {prior['intercept']:.8f} → {fitted['intercept']:.8f}.", "",
                  "Reproduce: `.venv/Scripts/python.exe -m project.recsys.ridge_selected_report --run project/artifacts/hybrid-60-20-20 --selected project/artifacts/hybrid-60-20-20/ridge-extended`."]
        (results / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps(metrics["test"]["regressor"]["aggregate"], indent=2), flush=True)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--selected", required=True)
    args = parser.parse_args()
    print(report(args.run, args.selected))
