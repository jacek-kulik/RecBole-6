"""Compare frozen OOF-hybrid and component models on aligned valid/test rankings.

No training or parameter selection happens here. For the current six models,
scores depend only on the fixed training history: test candidates are a subset
of validation candidates, so test scores are obtained by masking validation
interactions in the saved full-candidate score tables. Hybrid normalization is
recomputed over test candidates using the already fitted coefficients.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .component_cache import atomic_json, file_hash, run_lock
from .contracts import ScoreTable, require_aligned
from .hybrids import predict_weighted_hybrid
from .metrics.accuracy import evaluate_accuracy
from .oof_hybrid import COMPONENTS
from .raw_rating_target import restore_raw_rating_target
from .recbole_adapter import load_scores, _write_scores


def read_pairs(path):
    relevant = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            relevant.setdefault(row["user_id"], set()).add(row["item_id"])
    return relevant


def test_candidates(table, training, validation, testing, catalogue):
    """Keep model scores fixed while changing only the eligibility mask."""
    if table.partition != "valid" or table.scores.keys() != validation.keys():
        raise ValueError("Validation scores and relevance users differ")
    if validation.keys() != testing.keys() or training.keys() != testing.keys():
        raise ValueError("This report requires identical nonempty user cohorts")
    scores = {}
    for user, row in table.scores.items():
        if (training[user] & validation[user] or training[user] & testing[user]
                or validation[user] & testing[user]):
            raise ValueError("Outer partitions contain overlapping user-item pairs")
        if set(row) != catalogue - training[user]:
            raise ValueError("Validation scores do not have exactly the full eligible catalogue")
        candidates = catalogue - training[user] - validation[user]
        if not testing[user] <= candidates:
            raise ValueError("Some test positives are excluded")
        scores[user] = {item: value for item, value in row.items() if item in candidates}
    result = ScoreTable(table.model, table.split_id, "test", scores)
    result.validate()
    return result


def report(output, k=10):
    output = Path(output).resolve()
    with run_lock(output):
        return _report(output, k)


def _report(output, k):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest.get("prepare_only"):
        raise ValueError("Complete the validation-only OOF run before final evaluation")
    if manifest["components"] != list(COMPONENTS) or manifest["smoke_only"]:
        raise ValueError("The comparison requires the full six-component experiment")
    restore_raw_rating_target(output, k)
    fitted = json.loads((output / "hybrid.json").read_text())
    if fitted["components"] != list(COMPONENTS):
        raise ValueError("Regressor component order does not match")
    with np.load(output / "oof-progress.npz", allow_pickle=False) as saved:
        if saved["features"].shape != (len(saved["targets"]), 6) or not np.isfinite(saved["features"]).all():
            raise ValueError("OOF matrix is incomplete")
        if fitted["training_rows"] != len(saved["targets"]):
            raise ValueError("Regressor row count does not match the OOF matrix")
    selection = json.loads((output / "validation.json").read_text())
    chosen = max(selection["trials"], key=lambda row: row["score"])
    if chosen["alpha"] != fitted["alpha"] or fitted["alpha"] != manifest["selected_alpha"]:
        raise ValueError("Frozen alpha does not match validation selection")
    training, validation, testing = [read_pairs(output / "splits" / f"{part}.csv")
                                     for part in ("train", "valid", "test")]
    catalogue = set().union(*training.values(), *validation.values(), *testing.values())
    results = output / "evaluation"
    results.mkdir(exist_ok=True)
    proof = {"fitted_hybrid_sha256": file_hash(output / "hybrid.json"),
             "validation_selection_sha256": file_hash(output / "validation.json"),
             "report_source_sha256": file_hash(__file__),
             "accuracy_source_sha256": file_hash(Path(__file__).parent / "metrics/accuracy.py"),
             "split_id": manifest["split_id"], "k": k,
             "test_score_method": "fixed component validation scores with additional validation-history mask",
             "test_selection_performed": False, "models": {}}
    atomic_json(results / "frozen-selection.json", proof)
    metrics = {"valid": {}, "test": {}}
    test_tables = []
    for name in COMPONENTS:
        pointer = json.loads((output / f"cache-{name}.json").read_text())
        component = Path(pointer["path"])
        source = component / "scores-valid.json.gz"
        if file_hash(source) != json.loads((component / "full.complete.json").read_text())["sha256"]:
            raise ValueError(f"Corrupt component score cache: {name}")
        table = load_scores(source)
        if (table.model, table.split_id) != (name, manifest["split_id"]):
            raise ValueError("Component score metadata differs")
        metrics["valid"][name] = evaluate_accuracy(table.top_k(k), validation, k)
        test_table = test_candidates(table, training, validation, testing, catalogue)
        metrics["test"][name] = evaluate_accuracy(test_table.top_k(k), testing, k)
        _write_scores(test_table, results / f"scores-test-{name}.json.gz")
        test_tables.append(test_table)
        proof["models"][name] = {"cache_key": pointer["key"], "validation_scores_sha256": file_hash(source)}
        print(f"{name}: independent validation/test metrics complete", flush=True)
    require_aligned(test_tables)
    valid_hybrid = load_scores(output / "scores-valid-hybrid.json.gz")
    if valid_hybrid.split_id != manifest["split_id"] or valid_hybrid.partition != "valid":
        raise ValueError("Validation hybrid score metadata differs")
    metrics["valid"]["regressor"] = evaluate_accuracy(valid_hybrid.top_k(k), validation, k)
    if not np.isclose(metrics["valid"]["regressor"]["aggregate"][f"NDCG@{k}"], selection["selected_score"]):
        raise ValueError("Independent NDCG does not reproduce selection")
    test_hybrid = predict_weighted_hybrid(test_tables, fitted)
    metrics["test"]["regressor"] = evaluate_accuracy(test_hybrid.top_k(k), testing, k)
    _write_scores(test_hybrid, results / "scores-test-hybrid.json.gz")
    if file_hash(output / "hybrid.json") != proof["fitted_hybrid_sha256"]:
        raise ValueError("Hybrid changed during test evaluation")
    atomic_json(results / "frozen-selection.json", proof)
    atomic_json(results / "metrics.json", metrics)
    columns = [f"{name}@{k}" for name in ("Precision", "Recall", "F1", "MRR", "NDCG", "MAP")]
    lines = ["# Six components and OOF Ridge regressor", "",
             f"MovieLens-100K; per-user 60/20/20 split, seed 42; {len(testing)} users; binary relevance for every held-out interaction. Five-fold OOF regression targets raw ratings. Ridge alpha {fitted['alpha']} selected by validation NDCG@{k} before test evaluation.", "",
             "All component scores come from fixed-budget full-training refits. Test scores preserve those exact training histories and weights, excluding training and validation interactions. The other five component caches were reused; FISM uses the corrected logits and its newly selected parameters. The regressor uses all six candidate-normalized score features.", "",
             "The cached loader ratings were normalized to 0-1. Original 1-5 ratings were recovered by external user/item pair and Ridge refitted at the frozen alpha. Coefficients equal four times the normalized-target coefficients, and the intercept equals four times its old value plus one; validation rankings are verified unchanged. Original normalized artifacts are preserved. No component retraining or new alpha selection was needed.", "",
             "Metrics use the existing independent accuracy implementation. MAP divides by the full number of relevant items; F1 is averaged per user. Deterministic ties use external item IDs. These single-seed comparisons do not establish statistical significance."]
    for partition in ("valid", "test"):
        lines += ["", "## " + ("Validation (selection evidence)" if partition == "valid" else "Test (frozen models)"), "",
                  "| Model | " + " | ".join(columns) + " |",
                  "|---|" + "---:|" * len(columns)]
        with (results / f"metrics-{partition}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["Model", *columns], lineterminator="\n")
            writer.writeheader()
            for name in (*COMPONENTS, "regressor"):
                values = metrics[partition][name]["aggregate"]
                writer.writerow({"Model": name, **values})
                lines.append("| " + name + " | " + " | ".join(f"{values[c]:.4f}" for c in columns) + " |")
    lines += ["", "## Regressor parameters", "", f"Intercept: {fitted['intercept']:.8f}. Training rows: {fitted['training_rows']}.", "",
              "| Component | Coefficient |", "|---|---:|"]
    lines += [f"| {name} | {value:.8f} |" for name, value in zip(fitted["components"], fitted["coefficients"])]
    lines += ["", "Coefficients are unconstrained and apply to correlated, normalized scores; their magnitudes are not causal importance.", "",
              "Reproduce after the OOF run: `.venv/Scripts/python.exe -m project.recsys.hybrid_report --output project/artifacts/hybrid-60-20-20`."]
    (results / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    manifest.update(regressor_fitted=True, test_evaluated=True,
                    independent_evaluation_complete=True, evaluation_directory="evaluation",
                    component_selection_notes={"fism": "Corrected model retuned on 60/20/20; embedding_size=128, reg_weights=[0.001,0.001], alpha=0",
                                               "other_five": "Previously approved transfer of 80/10/10 hyperparameters; cached 60/20/20 predictions reused"})
    atomic_json(output / "manifest.json", manifest)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    print(report(args.output, args.k))


if __name__ == "__main__":
    main()
