"""Test evaluation of frozen, completed feature-augmentation experiments."""

import argparse
import csv
import json
from pathlib import Path

from .artifacts import new_run, write_json
from .component_cache import file_hash
from .contracts import require_aligned
from .feature_augmentation import read_pairs, load_component
from .hybrid_candidates import test_candidates
from .metrics.accuracy import evaluate_accuracy
from .recbole_adapter import load_scores, _write_scores


def report(runs, output):
    # Freeze every supplied experiment before loading test labels. No fitting.
    experiments = []
    for directory in map(lambda p: Path(p).resolve(), runs):
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete experiment: {directory}")
        for name, digest in manifest["outputs_sha256"].items():
            if file_hash(directory / name) != digest:
                raise ValueError(f"Experiment output changed: {directory / name}")
        experiments.append((directory, manifest, file_hash(manifest_path)))
    source = Path(experiments[0][1]["source"])
    split_id = experiments[0][1]["split_id"]
    for _, manifest, _ in experiments:
        if Path(manifest["source"]) != source or manifest["split_id"] != split_id:
            raise ValueError("Different source or split")
        for part in ("train", "valid"):
            if file_hash(source / f"splits/{part}.csv") != manifest["sha256"][part]:
                raise ValueError("Training/development split changed")
        if file_hash(Path(__file__).parent / "metrics/accuracy.py") != manifest["sha256"]["metrics"]:
            raise ValueError("Metrics changed since validation")
    output = new_run(output)
    proof = {"status": "running", "test_selection_performed": False,
             "test_evaluated": False, "split_id": split_id,
             "method": "frozen training-only scores; mask training and validation history; no retraining",
             "k": 10, "source": str(source),
             "frozen_runs": {str(p): digest for p, _, digest in experiments},
             "sha256": {"report": file_hash(__file__),
                        "masking": file_hash(Path(__file__).parent / "hybrid_candidates.py"),
                        "metrics": file_hash(Path(__file__).parent / "metrics/accuracy.py"),
                        **{part: file_hash(source / f"splits/{part}.csv")
                           for part in ("train", "valid", "test")}}}
    write_json(output / "manifest.json", proof)
    training, validation, testing = [read_pairs(source / f"splits/{part}.csv")
                                     for part in ("train", "valid", "test")]
    catalogue = set().union(*training.values(), *validation.values(), *testing.values())
    metrics, reference, baselines = {}, None, {}

    def evaluate(name, table):
        nonlocal reference
        if table.split_id != split_id:
            raise ValueError("Score split differs")
        masked = test_candidates(table, training, validation, testing, catalogue)
        if reference is None:
            reference = masked
        else:
            require_aligned([reference, masked])
        result = evaluate_accuracy(masked.top_k(10), testing, 10)
        metrics[name] = result
        _write_scores(masked, output / f"scores-test-{name}.json.gz")
        write_json(output / f"metrics-{name}.json", result)
        print(json.dumps({name: result["aggregate"]}), flush=True)

    knn, evidence = load_component(source, "userknn")
    proof["userknn"] = evidence
    evaluate("userknn", knn)
    for directory, manifest, _ in experiments:
        receiver = manifest["receiver"].lower()
        baseline = load_scores(directory / "scores-valid-baseline.json.gz")
        if receiver in baselines:
            require_aligned([baseline, baselines[receiver]])
            if baseline.scores != baselines[receiver].scores:
                raise ValueError("Refitted baselines differ across strengths")
        else:
            baselines[receiver] = baseline
            evaluate(f"{receiver}-baseline", baseline)
        label = "equal" if manifest["equal"] else f"strength-{manifest['strength']:g}"
        name = f"{receiver}-{label}"
        if name in metrics:
            raise ValueError("Duplicate augmented model label")
        evaluate(name, load_scores(directory / "scores-valid-augmented.json.gz"))
    columns = [f"{m}@10" for m in ("Precision", "Recall", "F1", "MRR", "NDCG", "MAP")]
    write_json(output / "metrics.json", metrics)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Model", *columns])
        writer.writeheader()
        for name, result in metrics.items():
            writer.writerow({"Model": name, **result["aggregate"]})
    lines = ["# Frozen feature-augmentation test results", "",
             "MovieLens 100K; seed 42; per-user 60/20/20; 943 users. Binary relevance.", "",
             "All completed variants are reported without test-based selection or parameter changes. "
             "Models, added-entry matrices and training histories are frozen. Test candidates "
             "exclude train and validation interactions. The equal-weight variant has not been run.", "",
             "| Model | " + " | ".join(columns) + " |",
             "| --- | " + " | ".join(["---:"] * len(columns)) + " |"]
    for name, result in metrics.items():
        lines.append("| " + name + " | " + " | ".join(f"{result['aggregate'][c]:.6f}" for c in columns) + " |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for directory, _, digest in experiments:
        if file_hash(directory / "manifest.json") != digest:
            raise ValueError("Frozen run changed during test evaluation")
    proof.update(status="complete", test_evaluated=True,
                 models=list(metrics), users=len(testing))
    proof["outputs_sha256"] = {p.name: file_hash(p) for p in output.iterdir() if p.name != "manifest.json"}
    write_json(output / "manifest.json", proof)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(report(args.runs, args.output))
