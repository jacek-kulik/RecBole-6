"""Evaluate validation-selected feature-combination checkpoints on frozen test."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import new_run, write_json
from .component_cache import file_hash
from .content_based import read_history
from .feature_combination import PairMLP, evaluate
from .recbole_adapter import load_scores


def run(run_directory, source):
    run_directory, source = Path(run_directory).resolve(), Path(source).resolve()
    output = new_run(run_directory / "test")
    selected = json.loads((run_directory / "results.json").read_text())
    manifest = json.loads((run_directory / "manifest.json").read_text())
    verified = json.loads((run_directory / "verification.json").read_text())
    mapping = json.loads((run_directory / "mapping.json").read_text())
    checks = {t["trial"]: t for t in verified["trial_checks"]}
    if manifest["status"] != "complete" or verified["status"] != "passed":
        raise ValueError("Completed, verified validation run required")
    frozen = {"split_id": mapping["split_id"], "test_selection_performed": False,
              "refitted": False, "selection_sha256": file_hash(run_directory / "results.json"),
              "validation_manifest_sha256": file_hash(run_directory / "manifest.json"),
              "source_sha256": file_hash(__file__), "models": {}}
    for variant in ("mlp", "mlp-content"):
        frozen["models"][variant] = []
        for trial in selected[variant]["trials"]:
            folder = run_directory / trial["directory"]
            check = checks[trial["directory"]]
            for name, key in (("model.pt", "checkpoint_sha256"), ("scores-valid.npz", "scores_sha256")):
                if file_hash(folder / name) != check[key]:
                    raise ValueError("Selected checkpoint or scores changed after validation")
            frozen["models"][variant].append({**trial, "checkpoint_sha256": check["checkpoint_sha256"],
                                               "scores_sha256": check["scores_sha256"]})
    # Persist the exact selected models before accessing test relevance labels.
    write_json(output / "frozen-selection.json", frozen)
    training, validation, testing = [read_history(source / f"splits/{part}.csv")
                                    for part in ("train", "valid", "test")]
    users, items = mapping["users"], mapping["items"]
    indices = {item: n for n, item in enumerate(items)}
    if set(training) != set(validation) or set(training) != set(testing) or set(testing) != set(users):
        raise ValueError("Split users differ")
    for part in ("train", "valid"):
        if file_hash(source / f"splits/{part}.csv") != manifest["sha256"][part]:
            raise ValueError("Development split changed")
    eligible = np.load(run_directory / "candidate-mask.npz")["eligible"].copy()
    for n, user in enumerate(users):
        if training[user] & validation[user] or training[user] & testing[user] or validation[user] & testing[user]:
            raise ValueError("Split overlap")
        if set(np.array(items)[eligible[n]]) != set(items) - training[user]:
            raise ValueError("Original validation mask differs")
        eligible[n, [indices[i] for i in validation[user]]] = False
        if not testing[user] <= set(np.array(items)[eligible[n]]):
            raise ValueError("Test positives excluded")
    np.savez_compressed(output / "candidate-mask.npz", eligible=eligible)
    baseline_scores = np.load(run_directory / "scores-valid-mf.npz")["scores"]
    reference = load_scores(source / "evaluation/scores-test-bpr.json.gz")
    if reference.split_id != mapping["split_id"] or reference.partition != "test":
        raise ValueError("Existing test reference differs")
    for n, user in enumerate(users):
        if set(reference.scores[user]) != set(np.array(items)[eligible[n]]):
            raise ValueError("Test candidates differ from established component protocol")
        for item, score in reference.scores[user].items():
            if abs(float(baseline_scores[n, indices[item]]) - score) > 1e-6:
                raise ValueError("MF test scores differ from reference")
    results = {"mf": {"aggregate": evaluate(baseline_scores, eligible, users, items, testing)["aggregate"], "seed_count": 1}}
    write_json(output / "metrics-mf.json", evaluate(baseline_scores, eligible, users, items, testing))
    torch.set_num_threads(4)
    for variant in ("mlp", "mlp-content"):
        rows = []
        for trial in selected[variant]["trials"]:
            folder = run_directory / trial["directory"]
            saved = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
            if saved["epoch"] != trial["selected_epoch"] or saved["hidden"] != selected[variant]["selected_width"]:
                raise ValueError("Checkpoint does not match frozen selection")
            model = PairMLP(saved["state_dict"]["users"], saved["state_dict"]["items"], saved["hidden"])
            model.load_state_dict(saved["state_dict"])
            model.eval()
            scores = model.full_scores()
            np.testing.assert_allclose(scores, np.load(folder / "scores-valid.npz")["scores"], rtol=0, atol=1e-6)
            metrics = evaluate(scores, eligible, users, items, testing)
            write_json(output / f"metrics-{trial['directory']}.json", metrics)
            rows.append({"seed": trial["seed"], "selected_epoch": trial["selected_epoch"], **metrics["aggregate"]})
        keys = results["mf"]["aggregate"]
        results[variant] = {"selected_width": selected[variant]["selected_width"], "seed_count": len(rows), "trials": rows,
                            "aggregate": {m: float(np.mean([r[m] for r in rows])) for m in keys},
                            "std": {m: float(np.std([r[m] for r in rows], ddof=1)) for m in keys}}
    results["paired_content_minus_mlp"] = [{"seed": a["seed"], **{m: b[m] - a[m] for m in results["mf"]["aggregate"]}}
        for a, b in zip(results["mlp"]["trials"], results["mlp-content"]["trials"])]
    write_json(output / "results.json", results)
    frozen.update(status="complete", test_evaluated=True, candidate_alignment_verified=True,
                  mask_policy="exclude training and validation history; no retraining",
                  users=len(users), items=len(items), test_interactions=sum(map(len, testing.values())),
                  test_sha256=file_hash(source / "splits/test.csv"),
                  mf_scores_sha256=file_hash(run_directory / "scores-valid-mf.npz"))
    if file_hash(run_directory / "results.json") != frozen["selection_sha256"]:
        raise ValueError("Validation selection changed during testing")
    write_json(output / "manifest.json", frozen)
    lines = ["# Frozen feature-combination test comparison", "",
             "The exact validation-selected checkpoints and widths were frozen before test relevance was loaded. No retraining or test-based selection. Test candidates exclude training and validation history, matching the established component evaluation. Frozen checkpoint predictions reproduce the saved dense scores.", "",
             "MLPs report mean +/- sample standard deviation over three scorer seeds, conditional on one shared pretrained MF checkpoint. All test interactions are relevant; external item-ID ties; K=10.", "",
             "| Model | Recall@10 | MRR@10 | NDCG@10 |", "|---|---:|---:|---:|"]
    for variant in ("mf", "mlp", "mlp-content"):
        row = results[variant]
        values = [f"{row['aggregate'][m]:.6f}" + (f" +/- {row['std'][m]:.6f}" if "std" in row else "")
                  for m in ("Recall@10", "MRR@10", "NDCG@10")]
        lines.append("| " + variant + " | " + " | ".join(values) + " |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({v: results[v]["aggregate"] for v in ("mf", "mlp", "mlp-content")}, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="project/artifacts/feature-combination-mf-content")
    parser.add_argument("--source", default="project/artifacts/hybrid-60-20-20")
    args = parser.parse_args()
    print(run(args.run, args.source))
