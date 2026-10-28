"""Reproduce Task 2.2 with fresh fits and verify both corrected baselines."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from project.recsys.artifacts import REPO_ROOT, new_run, provenance, write_json
from project.recsys.comparisons import compare_runs
from project.recsys.evaluation import run_evaluation
from project.recsys.recbole_adapter import _load_checkpoint, train

MODELS = (
    ("Random", "random"),
    ("Popularity", "pop"),
    ("BPR", "bpr"),
    ("EASE", "ease"),
    ("SLIM", "slimelastic"),
    ("ItemKNN", "itemknn"),
    ("UserKNN", "userknn"),
    ("FISM", "fism"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def reproduce(output: Path) -> Path:
    output = new_run(output)
    started = time.monotonic()
    evaluations = []
    runs = {}
    split_ids = set()
    for index, (label, name) in enumerate(MODELS, 1):
        print(f"[{index}/{len(MODELS)}] Training {label}", flush=True)
        config = REPO_ROOT / f"project/configs/models/{name}.yaml"
        run = train(config, output / "training" / name)
        manifest = read_json(run / "manifest.json")
        split_ids.add(manifest["split_id"])
        print(
            f"[{index}/{len(MODELS)}] Evaluating {label} on validation and test",
            flush=True,
        )
        for partition in ("valid", "test"):
            run_evaluation(run, output / partition / name, partition)
        evaluation = output / "test" / name
        metrics = read_json(evaluation / "metrics.json")
        evaluations.append(f"{label}={evaluation}")
        runs[label] = {
            "config_sha256": digest(config),
            "checkpoint_sha256": digest(run / manifest["checkpoint"]),
            "test_metrics_sha256": digest(evaluation / "metrics.json"),
            "test_rankings_sha256": digest(evaluation / "rankings.json"),
            "test_metrics": metrics,
        }
        elapsed = time.monotonic() - started
        eta = elapsed / index * (len(MODELS) - index)
        print(
            f"[{index}/{len(MODELS)}] {label}: NDCG@10={metrics['NDCG@10']:.6f}; "
            f"elapsed={elapsed:.0f}s, remaining approximately {eta:.0f}s",
            flush=True,
        )
    if len(split_ids) != 1:
        raise ValueError("Models reconstructed different splits")

    verification = {}
    for label, name in MODELS[:2]:
        print(f"Replaying {label} with batch size 128 instead of 1024", flush=True)
        run = output / "training" / name
        replay = run_evaluation(run, output / "replay" / name, "test", batch_size=128)
        original = output / "test" / name
        for filename in (
            "rankings.json",
            "relevant.json",
            "metrics.json",
            "per_user_metrics.json",
            "beyond_accuracy.json",
        ):
            if (replay / filename).read_bytes() != (original / filename).read_bytes():
                raise ValueError(f"{label}: batch size changed {filename}")
        rankings = read_json(original / "rankings.json")
        unique = len({tuple(items) for items in rankings.values()})
        verification[label] = {
            "batch_sizes": [1024, 128],
            "exact_replay": True,
            "users": len(rankings),
            "distinct_rankings": unique,
            "distinct_recommended_items": len(
                {i for row in rankings.values() for i in row}
            ),
        }
        if name == "random" and unique != len(rankings):
            raise ValueError("Random baseline reused user rankings")
        if name == "pop":
            source = read_json(run / "manifest.json")
            _, model, dataset, training, _, _ = _load_checkpoint(
                run / source["checkpoint"]
            )
            expected = torch.bincount(
                training.dataset.inter_feat[dataset.iid_field],
                minlength=dataset.item_num,
            )
            if not torch.equal(model.item_cnt.flatten(), expected):
                raise ValueError(
                    "Popularity checkpoint differs from training frequencies"
                )
            verification[label]["training_counts_verified"] = True
            verification[label]["training_interactions"] = int(expected.sum())

    comparison = compare_runs(evaluations, output / "comparison")
    source_files = [
        "project/configs/protocol.yaml",
        "project/scripts/reproduce_baseline_comparison.py",
        "recbole/model/general_recommender/pop.py",
        "recbole/model/general_recommender/random.py",
        "dataset/ml-100k/ml-100k.inter",
        "dataset/ml-100k/ml-100k.item",
        "dataset/ml-100k/ml-100k.user",
    ]
    write_json(
        output / "verification.json",
        {
            **provenance(),
            "status": "passed",
            "split_id": next(iter(split_ids)),
            "source_sha256": {path: digest(REPO_ROOT / path) for path in source_files},
            "fresh_fits": len(MODELS),
            "baseline_verification": verification,
            "runs": runs,
            "comparison_sha256": {
                name: digest(comparison / name)
                for name in ("comparison.csv", "comparison.json", "per_user.csv")
            },
            "elapsed_seconds": round(time.monotonic() - started, 2),
        },
    )
    print(f"Verified comparison: {comparison}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New artifact directory"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    reproduce(args.output)
