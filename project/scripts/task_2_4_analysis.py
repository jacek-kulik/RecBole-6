"""Run the Task 2.4 explanatory analysis over the test-partition runs.

Builds the two focused signals - per-user sparsity vs accuracy and model
disagreement - plus the Task 2.5 group facts they must stay consistent with,
writes the evidence JSON, and renders the two figures.

Usage:
    .venv/bin/python project/scripts/task_2_4_analysis.py

Requires matplotlib:
    .venv/bin/python -m pip install matplotlib
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from project.recsys.analysis.explanations import analyze
MODELS = ("bpr", "ease", "fism", "itemknn", "slimelastic", "userknn")
MODEL_COLORS = {"bpr": "#c7857f", "ease": "#7f9fc7", "fism": "#86a87c",
                "itemknn": "#c7a97f", "slimelastic": "#a587c7", "userknn": "#c7c785"}


def _run_number(directory: Path, model: str) -> bool:
    # Training runs are "<model>-001"-style; case-insensitive for ITEMKNN folders.
    import re
    return bool(re.fullmatch(r"\d+", directory.name.lower()[len(model) + 1:])) \
        and directory.name.lower().startswith(f"{model}-")


def _test_suffix(directory: Path, model: str) -> bool:
    import re
    return directory.name.lower().startswith(f"{model}-test-") \
        and bool(re.fullmatch(r"\d+", directory.name.lower()[len(model) + 6:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metric", default="NDCG@10", help="Per-user metric to explain")
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "project" / "artifacts")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "project" / "artifacts" / "explanations-001")
    args = parser.parse_args()

    records = {}
    for model in MODELS:
        # Latest matching runs win, so retrained models (e.g. bpr-002) are used.
        train_runs = [d for d in args.artifacts.iterdir()
                      if d.name.lower() == f"{model}-001" or _run_number(d, model)]
        train_runs = [d for d in train_runs if (d / "splits" / "train.csv").is_file()]
        test_runs = [d for d in args.artifacts.iterdir()
                     if _test_suffix(d, model)]
        test_runs = [d for d in test_runs if (d / "group_metrics.json").is_file()]
        if train_runs and test_runs:
            records[model] = {"train": sorted(train_runs)[-1],
                              "evaluation": sorted(test_runs)[-1]}
    if not records:
        raise SystemExit(f"No test evaluations found in {args.artifacts}")
    evidence = analyze(records, args.metric)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")

    figures = REPO_ROOT / "project" / "figures"
    figures.mkdir(exist_ok=True)

    # Figure 1: mean accuracy per history-length quintile, one line per model.
    figure, axis = plt.subplots(figsize=(6.5, 4))
    for model, curve in sorted(evidence["sparsity"].items()):
        xs = [b["mean_history"] for b in curve["bins"]]
        ys = [b["mean_accuracy"] for b in curve["bins"]]
        axis.plot(xs, ys, marker="o", label=f"{model} (r={curve['pearson_correlation']:.2f})",
                  color=MODEL_COLORS.get(model))
    axis.set_xlabel("mean training history length per bin (quintiles)")
    axis.set_ylabel(f"mean {args.metric}")
    axis.set_title("Accuracy grows with user profile length in every model")
    axis.grid(True, axis="y", alpha=.3)
    axis.legend(loc="upper left", fontsize=8, frameon=True, framealpha=.9)
    figure.tight_layout()
    figure.savefig(figures / "sparsity_vs_accuracy.png", dpi=200)

    # Figure 2: pairwise mean Jaccard distance of the top-k lists.
    models = sorted(records)
    matrix = [[0.0] * len(models) for _ in models]
    for (pair, stats) in evidence["disagreement"].items():
        left, right = pair.split("|")
        value = stats["mean_jaccard_distance"]
        matrix[models.index(left)][models.index(right)] = value
        matrix[models.index(right)][models.index(left)] = value
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(models)), models, rotation=45, fontsize=9)
    axis.set_yticks(range(len(models)), models, fontsize=9)
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                axis.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                          color="white" if matrix[i][j] > .5 else "#333333", fontsize=8)
    figure.colorbar(image, label="mean Jaccard distance of top-10 lists")
    axis.set_title("Model disagreement (higher = more complementary candidates)")
    figure.tight_layout()
    figure.savefig(figures / "model_disagreement.png", dpi=200)

    print(f"Wrote {args.output / 'results.json'} ({len(records)} models)")
    print(f"Wrote {figures / 'sparsity_vs_accuracy.png'}")
    print(f"Wrote {figures / 'model_disagreement.png'}")
    for model, curve in sorted(evidence["sparsity"].items()):
        first, last = curve["bins"][0], curve["bins"][-1]
        print(f"  {model:12s} r={curve['pearson_correlation']:+.3f}  "
              f"lowest-bin {first['mean_accuracy']:.3f} -> highest-bin {last['mean_accuracy']:.3f}")
    for pair, stats in sorted(evidence["disagreement"].items()):
        print(f"  {pair:24s} distance={stats['mean_jaccard_distance']:.3f}  "
              f"exclusive-item precision {stats['a_only_precision']:.3f} / {stats['b_only_precision']:.3f}")


if __name__ == "__main__":
    main()
