"""Plot saved six-model results. Requires matplotlib; does not load checkpoints."""

import argparse
import json
from pathlib import Path

METHODS = {
    "genre_calibration": ("Genre calibration", "GenreMiscalibration@10", "#0072B2"),
    "popularity_calibration": ("Popularity calibration", "UPD@10", "#D55E00"),
    "item_fairness": ("Item exposure", "ExposureGini@10", "#009E73"),
}


def plot(run: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    models = json.loads((run / "manifest.json").read_text())["models"]
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    validation, axes = plt.subplots(3, 2, figsize=(12, 12), layout="constrained")
    test, test_axes = plt.subplots(3, 2, figsize=(12, 12), layout="constrained")
    for model, ax, test_ax in zip(models, axes.flat, test_axes.flat):
        rows = json.loads((run / model / "valid/summary.json").read_text())
        baseline = rows[0]
        selection = json.loads((run / model / "valid/selection.json").read_text())
        tested = json.loads((run / model / "test/summary.json").read_text())
        original_test = tested[0]
        for index, (method, (label, metric, color)) in enumerate(METHODS.items()):
            points = [row for row in rows if row["method"] == method]
            if not points or not baseline[metric]:
                continue
            x = [100 * (row["ndcg"] / baseline["ndcg"] - 1) for row in points]
            y = [100 * (1 - row[metric] / baseline[metric]) for row in points]
            ax.plot(x, y, "o-", color=color, markersize=4, linewidth=1.2)
            if selection["methods"][method]["status"] != "complete":
                continue
            strength = selection["methods"][method]["selected"]["strength"]
            chosen = next(i for i, row in enumerate(points) if row["strength"] == strength)
            ax.scatter(x[chosen], y[chosen], marker="*", s=170, color=color, edgecolors="black", linewidths=.6, zorder=4)
            row = next((row for row in tested if row["method"] == method), None)
            if row is None:
                test_ax.text(index, .04, "Unavailable", transform=test_ax.get_xaxis_transform(),
                             ha="center", fontsize=8, color="#777777")
                continue
            ndcg_change = 100 * (row["ndcg"] / original_test["ndcg"] - 1)
            gain = 100 * (1 - row[metric] / original_test[metric])
            test_ax.bar(index - .18, ndcg_change, width=.35, color="#565B65")
            test_ax.bar(index + .18, gain, width=.35, color=color)
            test_ax.text(index, max(ndcg_change, gain, 0) + 1.2, f"strength {strength:g}", ha="center", fontsize=8)
        ax.axvline(-5, color="#777777", linestyle="--", linewidth=1)
        ax.axhline(0, color="#aaaaaa", linewidth=.7)
        ax.set(title=model.upper(), xlabel="Validation NDCG change (%)", ylabel="Target metric reduction (%)")
        ax.grid(alpha=.15)
        test_ax.axhline(-5, color="#777777", linestyle="--", linewidth=1)
        test_ax.axhline(0, color="#aaaaaa", linewidth=.7)
        test_ax.set(title=model.upper(), ylabel="Change relative to original (%)", xlim=(-.5, 2.5),
                    xticks=range(3), xticklabels=("Genre", "Popularity", "Exposure"))
        test_ax.margins(y=.2)
        test_ax.grid(axis="y", alpha=.15)
    validation.suptitle("Validation trade-offs: top 50 to top 10\nStars: selected under the 5% relative NDCG-loss budget", fontsize=14)
    handles = [Line2D([0], [0], color=color, marker="o", label=label) for label, _, color in METHODS.values()]
    axes[0, 0].legend(handles=handles, fontsize=8, loc="best")
    test.suptitle("Test effects of frozen validation choices\nGrey: NDCG change; colour: target metric reduction", fontsize=14)
    for figure, name in ((validation, "validation-tradeoffs"), (test, "test-effects")):
        figure.savefig(output / f"{name}.png", dpi=150)
        figure.savefig(output / f"{name}.pdf")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.run, args.output)


if __name__ == "__main__":
    main()
