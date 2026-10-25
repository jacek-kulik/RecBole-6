"""Plot the three diversification objectives across saved individual-model runs."""

import argparse
import json
from pathlib import Path

VARIANTS = {
    "diversification": ("Genre diversity", "ILD", "#CC79A7"),
    "popularity_diversification": ("Long-tail discovery", "MeanSelfInformation", "#E69F00"),
    "coverage_diversification": ("Catalogue coverage", "CatalogueCoverage", "#56B4E9"),
}


def read(path):
    return json.loads(path.read_text())


def change(value, baseline):
    """Percent change, or absolute metric units when the baseline is zero."""
    return 100 * (value - baseline) / abs(baseline) if baseline else value - baseline


def plot(run: Path, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = read(run / "manifest.json")
    if root["status"] != "complete" or not root["models"]:
        raise ValueError("Use a completed individual-model reranking run")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    validation, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
    tested, test_axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
    exported = []
    try:
        for (method, (title, prefix, color)), ax, test_ax in zip(VARIANTS.items(), axes, test_axes):
            metric = f"{prefix}@{root['k']}"
            labels = []
            for index, model in enumerate(root["models"]):
                folder = run / model
                valid = read(folder / "valid/summary.json")
                test = read(folder / "test/summary.json")
                selection = read(folder / "valid/selection.json")["methods"][method]
                labels.append(model.upper())
                if selection["status"] != "complete":
                    test_ax.text(index, .04, "Unavailable", transform=test_ax.get_xaxis_transform(),
                                 rotation=90, ha="center", fontsize=8)
                    continue
                if (selection["target_metric"], selection["direction"]) != (metric, "maximize"):
                    raise ValueError(f"Unexpected target for {model}/{method}")
                points = [row for row in valid if row["method"] == method]
                baseline, original_test = valid[0], test[0]
                selected = selection["selected"]["strength"]
                x = [-change(row["ndcg"], baseline["ndcg"]) for row in points]
                y = [change(row[metric], baseline[metric]) for row in points]
                model_color = plt.get_cmap("tab10")(index % 10)
                label = model.upper() + (" (abs)" if not baseline[metric] else "")
                ax.plot(x, y, "o-", color=model_color, label=label, linewidth=1.2, markersize=4)
                chosen = next(n for n, row in enumerate(points) if row["strength"] == selected)
                ax.scatter(x[chosen], y[chosen], color=model_color, marker="*", s=160,
                           edgecolors="black", linewidths=.6, zorder=5)
                result = next((row for row in test if row["method"] == method), None)
                if result is None:
                    test_ax.text(index, .04, "Unavailable", transform=test_ax.get_xaxis_transform(),
                                 rotation=90, ha="center", fontsize=8)
                    continue
                if result["strength"] != selected:
                    raise ValueError(f"Test changed a frozen strength for {model}/{method}")
                ndcg = change(result["ndcg"], original_test["ndcg"])
                gain = change(result[metric], original_test[metric])
                test_ax.bar(index - .18, ndcg, width=.35, color="#565B65", label="NDCG" if index == 0 else None,
                            edgecolor="#B2182B" if ndcg < -5 else "#565B65", linewidth=1.2)
                test_ax.bar(index + .18, gain, width=.35, color=color, label="Target" if index == 0 else None)
                labels[-1] += f"\ns={selected:g}" + (" (abs)" if not original_test[metric] else "")
                exported.append({"model": model, "method": method, "metric": metric,
                                 "selected_strength": selected, "test_ndcg_change": ndcg,
                                 "test_target_gain": gain, "test_target_absolute": not bool(original_test[metric]),
                                 "validation_points": [{"strength": row["strength"], "ndcg_loss": xx,
                                                        "target_gain": yy} for row, xx, yy in zip(points, x, y)]})
            ax.axvline(5, color="#777777", linestyle="--", linewidth=1)
            ax.axhline(0, color="#aaaaaa", linewidth=.7)
            ax.set(title=title, xlabel="Validation NDCG loss (%)", ylabel="Target gain (%; abs if baseline=0)")
            ax.grid(alpha=.15)
            ax.legend(fontsize=8)
            test_ax.axhline(-5, color="#777777", linestyle="--", linewidth=1)
            test_ax.axhline(0, color="#aaaaaa", linewidth=.7)
            test_ax.set(title=title, ylabel="Test change (%; abs if baseline=0)",
                        xticks=range(len(labels)), xticklabels=labels)
            test_ax.tick_params(axis="x", labelrotation=35, labelsize=8)
            test_ax.grid(axis="y", alpha=.15)
            test_ax.legend(fontsize=8)
        validation.suptitle("Diversification across individual models\nStars: highest target within 5% validation NDCG loss", fontsize=13)
        tested.suptitle("Test effects at frozen validation strengths\nGrey: NDCG change; colour: relative target gain; red outline: test loss >5%", fontsize=13)
        for figure, name in ((validation, "validation-tradeoffs"), (tested, "test-effects")):
            figure.savefig(output / f"{name}.png", dpi=160)
            figure.savefig(output / f"{name}.pdf")
        (output / "figure-data.json").write_text(json.dumps(exported, indent=2) + "\n", encoding="utf-8")
    finally:
        plt.close(validation)
        plt.close(tested)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.run, args.output)


if __name__ == "__main__":
    main()
