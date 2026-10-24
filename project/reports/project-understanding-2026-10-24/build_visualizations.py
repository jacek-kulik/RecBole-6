"""Package existing reranking figures and visualize recorded project evidence.

Run with system Python containing NumPy and Matplotlib. No training, tuning,
checkpoint loading, or new evaluation is performed.
"""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parents[2]
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GOLD = "#E6AB02"


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_records(paths):
    records = []
    for path in paths:
        try:
            name = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            name = str(path.resolve())
        records.append({"path": name, "sha256": sha256(path)})
    return records


def save_figure(figure, output, name, sources, data):
    svg, png = output / f"{name}.svg", output / f"{name}.png"
    figure.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    figure.savefig(png, dpi=160)
    plt.close(figure)
    return {
        "kind": "new visualization of recorded results",
        "sources": source_records(sources),
        "plotted_data": data,
        "files": {path.name: sha256(path) for path in (svg, png)},
    }


def popularity_figure(bundle, output):
    training = bundle / "training/itemknn-rerankers-20261022-001/splits/train.csv"
    definitions = bundle / "evaluation/itemknn/test/original/groups.json"
    groups = read_json(definitions)["item_popularity"]["members"]
    counts = dict.fromkeys(groups, 0)
    for row in read_csv(training):
        counts[row["item_id"]] += 1
    ordered = sorted(counts, key=lambda item: (-counts[item], item))
    n_items, total = len(ordered), sum(counts.values())
    item_counts = Counter(groups.values())
    interaction_counts = {
        group: sum(counts[item] for item in ordered if groups[item] == group)
        for group in ("head", "middle", "tail")
    }
    cumulative = np.cumsum([counts[item] for item in ordered]) / total * 100
    ends = [item_counts["head"], item_counts["head"] + item_counts["middle"], n_items]
    if n_items != 1682 or total != 60734 or list(item_counts[g] for g in ("head", "middle", "tail")) != [60, 485, 1137]:
        raise ValueError("The popularity inputs differ from this report's recorded run")
    tied_groups = {}
    for item in ordered:
        tied_groups.setdefault(counts[item], set()).add(groups[item])
    if any(len(members) != 1 for members in tied_groups.values()):
        raise ValueError("A training-count tie crosses popularity groups")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    colors = (ORANGE, BLUE, GREEN)
    start = 0
    for group, end, color in zip(("head", "middle", "tail"), ends, colors):
        axes[0].axvspan(100 * start / n_items, 100 * end / n_items, color=color, alpha=.12)
        start = end
    axes[0].plot(np.arange(1, n_items + 1) / n_items * 100, cumulative, color="#263B43", lw=2)
    for level in (20, 80):
        axes[0].axhline(level, color="#777777", ls="--", lw=.8)
    for end, color in zip(ends[:2], colors):
        axes[0].scatter(100 * end / n_items, cumulative[end - 1], color=color, zorder=3)
        axes[0].annotate(f"{end} items", (100 * end / n_items, cumulative[end - 1]),
                         xytext=(10, -18), textcoords="offset points", fontsize=9)
    axes[0].set(xlabel="Catalogue share, sorted by training count (%)",
                ylabel="Cumulative training interactions (%)", xlim=(0, 100), ylim=(0, 102),
                title="A small catalogue share accounts for much of the history")
    x = np.arange(3)
    item_share = [100 * item_counts[g] / n_items for g in ("head", "middle", "tail")]
    interaction_share = [100 * interaction_counts[g] / total for g in ("head", "middle", "tail")]
    for bars in (axes[1].bar(x - .18, item_share, .35, label="Share of catalogue items", color="#565B65"),
                 axes[1].bar(x + .18, interaction_share, .35, label="Share of training interactions", color=BLUE)):
        axes[1].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    axes[1].set(xticks=x, xticklabels=[f"{g.title()}\n{item_counts[g]} items" for g in ("head", "middle", "tail")],
                ylabel="Share (%)", ylim=(0, 80), title="Interaction-share bands are not equal-size item groups")
    axes[1].legend(loc="upper left", fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Training-only item popularity: 60,734 interactions; 1,682 catalogue movies", fontsize=13)
    return save_figure(fig, output, "training-popularity", [training, definitions], {
        "catalogue_items": n_items, "training_interactions": total,
        "groups": {g: {"items": item_counts[g], "interactions": interaction_counts[g],
                        "item_share_percent": item_share[i], "interaction_share_percent": interaction_share[i]}
                   for i, g in enumerate(("head", "middle", "tail"))},
    })


def tuning_figure(results, output):
    root = results / "individual-retuned-60-20-20"
    ease_paths = sorted((root / "EASE").glob("trial-*.json"))
    slim_paths = sorted((root / "SLIMElastic").glob("trial-*.json"))
    ease = [read_json(path) for path in ease_paths]
    slim = [read_json(path) for path in slim_paths]
    if len(ease) != 9 or len(slim) != 25 or any(row["status"] != "complete" or row["convergence_warnings"] for row in ease + slim):
        raise ValueError("Expected nine complete EASE and 25 converged SLIM trials")
    selections = [root / model / "selection.json" for model in ("EASE", "SLIMElastic")]
    selected_ease, selected_slim = [read_json(path) for path in selections]
    if any(record["selection_metric"] != "validation NDCG@10" for record in (selected_ease, selected_slim)):
        raise ValueError("The selected trials must use validation NDCG")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), layout="constrained")
    axes[0].plot([r["parameters"]["reg_weight"] for r in ease],
                 [r["validation"]["NDCG@10"] for r in ease], "o-", color=BLUE)
    for key, marker, color, label in (("baseline_same_solver", "s", "#555555", "Previous setting"),
                                       ("best", "*", GOLD, "NDCG-selected setting")):
        record = selected_ease[key]
        axes[0].scatter(record["parameters"]["reg_weight"], record["validation"]["NDCG@10"],
                        marker=marker, s=150 if marker == "*" else 65, color=color,
                        edgecolors="black", zorder=3, label=label)
    axes[0].set(xscale="log", xlabel="EASE regularization (log scale)",
                ylabel="Validation NDCG@10", title="EASE: the gain over the previous setting is small", ylim=(0, .33))
    axes[0].grid(alpha=.15)
    axes[0].legend(fontsize=8, loc="lower right")
    alphas = sorted({r["parameters"]["alpha"] for r in slim})
    ratios = sorted({r["parameters"]["l1_ratio"] for r in slim})
    grid = np.full((len(alphas), len(ratios)), np.nan)
    for row in slim:
        grid[alphas.index(row["parameters"]["alpha"]), ratios.index(row["parameters"]["l1_ratio"])] = row["validation"]["NDCG@10"]
    heat = axes[1].imshow(grid, cmap="viridis", vmin=float(grid.min()), vmax=float(grid.max()))
    for i, j in np.ndindex(grid.shape):
        axes[1].text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", fontsize=9,
                     color="white" if grid[i,j] < (grid.min() + grid.max()) / 2 else "black")
    p = selected_slim["parameters"]
    axes[1].scatter(ratios.index(p["l1_ratio"]), alphas.index(p["alpha"]), marker="s", s=850,
                    facecolors="none", edgecolors="#E69F00", linewidths=2.5, label="Selected cell")
    axes[1].set(xticks=range(len(ratios)), xticklabels=ratios, yticks=range(len(alphas)),
                yticklabels=alphas, xlabel="SLIM L1 ratio", ylabel="SLIM alpha", title="SLIM: 25 declared, converged configurations")
    fig.colorbar(heat, ax=axes[1], shrink=.85, label="Validation NDCG@10")
    fig.suptitle("Individual-model retuning: select on validation, not on test", fontsize=13)
    return save_figure(fig, output, "individual-tuning", ease_paths + slim_paths + selections, {
        "ease": [{"regularization": r["parameters"]["reg_weight"], "ndcg": r["validation"]["NDCG@10"]} for r in ease],
        "slim": [{**r["parameters"], "ndcg": r["validation"]["NDCG@10"]} for r in slim],
        "selected": {"ease": selected_ease["parameters"], "slim": selected_slim["parameters"]},
    })


def hybrid_figure(results, output):
    weighted_path = results / "oof-ridge-final-60-20-20/aggregate-metrics.json"
    alternate_path = results / "alternate-hybrids-60-20-20/metrics.csv"
    weighted = read_json(weighted_path)
    alternate = {(r["partition"], r["scope"], r["model"]): float(r["NDCG@10"]) for r in read_csv(alternate_path)}
    left = [(name, name.upper() if name != "regressor" else "OOF Ridge")
            for name in ("itemknn", "userknn", "bpr", "slimelastic", "ease", "fism", "regressor")]
    right = [("bpr", "BPR"), ("ease", "EASE"), ("slim", "Native SLIM"),
             ("paired-slim", "Paired SLIM"), ("aug-slim-weak", "UserKNN → SLIM, 0.25"),
             ("aug-slim-comparable", "UserKNN → SLIM, 1.0"), ("aug-ease-weak", "UserKNN → EASE, 0.25"),
             ("cascade", "Cascade"), ("meta", "Meta-level"), ("meta-control", "Meta control"),
             ("mlp-mlp", "Feature combination, CF*"), ("residual-mlp", "Residual combination, CF*")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), layout="constrained")
    plotted = {}
    for ax, entries, family, title in zip(axes, (left, right), ("weighted", "alternate"),
                                          ("OOF experiment: its own six retained components", "Alternate experiment: its own corrected fits")):
        y = np.arange(len(entries))
        for part, offset, color, label in (("valid", -.12, BLUE, "Validation"), ("test", .12, ORANGE, "Test")):
            values = [weighted[part][name]["NDCG@10"] if family == "weighted" else alternate[part, "full", name]
                      for name, _ in entries]
            ax.scatter(values, y + offset, color=color, s=45, label=label)
            plotted[f"{family}/{part}"] = {name: value for (name, _), value in zip(entries, values)}
        ax.set(yticks=y, yticklabels=[label for _, label in entries], xlim=(0, .45), xlabel="NDCG@10", title=title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
        ax.legend(loc="lower left", fontsize=9)
    fig.suptitle("Frozen full-catalogue comparisons — keep each experiment's component controls", fontsize=14)
    fig.supxlabel("* Means over three scorer seeds conditional on one BPR fit; no uncertainty bars shown. Warm switching is excluded.", fontsize=9)
    return save_figure(fig, output, "hybrid-comparisons", [weighted_path, alternate_path], plotted)


def ablation_figure(results, output):
    fit_path = results / "oof-ridge-final-60-20-20/hybrid.json"
    rows_path = results / "task-2.3-coefficient-analysis/weighted.csv"
    fitted = read_json(fit_path)
    rows = {r["variant"]: r for r in read_csv(rows_path) if r["partition"] == "test"}
    names = fitted["components"]
    if any(float(row["alpha"]) != fitted["alpha"] for row in rows.values()):
        raise ValueError("The ablation and coefficient fit use different Ridge penalties")
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    axes[0].barh(y, fitted["coefficients"], color=BLUE, height=.55)
    for i, coefficient in enumerate(fitted["coefficients"]):
        axes[0].text(coefficient + .0008, i, f"{coefficient:.4f}", va="center", fontsize=9)
    axes[0].set(yticks=y, yticklabels=[n.upper() for n in names], xlim=(0, .06), xlabel="Coefficient on per-user z-scored scores",
                title="Fitted raw-rating regression weights")
    data = {"coefficients": dict(zip(names, fitted["coefficients"])), "removals": {}}
    for method, prefix, color, offset in (("Frozen removal", "zero-", ORANGE, -.12),
                                           ("OOF refit, fixed alpha", "refit-", GREEN, .12)):
        records = [rows[prefix + name] for name in names]
        values = np.array([float(r["delta_NDCG@10"]) for r in records])
        low = np.array([float(r["delta_NDCG_ci_low"]) for r in records])
        high = np.array([float(r["delta_NDCG_ci_high"]) for r in records])
        if np.any(values < low) or np.any(values > high):
            raise ValueError("A recorded bootstrap interval excludes its point estimate")
        axes[1].errorbar(values, y + offset, xerr=[values - low, high - values], fmt="o", color=color,
                         capsize=3, ms=5, label=method)
        data["removals"][method] = {name: {"delta_ndcg": float(value), "ci_low": float(lo), "ci_high": float(hi)}
                                      for name, value, lo, hi in zip(names, values, low, high)}
    axes[1].axvline(0, color="#555555", lw=.9)
    axes[1].set(yticks=y, yticklabels=[n.upper() for n in names], xlabel="Test NDCG change after removing the component",
                title="Positive means removal improved this fitted system")
    axes[1].legend(loc="lower right", fontsize=8)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("Coefficient magnitude and ranking contribution answer different questions", fontsize=13)
    fig.supxlabel("Error bars: recorded 95% paired user-bootstrap intervals, conditional on the same split and trained fits.", fontsize=9)
    return save_figure(fig, output, "coefficient-ablation", [fit_path, rows_path], data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reranking-bundle", type=Path, default=REPO_ROOT / "project/artifacts/reranking-20261022-001")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "hyperparameters/results")
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "recbole-study-20261024"})
    figures = {}
    for name in ("validation-tradeoffs", "test-effects"):
        source = args.reranking_bundle / "evaluation/figures" / f"{name}.png"
        destination = args.output / source.name
        shutil.copyfile(source, destination)
        figures[name] = {"kind": "existing project figure copied without alteration",
                         "sources": source_records([source]), "files": {destination.name: sha256(destination)},
                         "source_code": "PR24 at 580bfaa63: project/scripts/plot_reranking.py"}
    for name, generate, inputs in (("training-popularity", popularity_figure, args.reranking_bundle),
                                    ("individual-tuning", tuning_figure, args.results_root),
                                    ("hybrid-comparisons", hybrid_figure, args.results_root),
                                    ("coefficient-ablation", ablation_figure, args.results_root)):
        figures[name] = generate(inputs, args.output)
    manifest = {"kind": "report figure provenance", "report_source_snapshot": "54e60bffb",
                "generator_sha256": sha256(Path(__file__)),
                "training_or_selection_performed": False, "figures": figures}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(figures)} figures in {args.output}")


if __name__ == "__main__":
    main()
