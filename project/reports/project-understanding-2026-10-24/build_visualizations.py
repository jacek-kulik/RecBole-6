"""Package existing reranking figures and visualize recorded project evidence.

Run with system Python containing NumPy and Matplotlib. No training, tuning,
checkpoint loading, or new evaluation is performed.
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
    for key, marker, color, label in (("baseline_same_solver", "s", "#555555", "Reference setting (250)"),
                                       ("best", "*", GOLD, "NDCG-selected setting")):
        record = selected_ease[key]
        axes[0].scatter(record["parameters"]["reg_weight"], record["validation"]["NDCG@10"],
                        marker=marker, s=150 if marker == "*" else 65, color=color,
                        edgecolors="black", zorder=3, label=label)
    axes[0].set(xscale="log", xlabel="EASE regularization (log scale)",
                ylabel="Validation NDCG@10", title="EASE: nearby strong settings differ little", ylim=(0, .33))
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


def diagram_canvas(title, subtitle, height=5.4):
    figure, axis = plt.subplots(figsize=(12, height), layout="constrained")
    axis.set(xlim=(0, 12), ylim=(0, 6))
    axis.axis("off")
    figure.suptitle(title, fontsize=15, weight="bold")
    figure.supxlabel(subtitle, fontsize=10, color="#475569")
    return figure, axis


def box(axis, x, y, width, height, text, color=BLUE, fontsize=11):
    axis.add_patch(FancyBboxPatch((x, y), width, height,
                                 boxstyle="round,pad=0.03,rounding_size=0.12",
                                 facecolor=color + "12", edgecolor=color, linewidth=1.3))
    axis.text(x + width / 2, y + height / 2, text,
              ha="center", va="center", fontsize=fontsize, linespacing=1.5, color="#172A3A")


def arrow(axis, start, end, color="#64748B"):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                  mutation_scale=16, linewidth=1.4, color=color))


def conceptual_record(figure, output, name, sources, data):
    record = save_figure(figure, output, name, sources, data)
    record["kind"] = "illustrative explanation of implemented behavior; not an experimental measurement"
    return record


def evaluation_flow(output):
    figure, axis = diagram_canvas(
        "Which information may influence each decision?",
        "Conceptual map. Validation history joins the test mask; it does not refit the recorded training model.",
        height=6.2)
    rows = [(4.35, GREEN, "TRAIN", ["Training interactions\n+ static metadata", "Fit model parameters\nBuild profiles / counts", "Model scores\n+ training-only context"]),
            (2.45, BLUE, "VALIDATE", ["Held-out validation\nlabels + predictions", "Measure NDCG\nand method target", "Choose checkpoint\nand operating strength"]),
            (.55, ORANGE, "TEST", ["Frozen model, context\nand selected settings", "Mask train + valid\nApply frozen pipeline", "Test labels measure\nthe final saved lists"])]
    for y, color, label, texts in rows:
        axis.text(.1, y + .58, label, ha="left", va="center", color=color, weight="bold", fontsize=10)
        for x, text in zip((1.45, 5.0, 8.55), texts):
            box(axis, x, y, 3.15, 1.2, text, color)
        arrow(axis, (4.65, y + .6), (4.95, y + .6), color)
        arrow(axis, (8.2, y + .6), (8.5, y + .6), color)
    axis.text(6.55, 2.08, "FREEZE the selected checkpoint, settings and input identity before final test",
              ha="center", fontsize=10, weight="bold", color=ORANGE)
    return conceptual_record(figure, output, "evaluation-flow",
                             [REPO_ROOT / "project/recsys/rerank.py", REPO_ROOT / "project/recsys/recbole_adapter.py"],
                             {"training_context_uses": "training interactions and static metadata",
                              "selection_uses": "validation outcomes", "final_measurement_uses": "test outcomes",
                              "test_history_mask": "training plus validation", "test_reselection": False})


def calibration_step(output):
    profile = np.array([.75, .25])
    appended = [np.array([1., 0.]), np.array([.5, .5])]
    smoothed = [.99 * distribution + .01 * profile for distribution in appended]
    divergences = [float(np.sum(profile * np.log(profile / distribution))) for distribution in smoothed]
    objectives = [.5 * relevance - .5 * divergence for relevance, divergence in zip((1., .8), divergences)]
    figure, axes = plt.subplots(1, 3, figsize=(12, 5), layout="constrained")
    titles = ["User's training profile", "Append A-only candidate x", "Append B-only candidate y"]
    for axis, values, title in zip(axes, [profile] + appended, titles):
        bars = axis.bar([0, 1], values, color=[BLUE, ORANGE], width=.55)
        axis.bar_label(bars, labels=[f"{value:.0%}" for value in values], padding=4)
        axis.set(xticks=[0, 1], xticklabels=["Genre A", "Genre B"], ylim=(0, 1.15),
                 ylabel="Fractional genre mass", title=title)
        axis.grid(axis="y", alpha=.15)
    axes[0].set_xlabel("Growing list already contains one A-only item", fontsize=9)
    for i, axis in enumerate(axes[1:]):
        axis.set_xlabel(f"Relevance: {1. if i == 0 else .8:.1f}   |   smoothed KL: {divergences[i]:.6f}\n"
                        f"Objective at strength 0.5: {objectives[i]:.6f}" + ("  → CHOOSE y" if i else ""),
                        fontsize=9, color=GREEN if i else "#334155")
    figure.suptitle("One greedy calibration decision: profile fit can outweigh relevance", fontsize=14, weight="bold")
    figure.supxlabel("Toy example, not a measured MovieLens result. Objectives use already-normalized pool relevance and natural-log KL.", fontsize=9)
    return conceptual_record(figure, output, "calibration-step", [REPO_ROOT / "project/recsys/rerankers/calibration.py"],
                             {"profile": profile.tolist(), "candidate_distributions": [d.tolist() for d in appended],
                              "smoothed_distributions": [d.tolist() for d in smoothed],
                              "smoothed_kl": divergences, "strength": .5, "objectives": objectives, "chosen": "y"})


def exposure_round(output):
    figure, axis = diagram_canvas(
        "One user's recommendation changes the next user's choice",
        "Toy example: K=1, strength=1, both users prefer a to b. At strength=0 both receive a.")
    steps = [(0.25, "START", "Exposure (a, b) = (0, 0)\nBonuses: a=1, b=1\nNo item has been recommended"),
             (4.3, "USER 1 PICKS a", "Bonuses tie; original score wins\nExposure becomes (1, 0)\nRank-one increment = 1"),
             (8.35, "USER 2 PICKS b", "Bonus a=1/2; bonus b=1\nExposure becomes (1, 1)\nShared state changes the choice")]
    for x, label, text in steps:
        axis.text(x + 1.7, 5.1, label, ha="center", weight="bold", color=BLUE, fontsize=11)
        box(axis, x, 2.45, 3.4, 2.1, text, GREEN if x > 8 else BLUE, fontsize=10)
    arrow(axis, (3.75, 3.5), (4.2, 3.5))
    arrow(axis, (7.8, 3.5), (8.25, 3.5))
    axis.text(6, 1.45, "Real calls: rank rounds → sorted external user IDs → discounted exposure update",
              ha="center", fontsize=11, weight="bold")
    axis.text(6, .65, "A new call starts a new exposure map. Coverage instead tracks an ever-recommended set.",
              ha="center", fontsize=10, color="#475569")
    return conceptual_record(figure, output, "exposure-round", [REPO_ROOT / "project/recsys/rerankers/item_fairness.py"],
                             {"k": 1, "strength": 1, "original_preference": ["a", "b"],
                              "exposure_states": [[0, 0], [1, 0], [1, 1]], "chosen_items": ["a", "b"]})


def pipeline_orders(output):
    figure, axis = diagram_canvas(
        "The order experiment changes one controlled operation",
        "Conceptual weighted-hybrid paths. Fixed routing replaces score fusion in the separate switching experiment.", height=6.1)
    box(axis, .6, 4.8, 10.8, .85,
        "Shared start: full eligible-candidate z-scores → unchanged hybrid chooses the same top-50 pool",
        GREEN, fontsize=11)
    axis.text(.2, 3.55, "PRE", color=BLUE, weight="bold")
    for x, text in [(1.15, "Rerank each of the\nsix component lists"),
                    (4.75, "Promote prefix; assign\noriginal sorted scores"),
                    (8.35, "Frozen Ridge fusion\n→ hybrid top 10")]:
        box(axis, x, 2.9, 3.25, 1.25, text, BLUE, fontsize=10.5)
    arrow(axis, (4.45, 3.52), (4.7, 3.52))
    arrow(axis, (8.05, 3.52), (8.3, 3.52))
    axis.text(.2, 1.55, "POST", color=ORANGE, weight="bold")
    for x, text in [(1.15, "Original standardized\ncomponent scores"),
                    (4.75, "Frozen Ridge fusion\n(no second normalization)"),
                    (8.35, "Rerank hybrid once\n→ hybrid top 10")]:
        box(axis, x, .9, 3.25, 1.25, text, ORANGE, fontsize=10.5)
    arrow(axis, (4.45, 1.52), (4.7, 1.52))
    arrow(axis, (8.05, 1.52), (8.3, 1.52))
    axis.text(6.3, .25, "Same pool, fits, profiles and relevance denominator; select each method/order on validation, then freeze.",
              ha="center", fontsize=9.5, color="#475569")
    return conceptual_record(figure, output, "pipeline-orders", [REPO_ROOT / "project/recsys/experiments/task3.py"],
                             {"candidate_count": 50, "k": 10, "pre_score_conversion": "score-preserving permutation",
                              "pre_strength": "one shared strength for all six components", "second_normalization": False})


def overview_figure(output):
    figure, axis = diagram_canvas(
        "From interaction evidence to an evaluated top-ten list",
        "Scores express a preference order. Reranking changes the supplied candidates; evaluation measures the final output.",
        height=4.8)
    labels = ["Training\ninteractions", "Individual models\nor a hybrid", "Eligible\ncandidate scores", "Optional\nreranking", "Final ten\n→ metrics"]
    for i, label in enumerate(labels):
        x = .15 + 2.4 * i
        box(axis, x, 2.45, 2.05, 1.65, label, GREEN if i == 0 else BLUE if i < 3 else ORANGE, 10)
        if i < 4:
            arrow(axis, (x + 2.1, 3.25), (x + 2.35, 3.25))
    axis.text(6, 1.2, "Training fits parameters  •  Validation chooses settings  •  Test measures fixed choices",
              ha="center", fontsize=10)
    return conceptual_record(figure, output, "system-overview",
                             [REPO_ROOT / "project/recsys/contracts.py"],
                             {"stages": labels, "final_list_length": 10})


def accuracy_figure(output):
    figure, axis = diagram_canvas(
        "One ranking can be judged in several useful ways",
        "Toy example: K=3, recommendations [a,b,c], relevant items {a,c,d,e,f}.",
        height=4.8)
    for i, (item, hit) in enumerate([("a", True), ("b", False), ("c", True)]):
        x = 1 + i * 3.5
        box(axis, x, 3.25, 2.7, 1.65, f"Rank {i+1}: {item}\n" + ("HIT" if hit else "MISS"),
            GREEN if hit else "#64748B", 13)
    axis.text(6, 2.3, "Precision = 2/3       Recall = 2/5       F1 = 1/2       Reciprocal rank = 1",
              ha="center", fontsize=11)
    ndcg = 1.5 / sum(1 / np.log2(rank+1) for rank in range(1, 4))
    axis.text(6, 1.25, f"AP = (1 + 2/3)/5 = 1/3       DCG = 1 + 1/2 = 1.5       NDCG ≈ {ndcg:.3f}",
              ha="center", fontsize=11)
    return conceptual_record(figure, output, "accuracy-example",
                             [REPO_ROOT / "project/recsys/metrics/accuracy.py"],
                             {"ranking": ["a","b","c"], "relevant": ["a","c","d","e","f"],
                              "precision": 2/3, "recall": .4, "f1": .5, "rr": 1, "ap": 1/3,
                              "dcg": 1.5, "ndcg": float(ndcg)})


def baseline_figure(output):
    source = REPORT_DIR / "data/baselines.csv"
    rows = read_csv(source)
    names = [row["label"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    data = {}
    for axis, metric, factor, title, limit in [
            (axes[0], "NDCG@10", 1, "Recovery of held-out interactions", .45),
            (axes[1], "CatalogueCoverage@10", 100, "Fraction of the catalogue reached", 110)]:
        values = [factor * float(row[metric]) for row in rows]
        colors = [ORANGE if name in ["Random","Popularity"] else BLUE for name in names]
        bars = axis.barh(names, values, color=colors, height=.65)
        axis.bar_label(bars, labels=[f"{value:.4f}" if factor == 1 else f"{value:.2f}%" for value in values],
                       padding=4, fontsize=9)
        axis.set(xlim=(0,limit), xlabel=metric if factor == 1 else "Catalogue coverage (%)", title=title)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=.15)
        data[metric] = dict(zip(names, values))
    figure.suptitle("Accuracy and catalogue breadth favor different behaviors", fontsize=14, weight="bold")
    figure.supxlabel("Supplied individual configurations; the same 943 users, full eligible catalogue and K=10.", fontsize=9)
    return save_figure(figure, output, "baseline-comparison", [source], data)


def project_reranking_figures(output):
    sys.path.insert(0, str(REPO_ROOT))
    from project.scripts.plot_reranking import plot as calibration_plot
    from project.scripts.plot_diversification import plot as diversification_plot

    source = REPORT_DIR / "data/reranking.json"
    evidence = read_json(source)
    records = {}
    with tempfile.TemporaryDirectory(prefix="recbole-study-plots-") as temporary:
        root = Path(temporary)
        (root / "manifest.json").write_text(json.dumps(
            {"status": "complete", "models": list(evidence["models"]), "k": evidence["k"]}))
        for name, model in evidence["models"].items():
            for partition, field in [("valid", "validation"), ("test", "test")]:
                folder = root / name / partition
                folder.mkdir(parents=True)
                (folder / "summary.json").write_text(json.dumps(model[field]))
            (root / name / "valid/selection.json").write_text(json.dumps(model["selection"]))
        for plotter, prefix, module in [
                (calibration_plot, "", "plot_reranking.py"),
                (diversification_plot, "diversification-", "plot_diversification.py")]:
            target = root / ("calibration" if not prefix else "diversification")
            plotter(root, target)
            for stem in ("validation-tradeoffs", "test-effects"):
                name = prefix + stem
                destination = output / (name + ".png")
                shutil.copyfile(target / (stem + ".png"), destination)
                records[name] = {"kind": "project plotting code applied to saved tuned-model results",
                                 "sources": source_records([source, REPO_ROOT / "project/scripts" / module]),
                                 "files": {destination.name: sha256(destination)}}
            if prefix:
                destination = output / "diversification-figure-data.json"
                shutil.copyfile(target / "figure-data.json", destination)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reranking-bundle", type=Path,
                        default=REPO_ROOT / "project/artifacts/reranking-20261022-001")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "hyperparameters/results")
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "recbole-study"})
    figures = {}
    for name, generate, inputs in [
            ("training-popularity", popularity_figure, args.reranking_bundle),
            ("individual-tuning", tuning_figure, args.results_root),
            ("hybrid-comparisons", hybrid_figure, args.results_root),
            ("coefficient-ablation", ablation_figure, args.results_root)]:
        figures[name] = generate(inputs, args.output)
    for name, generate in [("evaluation-flow", evaluation_flow), ("calibration-step", calibration_step),
                            ("exposure-round", exposure_round), ("pipeline-orders", pipeline_orders),
                            ("system-overview", overview_figure), ("accuracy-example", accuracy_figure),
                            ("baseline-comparison", baseline_figure)]:
        figures[name] = generate(args.output)
    figures.update(project_reranking_figures(args.output))
    group_plot = args.output / "reranking-groups.png"
    figures["reranking-groups"] = {
        "kind": "existing project illustration copied without alteration",
        "sources": source_records([REPORT_DIR / "data/groups.json", REPORT_DIR / "data/reranking.json"]),
        "files": {group_plot.name: sha256(group_plot)}}
    manifest = {"kind": "supporting figure verification",
                "report_source_snapshot": "193a52a9d1d004475bd1afa64c6c34f57ec1bd0a",
                "generator_sha256": sha256(Path(__file__)),
                "training_or_selection_performed": False, "figures": figures}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(figures)} figures in {args.output}")


if __name__ == "__main__":
    main()
