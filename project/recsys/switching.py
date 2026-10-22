"""Validation-only UserKNN/BPR hard switch on users/items with training evidence."""

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .artifacts import new_run, write_json
from .component_cache import file_hash
from .content_based import read_history
from .contracts import ScoreTable, require_aligned
from .metrics.accuracy import evaluate_accuracy
from .recbole_adapter import load_scores, _write_scores


OVERLAPS = (1, 2, 5, 10, 20)
NEIGHBORS = (1, 5, 10, 20, 40, 51)
HISTORIES = (0, 10, 20, 40, 80, 160, 320, 100000)


def training_features(training, users, neighbor_k=50):
    """Support proxy: overlap among top-K binary cosine training neighbors.

    Self excluded. Equal similarities use external user-ID order. This is a
    reproducible routing proxy, not a reload of a cached model's neighbor matrix.
    """
    if neighbor_k <= 0 or not set(users) <= training.keys():
        raise ValueError("Positive neighbor count and training users required")
    population = sorted(training)
    items = sorted(set().union(*training.values()))
    indices = {item: i for i, item in enumerate(items)}
    rows, columns = [], []
    for row, user in enumerate(population):
        for item in sorted(training[user]):
            rows.append(row)
            columns.append(indices[item])
    matrix = csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(population), len(items)))
    common = (matrix @ matrix.T).toarray()
    lengths = np.array([len(training[u]) for u in population])
    denominator = np.sqrt(lengths[:, None] * lengths[None, :])
    cosine = np.divide(common, denominator, out=np.zeros_like(common), where=denominator != 0)
    np.fill_diagonal(cosine, -np.inf)
    features = {}
    for user in users:
        row = population.index(user)
        neighbors = np.argsort(-cosine[row], kind="stable")[:min(neighbor_k, len(population) - 1)]
        features[user] = {"history_length": int(lengths[row]),
                          "neighbor_overlaps": [int(common[row, i]) for i in neighbors]}
    return features


def use_userknn(features, overlap, neighbors, history):
    support = sum(n >= overlap for n in features["neighbor_overlaps"])
    return support >= neighbors or features["history_length"] < history


def warm_tables(tables, training, relevant):
    require_aligned(tables)
    counts = Counter(i for seen in training.values() for i in seen)
    positives = {u: {i for i in truth if counts[i] > 0}
                 for u, truth in relevant.items() if training.get(u)}
    positives = {u: truth for u, truth in positives.items() if truth}
    warm = []
    for table in tables:
        warm.append(ScoreTable(table.model, table.split_id, table.partition,
                               {u: {i: s for i, s in table.scores[u].items() if counts[i] > 0}
                                for u in positives}))
    require_aligned(warm)
    if any(not truth <= warm[0].scores[u].keys() for u, truth in positives.items()):
        raise ValueError("Eligible warm positives absent from candidates")
    return warm, positives, counts


def select_rule(users, features, knn_metrics, bpr_metrics):
    """Search fixed grids by macro validation NDCG@10; prefer constants on ties."""
    trials = []
    candidates = [({"kind": "constant", "model": "userknn"}, np.ones(len(users), dtype=bool)),
                  ({"kind": "constant", "model": "bpr"}, np.zeros(len(users), dtype=bool))]
    for overlap, neighbors, history in itertools.product(OVERLAPS, NEIGHBORS, HISTORIES):
        mask = np.array([use_userknn(features[u], overlap, neighbors, history) for u in users])
        candidates.append(({"kind": "switch", "min_overlap": overlap,
                            "min_reliable_neighbors": neighbors, "min_bpr_history": history}, mask))
    best, mixed = None, None
    for index, (rule, mask) in enumerate(candidates):
        score = float(np.where(mask, knn_metrics, bpr_metrics).mean())
        n_knn = int(mask.sum())
        row = {**rule, "NDCG@10": score, "userknn_users": n_knn, "bpr_users": len(users) - n_knn}
        trials.append(row)
        key = (score, rule["kind"] == "constant", -index)
        if best is None or key > best[0]:
            best = (key, row, mask.copy())
        if 0 < n_knn < len(users) and (mixed is None or key > mixed[0]):
            mixed = (key, row, mask.copy())
    return best[1:], mixed[1:] if mixed else None, trials


def run(source, output):
    source, output = Path(source).resolve(), new_run(output)
    manifest = json.loads((source / "manifest.json").read_text())
    training, relevant = [read_history(source / f"splits/{p}.csv") for p in ("train", "valid")]
    if any(training.get(u, set()) & truth for u, truth in relevant.items()):
        raise ValueError("Training/validation overlap")
    tables, score_hashes = [], {}
    for name in ("userknn", "bpr"):
        pointer = json.loads((source / f"cache-{name}.json").read_text())
        component = Path(pointer["path"])
        path = component / "scores-valid.json.gz"
        digest = file_hash(path)
        if digest != json.loads((component / "full.complete.json").read_text())["sha256"]:
            raise ValueError(f"Corrupt {name} cache")
        table = load_scores(path)
        if (table.model, table.split_id, table.partition) != (name, manifest["split_id"], "valid"):
            raise ValueError("Component identity differs")
        tables.append(table)
        score_hashes[name] = {"path": str(path), "sha256": digest, "key": pointer["key"]}
    require_aligned(tables)
    catalogue = set().union(*training.values(), *(set(row) for row in tables[0].scores.values()))
    if tables[0].scores.keys() != relevant.keys() or any(
            set(row) != catalogue - training.get(u, set()) for u, row in tables[0].scores.items()):
        raise ValueError("Reference users/candidate mask differ")
    tables, relevant_warm, counts = warm_tables(tables, training, relevant)
    users = sorted(relevant_warm)
    features = training_features(training, users)
    write_json(output / "user-features.json", features)
    proof = {"status": "running", "split_id": manifest["split_id"], "test_evaluated": False,
             "selection_metric": "NDCG@10", "routing_unit": "whole user ranking",
             "rule": "UserKNN if reliable_neighbors >= threshold OR history < min_bpr_history; BPR otherwise",
             "support": "Among top 50 binary-cosine neighbors, count those sharing at least min_overlap training items",
             "scope": "training-supported users and items; no cold-start solution",
             "grids": {"overlap": OVERLAPS, "neighbors": NEIGHBORS, "history": HISTORIES},
             "tie_break": "prefer constant model; then first grid entry; score ties use external item IDs",
             "component_scores": score_hashes,
             "sha256": {"source": file_hash(__file__), "train": file_hash(source / "splits/train.csv"),
                        "valid": file_hash(source / "splits/valid.csv"),
                        "accuracy": file_hash(Path(__file__).parent / "metrics/accuracy.py"),
                        "features": file_hash(output / "user-features.json")}}
    write_json(output / "manifest.json", proof)
    component_metrics = {name: evaluate_accuracy(table.top_k(10), relevant_warm, 10)
                         for name, table in zip(("userknn", "bpr"), tables)}
    values = [np.array([component_metrics[name]["per_user"][u]["NDCG@10"] for u in users])
              for name in ("userknn", "bpr")]
    (selected, mask), mixed, trials = select_rule(users, features, *values)
    write_json(output / "trials.json", trials)
    write_json(output / "selected-rule.json", selected)
    if mixed:
        write_json(output / "best-nonconstant-rule.json", mixed[0])
    scores, routes = {}, {}
    for u, knn in zip(users, mask):
        scores[u] = tables[0 if knn else 1].scores[u].copy()
        routes[u] = "userknn" if knn else "bpr"
    hybrid = ScoreTable("switch-userknn-bpr", tables[0].split_id, "valid", scores)
    require_aligned([hybrid, *tables])
    final_metrics = evaluate_accuracy(hybrid.top_k(10), relevant_warm, 10)
    if not np.isclose(final_metrics["aggregate"]["NDCG@10"], selected["NDCG@10"]):
        raise ValueError("Exported switch differs from search")
    _write_scores(hybrid, output / "scores-valid.json.gz")
    write_json(output / "routes.json", routes)
    metrics = {**component_metrics, "switch": final_metrics}
    write_json(output / "metrics.json", metrics)
    # Conditional diagnostics do not influence routing or threshold selection.
    route_diagnostics = {}
    for name in ("userknn", "bpr"):
        cohort = [u for u in users if routes[u] == name]
        route_diagnostics[name] = {"users": len(cohort), "component_mean_ndcg": {
            model: float(np.mean([component_metrics[model]["per_user"][u]["NDCG@10"] for u in cohort]))
            if cohort else None for model in ("userknn", "bpr")}}
    coverage = {"evaluation_users": len(users), "excluded_users": len(relevant) - len(users),
                "all_validation_interactions": sum(map(len, relevant.values())),
                "supported_validation_interactions": sum(map(len, relevant_warm.values())),
                "excluded_zero_training_item_interactions": sum(counts[i] == 0 for truth in relevant.values() for i in truth),
                "catalogue_items": len(catalogue), "supported_items": len(counts),
                "route_diagnostics": route_diagnostics,
                "userknn_wins": int(np.sum(values[0] > values[1])), "bpr_wins": int(np.sum(values[1] > values[0])),
                "ties": int(np.sum(values[0] == values[1]))}
    write_json(output / "diagnostics.json", coverage)
    lines = ["# UserKNN/BPR switching validation", "",
             "Per-user hard switch; existing trained components are reused. Train-only routing features. "
             "No blending or cross-model score calibration. Test split is not read or evaluated.", "",
             f"Warm scope: {len(users)} users; {coverage['supported_items']}/{len(catalogue)} supported items; "
             f"{coverage['supported_validation_interactions']}/{coverage['all_validation_interactions']} validation positives. "
             "Unsupported items are removed from both candidates and relevance for all compared models. "
             "These numbers are not directly comparable with previous full-catalogue evaluations.", "",
             "Selected rule:", "", "```json", json.dumps(selected, indent=2), "```", "",
             "| Model | Recall@10 | MRR@10 | NDCG@10 |", "|---|---:|---:|---:|"]
    for name, result in metrics.items():
        row = result["aggregate"]
        lines.append("| " + name + " | " + " | ".join(f"{row[m+'@10']:.6f}" for m in ("Recall", "MRR", "NDCG")) + " |")
    if mixed:
        lines += ["", "Best rule using both models (development diagnostic):", "", "```json", json.dumps(mixed[0], indent=2), "```"]
    lines += ["", f"Searched {len(trials)} rules, including constant-model controls. "
              "All performance shown is development evidence used for threshold selection; "
              "gains are not independent evidence of generalization. Neighborhood support is a deterministic "
              "proxy reconstructed from binary training histories, not the cached model's exact tie-dependent neighbors. "
              "Cold-start coverage is explicitly excluded, not solved. Fixed fallback for short histories is UserKNN. "
              "NDCG@10 is the explicit selection criterion consistent with previous hybrid experiments; "
              "the shared provisional YAML also lists MRR@10, reported here separately."]
    (output / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    proof.update(status="complete", selected_rule=selected, candidate_alignment_verified=True)
    write_json(output / "manifest.json", proof)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="project/artifacts/hybrid-60-20-20")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(run(args.source, args.output))
