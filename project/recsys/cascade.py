"""Joint validation tuning of ItemKNN retrieval and BPR reranking.

Run: python -m project.recsys.cascade --output project/artifacts/cascade-itemknn-bpr
Completed component fits are hash-verified and reusable with --resume.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

from .artifacts import REPO_ROOT, fingerprint, new_run
from .component_cache import atomic_json, file_hash, run_lock
from .contracts import ScoreTable, require_aligned
from .metrics.accuracy import evaluate_accuracy
from .recbole_adapter import _export_splits, _write_scores, load_scores


SHORTLISTS = (10, 20, 50, 100, 200, 500, 1000, None)
ITEM_GRID = [dict(k=k, shrink=s) for k, s in
             itertools.product((10, 20, 50, 100, 200), (0.0, 10.0, 100.0, 200.0))]
BPR_GRID = [dict(embedding_size=d, learning_rate=lr, epochs=e, train_batch_size=512)
            for d, lr, e in itertools.product((32, 64, 128), (0.0005, 0.0015), (20, 50))]


def read_pairs(path):
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            items = result.setdefault(row["user_id"], set())
            if row["item_id"] in items:
                raise ValueError("Duplicate split pair")
            items.add(row["item_id"])
    return result


class CascadeSpace:
    """Dense evaluation workspace; external-token order resolves every score tie."""

    def __init__(self, reference, relevant, k=10):
        reference.validate()
        if reference.scores.keys() != relevant.keys() or k <= 0:
            raise ValueError("Evaluation users or cutoff differ")
        self.reference, self.relevant, self.k = reference, relevant, k
        self.users = sorted(reference.scores)
        self.items = sorted(set().union(*(set(row) for row in reference.scores.values())))
        self.lookup = {item: i for i, item in enumerate(self.items)}
        self.lengths = np.array([len(reference.scores[u]) for u in self.users])
        if np.any(self.lengths < k):
            raise ValueError("Too few eligible candidates")
        self.truth = np.zeros((len(self.users), len(self.items)), dtype=bool)
        for n, user in enumerate(self.users):
            if not relevant[user] or not relevant[user] <= reference.scores[user].keys():
                raise ValueError("Positive items absent from eligible catalogue")
            self.truth[n, [self.lookup[i] for i in relevant[user]]] = True
        self.rows = np.arange(len(self.users))[:, None]
        self.positive_counts = self.truth.sum(axis=1)
        self.discount = 1 / np.log2(np.arange(k) + 2)
        self.ideal = np.array([self.discount[:min(k, n)].sum() for n in self.positive_counts])

    def matrix(self, table):
        require_aligned([self.reference, table])
        values = np.full(self.truth.shape, -np.inf)
        for n, user in enumerate(self.users):
            row = table.scores[user]
            values[n, [self.lookup[i] for i in row]] = list(row.values())
        return values

    def order(self, values):
        return np.argsort(-values, axis=1, kind="stable")

    def shortlist(self, retrieval_order, size):
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < self.k):
            raise ValueError("Shortlist size must be an integer at least the final cutoff")
        return retrieval_order[:, :size]

    def rerank(self, retrieval_order, bpr_values, size):
        candidates = self.shortlist(retrieval_order, size)
        values = np.take_along_axis(bpr_values, candidates, axis=1)
        order = np.lexsort((candidates, -values), axis=1)[:, :self.k]
        return np.take_along_axis(candidates, order, axis=1)

    def metrics(self, top):
        hits = self.truth[self.rows, top]
        dcg = (hits * self.discount).sum(axis=1)
        reciprocal = (hits / np.arange(1, self.k + 1)).max(axis=1)
        return {f"NDCG@{self.k}": float(np.mean(dcg / self.ideal)),
                f"Recall@{self.k}": float(np.mean(hits.sum(axis=1) / self.positive_counts)),
                f"MRR@{self.k}": float(reciprocal.mean())}

    def retrieval_metrics(self, order, size):
        candidates = self.shortlist(order, size)
        hits = self.truth[self.rows, candidates].sum(axis=1)
        counts = self.lengths if size is None else np.minimum(size, self.lengths)
        ideal_retrieved = np.array([self.discount[:min(self.k, n)].sum() for n in hits])
        return {"candidate_recall": float(np.mean(hits / self.positive_counts)),
                "oracle_ndcg_at_10": float(np.mean(ideal_retrieved / self.ideal)),
                "mean_bpr_candidates": float(counts.mean()),
                "candidate_fraction": float(counts.sum() / self.lengths.sum())}

    def rankings(self, top):
        return {u: [self.items[i] for i in row] for u, row in zip(self.users, top)}


def cascade_scores(retriever, ranker, size, k=10):
    """Return BPR scores on only the ItemKNN shortlist; no score blending."""
    require_aligned([retriever, ranker])
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < k):
        raise ValueError("Invalid shortlist size")
    scores = {}
    for u, row in retriever.scores.items():
        if len(row) < k:
            raise ValueError("Too few candidates")
        selected = sorted(row, key=lambda i: (-row[i], i))[:size]
        scores[u] = {i: ranker.scores[u][i] for i in selected}
    result = ScoreTable("cascade-itemknn-bpr", retriever.split_id, retriever.partition, scores)
    result.validate()
    return result


def load_reference(source):
    manifest = json.loads((source / "manifest.json").read_text())
    tables, evidence = [], {}
    for name in ("itemknn", "bpr"):
        pointer = json.loads((source / f"cache-{name}.json").read_text())
        directory = Path(pointer["path"])
        path = directory / "scores-valid.json.gz"
        digest = file_hash(path)
        if digest != json.loads((directory / "full.complete.json").read_text())["sha256"]:
            raise ValueError("Corrupt reference score cache")
        table = load_scores(path)
        if (table.model, table.split_id, table.partition) != (name, manifest["split_id"], "valid"):
            raise ValueError("Reference score identity differs")
        tables.append(table)
        evidence[name] = {"path": str(path), "sha256": digest, "key": pointer["key"]}
    require_aligned(tables)
    return manifest, tables, evidence


def fit_component(name, params, source, output, split_id, identity):
    """Fit independently of labels used for tuning; fixed epochs, no early stopping."""
    from recbole.config import Config
    from recbole.data import create_dataset
    from recbole.utils import init_seed
    from .hybrid_training import train_and_score

    key = fingerprint({"name": name, "params": params, "identity": identity})
    directory = output / "components" / name / key
    marker, scores = directory / "complete.json", directory / "scores-valid.json.gz"
    if marker.exists():
        if file_hash(scores) != json.loads(marker.read_text())["sha256"]:
            raise ValueError("Corrupt completed cascade cache")
        table = load_scores(scores)
        if (table.model, table.partition, table.split_id) != (name, "valid", split_id):
            raise ValueError("Cascade cache identity differs")
        return table, scores
    directory.mkdir(parents=True, exist_ok=True)
    argv = sys.argv
    try:
        sys.argv = argv[:1]
        config = Config(config_file_list=[str(REPO_ROOT / f"project/configs/models/{name}.yaml"),
                                          str(REPO_ROOT / "project/configs/protocol.yaml")],
                        config_dict={**params, "seed": 42, "use_gpu": False,
                                     "data_path": str(REPO_ROOT / "dataset"),
                                     "checkpoint_dir": str(directory),
                                     "save_dataset": False, "save_dataloaders": False})
    finally:
        sys.argv = argv
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    parts = dataset.build()
    # Compare exported bytes, including order/ratings, with the original split.
    with tempfile.TemporaryDirectory() as tmp:
        generated = Path(tmp) / "splits"
        actual = _export_splits(dataset, [SimpleNamespace(dataset=d) for d in parts], generated, config)
        if actual != split_id or any((generated / f"{p}.csv").read_bytes() !=
                                    (source / f"splits/{p}.csv").read_bytes()
                                    for p in ("train", "valid", "test")):
            raise ValueError("Reconstructed split differs from reference")
    atomic_json(directory / "parameters.json", params)
    atomic_json(directory / "resolved-config.json", json.loads(json.dumps(config.final_config_dict, default=str)))
    table = train_and_score(config, dataset, parts[0], parts[1], directory / "training", split_id, name)
    temporary = directory / "scores-valid.tmp.gz"
    _write_scores(table, temporary)
    temporary.replace(scores)
    atomic_json(marker, {"sha256": file_hash(scores), "key": key})
    return table, scores


def input_identity(source, references):
    paths = [Path(__file__), REPO_ROOT / "project/recsys/hybrid_training.py",
             REPO_ROOT / "project/recsys/recbole_adapter.py",
             REPO_ROOT / "project/recsys/contracts.py",
             REPO_ROOT / "project/recsys/metrics/accuracy.py",
             REPO_ROOT / "project/recsys/metrics/_validation.py",
             REPO_ROOT / "project/configs/protocol.yaml"]
    paths += [REPO_ROOT / f"project/configs/models/{name}.yaml" for name in ("itemknn", "bpr")]
    paths += [REPO_ROOT / p for p in ("recbole/model/general_recommender/itemknn.py",
              "recbole/model/general_recommender/bpr.py", "recbole/trainer/trainer.py",
              "recbole/data/dataset/dataset.py", "recbole/data/utils.py",
              "recbole/model/abstract_recommender.py", "recbole/utils/case_study.py")]
    paths += list((REPO_ROOT / "dataset/ml-100k").glob("*.inter"))
    paths += [source / f"splits/{p}.csv" for p in ("train", "valid", "test")]
    return {"hashes": {str(p): file_hash(p) for p in paths}, "references": references,
            "item_grid": ITEM_GRID, "bpr_grid": BPR_GRID, "shortlists": SHORTLISTS,
            "selection_metric": "NDCG@10", "seed": 42}


def run(source, output, resume=False):
    import torch
    torch.set_num_threads(2)
    source, output = Path(source).resolve(), Path(output).resolve()
    if not resume:
        new_run(output)
    elif not (output / "identity.json").exists():
        raise ValueError("Resume requires an existing cascade run")
    with run_lock(output):
        return _run(source, output)


def _run(source, output):
    manifest, references, evidence = load_reference(source)
    identity = input_identity(source, evidence)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != json.loads(json.dumps(identity)):
        raise ValueError("Source, configuration or data changed; use a new output directory")
    atomic_json(identity_path, identity)
    proof = {"status": "running", "split_id": manifest["split_id"], "test_evaluated": False,
             "selection_metric": "NDCG@10", "seed": 42, "identity_sha256": file_hash(identity_path),
             "training": "Independent train-only fits; fixed epochs; joint cascade validation selection",
             "scope": "Full catalogue; all held-out interactions positive; same users/masks as Task 1.3",
             "budget": {"itemknn_fits": len(ITEM_GRID), "bpr_fits": len(BPR_GRID),
                        "cascade_trials": len(ITEM_GRID) * len(BPR_GRID) * len(SHORTLISTS)},
             "tie_break": "Highest validation NDCG@10, then smaller shortlist, then grid order",
             "test_policy": "Freeze selection before scoring test; no train+validation refit"}
    atomic_json(output / "manifest.json", proof)
    training, validation = [read_pairs(source / f"splits/{p}.csv") for p in ("train", "valid")]
    catalogue = set().union(*training.values(), *(set(row) for row in references[0].scores.values()))
    if any(training.get(u, set()) & truth for u, truth in validation.items()) or any(
            set(row) != catalogue - training.get(u, set()) for u, row in references[0].scores.items()):
        raise ValueError("Invalid reference full-catalogue mask")
    space = CascadeSpace(references[0], validation)
    item_orders, bpr_values, paths = [], [], {"itemknn": [], "bpr": []}
    for name, grid in (("itemknn", ITEM_GRID), ("bpr", BPR_GRID)):
        for n, params in enumerate(grid):
            print(f"{name} {n+1}/{len(grid)}: {params}", flush=True)
            table, path = fit_component(name, params, source, output, manifest["split_id"], identity)
            values = space.matrix(table)
            if name == "itemknn":
                item_orders.append(space.order(values))
            else:
                bpr_values.append(values)
            paths[name].append(str(path))
    atomic_json(output / "component-paths.json", paths)
    trials, best, finite_best = [], None, None
    for i, order in enumerate(item_orders):
        diagnostics = {size: space.retrieval_metrics(order, size) for size in SHORTLISTS}
        for j, values in enumerate(bpr_values):
            for size in SHORTLISTS:
                metrics = space.metrics(space.rerank(order, values, size))
                row = {"itemknn_index": i, "bpr_index": j, "shortlist_size": size,
                       **metrics, **diagnostics[size]}
                trials.append(row)
                key = (metrics["NDCG@10"], -(size if size is not None else len(space.items)))
                if best is None or key > best[0]:
                    best = (key, row)
                if size is not None and (finite_best is None or key > finite_best[0]):
                    finite_best = (key, row)
        print(f"Searched retriever {i+1}/{len(item_orders)}; best NDCG@10={best[1]['NDCG@10']:.6f}", flush=True)
    atomic_json(output / "trials.json", trials)
    selected = best[1]
    i, j, size = selected["itemknn_index"], selected["bpr_index"], selected["shortlist_size"]
    frozen = {"selection": selected, "itemknn": ITEM_GRID[i], "bpr": BPR_GRID[j],
              "best_finite_cascade": finite_best[1], "selection_metric": "validation NDCG@10",
              "trials_sha256": file_hash(output / "trials.json"),
              "components": {name: {"path": paths[name][index], "sha256": file_hash(paths[name][index])}
                             for name, index in (("itemknn", i), ("bpr", j))}}
    # This file is written before loading test relevance or calculating test metrics.
    atomic_json(output / "frozen-selection.json", frozen)
    frozen_hash = file_hash(output / "frozen-selection.json")
    selected_tables = [load_scores(paths[name][index]) for name, index in (("itemknn", i), ("bpr", j))]
    selected_cascade = cascade_scores(*selected_tables, size)
    _write_scores(selected_cascade, output / "scores-valid.json.gz")
    rankings = selected_cascade.top_k(10)
    valid_metrics = evaluate_accuracy(rankings, validation, 10)
    if not np.isclose(valid_metrics["aggregate"]["NDCG@10"], selected["NDCG@10"], atol=1e-12):
        raise ValueError("Vectorized search disagrees with independent evaluator")
    if rankings != space.rankings(space.rerank(item_orders[i], bpr_values[j], size)):
        raise ValueError("Exported cascade ranking differs from search")
    # Standalone controls are also selected on validation, independently of cascade.
    standalone_i = max(range(len(item_orders)), key=lambda n: space.metrics(item_orders[n][:, :10])["NDCG@10"])
    standalone_j = max(range(len(bpr_values)), key=lambda n: space.metrics(space.order(bpr_values[n])[:, :10])["NDCG@10"])
    controls = {"task13-itemknn": references[0], "task13-bpr": references[1],
                "retuned-itemknn": load_scores(paths["itemknn"][standalone_i]),
                "retuned-bpr": load_scores(paths["bpr"][standalone_j])}
    atomic_json(output / "standalone-selection.json", {"itemknn": ITEM_GRID[standalone_i],
                                                      "bpr": BPR_GRID[standalone_j]})
    # Test candidates remove training + validation history; weights stay fixed.
    testing = read_pairs(source / "splits/test.csv")
    def test_table(table):
        if testing.keys() != table.scores.keys() or any(
                training.get(u, set()) & truth or validation[u] & truth for u, truth in testing.items()):
            raise ValueError("Test users differ or partitions overlap")
        result = ScoreTable(table.model, table.split_id, "test",
                            {u: {item: score for item, score in row.items() if item not in validation[u]}
                             for u, row in table.scores.items()})
        result.validate()
        return result
    test_components = [test_table(t) for t in selected_tables]
    test_cascade = cascade_scores(*test_components, size)
    _write_scores(test_cascade, output / "scores-test.json.gz")
    results = {"valid": {"cascade": valid_metrics},
               "test": {"cascade": evaluate_accuracy(test_cascade.top_k(10), testing, 10)}}
    for name, table in controls.items():
        results["valid"][name] = evaluate_accuracy(table.top_k(10), validation, 10)
        results["test"][name] = evaluate_accuracy(test_table(table).top_k(10), testing, 10)
    test_space = CascadeSpace(test_components[0], testing)
    test_order = test_space.order(test_space.matrix(test_components[0]))
    diagnostics = {"valid": space.retrieval_metrics(item_orders[i], size),
                   "test": test_space.retrieval_metrics(test_order, size)}
    atomic_json(output / "metrics.json", results)
    atomic_json(output / "retrieval.json", diagnostics)
    if file_hash(output / "frozen-selection.json") != frozen_hash:
        raise ValueError("Selection changed during test evaluation")
    lines = ["# ItemKNN to BPR cascade", "",
             "Train-only independent component fits, selected jointly by final validation NDCG@10. "
             "MovieLens 100K, per-user 60/20/20, seed 42, full catalogue, binary relevance. "
             "No score blending and no candidate-conditioned negative sampling.", "",
             f"Search: {len(ITEM_GRID)} ItemKNN fits x {len(BPR_GRID)} BPR fits x {len(SHORTLISTS)} shortlist sizes "
             f"= {len(trials)} validation trials. Every new configuration was fitted from scratch. "
             "BPR batch size is fixed at 512; its embedding dimension, learning rate and epoch budget vary. "
             "This is a bounded grid, not an exhaustive optimization of all training settings.", "",
             "Frozen selection:", "", "```json", json.dumps(frozen, indent=2), "```", "",
             "Full-catalogue shortlist is a BPR-only control and can win. A finite shortlist restricts "
             "BPR to ItemKNN candidates. Shortlist sizes are capped by each user's eligible catalogue. "
             "Shortlists are rebuilt after applying the test history mask."]
    for partition in ("valid", "test"):
        lines += ["", "## " + partition, "", "| Model | Recall@10 | MRR@10 | NDCG@10 |",
                  "|---|---:|---:|---:|"]
        for name, metric in results[partition].items():
            a = metric["aggregate"]
            lines.append(f"| {name} | {a['Recall@10']:.6f} | {a['MRR@10']:.6f} | {a['NDCG@10']:.6f} |")
        lines += ["", "Retrieval: " + json.dumps(diagnostics[partition])]
    lines += ["", "Test was scored only after validation choices were frozen. Existing Task 1.3 test "
              "results were already available; this is the same benchmark holdout, not a new untouched dataset. "
              "Standalone retuned controls use the same component grids, selected by validation NDCG@10. "
              "Validation gains reflect selection and are not independent generalization evidence. "
              "Single seed; no significance claim. Candidate fraction estimates BPR scoring volume only: "
              "the experiment caches full scores to reuse fits and does not measure production latency. "
              "All catalogue items remain eligible, including items without training evidence. "
              "NDCG@10 follows earlier hybrid experiments; shared YAML's provisional MRR@10 is also reported.", "",
              "Reproduce: `.venv/Scripts/python.exe -m project.recsys.cascade "
              "--output project/artifacts/cascade-itemknn-bpr --resume`."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    proof.update(status="complete", test_evaluated=True, selected=frozen,
                 frozen_selection_sha256=frozen_hash, independent_evaluation_verified=True)
    atomic_json(output / "manifest.json", proof)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="project/artifacts/hybrid-60-20-20")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(run(args.source, args.output, args.resume))
