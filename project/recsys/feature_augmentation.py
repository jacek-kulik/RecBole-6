"""UserKNN-augmented EASE/SLIM inputs with unchanged binary targets."""

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .artifacts import new_run, write_json
from .component_cache import file_hash
from .contracts import ScoreTable, require_aligned
from .metrics.accuracy import evaluate_accuracy
from .recbole_adapter import _write_scores, load_scores


def read_pairs(path):
    pairs = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            items = pairs.setdefault(row["user_id"], set())
            if row["item_id"] in items:
                raise ValueError("Duplicate interaction")
            items.add(row["item_id"])
    return pairs


def augment(r, predictions, top_n, strength, equal=False):
    """Choose positive unseen predictions; normalize by each user's unseen max."""
    if (r.shape != predictions.shape or not np.isfinite(predictions).all()
            or not np.isin(r, [0, 1]).all() or top_n < 1
            or not np.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError("Invalid augmentation inputs")
    additions = np.zeros_like(r, dtype=float)
    for u in range(len(r)):
        eligible = np.flatnonzero((r[u] == 0) & (predictions[u] > 0))
        order = np.argsort(-predictions[u, eligible], kind="stable")
        selected = eligible[order[:top_n]]
        if len(selected):
            additions[u, selected] = strength * (1 if equal else
                predictions[u, selected] / predictions[u, selected].max())
    return r + additions, additions


def fit_ease(x, targets, reg):
    """Minimize ||targets-XB||^2 + reg ||B||^2, subject to diag(B)=0.

    The constrained optimum uses A=X'X+reg*I and C=X'targets.
    Correct each unconstrained column using its Lagrange multiplier.
    """
    if (x.shape != targets.shape or reg <= 0 or not np.isfinite(reg)
            or not np.isfinite(x).all() or not np.isfinite(targets).all()):
        raise ValueError("Invalid fit inputs")
    gram = x.T @ x
    gram.flat[::len(gram) + 1] += reg
    factor = cho_factor(gram, check_finite=False)
    inverse = cho_solve(factor, np.eye(len(gram)), check_finite=False)
    weights = cho_solve(factor, x.T @ targets, check_finite=False)
    weights -= inverse * (np.diag(weights) / np.diag(inverse))[None, :]
    np.fill_diagonal(weights, 0)
    return weights


def load_component(source, name):
    pointer = json.loads((source / f"cache-{name}.json").read_text())
    directory = Path(pointer["path"])
    path = directory / "scores-valid.json.gz"
    digest = file_hash(path)
    if digest != json.loads((directory / "full.complete.json").read_text())["sha256"]:
        raise ValueError("Component cache hash mismatch")
    table = load_scores(path)
    if table.model != name or table.partition != "valid":
        raise ValueError("Unexpected component identity")
    return table, {"path": str(path), "sha256": digest,
                   "identity_sha256": file_hash(directory / "identity.json")}


def fit_slim(x, targets, alpha, l1_ratio):
    """Nonnegative elastic-net columns, hiding each target's input column."""
    from scipy.sparse import csc_matrix
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNet

    if (x.shape != targets.shape or not np.isfinite(x).all()
            or not np.isfinite(targets).all() or alpha <= 0
            or not np.isfinite(alpha) or not 0 <= l1_ratio <= 1):
        raise ValueError("Invalid SLIM inputs")
    features = csc_matrix(x, dtype=np.float64)
    weights = np.zeros((x.shape[1], x.shape[1]))
    failed = []
    for j in range(x.shape[1]):
        start, end = features.indptr[j:j+2]
        original = features.data[start:end].copy()
        features.data[start:end] = 0
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=True,
                           fit_intercept=False, copy_X=False, selection="random",
                           random_state=42, max_iter=1000, tol=1e-4)
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(features, targets[:, j])
            if any(issubclass(w.category, ConvergenceWarning) for w in captured):
                failed.append(j)
            weights[:, j] = model.coef_
        finally:
            features.data[start:end] = original
        if (j + 1) % 200 == 0:
            print(f"SLIM columns {j + 1}/{x.shape[1]}", flush=True)
    if failed:
        raise ValueError(f"SLIM did not converge for {len(failed)} columns")
    np.fill_diagonal(weights, 0)
    return weights


def run(args):
    source = Path(args.source).resolve()
    knn, knn_evidence = load_component(source, "userknn")
    baseline, receiver_evidence = load_component(source, args.receiver)
    require_aligned([knn, baseline])
    manifest = json.loads((source / "manifest.json").read_text())
    if knn.split_id != manifest["split_id"]:
        raise ValueError("Split identity mismatch")
    training, relevant = [read_pairs(source / f"splits/{p}.csv") for p in ("train", "valid")]
    users = sorted(knn.scores)
    items = sorted(set().union(*training.values(), *(set(r) for r in knn.scores.values())))
    index = {item: i for i, item in enumerate(items)}
    if set(users) != set(training) or set(users) != set(relevant):
        raise ValueError("User sets differ")
    r = np.zeros((len(users), len(items)))
    predictions = np.zeros_like(r)
    for u, user in enumerate(users):
        if (set(knn.scores[user]) != set(items) - training[user]
                or not relevant[user] <= knn.scores[user].keys()):
            raise ValueError("Candidate mask or held-out positives differ")
        r[u, [index[i] for i in training[user]]] = 1
        for item, score in knn.scores[user].items():
            predictions[u, index[item]] = score
    output = new_run(args.output)
    objective = ("||R-XB||_F^2 + lambda*||B||_F^2; diag(B)=0" if args.receiver == "ease" else
                 "||R-XB||_F^2/(2*n_users) + alpha*l1_ratio*||B||_1 + alpha*(1-l1_ratio)*||B||_F^2/2; diag(B)=0; B>=0")
    proof = {"status": "running", "receiver": args.receiver, "split_id": knn.split_id,
             "test_evaluated": False, "target": "unchanged binary training matrix, including zero entries",
             "objective": objective,
             "source": str(source), "reg": args.reg, "top_n": args.top_n,
             "strength": args.strength, "equal": args.equal,
             "prediction_scale": "per-user positive unseen maximum; top-N external-token tie break",
             "userknn": knn_evidence, "receiver_cache": receiver_evidence,
             "slim_alpha": args.slim_alpha, "slim_l1_ratio": args.slim_l1_ratio,
             "slim_solver": {"max_iter": 1000, "tol": 1e-4, "random_state": 42,
                             "selection": "random", "positive": True},
             "sha256": {"implementation": file_hash(__file__),
                        "train": file_hash(source / "splits/train.csv"),
                        "valid": file_hash(source / "splits/valid.csv"),
                        "metrics": file_hash(Path(__file__).parent / "metrics/accuracy.py")}}
    write_json(output / "manifest.json", proof)
    started = time.perf_counter()
    x, additions = augment(r, predictions, args.top_n, args.strength, args.equal)
    results = {}
    for name, inputs in (("baseline", r), ("augmented", x)):
        print(f"Fitting {args.receiver} {name}: {r.shape}", flush=True)
        weights = (fit_ease(inputs, r, args.reg) if args.receiver == "ease" else
                   fit_slim(inputs, r, args.slim_alpha, args.slim_l1_ratio))
        values = inputs @ weights
        table = ScoreTable(f"{args.receiver}-{name}", knn.split_id, "valid", {
            user: {i: float(values[u, index[i]]) for i in knn.scores[user]}
            for u, user in enumerate(users)})
        require_aligned([table, baseline])
        results[name] = evaluate_accuracy(table.top_k(10), relevant, 10)
        if name == "baseline":
            error = max(abs(score - baseline.scores[u][i])
                        for u, row in table.scores.items() for i, score in row.items())
            proof["cached_baseline_max_absolute_error"] = error
            if (args.receiver == "ease" and
                    args.reg == manifest["component_parameters"]["ease"]["reg_weight"] and error > 1e-5):
                raise ValueError(f"Reconstructed baseline differs from cache: {error}")
        else:
            np.savez_compressed(output / "model.npz", weights=weights, additions=additions,
                                users=np.array(users), items=np.array(items))
        _write_scores(table, output / f"scores-valid-{name}.json.gz")
        write_json(output / f"metrics-{name}.json", results[name])
        print(json.dumps({name: results[name]["aggregate"]}), flush=True)
    proof.update(status="complete", seconds=time.perf_counter() - started,
                 added_entries=int(np.count_nonzero(additions)),
                 mean_added_mass=float(additions.sum(axis=1).mean()),
                 mean_observed_mass=float(r.sum(axis=1).mean()))
    comparison = {name: value["aggregate"] for name, value in results.items()}
    comparison["userknn"] = evaluate_accuracy(knn.top_k(10), relevant, 10)["aggregate"]
    write_json(output / "comparison.json", comparison)
    proof["outputs_sha256"] = {p.name: file_hash(p) for p in output.iterdir() if p.name != "manifest.json"}
    write_json(output / "manifest.json", proof)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="project/artifacts/hybrid-60-20-20")
    parser.add_argument("--output", required=True)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--reg", type=float, default=250)
    parser.add_argument("--receiver", choices=("ease", "slimelastic"), default="ease")
    parser.add_argument("--slim-alpha", type=float, default=0.2)
    parser.add_argument("--slim-l1-ratio", type=float, default=0.01)
    parser.add_argument("--equal", action="store_true")
    run(parser.parse_args())
