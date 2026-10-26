"""Validation comparison of frozen BPR, concatenation MLP, and metadata MLP.

Run with python -m project.recsys.feature_combination --output <new directory>.
No test split is opened. The original dataset is loaded only to reconstruct the
checkpoint's token mapping; its interaction values are not used for fitting.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .artifacts import REPO_ROOT, new_run, write_json
from .component_cache import file_hash
from .content_based import read_genres, read_history
from .genre_ridge import read_years, year_feature
from .metrics.accuracy import evaluate_accuracy
from .recbole_adapter import load_scores


class PairMLP(nn.Module):
    """Frozen feature buffers and two trainable ReLU hidden layers."""

    def __init__(self, users, items, hidden):
        super().__init__()
        self.register_buffer("users", users.clone())
        self.register_buffer("items", items.clone())
        self.first = nn.Linear(users.shape[1] + items.shape[1], hidden)
        self.second = nn.Linear(hidden, hidden // 2)
        self.output = nn.Linear(hidden // 2, 1)

    def forward(self, users, items):
        features = torch.cat([self.users[users], self.items[items]], dim=1)
        return self.output(F.relu(self.second(F.relu(self.first(features))))).squeeze(-1)

    def full_scores(self, batch_size=16):
        """Algebraically equivalent to forward over every user-item pair."""
        rows = []
        with torch.no_grad():
            d = self.users.shape[1]
            user_part = F.linear(self.users, self.first.weight[:, :d], self.first.bias)
            item_part = F.linear(self.items, self.first.weight[:, d:])
            for start in range(0, len(self.users), batch_size):
                hidden = F.relu(user_part[start:start + batch_size, None] + item_part[None])
                rows.append(self.output(F.relu(self.second(hidden))).squeeze(-1))
        return torch.cat(rows).cpu().numpy()


def standardize(values, fit_indices):
    fit = values[fit_indices]
    mean, scale = fit.mean(0), fit.std(0)
    scale[scale < 1e-8] = 1.0
    return ((values - mean) / scale).astype(np.float32), {"mean": mean.tolist(), "scale": scale.tolist()}


def sample_negatives(rng, users, seen):
    """Uniform catalogue negatives, excluding training positives only."""
    negatives = rng.integers(seen.shape[1], size=len(users))
    invalid = seen[users, negatives]
    while invalid.any():
        negatives[invalid] = rng.integers(seen.shape[1], size=int(invalid.sum()))
        invalid = seen[users, negatives]
    return negatives


def evaluate(scores, mask, users, items, relevant):
    if scores.shape != mask.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid full-catalogue scores")
    masked = np.where(mask, scores, -np.inf)
    # Columns are sorted by external token, so stable sorting matches ScoreTable.
    top = np.argsort(-masked, axis=1, kind="stable")[:, :10]
    rankings = {u: [items[i] for i in row] for u, row in zip(users, top)}
    return evaluate_accuracy(rankings, relevant, 10)


def train_trial(user_features, item_features, pairs, seen, mask, users, items,
                relevant, hidden, seed, epochs, output):
    torch.manual_seed(seed)
    model = PairMLP(user_features, item_features, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    rng = np.random.default_rng(seed)
    best, history = None, []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(pairs))
        negatives = sample_negatives(rng, pairs[:, 0], seen)
        loss_sum = 0.0
        model.train()
        for start in range(0, len(order), 2048):
            indices = order[start:start + 2048]
            u, positive = torch.from_numpy(pairs[indices].T.copy())
            negative = torch.from_numpy(negatives[indices])
            optimizer.zero_grad(set_to_none=True)
            # Same pairwise objective as BPR; one resampled negative per positive.
            loss = F.softplus(model(u, negative) - model(u, positive)).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            scores = model.full_scores()
            metrics = evaluate(scores, mask, users, items, relevant)
            ndcg = metrics["aggregate"]["NDCG@10"]
            history.append({"epoch": epoch, "loss": loss_sum / len(pairs), **metrics["aggregate"]})
            if best is None or ndcg > best["metrics"]["aggregate"]["NDCG@10"]:
                best = {"epoch": epoch, "metrics": metrics, "scores": scores.copy(),
                        "state": copy.deepcopy(model.state_dict())}
            print(f"  epoch={epoch} loss={loss_sum / len(pairs):.4f} NDCG@10={ndcg:.6f}", flush=True)
            write_json(output / "history.json", history)
    torch.save({"state_dict": best.pop("state"), "hidden": hidden, "seed": seed,
                "epoch": best["epoch"]}, output / "model.pt")
    np.savez_compressed(output / "scores-valid.npz", scores=best.pop("scores"))
    write_json(output / "metrics.json", best["metrics"])
    return {"seed": seed, "hidden": hidden, "selected_epoch": best["epoch"],
            "directory": output.name, **best["metrics"]["aggregate"]}


def run(source, output, seeds=(42, 43, 44), widths=(64, 128), epochs=40):
    from recbole.data import create_dataset

    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    source, output = Path(source).resolve(), new_run(output)
    training = read_history(source / "splits/train.csv")
    relevant = read_history(source / "splits/valid.csv")
    pointer = json.loads((source / "cache-bpr.json").read_text())
    cache = Path(pointer["path"])
    checkpoint_paths = list((cache / "full-training").glob("*.pth"))
    if len(checkpoint_paths) != 1:
        raise ValueError("Expected exactly one full-training BPR checkpoint")
    checkpoint_path = checkpoint_paths[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset = create_dataset(checkpoint["config"])
    users = sorted(relevant)
    items = sorted(str(i) for i in dataset.field2id_token[dataset.iid_field][1:])
    uid, iid = {u: n for n, u in enumerate(users)}, {i: n for n, i in enumerate(items)}
    if set(training) != set(users):
        raise ValueError("Training/validation users differ")
    if any(training[u] & relevant[u] for u in users):
        raise ValueError("Training/validation overlap")
    pairs = np.array([(uid[u], iid[i]) for u in users for i in sorted(training[u])], dtype=np.int64)
    seen = np.zeros((len(users), len(items)), dtype=bool)
    seen[pairs[:, 0], pairs[:, 1]] = True
    mask = ~seen
    if np.any(seen.all(axis=1)):
        raise ValueError("A user has no negative candidates")
    reference_path = source / "scores-valid-bpr.json.gz"
    reference = load_scores(reference_path)
    if set(reference.scores) != set(users):
        raise ValueError("Reference users differ")
    for u in users:
        if set(reference.scores[u]) != set(items) - training[u] or not relevant[u] <= reference.scores[u].keys():
            raise ValueError("Candidate mask or validation positives differ")
    internal_users = dataset.token2id(dataset.uid_field, users)
    internal_items = dataset.token2id(dataset.iid_field, items)
    p = checkpoint["state_dict"]["user_embedding.weight"][internal_users].numpy()
    q = checkpoint["state_dict"]["item_embedding.weight"][internal_items].numpy()
    baseline_scores = p @ q.T
    # Prove that token mapping and checkpoint match the existing score cache.
    max_error = max(float(np.max(np.abs(baseline_scores[uid[u], [iid[i] for i in row]] - np.array(list(row.values())))))
                    for u, row in reference.scores.items())
    if max_error > 1e-5:
        raise ValueError(f"Checkpoint/cache score mismatch: {max_error}")
    baseline = evaluate(baseline_scores, mask, users, items, relevant)
    np.savez_compressed(output / "scores-valid-mf.npz", scores=baseline_scores)
    write_json(output / "metrics-mf.json", baseline)
    metadata = REPO_ROOT / "dataset/ml-100k/ml-100k.item"
    genres, years = read_genres(metadata), read_years(metadata)
    vocabulary = sorted(set().union(*(genres[i] for i in items)))
    genre_features = np.array([[float(g in genres[i]) for g in vocabulary] for i in items], dtype=np.float32)
    year_values, year_scaler = year_feature(years, set().union(*training.values()), items)
    p_scaled, p_scaler = standardize(p, np.unique(pairs[:, 0]))
    q_scaled, q_scaler = standardize(q, np.unique(pairs[:, 1]))
    content_items = np.column_stack([q_scaled, genre_features, year_values]).astype(np.float32)
    write_json(output / "mapping.json", {"users": users, "items": items, "genres": vocabulary,
                                        "split_id": reference.split_id,
                                        "score_files": "dense scores; apply candidate-mask.npz eligible mask"})
    np.savez_compressed(output / "candidate-mask.npz", eligible=mask)
    write_json(output / "scalers.json", {"user_factors": p_scaler, "item_factors": q_scaler, "year": year_scaler})
    proof = {"status": "running", "split_id": reference.split_id, "test_evaluated": False,
             "test_split_read": False, "frozen_embeddings": True,
             "dataset_access": "create_dataset for original checkpoint token mapping; no build/split or fitting on its interactions",
             "training_pairs": len(pairs), "users": len(users), "items": len(items),
             "selection_metric": "validation NDCG@10", "seeds": list(seeds), "widths": list(widths),
             "max_epochs": epochs, "evaluate_every": 5,
             "architecture": "concat -> Linear(h) -> ReLU -> Linear(h/2) -> ReLU -> Linear(1)",
             "optimizer": {"name": "Adam", "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 2048},
             "loss": "pairwise softplus(negative score - positive score)",
             "negative_sampling": "one uniform catalogue item per positive per epoch, reject training positives only; paired across variants",
             "selection": "best epoch per trial; width selected by mean validation NDCG across seeds; ties prefer earlier epoch/smaller width",
             "baseline_checkpoint_seed": checkpoint["config"]["seed"],
             "baseline_checkpoint_epoch": checkpoint["epoch"], "baseline_cache_max_abs_error": max_error,
             "sha256": {"source": file_hash(__file__), "checkpoint": file_hash(checkpoint_path),
                        "train": file_hash(source / "splits/train.csv"), "valid": file_hash(source / "splits/valid.csv"),
                        "metadata": file_hash(metadata), "reference": file_hash(reference_path),
                        "raw_catalogue_mapping_source": file_hash(REPO_ROOT / "dataset/ml-100k/ml-100k.inter"),
                        "accuracy": file_hash(Path(__file__).parent / "metrics/accuracy.py")},
             "checkpoint": str(checkpoint_path), "torch_version": torch.__version__, "threads": 4}
    write_json(output / "manifest.json", proof)
    trials = []
    user_tensor = torch.from_numpy(p_scaled)
    for variant, features in (("mlp", q_scaled), ("mlp-content", content_items)):
        for hidden in widths:
            for seed in seeds:
                name = f"{variant}-h{hidden}-seed{seed}"
                trial_output = output / name
                trial_output.mkdir()
                print(name, flush=True)
                result = train_trial(user_tensor, torch.from_numpy(features), pairs, seen, mask,
                                     users, items, relevant, hidden, seed, epochs, trial_output)
                trials.append({"variant": variant, **result})
                write_json(output / "trials.json", trials)
    results = {"mf": {"aggregate": baseline["aggregate"], "seed_count": 1}}
    selected_rows = {}
    for variant in ("mlp", "mlp-content"):
        means = {h: float(np.mean([t["NDCG@10"] for t in trials if t["variant"] == variant and t["hidden"] == h])) for h in widths}
        selected_width = max(sorted(widths), key=lambda h: means[h])
        selected = [t for t in trials if t["variant"] == variant and t["hidden"] == selected_width]
        selected_rows[variant] = selected
        metrics = baseline["aggregate"].keys()
        results[variant] = {"selected_width": selected_width, "width_mean_ndcg": means, "seed_count": len(seeds),
                            "aggregate": {m: float(np.mean([t[m] for t in selected])) for m in metrics},
                            "std": {m: float(np.std([t[m] for t in selected], ddof=1)) if len(seeds) > 1 else 0.0 for m in metrics},
                            "trials": selected}
    paired = []
    for seed in seeds:
        a, b = [next(t for t in selected_rows[v] if t["seed"] == seed) for v in ("mlp", "mlp-content")]
        paired.append({"seed": seed, **{m: b[m] - a[m] for m in baseline["aggregate"]}})
    results["paired_content_minus_mlp"] = paired
    write_json(output / "results.json", results)
    lines = ["# Feature combination: validation comparison", "",
             f"Frozen {p.shape[1]}-dimensional BPR user/item embeddings. Content adds {len(vocabulary)} multi-hot genres and one training-standardized release year. No explicit content user profile.", "",
             "Same 60/20/20 split and full-catalogue validation masks; all held-out interactions relevant; external token ties. Test split not opened or evaluated.", "",
             "MLPs use two ReLU hidden layers, BPR pairwise loss, Adam lr=0.001, weight decay=0.0001, batch size 2048. Uniform negatives exclude training positives only. Training order/negatives are paired across variants.", "",
             f"Widths {list(widths)}, seeds {list(seeds)}, up to {epochs} epochs, validation every five epochs. Select epoch per trial, then width by mean validation NDCG@10. Report mean and sample standard deviation across MLP seeds. The MF baseline is one shared pretrained checkpoint.", "",
             "| Model | Recall@10 | MRR@10 | NDCG@10 |", "|---|---:|---:|---:|"]
    for variant in ("mf", "mlp", "mlp-content"):
        result = results[variant]
        cells = [f"{result['aggregate'][m]:.6f}" + (f" +/- {result['std'][m]:.6f}" if "std" in result else "")
                 for m in ("Recall@10", "MRR@10", "NDCG@10")]
        lines.append("| " + variant + " | " + " | ".join(cells) + " |")
    lines += ["", "## Selected trials", "", "| Variant | Width | Seed | Epoch | NDCG@10 |", "|---|---:|---:|---:|---:|"]
    for variant, rows in selected_rows.items():
        lines += [f"| {variant} | {t['hidden']} | {t['seed']} | {t['selected_epoch']} | {t['NDCG@10']:.6f} |" for t in rows]
    lines += ["", "## Interpretation limits", "",
              "Development results: validation selected both checkpoints and width. Seed variation covers the MLP only, not MF retraining or split variation. No independent significance claim. MF embeddings are fixed, so the content model can exploit only user information retained by MF. MLP vs dot product also changes the scorer capacity and training stage; it does not isolate nonlinearity alone.", "",
              "Dense candidate score matrices, masks, external-ID mappings, fitted feature scalers, selected trial checkpoints, histories and per-user metrics are retained for audit/reuse. Each trial score file contains the selected epoch, with ineligible scores excluded using the shared mask."]
    (output / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    proof.update(status="complete", elapsed_seconds=time.monotonic() - started,
                 candidate_alignment_verified=True, selected_widths={v: results[v]["selected_width"] for v in selected_rows})
    write_json(output / "manifest.json", proof)
    print(json.dumps({v: results[v]["aggregate"] for v in ("mf", "mlp", "mlp-content")}, indent=2), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="project/artifacts/hybrid-60-20-20")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    print(run(args.source, args.output, epochs=args.epochs))
