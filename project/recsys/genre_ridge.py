"""Per-user Ridge genre rating predictors; validation-only alpha selection."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .artifacts import REPO_ROOT, new_run, write_json
from .component_cache import file_hash
from .content_based import (
    cohort_metrics,
    read_genres,
    read_history,
    read_training_ratings,
)
from .contracts import ScoreTable, require_aligned
from .metrics.accuracy import evaluate_accuracy
from .recbole_adapter import _write_scores, load_scores

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def read_years(path):
    years = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            token = row["release_year:token"].strip()
            years[row["item_id:token"]] = int(token) if token.isdigit() else None
    return years


def year_feature(years, training_items, catalogue):
    """Fit one year z-score on distinct training items; missing -> training mean."""
    known = [years[i] for i in sorted(training_items) if years.get(i) is not None]
    if not known:
        raise ValueError("No known training release years")
    mean, scale = float(np.mean(known)), float(np.std(known))
    scale = scale or 1.0
    values = np.array([(years[i] - mean) / scale if years.get(i) is not None else 0.0
                       for i in catalogue])
    return values, {"mean": mean, "scale": scale, "known_training_items": len(known),
                    "missing_catalogue_items": [i for i in catalogue if years.get(i) is None],
                    "missing_policy": "impute training-item mean; standardized value zero"}


def fit_user(x, y, alpha):
    """Minimize sum squared rating errors + alpha * squared coefficient norm.

    Intercept is unpenalized; features are raw genre indicators, not normalized.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if len(y) == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Supply finite nonempty training data")
    mean_x, mean_y = x.mean(axis=0), y.mean()
    centered_x, centered_y = x - mean_x, y - mean_y
    weights = np.linalg.solve(centered_x.T @ centered_x + alpha * np.eye(x.shape[1]),
                              centered_x.T @ centered_y)
    return weights, float(mean_y - mean_x @ weights)


def score_ridge(training, genres, ratings, reference, alpha, years=None):
    reference.validate()
    if reference.partition != "valid":
        raise ValueError("Validation reference required")
    catalogue = sorted(set().union(*training.values(), *(set(r) for r in reference.scores.values())))
    if not set(catalogue) <= genres.keys():
        raise ValueError("Missing metadata")
    vocabulary = sorted(set().union(*(genres[i] for i in catalogue)))
    indices = {item: j for j, item in enumerate(catalogue)}
    x = np.array([[float(g in genres[i]) for g in vocabulary] for i in catalogue])
    scaler = None
    if years is not None:
        values, scaler = year_feature(years, set().union(*training.values()), catalogue)
        x = np.column_stack([x, values])
    counts = Counter(i for seen in training.values() for i in seen)
    models, scores = {}, {}
    for user, candidates in reference.scores.items():
        seen = sorted(training.get(user, set()))
        if set(candidates) != set(catalogue) - set(seen):
            raise ValueError("Candidate mask differs from training history")
        if seen:
            y = np.array([ratings[user, i] for i in seen])
            if not np.isfinite(y).all() or np.any(y < 1) or np.any(y > 5):
                raise ValueError("Raw training ratings must be in 1-5")
            weights, intercept = fit_user(x[[indices[i] for i in seen]], y, alpha)
            values = x @ weights + intercept
            models[user] = {"intercept": intercept, "coefficients": weights.tolist()}
        else:
            values = np.array([float(counts[i]) for i in catalogue])
            models[user] = {"fallback": "training popularity"}
        scores[user] = {i: float(values[indices[i]]) for i in candidates}
    label = "content-genre-year-ridge" if years is not None else "content-genre-ridge"
    table = ScoreTable(label, reference.split_id, "valid", scores)
    require_aligned([table, reference])
    return table, {"genres": vocabulary, "alpha": alpha, "users": models,
                   "feature_order": vocabulary + (["release_year_z"] if years is not None else []),
                   "year_scaler": scaler}, counts


def run(source, output, include_year=False):
    source, output = Path(source).resolve(), new_run(output)
    metadata = REPO_ROOT / "dataset/ml-100k/ml-100k.item"
    raw = REPO_ROOT / "dataset/ml-100k/ml-100k.inter"
    training, relevant = [read_history(source / f"splits/{part}.csv") for part in ("train", "valid")]
    reference_path = source / "scores-valid-itemknn.json.gz"
    reference = load_scores(reference_path)
    manifest = json.loads((source / "manifest.json").read_text())
    if reference.split_id != manifest["split_id"] or reference.scores.keys() != relevant.keys():
        raise ValueError("Reference split/users differ")
    if any(training.get(u, set()) & truth or not truth <= reference.scores[u].keys()
           for u, truth in relevant.items()):
        raise ValueError("Validation overlap or absent positives")
    hashes = {"metadata": file_hash(metadata), "train": file_hash(source / "splits/train.csv"),
              "valid": file_hash(source / "splits/valid.csv"), "candidate_reference": file_hash(reference_path),
              "accuracy": file_hash(Path(__file__).parent / "metrics/accuracy.py")}
    baselines = {}
    for variant in ("v1", "v3", "v4"):
        path = source.parent / f"content-genres-{variant}"
        previous = json.loads((path / "manifest.json").read_text())
        if previous["status"] != "complete" or any(previous["sha256"][key] != val for key, val in hashes.items()):
            raise ValueError(f"Baseline inputs changed: {variant}")
        baselines[variant] = load_scores(path / "scores-valid.json.gz")
        require_aligned([reference, baselines[variant]])
    if include_year:
        path = source.parent / "content-genre-ridge"
        previous = json.loads((path / "manifest.json").read_text())
        if previous["status"] != "complete" or any(previous["sha256"][key] != val for key, val in hashes.items()):
            raise ValueError("Genre Ridge baseline inputs changed")
        baselines["genre-ridge"] = load_scores(path / "scores-valid.json.gz")
        require_aligned([reference, baselines["genre-ridge"]])
    label = "content-genre-year-ridge" if include_year else "content-genre-ridge"
    feature_description = "raw multi-hot genres" + (" + training-standardized year" if include_year else "; no year")
    proof = {"status": "running", "model": label, "split_id": reference.split_id,
             "test_evaluated": False, "selection_metric": "NDCG@10", "alpha_grid": ALPHAS,
             "objective": "sum squared rating errors + alpha * sum squared feature coefficients",
             "intercept": "per-user, unpenalized", "features": feature_description + "; no IDF",
             "tie_break": "external item ID; equal validation scores select larger alpha",
             "sha256": {**hashes, "source": file_hash(__file__), "raw_ratings": file_hash(raw),
                        "content_helpers": file_hash(Path(__file__).parent / "content_based.py")}}
    write_json(output / "manifest.json", proof)
    ratings, genres = read_training_ratings(raw, training), read_genres(metadata)
    years = read_years(metadata) if include_year else None
    trials, best = [], None
    for alpha in ALPHAS:
        table, fitted, counts = score_ridge(training, genres, ratings, reference, alpha, years)
        metrics = evaluate_accuracy(table.top_k(10), relevant, 10)
        score = metrics["aggregate"]["NDCG@10"]
        trials.append({"alpha": alpha, **metrics["aggregate"]})
        print(f"alpha={alpha:g}: validation NDCG@10={score:.6f}", flush=True)
        if best is None or (score, alpha) > best[:2]:
            best = (score, alpha, table, fitted)
        write_json(output / "trials.json", trials)
    _, alpha, table, fitted = best
    _write_scores(table, output / "scores-valid.json.gz")
    write_json(output / "model.json", fitted)
    popularity = ScoreTable("popularity", reference.split_id, "valid",
                            {u: {i: float(counts[i]) for i in row} for u, row in reference.scores.items()})
    results = {}
    for name, candidate in {"ridge": table, **baselines, "popularity": popularity, "itemknn": reference}.items():
        rankings = candidate.top_k(10)
        results[name] = {"accuracy": evaluate_accuracy(rankings, relevant, 10),
                         "item_history_cohorts": cohort_metrics(rankings, relevant, counts, 10)}
    write_json(output / "metrics.json", results)
    # Exact ties at the K boundary: count tied candidates excluded by fixed ID order.
    boundary_ties = []
    for user, ranking in table.top_k(10).items():
        boundary = table.scores[user][ranking[-1]]
        equal = sum(v == boundary for v in table.scores[user].values())
        selected_equal = sum(table.scores[user][i] == boundary for i in ranking)
        boundary_ties.append(equal - selected_equal)
    patterns = Counter(tuple(sorted(genres[i])) for i in set().union(*(set(r) for r in reference.scores.values())))
    diagnostics = {"genre_patterns": len(patterns), "users_with_ties_crossing_top10": sum(n > 0 for n in boundary_ties),
                   "mean_extra_tied_candidates_at_boundary": float(np.mean(boundary_ties)),
                   "negative_coefficient_count": sum(w < 0 for m in fitted["users"].values()
                                                     for w in m.get("coefficients", [])[:len(fitted['genres'])])}
    if include_year:
        catalogue = set().union(*(set(r) for r in reference.scores.values()))
        diagnostics["genre_year_patterns"] = len({(tuple(sorted(genres[i])), years.get(i)) for i in catalogue})
        diagnostics["year_scaler"] = fitted["year_scaler"]
    write_json(output / "diagnostics.json", diagnostics)
    lines = [f"# Regularized {'genre + year' if include_year else 'genre'} rating predictor: validation", "",
             (f"Per-user Ridge; {feature_description}; unpenalized intercept; selected alpha {alpha:g} "
             "by validation NDCG@10. Scores are predicted ratings, without cosine or clipping. "
             "Profiles use raw training ratings only. No IDF or external metadata. Test split not read."), "",
             "| Model | Recall@10 | MRR@10 | NDCG@10 |", "|---|---:|---:|---:|"]
    for name, result in results.items():
        row = result["accuracy"]["aggregate"]
        lines.append("| " + name + " | " + " | ".join(f"{row[m+'@10']:.4f}" for m in ("Recall", "MRR", "NDCG")) + " |")
    lines += ["", "## Alpha search", "", "| Alpha | Validation NDCG@10 |", "|---:|---:|"]
    lines += [f"| {row['alpha']:g} | {row['NDCG@10']:.6f} |" for row in trials]
    cold = results["ridge"]["item_history_cohorts"]["zero"]
    lines += ["", (f"Cold-item retrieval: {cold['hits']}/{cold['positive_interactions']} held-out interactions. "
              "This small random-split cohort does not establish cold-start effectiveness."), "",
              (f"Distinct genre patterns: {diagnostics['genre_patterns']}. Users with exact ties crossing top-10: "
              f"{diagnostics['users_with_ties_crossing_top10']}/{len(relevant)}. "
              "Identical input feature vectors necessarily receive identical predictions for a user."), "",
              ("These are development results used to select alpha, not an independent final evaluation. "
              "Failure of a genre-only rating predictor cannot establish that content-based recommendation is impossible. "
              "The evaluation target is held-out interactions; Ridge optimizes explicit rating error, a different objective.")]
    (output / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    proof.update(status="complete", selected_alpha=alpha, candidate_alignment_verified=True,
                 training_rating_pairs=len(ratings))
    write_json(output / "manifest.json", proof)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="project/artifacts/hybrid-60-20-20")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-year", action="store_true")
    args = parser.parse_args()
    print(run(args.source, args.output, args.include_year))
