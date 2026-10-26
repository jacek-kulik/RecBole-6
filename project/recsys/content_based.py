"""Genre-only V1-V4: training-history genre profiles and cosine scoring.

Validation only. Reuses the exact candidate sets of an existing component run.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .artifacts import REPO_ROOT, new_run, write_json
from .component_cache import file_hash
from .contracts import ScoreTable, require_aligned
from .metrics.accuracy import evaluate_accuracy
from .recbole_adapter import _write_scores, load_scores


def read_history(path):
    history = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            user, item = row["user_id"], row["item_id"]
            items = history.setdefault(user, set())
            if item in items:
                raise ValueError("Duplicate user-item interaction")
            items.add(item)
    return history


def read_genres(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return {row["item_id:token"]: set(row["class:token_seq"].split()) - {"unknown"}
                for row in csv.DictReader(stream, delimiter="\t")}


def read_training_ratings(path, training):
    """Recover raw 1-5 ratings by external IDs, retaining training pairs only."""
    ratings = {}
    wanted = {(u, i) for u, seen in training.items() for i in seen}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            pair = (row["user_id:token"], row["item_id:token"])
            if pair in wanted:
                if pair in ratings:
                    raise ValueError("Duplicate training rating in original interactions")
                value = float(row["rating:float"])
                if not np.isfinite(value) or not 1 <= value <= 5:
                    raise ValueError("Expected finite raw 1-5 training ratings")
                ratings[pair] = value
    if ratings.keys() != wanted:
        raise ValueError("Original interactions missing training ratings")
    return ratings


def score_v1(training, genres, reference):
    return score_content(training, genres, reference, variant="v1")


def score_content(training, genres, reference, variant="v1", ratings=None):
    """Ignore reference score values; reuse only its users/candidate IDs/split ID.

    V1 sums raw multi-hot vectors; V2 sums L2-normalized item vectors.
    V3 weights genre dimensions by IDF fitted on distinct training items only,
    before summing; no per-item normalization before aggregation.
    V4 sums raw multi-hot vectors weighted by the raw 1-5 training rating.
    Zero profiles or candidates without genres use training item popularity.
    A missing candidate genre receives a score below every cosine score; within
    that fallback tier, popularity determines ordering. Zero-profile users use
    popularity across their entire candidate set.
    """
    if variant not in {"v1", "v2", "v3", "v4"}:
        raise ValueError("Unknown content variant")
    if variant == "v4" and (
        ratings is None or any((u, i) not in ratings or not np.isfinite(ratings[u, i])
                               or not 1 <= ratings[u, i] <= 5
                               for u, seen in training.items() for i in seen)
    ):
        raise ValueError("V4 requires raw 1-5 ratings for every training pair")
    reference.validate()
    if reference.partition != "valid":
        raise ValueError("This runner accepts validation candidates only")
    catalogue = set().union(*training.values(), *[set(r) for r in reference.scores.values()])
    if not catalogue <= genres.keys():
        raise ValueError("Catalogue items missing from metadata")
    vocabulary = sorted(set().union(*(genres[i] for i in catalogue)))
    if not vocabulary:
        raise ValueError("No usable genre features")
    items = sorted(catalogue)
    indices = {item: index for index, item in enumerate(items)}
    features = np.array([[float(g in genres[i]) for g in vocabulary] for i in items])
    idf_info = {}
    if variant == "v3":
        training_items = sorted(set().union(*training.values()))
        if not training_items:
            raise ValueError("IDF requires a nonempty training-item catalogue")
        frequencies = features[[indices[i] for i in training_items]].sum(axis=0)
        weights = np.log((1 + len(training_items)) / (1 + frequencies)) + 1
        features = features * weights
        idf_info = {"idf": {g: float(w) for g, w in zip(vocabulary, weights)},
                    "genre_training_item_counts": {g: int(n) for g, n in zip(vocabulary, frequencies)},
                    "idf_training_items": len(training_items),
                    "idf_formula": "1 + ln((1 + N_training_items) / (1 + genre_training_items))"}
    lengths = np.linalg.norm(features, axis=1)
    normalized = np.divide(features, lengths[:, None], out=np.zeros_like(features),
                           where=lengths[:, None] != 0)
    counts = Counter(i for seen in training.values() for i in seen)
    popularity = np.array([counts[i] / max(1, len(training)) for i in items])
    scores, fallback_users = {}, []
    for user, candidates in reference.scores.items():
        seen = training.get(user, set())
        if set(candidates) != catalogue - seen:
            raise ValueError("Candidates must be exactly catalogue minus training history")
        profile_features = normalized if variant == "v2" else features
        history_items = sorted(seen)
        history_features = profile_features[[indices[i] for i in history_items]]
        if variant == "v4":
            history_features = history_features * np.array([ratings[user, i] for i in history_items])[:, None]
        profile = history_features.sum(axis=0)
        norm = np.linalg.norm(profile)
        if norm:
            values = normalized @ (profile / norm)
            values[lengths == 0] = -2 + popularity[lengths == 0]
        else:
            values = popularity.copy()
            fallback_users.append(user)
        scores[user] = {i: float(values[indices[i]]) for i in candidates}
    result = ScoreTable(f"content-genres-{variant}", reference.split_id, "valid", scores)
    require_aligned([reference, result])
    return result, counts, {"genres": vocabulary, "catalogue_items": len(items),
                            "items_without_genres": sum(lengths == 0).item(),
                            "fallback_users": fallback_users, **idf_info}


def cohort_metrics(rankings, relevant, counts, k):
    """Conditional retrieval metrics on unchanged full-catalogue top-K lists."""
    result = {}
    for name, lower, upper in (("zero", 0, 0), ("1-5", 1, 5),
                               ("6-20", 6, 20), ("21+", 21, float("inf"))):
        positives = {u: {i for i in truth if lower <= counts[i] <= upper}
                     for u, truth in relevant.items()}
        positives = {u: truth for u, truth in positives.items() if truth}
        hits = sum(len(set(rankings[u]) & truth) for u, truth in positives.items())
        total = sum(map(len, positives.values()))
        result[name] = {
            "users": len(positives), "positive_interactions": total,
            "distinct_positive_items": len(set().union(*positives.values())),
            "hits": hits, "micro_recall": hits / total if total else None,
            "metrics": evaluate_accuracy({u: rankings[u] for u in positives}, positives, k)["aggregate"]
            if positives else None,
        }
    return result


def run(source, metadata, output, k=10, variant="v1", baseline=None):
    source, metadata = Path(source).resolve(), Path(metadata).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    training = read_history(source / "splits/train.csv")
    relevant = read_history(source / "splits/valid.csv")
    if any(training.get(u, set()) & truth for u, truth in relevant.items()):
        raise ValueError("Training and validation overlap")
    reference_path = source / "scores-valid-itemknn.json.gz"
    reference = load_scores(reference_path)
    if reference.split_id != manifest["split_id"] or reference.scores.keys() != relevant.keys():
        raise ValueError("Reference split/users differ")
    if any(not truth <= reference.scores[u].keys() for u, truth in relevant.items()):
        raise ValueError("Validation positives absent from candidate sets")
    output = new_run(output)
    proof = {"model": f"content-genres-{variant}", "status": "running", "split_id": reference.split_id,
             "k": k, "test_evaluated": False, "training_ratings_used": variant == "v4",
             "normalize_items_before_sum": variant == "v2",
             "genre_idf": variant == "v3",
             "source_run": str(source), "metadata": str(metadata),
             "sha256": {"source": file_hash(__file__), "metadata": file_hash(metadata),
                        "train": file_hash(source / "splits/train.csv"),
                        "valid": file_hash(source / "splits/valid.csv"),
                        "candidate_reference": file_hash(reference_path),
                        "accuracy": file_hash(Path(__file__).parent / "metrics/accuracy.py")}}
    write_json(output / "manifest.json", proof)
    ratings = None
    if variant == "v4":
        rating_source = REPO_ROOT / "dataset/ml-100k/ml-100k.inter"
        ratings = read_training_ratings(rating_source, training)
        proof["rating_weights"] = "raw 1-5; no centering; training pairs only"
        proof["sha256"]["raw_rating_source"] = file_hash(rating_source)
        proof["training_rating_pairs"] = len(ratings)
        write_json(output / "manifest.json", proof)
    table, counts, diagnostics = score_content(training, read_genres(metadata), reference, variant, ratings)
    popularity = ScoreTable("popularity", table.split_id, "valid",
                            {u: {i: float(counts[i]) for i in row} for u, row in table.scores.items()})
    results = {}
    models = [(f"content-{variant}", table), ("popularity", popularity), ("itemknn", reference)]
    if baseline is not None:
        baseline = Path(baseline).resolve()
        previous = json.loads((baseline / "manifest.json").read_text())
        if previous["model"] != "content-genres-v1" or previous["k"] != k:
            raise ValueError("Baseline must be V1 with identical cutoff")
        for key in ("metadata", "train", "valid", "candidate_reference", "accuracy"):
            if previous["sha256"][key] != proof["sha256"][key]:
                raise ValueError(f"Baseline input changed: {key}")
        baseline_table = load_scores(baseline / "scores-valid.json.gz")
        require_aligned([table, baseline_table])
        models.append(("content-v1", baseline_table))
        proof["baseline"] = {"directory": str(baseline),
                             "scores_sha256": file_hash(baseline / "scores-valid.json.gz")}
    for name, model in models:
        rankings = model.top_k(k)
        results[name] = {"accuracy": evaluate_accuracy(rankings, relevant, k),
                         "item_history_cohorts": cohort_metrics(rankings, relevant, counts, k)}
    _write_scores(table, output / "scores-valid.json.gz")
    write_json(output / "metrics.json", results)
    write_json(output / "diagnostics.json", diagnostics)
    description = {
        "v1": "Raw multi-hot vectors summed into user profiles. No IDF or item normalization before summing.",
        "v2": "L2-normalized movie genre vectors summed into user profiles; only change from V1. No IDF.",
        "v3": "Genre vectors weighted by training-item smoothed IDF before summing and cosine scoring; "
              "only change from V1. No item normalization before summing. "
              "IDF(g) = 1 + ln((1 + N_training_items) / (1 + genre_training_items)).",
        "v4": "Raw multi-hot vectors weighted by each movie's raw 1-5 training rating before summing; "
              "only change from V1. No IDF, centering, or item normalization before summing.",
    }[variant]
    lines = [f"# Content-based {variant.upper()} validation", "",
             description + " Cosine scoring; no validation-fitted parameters.", "",
             ("Exact saved 60/20/20 split and ItemKNN validation candidate masks. All held-out "
             "interactions are relevant. Top-K ties use external item IDs. Test split was not read; "
             "held-out ratings are not used in profiles or evaluation relevance."), "",
             (f"Users: {len(relevant)}. Catalogue: {diagnostics['catalogue_items']}. "
             f"Genres: {len(diagnostics['genres'])}. Popularity fallback users: {len(diagnostics['fallback_users'])}."), "",
             "| Model | Precision@10 | Recall@10 | MRR@10 | NDCG@10 | MAP@10 |",
             "|---|---:|---:|---:|---:|---:|"]
    # Column labels use the actual requested cutoff.
    lines = [line.replace("@10", f"@{k}") for line in lines]
    for name, result in results.items():
        values = result["accuracy"]["aggregate"]
        lines.append("| " + name + " | " + " | ".join(f"{values[f'{m}@{k}']:.4f}"
                     for m in ("Precision", "Recall", "MRR", "NDCG", "MAP")) + " |")
    lines += ["", "## Item-history diagnostics", "",
              ("Bins are descriptive, fixed before inspecting results; they are not switching thresholds. "
              "Recall below counts only positives in the bin, using the unchanged full-catalogue rankings. "
              "The denominator is the number of validation interactions in that bin."), "",
              f"| Training count | Users | Positive interactions | Items | {variant.upper()} hits | {variant.upper()} micro recall | Popularity micro recall | ItemKNN micro recall |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cohort, row in results[f"content-{variant}"]["item_history_cohorts"].items():
        recalls = [results[m]["item_history_cohorts"][cohort]["micro_recall"]
                   for m in (f"content-{variant}", "popularity", "itemknn")]
        lines.append(f"| {cohort} | {row['users']} | {row['positive_interactions']} | "
                     f"{row['distinct_positive_items']} | {row['hits']} | "
                     + " | ".join("n/a" if v is None else f"{v:.4f}" for v in recalls) + " |")
    lines += ["", ("These are single-split development results, not evidence of statistical significance. "
              "A random interaction split is not a dedicated item cold-start experiment. "
              "The model can score unseen items, but any cold-start claim requires sufficient item-held-out evidence. "
              "Genre-only items can tie exactly; the fixed ID tie-break can affect top-K results.")]
    (output / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    proof.update(status="complete", diagnostics=diagnostics, candidate_alignment_verified=True)
    write_json(output / "manifest.json", proof)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="project/artifacts/hybrid-60-20-20")
    parser.add_argument("--metadata", default=str(REPO_ROOT / "dataset/ml-100k/ml-100k.item"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--variant", choices=("v1", "v2", "v3", "v4"), default="v1")
    parser.add_argument("--baseline", help="Existing V1 run for an input-verified comparison")
    args = parser.parse_args()
    print(run(args.source, args.metadata, args.output, args.k, args.variant, args.baseline))


if __name__ == "__main__":
    main()
