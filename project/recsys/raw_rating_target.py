"""Restore the agreed 1-5 regression target from original external-ID ratings.

RecBole's MovieLens defaults normalize float fields, including ratings. This
post-fit correction preserves all component caches and the frozen Ridge alpha.
With an intercept, replacing y by 4*y+1 scales Ridge coefficients by four and
changes the intercept to 4*b+1, leaving the ranking objective unchanged.
"""

import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import REPO_ROOT
from .component_cache import atomic_json, file_hash
from .contracts import ScoreTable
from .hybrids import fit_oof_regression
from .recbole_adapter import load_scores, _write_scores


def raw_targets(pairs, observed, source):
    with Path(source).open(newline="") as stream:
        ratings = {(row["user_id:token"], row["item_id:token"]): float(row["rating:float"])
                   for row in csv.DictReader(stream, delimiter="\t")}
    target = np.array([ratings[(row["user"], row["item"])] for row in pairs])
    if not (np.allclose(observed, target) or np.allclose(observed, (target - 1) / 4)):
        raise ValueError("OOF targets do not align with the original raw rating pairs")
    return target


def restore_raw_rating_target(output, k=10):
    output = Path(output)
    source = REPO_ROOT / "dataset/ml-100k/ml-100k.inter"
    pairs = json.loads((output / "folds.json").read_text())
    features_path = output / "oof-progress.npz"
    backup = output / "oof-normalized-rating-targets.npz"
    hybrid_path = output / "hybrid.json"
    hybrid_backup = output / "hybrid-normalized-rating-targets.json"
    cache_keys = {name: json.loads(path.read_text())["key"]
                  for path in output.glob("cache-*.json")
                  if (name := path.stem.removeprefix("cache-")) != "run"}
    proof_path = output / "raw-rating-target.json"
    if proof_path.exists():
        previous = json.loads(proof_path.read_text())
        if "cache_keys" in previous and previous["cache_keys"] != cache_keys:
            raise ValueError("Component caches changed; use a new target-correction output")
    with np.load(backup if backup.exists() else features_path, allow_pickle=False) as saved:
        features, observed, folds = [saved[name].copy() for name in ("features", "targets", "folds")]
    if backup.exists():
        with np.load(features_path, allow_pickle=False) as current_arrays:
            if (not np.array_equal(features, current_arrays["features"])
                    or not np.array_equal(folds, current_arrays["folds"])):
                raise ValueError("OOF features changed; do not reuse the target-correction backup")
            raw_targets(pairs, current_arrays["targets"], source)
    if not np.array_equal(folds, [row["fold"] for row in pairs]):
        raise ValueError("OOF fold mapping changed")
    target = raw_targets(pairs, observed, source)
    current = json.loads(hybrid_path.read_text())
    if np.allclose(observed, target):
        if current["target"] != "raw held-out rating":
            raise ValueError("Unexpected target definition")
        return
    normalized = json.loads((hybrid_backup if hybrid_backup.exists() else hybrid_path).read_text())
    if current["alpha"] != normalized["alpha"]:
        raise ValueError("Ridge alpha changed after validation selection")
    fitted = fit_oof_regression(features, target, normalized["components"], normalized["alpha"])
    np.testing.assert_allclose(fitted["coefficients"], 4 * np.asarray(normalized["coefficients"]), rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(fitted["intercept"], 4 * normalized["intercept"] + 1, rtol=1e-8, atol=1e-9)
    if not backup.exists():
        backup.write_bytes(features_path.read_bytes())
        hybrid_backup.write_bytes(hybrid_path.read_bytes())
    # If recovering after a partial correction, always transform the original
    # normalized scores rather than multiplying already corrected scores again.
    valid_path = output / "scores-valid-hybrid.json.gz"
    valid_backup = output / "scores-valid-hybrid-normalized-targets.json.gz"
    if not valid_backup.exists():
        valid_backup.write_bytes(valid_path.read_bytes())
    old = load_scores(valid_backup)
    corrected = ScoreTable(old.model, old.split_id, old.partition,
                           {u: {i: 4 * value + 1 for i, value in row.items()} for u, row in old.scores.items()})
    if old.top_k(k) != corrected.top_k(k):
        raise ValueError("Target-unit correction unexpectedly changed validation rankings")
    temporary = output / "oof-raw-targets.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, features=features, targets=target, folds=folds)
    temporary.replace(features_path)
    score_temp = output / "scores-valid-hybrid-raw-targets.tmp.gz"
    _write_scores(corrected, score_temp)
    score_temp.replace(valid_path)
    atomic_json(hybrid_path, fitted)
    atomic_json(output / "raw-rating-target.json", {
        "cache_keys": cache_keys,
        "original_rating_source_sha256": file_hash(source),
        "source_code_sha256": file_hash(__file__),
        "fold_mapping_sha256": file_hash(output / "folds.json"),
        "original_normalized_oof_sha256": file_hash(backup),
        "raw_rating_min": float(target.min()), "raw_rating_max": float(target.max()),
        "training_rows": len(target), "target_transform": "raw_rating = 4 * normalized_rating + 1",
        "alpha_changed": False, "validation_top_k_unchanged": True,
        "coefficient_scaling_verified_against_direct_raw_target_refit": True,
    })
