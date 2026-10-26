"""Bogdan: Tasks 1.3-1.5; Gabriel reviews 1.3, Caio reviews 1.4/1.5."""

import numpy as np
from sklearn.linear_model import Ridge

from .contracts import ScoreTable, require_aligned


def candidate_features(tables):
    """Z-score each component over each user's eligible candidates, without labels.

    Constant score vectors map to zero. The same rule applies to fold models and
    full-training models; candidate-relative transforms need no held-out targets.
    """
    require_aligned(tables)
    names = [table.model for table in tables]
    if len(set(names)) != len(names):
        raise ValueError("Component names must be unique")
    pairs, blocks = [], []
    for user in sorted(tables[0].scores):
        items = sorted(tables[0].scores[user])
        values = np.array([[t.scores[user][item] for t in tables] for item in items], dtype=float)
        deviation = values.std(axis=0)
        active = deviation > 1e-12
        block = np.zeros_like(values)
        block[:, active] = (values[:, active] - values[:, active].mean(axis=0)) / deviation[active]
        blocks.append(block)
        pairs.extend((user, item) for item in items)
    return pairs, np.concatenate(blocks)


def fit_oof_regression(features, ratings, component_names, alpha=1.0):
    """Fit raw rating targets; caller must supply genuinely out-of-fold features."""
    x, y = np.asarray(features, dtype=float), np.asarray(ratings, dtype=float)
    if (x.ndim != 2 or x.shape[1] != len(component_names) or len(x) < 2
            or y.shape != (len(x),) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isfinite(alpha) or alpha < 0):
        raise ValueError("Invalid regression data or ridge penalty")
    model = Ridge(alpha=alpha).fit(x, y)
    return {"components": list(component_names), "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_), "alpha": float(alpha),
            "target": "raw held-out rating", "transform": "per-user candidate z-score",
            "training_rows": len(y)}


def fit_weighted_hybrid(development_scores, targets, alpha=1.0):
    """Fit held-out scores against a mapping of (user, item) to raw rating.

    Caller guarantees components did not train on these targets. The OOF runner
    uses fit_oof_regression after concatenating features from distinct folds.
    Candidate scores may include unrated items; those are not zero targets.
    """
    require_aligned(development_scores)
    if any(table.partition == "test" for table in development_scores):
        raise ValueError("Test scores must not be used to fit a hybrid.")
    pairs, values = candidate_features(development_scores)
    lookup = {pair: index for index, pair in enumerate(pairs)}
    if not targets or any(pair not in lookup for pair in targets):
        raise ValueError("Every target must be an eligible scored user-item pair")
    selected = sorted(targets)
    return fit_oof_regression(values[[lookup[pair] for pair in selected]],
                              [targets[pair] for pair in selected],
                              [table.model for table in development_scores], alpha)


def predict_weighted_hybrid(component_scores, fitted_hybrid) -> ScoreTable:
    """Apply the saved coefficient order and candidate-relative score transform."""
    if [t.model for t in component_scores] != fitted_hybrid["components"]:
        raise ValueError("Component names/order differ from fitted hybrid")
    pairs, x = candidate_features(component_scores)
    predictions = x @ np.asarray(fitted_hybrid["coefficients"]) + fitted_hybrid["intercept"]
    scores = {}
    for (user, item), value in zip(pairs, predictions):
        scores.setdefault(user, {})[item] = float(value)
    first = component_scores[0]
    result = ScoreTable("ridge-hybrid", first.split_id, first.partition, scores)
    result.validate()
    return result


def predict_other_hybrid(component_scores, settings) -> ScoreTable:
    # TODO Task 1.4: add the alternatives from lectures, each with its own name
    # and configuration. Document how missing candidates or rank fusion work.
    require_aligned(component_scores)
    raise NotImplementedError("Bogdan: implement the agreed alternative hybrids.")
