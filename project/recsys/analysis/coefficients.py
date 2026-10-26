"""Controlled Task 2.3 coefficient and component-removal analysis."""

import numpy as np
from sklearn.linear_model import Ridge


def candidate_zscores(values, eligible):
    """Normalize (user,item,component) scores on eligible candidates, without labels."""
    if (values.ndim != 3 or eligible.shape != values.shape[:2]
            or eligible.dtype != bool or not np.isfinite(values).all()
            or np.any(eligible.sum(axis=1) == 0)):
        raise ValueError('Invalid candidate scores or mask')
    result = np.zeros_like(values, dtype=float)
    for u in range(len(values)):
        block = values[u, eligible[u]]
        deviation = block.std(axis=0)
        active = deviation > 1e-12
        normalized = np.zeros_like(block)
        normalized[:, active] = ((block[:, active] - block[:, active].mean(axis=0))
                                 / deviation[active])
        result[u, eligible[u]] = normalized
    return result


def removal_fits(features, targets, fitted):
    """Return a frozen removal and a fixed-penalty OOF refit per component."""
    x, y = np.asarray(features), np.asarray(targets)
    names = fitted['components']
    weights = np.asarray(fitted['coefficients'], dtype=float)
    if (x.ndim != 2 or x.shape != (len(y), len(names))
            or weights.shape != (len(names),) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isfinite(weights).all()):
        raise ValueError('Invalid OOF design or component ordering')
    variants = {'original': fitted}
    for j, name in enumerate(names):
        frozen = weights.copy()
        frozen[j] = 0
        variants[f'zero-{name}'] = {**fitted, 'coefficients': frozen.tolist(),
                                  'removed': name, 'method': 'frozen removal'}
        keep = np.arange(len(names)) != j
        model = Ridge(alpha=fitted['alpha']).fit(x[:, keep], y)
        refitted = np.zeros_like(weights)
        refitted[keep] = model.coef_
        variants[f'refit-{name}'] = {
            **fitted, 'coefficients': refitted.tolist(), 'intercept': float(model.intercept_),
            'removed': name, 'method': 'OOF refit at original fixed penalty',
        }
    return variants


def paired_summary(accuracy, reference, rankings, reference_rankings, seed=42):
    """Paired user bootstrap; conditional on this split and these trained fits."""
    users = sorted(reference['per_user'])
    if (set(users) != set(accuracy['per_user']) or set(users) != set(rankings)
            or set(users) != set(reference_rankings)):
        raise ValueError('Comparison user sets differ')
    delta = np.array([accuracy['per_user'][u]['NDCG@10']
                      - reference['per_user'][u]['NDCG@10'] for u in users])
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(len(users), size=(2000, len(users)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        'delta_NDCG@10': float(delta.mean()),
        'delta_Recall@10': accuracy['aggregate']['Recall@10'] - reference['aggregate']['Recall@10'],
        'delta_MRR@10': accuracy['aggregate']['MRR@10'] - reference['aggregate']['MRR@10'],
        'delta_NDCG_ci_low': float(lo), 'delta_NDCG_ci_high': float(hi),
        'users_improved': int((delta > 1e-12).sum()),
        'users_worse': int((delta < -1e-12).sum()),
        'users_tied': int((np.abs(delta) <= 1e-12).sum()),
        'top10_overlap': float(np.mean([len(set(rankings[u]) & set(reference_rankings[u])) / 10
                                        for u in users])),
    }
