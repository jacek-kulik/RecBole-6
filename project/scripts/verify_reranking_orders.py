"""Audit saved Task 3.3 lists, masks, metrics and validation-only selection."""

import argparse
import csv
import json
from collections import Counter
from math import isfinite, log, log2
from pathlib import Path

import numpy as np

from project.recsys.artifacts import fingerprint, write_json
from project.recsys.comparisons import compare_runs
from project.recsys.component_cache import file_hash
from project.recsys.data import prepare_metadata, read_training_interactions
from project.recsys.hybrid_bundle import HybridBundle
from project.recsys.metrics.accuracy import evaluate_accuracy
from project.recsys.metrics.beyond_accuracy import evaluate_beyond_accuracy
from project.recsys.rerank import select_strength
from project.scripts.analyze_hybrid_coefficients import WEIGHTED


def read(path):
    return json.loads(Path(path).read_text())


def verify_metric_values(actual, expected, label):
    """Compare saved numeric fields, including unavailable values and NaNs."""
    max_error = 0.
    for key, value in expected.items():
        if key not in actual:
            raise ValueError(f'{label}: missing {key}')
        saved = actual[key]
        if value is None:
            if saved is not None:
                raise ValueError(f'{label}: {key} differs')
            continue
        if isinstance(saved, bool) or not isinstance(saved, (int, float)) or not isfinite(saved):
            raise ValueError(f'{label}: {key} must be finite')
        error = abs(saved - value)
        if error > 1e-12:
            raise ValueError(f'{label}: {key} differs')
        max_error = max(max_error, error)
    return max_error


def verify_summary_csv(path, rows):
    """The hashed CSV and the JSON used by selection/reports must agree."""
    with path.open(newline='', encoding='utf-8') as stream:
        saved = list(csv.DictReader(stream))
    if len(saved) != len(rows):
        raise ValueError('CSV and JSON summary lengths differ')
    for actual, expected in zip(saved, rows):
        if actual.keys() != expected.keys():
            raise ValueError('CSV and JSON summary fields differ')
        numbers = {}
        for key, value in expected.items():
            if isinstance(value, (int, float)):
                numbers[key] = float(actual[key])
            elif actual[key] != ('' if value is None else value):
                raise ValueError(f'CSV and JSON summaries differ: {key}')
        verify_metric_values(numbers, {k: expected[k] for k in numbers}, 'CSV summary')


def verify_saved_metrics(folder, row, accuracy, beyond, baseline_ndcg, target_metric):
    """Check canonical files and every metric alias consumed by reports."""
    ndcg = accuracy['aggregate']['NDCG@10']
    original = row['method'] == 'original'
    expected = {
        **accuracy['aggregate'], **beyond['aggregate'], 'ndcg': ndcg,
        'relative_ndcg_loss': 0. if original else 1 - ndcg / baseline_ndcg,
        'target': None if original else beyond['aggregate'][target_metric],
    }
    return max(
        verify_metric_values(row, expected, str(folder)),
        verify_metric_values(read(folder / 'metrics.json'), accuracy['aggregate'], 'Saved accuracy'),
        verify_metric_values(read(folder / 'beyond_accuracy.json')['aggregate'],
                             beyond['aggregate'], 'Saved beyond-accuracy'),
    )


def independent_targets(ranked, stats):
    kl, js = [], []
    exposure = dict.fromkeys(stats.catalogue, 0.)
    for user, items in ranked.items():
        genres, groups = Counter(), Counter()
        for rank, item in enumerate(items, 1):
            names = stats.item_features[item]
            for genre in names: genres[genre] += 1 / len(names)
            groups[stats.item_groups[item]] += 1
            exposure[item] += 1 / log2(rank + 1)
        p = stats.genre_profiles[user]
        q = {g: n / len(items) for g, n in genres.items()}
        kl.append(sum(v * log(v / (.99 * q.get(g, 0) + .01 * v)) for g, v in sorted(p.items()) if v))
        p = stats.popularity_profiles[user]
        q = {g: n / len(items) for g, n in groups.items()}
        value = 0.
        for group in sorted(p.keys() | q.keys()):
            a, b = p.get(group, 0), q.get(group, 0)
            middle = (a + b) / 2
            if a: value += .5 * a * log2(a / middle)
            if b: value += .5 * b * log2(b / middle)
        js.append(value)
    amounts = np.sort(list(exposure.values()))
    lorenz = np.r_[0., amounts.cumsum()] / amounts.sum()
    # Uniform-width trapezoids work on both NumPy 1.x and 2.x.
    gini = 1 - np.sum(lorenz[:-1] + lorenz[1:]) / len(amounts)
    return {'GenreMiscalibration@10': float(np.mean(kl)), 'UPD@10': float(np.mean(js)),
            'ExposureGini@10': float(gini),
            'CatalogueCoverage@10': sum(x > 0 for x in exposure.values()) / len(exposure)}


def compact_report(rows):
    tested = [r for r in rows if r['partition'] == 'test']
    base = next(r for r in tested if r['order'] == 'baseline')
    targets = {'genre': 'GenreMiscalibration@10', 'popularity': 'UPD@10', 'exposure': 'ExposureGini@10'}
    lines = ['# Task 3.3: rerank before or after hybrid combination', '',
             (f"Baseline test NDCG@10: **{base['ndcg']:.6f}**. Pre: rerank components then combine. "
              'Post: combine then rerank. Target reductions are relative to the corresponding baseline metric.'), '',
             '| Target | Order | Strength | Test NDCG@10 | NDCG loss | Target reduction |',
             '|---|---|---:|---:|---:|---:|']
    for r in tested:
        if r['order'] == 'baseline': continue
        t = targets[r['method']]
        reduction = 100 * (1 - r[t] / base[t])
        lines.append(f"| {r['method']} | {r['order']} | {r['strength']:g} | {r['ndcg']:.6f} | "
                     f"{100*r['relative_ndcg_loss']:.2f}% | {reduction:.2f}% |")
    lines += ['', (f"Baseline targets: genre KL {base['GenreMiscalibration@10']:.4f}; "
                   f"popularity JS {base['UPD@10']:.4f}; exposure Gini {base['ExposureGini@10']:.4f}."), '',
              '**Discussion**', '']
    discussion = (
        'We compare both orders using the corrected six-component Ridge hybrid, the same 943 users, '
        'training-derived profiles and the same hybrid-selected top-50 eligible candidates. The pre order '
        'promotes each component\'s reranked top ten, appends its original-order remainder and assigns '
        'original descending standardized scores to those positions. Frozen Ridge coefficients then combine '
        'the reassigned scores. This explicit conversion preserves score distributions and reproduces the '
        'original hybrid at zero strength. '
        'Each method/order selects strength on validation under a 5% NDCG-loss budget; all choices are frozen '
        'before test. Paired orders select identical strengths. Post-reranking produces larger reductions in '
        'every designated target but incurs greater NDCG loss. Both genre pipelines and post-exposure '
        'reranking exceed the validation loss budget on test, illustrating that the constraint does not '
        'guarantee test accuracy. Pre-reranking retains more accuracy while subsequent combination partly '
        'dilutes the component rerankers\' objectives. These descriptive findings are conditional on the '
        'conversion rule, candidate pool and single previously inspected split. Target reduction percentages '
        'refer to different metrics and are not directly comparable across targets.'
    )
    assert len(discussion.split()) <= 200
    lines.append(discussion)
    return '\n'.join(lines) + '\n', len(discussion.split())


def verify(out, results):
    manifest = read(out / 'manifest.json')
    protocol = read(out / 'frozen-protocol.json')
    selection = read(out / 'selection.json')
    if manifest['status'] != 'complete' or fingerprint(protocol) != manifest['identity']:
        raise ValueError('Incomplete or altered run')
    if file_hash(out / 'selection.json') != manifest['selection_sha256']:
        raise ValueError('Frozen selection changed')
    if file_hash(out / 'metrics.csv') != manifest['metrics_sha256']:
        raise ValueError('Aggregate CSV changed')
    for path, digest in protocol['code_sha256'].items():
        if file_hash(path) != digest: raise ValueError(f'Numerical source changed: {path}')
    for path, digest in read(out / 'weighted-inputs.json').items():
        if file_hash(path) != digest: raise ValueError(f'Weighted source changed: {path}')
    bundle = HybridBundle()
    stats = prepare_metadata(read_training_interactions(WEIGHTED / 'splits/train.csv'),
                             'dataset/ml-100k/ml-100k.item', 'dataset/ml-100k/ml-100k.user', bundle.items)
    rows = read(out / 'metrics.json')
    if len(rows) != manifest['evaluation_count'] or len(rows) != 38:
        raise ValueError('Expected all 38 predefined evaluations')
    if len({(r['partition'], r['name']) for r in rows}) != 38:
        raise ValueError('Duplicate evaluations')
    verify_summary_csv(out / 'metrics.csv', rows)
    baselines = {p: next(r['NDCG@10'] for r in rows if r['partition'] == p and r['order'] == 'baseline')
                 for p in ('valid', 'test')}
    masks = {p: bundle.mask(p) for p in ('valid', 'test')}
    pools = {p: read(out / f'{p}-pool.json') for p in masks}
    ui, ii = {u: n for n, u in enumerate(bundle.users)}, {i: n for n, i in enumerate(bundle.items)}
    originals = {p: read(out / p / 'original/rankings.json') for p in masks}
    max_error, zero_controls, list_count = 0., 0, 0
    for row in rows:
        p = row['partition']; folder = out / p / row['name']
        child = read(folder / 'manifest.json')
        if child['status'] != 'complete' or child['input_identity'] != manifest['identity']:
            raise ValueError('Evaluation identity differs')
        ranked = read(folder / 'rankings.json')
        if set(ranked) != set(bundle.users): raise ValueError('Evaluation cohort differs')
        relevant = {u: set(items) for u, items in read(folder / 'relevant.json').items()}
        if relevant != bundle.truth(p): raise ValueError('Relevance was filtered or altered')
        for user, items in ranked.items():
            if len(items) != 10 or len(set(items)) != 10: raise ValueError('Invalid top 10')
            if not set(items) <= set(pools[p][user]): raise ValueError('Recommendation outside fixed pool')
            if not all(masks[p][ui[user], ii[i]] for i in items): raise ValueError('History item recommended')
        acc = evaluate_accuracy(ranked, relevant, 10)
        if read(folder / 'per_user_metrics.json') != acc['per_user']:
            raise ValueError('Per-user accuracy differs')
        beyond = evaluate_beyond_accuracy(ranked, stats, acc['per_user'], 10)
        target = selection['target_metrics'].get(row['method'])
        max_error = max(max_error, verify_saved_metrics(folder, row, acc, beyond, baselines[p], target),
                        verify_metric_values(row, independent_targets(ranked, stats), 'Independent targets'))
        if row['strength'] == 0:
            if ranked != originals[p]: raise ValueError('Zero-strength control differs')
            zero_controls += 1
        list_count += len(ranked)
    for label, chosen in selection['choices'].items():
        method, order = label.split('/')
        candidates = [r for r in rows if r['partition'] == 'valid' and r['method'] == method and r['order'] == order]
        if sorted(r['strength'] for r in candidates) != protocol['strengths']:
            raise ValueError('Incomplete validation grid')
        baseline = next(r for r in rows if r['partition'] == 'valid' and r['order'] == 'baseline')['ndcg']
        expected = select_strength(candidates, baseline)
        if chosen != {k: expected[k] for k in chosen}: raise ValueError('Selection differs')
        tested = next(r for r in rows if r['partition'] == 'test' and r['method'] == method and r['order'] == order)
        if tested['strength'] != chosen['strength']: raise ValueError('Test strength differs')
    # Ensure the six selections were written after all validation and before test.
    frozen_time = (out / 'selection.json').stat().st_mtime_ns
    if max((out / 'valid' / r['name'] / 'manifest.json').stat().st_mtime_ns for r in rows if r['partition'] == 'valid') > frozen_time:
        raise ValueError('Validation written after freezing')
    if min((out / 'test' / r['name'] / 'manifest.json').stat().st_mtime_ns for r in rows if r['partition'] == 'test') < frozen_time:
        raise ValueError('Test written before freezing')
    compare_path = out / 'comparison'
    if not compare_path.exists():
        compare_runs([f"{r['name']}={out / 'test' / r['name']}" for r in rows if r['partition'] == 'test'], compare_path, 10)
    text, words = compact_report(rows)
    report = Path('project/reports/task-3.3-weighted-reranking-orders.md')
    report.write_text(text, encoding='utf-8')
    (out / 'compact-report.md').write_text(text, encoding='utf-8')
    proof = {'status': 'passed', 'evaluations': 38, 'recommendation_lists': list_count,
             'max_metric_error': max_error, 'zero_strength_controls': zero_controls,
             'all_candidates_and_full_relevance_checked': True,
             'independent_target_calculations': ['fractional-genre KL', 'popularity JS', 'Lorenz-curve exposure Gini', 'coverage'],
             'all_six_selections_reproduced_from_validation': True,
             'all_validation_precedes_test': True, 'selection_unchanged_through_test': True,
             'discussion_words': words, 'report_path': str(report), 'report_sha256': file_hash(report),
             'verifier_sha256': file_hash(__file__)}
    write_json(out / 'verification.json', proof)
    write_json(results / 'verification.json', proof)
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    verify(args.run, args.results)
