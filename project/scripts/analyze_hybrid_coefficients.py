"""Run predefined Task 2.3 experiments, preserving the original fits.

python -m project.scripts.analyze_hybrid_coefficients --output <fresh directory>
"""

import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import time

import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from project.recsys.analysis.coefficients import candidate_zscores, removal_fits, paired_summary
from project.recsys.artifacts import write_json
from project.recsys.component_cache import file_hash, load_fold
from project.recsys.feature_augmentation import augment, fit_ease, fit_slim, read_pairs
from project.recsys.hybrid_bundle import HybridBundle
from project.recsys.metrics.accuracy import evaluate_accuracy
from project.recsys.recbole_adapter import load_scores
from project.scripts.retune_ease_slim import rankings


WEIGHTED = Path('project/artifacts/hybrid-60-20-20-fism-self-excluded/final')
RESULTS = Path('hyperparameters/results/task-2.3-coefficient-analysis')
REPORT = Path('project/reports/task-2.3-coefficient-analysis.md')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_out(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_metrics(actual, expected, label):
    error = max(abs(actual[k] - expected[k]) for k in expected)
    if error > 1e-12:
        raise ValueError(f'{label}: baseline metric replay error {error}')
    return error


def save_evaluation(output, family, name, part, values, eligible, bundle, reference=None):
    folder = output / family / name / part
    folder.mkdir(parents=True, exist_ok=True)
    ranked = rankings(values, eligible, bundle.users, bundle.items)
    accuracy = evaluate_accuracy(ranked, bundle.truth(part), 10)
    write_json(folder / 'metrics.json', accuracy['aggregate'])
    with gzip.open(folder / 'per-user-and-rankings.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump({'accuracy': accuracy['per_user'], 'rankings': ranked}, stream, sort_keys=True)
    np.savez_compressed(folder / 'predictions.npz', scores=values, eligible=eligible)
    if reference is None:
        reference = (accuracy, ranked)
    row = {'family': family, 'variant': name, 'partition': part, **accuracy['aggregate'],
           **paired_summary(accuracy, reference[0], ranked, reference[1])}
    print(f'{family}/{name}/{part}: NDCG={row["NDCG@10"]:.6f} delta={row["delta_NDCG@10"]:+.6f}', flush=True)
    return row, (accuracy, ranked)


def verify_weighted(bundle, output):
    manifest, fitted = read(WEIGHTED / 'manifest.json'), read(WEIGHTED / 'hybrid.json')
    frozen = read(WEIGHTED / 'evaluation/frozen-selection.json')
    if (manifest['status'] != 'complete' or manifest['provisional_fism']
            or manifest['split_id'] != bundle.manifest['split_id']
            or frozen['fit_sha256'] != file_hash(WEIGHTED / 'hybrid.json')
            or frozen['selection_sha256'] != file_hash(WEIGHTED / 'selection.json')
            or fitted['alpha'] != 30000 or fitted['transform'] != 'per-user candidate z-score'):
        raise ValueError('Incorrect or modified weighted source')
    hashes = {str(WEIGHTED / name): file_hash(WEIGHTED / name) for name in
              ('hybrid.json', 'selection.json', 'manifest.json', 'oof-progress.npz', 'inputs.json', 'folds.json')}
    proof = read(WEIGHTED / 'inputs.json')
    if hashes[str(WEIGHTED / 'folds.json')] != proof['folds_sha256']:
        raise ValueError('Weighted fold identity changed')
    old, prepared = Path(proof['old_run']), WEIGHTED.parent / 'fism-prepared'
    for path, expected in ((old / 'oof-progress.npz', proof['old_oof_sha256']),
                           (prepared / 'oof-progress.npz', proof['fism_prepared_oof_sha256'])):
        if file_hash(path) != expected:
            raise ValueError(f'OOF source changed: {path}')
        hashes[str(path)] = expected
    with np.load(WEIGHTED / 'oof-progress.npz', allow_pickle=False) as saved:
        x, y, folds = (saved[k].copy() for k in ('features', 'targets', 'folds'))
    if x.shape != (60734, 6) or y.min() < 1 or y.max() > 5:
        raise ValueError('Wrong OOF rows or rating target')
    with np.load(old / 'oof-progress.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(x[:, :5], saved['features'][:, :5])
        np.testing.assert_array_equal(y, saved['targets'])
        np.testing.assert_array_equal(folds, saved['folds'])
    with np.load(prepared / 'oof-progress.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(x[:, 5], saved['features'][:, 0])
        np.testing.assert_allclose(y, 4 * saved['targets'] + 1, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(folds, saved['folds'])
    reconstructed = Ridge(alpha=fitted['alpha']).fit(x, y)
    np.testing.assert_allclose(reconstructed.coef_, fitted['coefficients'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(reconstructed.intercept_, fitted['intercept'], rtol=0, atol=1e-12)
    for part in ('train', 'valid', 'test'):
        path = WEIGHTED / 'splits' / f'{part}.csv'
        expected = ({u: {bundle.items[i] for i in np.flatnonzero(bundle.train[n])}
                     for n, u in enumerate(bundle.users)} if part == 'train' else bundle.truth(part))
        if read_pairs(path) != expected:
            raise ValueError(f'Weighted and augmentation splits differ: {part}')
        hashes[str(path)] = file_hash(path)
    if bundle.users != sorted(bundle.users) or bundle.items != sorted(bundle.items):
        raise ValueError('Expected lexicographic external-token order')
    raw = np.zeros((*bundle.train.shape, 6), dtype=np.float64)
    valid_mask = bundle.mask('valid')
    for j, name in enumerate(fitted['components']):
        cache = Path(read(WEIGHTED / f'cache-{name}.json')['path'])
        for fold in range(5):
            selected = np.flatnonzero(folds == fold)
            column = load_fold(cache / f'fold-{fold}.npz', selected)
            if column is None:
                raise ValueError('Missing verified OOF fold')
            np.testing.assert_array_equal(x[selected, j], column)
        path = cache / 'scores-valid.json.gz'
        expected = read(cache / 'full.complete.json')['sha256']
        if file_hash(path) != expected:
            raise ValueError(f'Changed component scores: {name}')
        hashes[str(path)] = expected
        table = load_scores(path)
        if (table.model != name or table.partition != 'valid'
                or table.split_id != manifest['split_id'] or set(table.scores) != set(bundle.users)):
            raise ValueError('Component identity or users differ')
        for u, user in enumerate(bundle.users):
            items = [bundle.items[i] for i in np.flatnonzero(valid_mask[u])]
            if set(table.scores[user]) != set(items):
                raise ValueError('Component candidate identity differs')
            raw[u, valid_mask[u], j] = [table.scores[user][i] for i in items]
        del table
        print(f'Verified weighted component: {name}', flush=True)
    write_json(output / 'weighted-inputs.json', hashes)
    return fitted, x, y, raw


def weighted_experiments(bundle, output):
    fitted, x, y, raw = verify_weighted(bundle, output)
    fits = removal_fits(x, y, fitted)
    write_json(output / 'weighted-fits.json', fits)
    originals = read(WEIGHTED / 'evaluation/metrics.json')
    rows, replay = [], {}
    for part in ('valid', 'test'):
        eligible = bundle.mask(part)
        design = candidate_zscores(raw, eligible)
        reference = None
        for name, fit in fits.items():
            values = design @ np.asarray(fit['coefficients']) + fit['intercept']
            row, evaluated = save_evaluation(output, 'weighted', name, part, values, eligible, bundle, reference)
            if name == 'original':
                reference = evaluated
                replay[part] = verify_metrics(evaluated[0]['aggregate'], originals[part]['regressor']['aggregate'],
                                              f'weighted/{part}')
            row.update(removed=fit.get('removed', ''), alpha=fit['alpha'])
            rows.append(row)
        del design
    write_json(output / 'weighted-replay.json', replay)
    return rows


def augmentation_experiments(bundle, output):
    r = bundle.train.astype(np.float64)
    # Original JSON score tables populate a float64 array before normalization.
    # Promote the portable float32 scores BEFORE division, not after it.
    _, additions = augment(r, bundle.scores('userknn').astype(np.float64), 50, 1.0)
    np.savez_compressed(output / 'augmentation-inputs.npz', targets=r, additions=additions,
                        users=bundle.users, items=bundle.items)
    from scipy.sparse import load_npz
    for name, strength in (('aug-ease-weak', 0.25), ('aug-slim-weak', 0.25),
                           ('aug-slim-comparable', 1.0)):
        saved = load_npz(bundle.root / bundle.models[name]['additions']).toarray()
        np.testing.assert_allclose(strength * additions, saved, rtol=0, atol=1e-12)
    rows, evidence = [], {}
    sources = Path('hyperparameters/results/alternate-hybrids-60-20-20/evaluations')
    for receiver in ('ease', 'slim'):
        references = {}
        for strength in (0.0, 0.25, 1.0):
            started = time.monotonic()
            name = f'{receiver}-strength-{strength:g}'
            x = r + strength * additions
            weights = fit_ease(x, r, 500.0) if receiver == 'ease' else fit_slim(x, r, 0.2, 0.005)
            values = x @ weights
            folder = output / 'augmentation' / name
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / 'weights.npz', weights=weights)
            seconds = time.monotonic() - started
            existing = ({0.0: 'ease', 0.25: 'aug-ease-weak'}.get(strength) if receiver == 'ease'
                        else {0.0: 'paired-slim', 0.25: 'aug-slim-weak', 1.0: 'aug-slim-comparable'}[strength])
            if existing:
                # Native EASE uses float32 weights; the constrained paired refit is
                # float64. Its documented sub-micro score difference is harmless
                # only if all six ranking metrics still replay exactly below.
                tolerance = 1e-6 if receiver == 'ease' and strength == 0 else 1e-8
                # Observed-item scores are never ranked and were not part of the
                # historical score-table comparison. Compare eligible entries.
                difference = np.abs(values - bundle.scores(existing))[bundle.mask('valid')]
                if float(difference.max()) > tolerance:
                    raise ValueError(f'{name}: candidate score replay error {difference.max()}')
            for part in ('valid', 'test'):
                row, evaluated = save_evaluation(output, 'augmentation', name, part, values, bundle.mask(part),
                                                 bundle, references.get(part))
                if strength == 0:
                    references[part] = evaluated
                if existing:
                    error = verify_metrics(evaluated[0]['aggregate'],
                                           read(sources / part / 'full' / existing / 'metrics.json'), f'{name}/{part}')
                    evidence[f'{name}/{part}'] = {
                        'existing_model': existing, 'metric_error': error,
                        'eligible_score_max_absolute_error': float(difference.max()),
                    }
                row.update(receiver=receiver, strength=strength, top_n=50,
                           selected_entries=int(np.count_nonzero(additions)),
                           added_entries=int(np.count_nonzero(strength * additions)),
                           mean_added_mass=float((strength * additions).sum(axis=1).mean()),
                           mean_observed_mass=float(r.sum(axis=1).mean()), fit_seconds=seconds)
                rows.append(row)
    write_json(output / 'augmentation-replay.json', evidence)
    return rows


def resume_weighted(bundle, output):
    """Verify saved weighted fits, sources, predictions and metrics before reuse."""
    for path, digest in read(output / 'weighted-inputs.json').items():
        if file_hash(path) != digest:
            raise ValueError(f'Resume source changed: {path}')
    with np.load(WEIGHTED / 'oof-progress.npz', allow_pickle=False) as saved:
        fits = removal_fits(saved['features'], saved['targets'], read(WEIGHTED / 'hybrid.json'))
    if fits != read(output / 'weighted-fits.json'):
        raise ValueError('Resume weighted fits differ')
    strings = {'family', 'variant', 'partition', 'removed'}
    with (output / 'weighted.csv').open(newline='') as stream:
        rows = [{k: v if k in strings else float(v) for k, v in row.items()}
                for row in csv.DictReader(stream)]
    if {(r['variant'], r['partition']) for r in rows} != {
            (name, part) for name in fits for part in ('valid', 'test')} or len(rows) != 26:
        raise ValueError('Incomplete weighted stage')
    reference = {}
    originals = read(WEIGHTED / 'evaluation/metrics.json')
    for row in rows:
        part = row['partition']
        folder = output / 'weighted' / row['variant'] / part
        with np.load(folder / 'predictions.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['eligible'], bundle.mask(part))
            ranked = rankings(saved['scores'], saved['eligible'], bundle.users, bundle.items)
        accuracy = evaluate_accuracy(ranked, bundle.truth(part), 10)
        verify_metrics(accuracy['aggregate'], {k: row[k] for k in accuracy['aggregate']}, 'resume weighted')
        if row['variant'] == 'original':
            reference[part] = (accuracy, ranked)
            verify_metrics(accuracy['aggregate'], originals[part]['regressor']['aggregate'], 'resume baseline')
        for k, v in paired_summary(accuracy, reference[part][0], ranked, reference[part][1]).items():
            if abs(row[k] - v) > 1e-12:
                raise ValueError('Resume comparison differs')
    print('Verified and resumed all 26 completed weighted evaluations', flush=True)
    return rows


def report(output, weighted, augmentation, destination=REPORT):
    fitted = read(output / 'weighted-fits.json')['original']
    lines = ['# Task 2.3: hybrid coefficients and component contributions', '',
             'MovieLens-100K, 60/20/20, seed 42; 943 users, 1,682 items. NDCG@10 throughout. '
             'Removal deltas are relative to the original weighted hybrid; positive means improvement.', '',
             '**Weighted hybrid: test contribution analysis**', '',
             '| Removed component | Original coefficient | Frozen removal delta | Refit delta |',
             '|---|---:|---:|---:|']
    lookup = {(r['variant'], r['partition']): r for r in weighted}
    for name, coefficient in zip(fitted['components'], fitted['coefficients']):
        deltas = [lookup[(f'{method}-{name}', 'test')]['delta_NDCG@10']
                  for method in ('zero', 'refit')]
        lines.append(f'| {name} | {coefficient:.6f} | ' + ' | '.join(f'{d:+.6f}' for d in deltas) + ' |')
    lines += ['', 'Original hybrid NDCG@10: ' + ', '.join(
        f"{p} {lookup[('original', p)]['NDCG@10']:.6f}" for p in ('valid', 'test')) + '.', '',
        '**UserKNN feature augmentation: receiver retrained at each strength**', '',
        '| Receiver | Strength | Validation NDCG | Test NDCG |',
        '|---|---:|---:|---:|']
    aug = {(r['receiver'], r['strength'], r['partition']): r for r in augmentation}
    for receiver in ('ease', 'slim'):
        for strength in (0.0, 0.25, 1.0):
            v, t = (aug[(receiver, strength, p)] for p in ('valid', 'test'))
            lines.append(f"| {receiver.upper()} | {strength:g} | {v['NDCG@10']:.6f} | {t['NDCG@10']:.6f} |")
    discussion = (
        'Coefficients multiply per-user standardized component scores; their sizes are not contribution percentages. '
        'Frozen removals set one coefficient to zero without changing the others. Refits remove the same component '
        'and relearn the remaining coefficients from identical out-of-fold rating targets, holding the Ridge penalty '
        'at 30,000. Component fits and candidate masks remain fixed. '
        'Despite receiving the largest coefficient, ItemKNN lowers ranking accuracy in this fitted hybrid: both '
        'removal procedures improve validation and test NDCG. FISM and SLIM cause the largest frozen test losses; '
        'refitting does not fully recover either loss. Thus coefficient magnitude does not establish ranking utility, '
        'and the rating-regression objective need not align perfectly with NDCG. '
        'Augmentation scales the same top-50 normalized UserKNN predictions, keeps observed targets unchanged, '
        'and retrains each receiver with fixed regularization. Strength 0.25 helps SLIM more than strength 1; '
        'EASE gains slightly on validation but loses on test at both strengths. More transferred prediction mass '
        'therefore need not help. '
        'These are descriptive results on one previously inspected test split, with no new configuration selection; '
        'they do not establish generalizable improvements.'
    )
    if len(discussion.split()) > 200:
        raise ValueError('Discussion exceeds assignment word limit')
    lines += ['', '**Discussion**', '', discussion]
    text = '\n'.join(lines) + '\n'
    (output / 'report.md').write_text(text, encoding='utf-8')
    Path(destination).write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--report-only', action='store_true', help='Render existing CSVs; never fit/evaluate')
    parser.add_argument('--resume', action='store_true', help='Verify/reuse a completed weighted stage')
    args = parser.parse_args()
    output = args.output
    if args.report_only:
        def rows(name):
            with (output / name).open(newline='') as stream:
                return [{k: v if k in {'family', 'variant', 'partition', 'removed', 'receiver'} else float(v)
                         for k, v in row.items()} for row in csv.DictReader(stream)]
        report(output, rows('weighted.csv'), rows('augmentation.csv'), args.report)
        return
    output.mkdir(parents=True, exist_ok=args.resume)
    if args.results.exists():
        raise ValueError('Results already exist; preserve them before a new run')
    bundle = HybridBundle()
    package = read('hyperparameters/results/alternate-hybrids-60-20-20/package.json')
    for name, digest in package['files_sha256'].items():
        if file_hash(bundle.root / name) != digest:
            raise ValueError(f'Changed portable source: {name}')
    protocol = {
        'status': 'running', 'split_id': bundle.manifest['split_id'], 'seed': 42,
        'scope': 'full', 'k': 10, 'partitions': ['valid', 'test'],
        'weighted_source': str(WEIGHTED), 'weighted_penalty': 30000,
        'weighted_variants': ['original', 'six frozen removals', 'six fixed-penalty OOF refits'],
        'weighted_components_retuned': False, 'weighted_regularization_retuned': False,
        'augmentation_sender': {'model': 'userknn', 'k': 160, 'shrink': 0},
        'augmentation_strengths': [0.0, 0.25, 1.0], 'top_n': 50, 'equal_additions': False,
        'ease_reg': 500, 'slim_alpha': 0.2, 'slim_l1_ratio': 0.005,
        'slim_solver': {'max_iter': 1000, 'random_state': 42, 'selection': 'random',
                        'tol': 1e-4, 'positive': True, 'fit_intercept': False},
        'receiver_retrained_each_strength': True, 'targets': 'unchanged binary training matrix',
        'candidate_masks': 'valid excludes train; test excludes train and valid',
        'weighted_normalization': 'per-user candidate z-score recomputed separately on valid/test masks',
        'tie_break': 'lexicographic external item token', 'test_selection_performed': False,
        'test_previously_inspected': True, 'new_hyperparameter_selection': False,
        'bootstrap': {'paired_users': True, 'replicates': 2000, 'seed': 42,
                      'scope': 'conditional on existing split/fits; descriptive, no multiple-comparison correction'},
        'source_bundle_package_sha256': file_hash('hyperparameters/results/alternate-hybrids-60-20-20/package.json'),
        'implementation_sha256': {str(p): file_hash(p) for p in
                                  (Path(__file__), Path('project/recsys/analysis/coefficients.py'),
                                   Path('project/recsys/feature_augmentation.py'),
                                   Path('project/recsys/metrics/accuracy.py'))},
    }
    if args.resume:
        prior = read(output / 'protocol.json')
        for key in protocol.keys() - {'implementation_sha256', 'status'}:
            if protocol[key] != prior[key]:
                raise ValueError(f'Resume protocol differs: {key}')
        helper = str(Path('project/recsys/analysis/coefficients.py'))
        if prior['implementation_sha256'][helper] != protocol['implementation_sha256'][helper]:
            raise ValueError('Weighted analysis implementation changed')
        protocol['weighted_stage_original_implementation_sha256'] = prior['implementation_sha256']
    write_json(output / 'protocol.json', protocol)
    with threadpool_limits(limits=2):
        weighted = resume_weighted(bundle, output) if args.resume else weighted_experiments(bundle, output)
        csv_out(output / 'weighted.csv', weighted)
        augmentation = augmentation_experiments(bundle, output)
        csv_out(output / 'augmentation.csv', augmentation)
    protocol['status'] = 'complete'
    write_json(output / 'manifest.json', {**protocol, 'local_run': str(output.resolve()),
                                         'result_sha256': {n: file_hash(output / n) for n in
                                                           ('weighted.csv', 'augmentation.csv', 'weighted-fits.json')}})
    report(output, weighted, augmentation, args.report)
    args.results.mkdir(parents=True)
    for name in ('protocol.json', 'manifest.json', 'weighted.csv', 'augmentation.csv',
                 'weighted-fits.json', 'weighted-inputs.json', 'weighted-replay.json', 'augmentation-replay.json'):
        shutil.copyfile(output / name, args.results / name)
    print(f'Completed. Report: {args.report}', flush=True)


if __name__ == '__main__':
    main()
