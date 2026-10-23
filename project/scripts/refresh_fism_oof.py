"""Retune self-excluded FISM, refresh its OOF column, freeze Ridge, evaluate test."""

import argparse
import csv
import gc
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from project.recsys.artifacts import REPO_ROOT
from project.recsys.component_cache import atomic_json, file_hash, load_fold, run_lock
from project.recsys.contracts import ScoreTable, require_aligned
from project.recsys.hybrid_report import read_pairs, test_candidates
from project.recsys.hybrids import candidate_features, fit_oof_regression
from project.recsys.metrics.accuracy import evaluate_accuracy
from project.recsys.oof_hybrid import COMPONENTS, external_pairs, run, user_folds
from project.recsys.recbole_adapter import _export_splits, _write_scores, load_scores
from project.recsys.ridge_search import DEFAULT_ALPHAS, candidate_blocks, rank_predictions
from project.scripts.retune_fism import search_space


def config_for(params, directory):
    from recbole.config import Config
    argv = sys.argv
    try:
        sys.argv = argv[:1]
        return Config(config_file_list=[str(REPO_ROOT / 'project/configs/models/fism.yaml'),
                                       str(REPO_ROOT / 'project/configs/protocol.yaml')],
                      config_dict={**params, 'seed': 42, 'data_path': str(REPO_ROOT / 'dataset'),
                                   'checkpoint_dir': str(directory), 'save_dataset': False,
                                   'save_dataloaders': False})
    finally:
        sys.argv = argv


def retune(old, root):
    from recbole.data import create_dataset
    from recbole.utils import init_seed
    from project.recsys.oof_hybrid import train_and_score
    tuning = root / 'tuning'
    tuning.mkdir(parents=True, exist_ok=True)
    inputs = {'source': json.loads((REPO_ROOT / 'source-provenance.json').read_text()),
              'runner_sha256': file_hash(__file__), 'protocol_sha256': file_hash(REPO_ROOT / 'project/configs/protocol.yaml'),
              'grid_sha256': file_hash(REPO_ROOT / 'hyperparameters/hyper-FISM.test'),
              'dataset_sha256': file_hash(REPO_ROOT / 'dataset/ml-100k/ml-100k.inter'),
              'epochs': 20, 'selection_metric': 'independent validation NDCG@10',
              'checkpoint_policy': 'fixed-budget final epoch; no validation early stopping',
              'test_evaluated': False, 'torch': torch.__version__, 'threads': 2}
    identity_path = tuning / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != inputs:
        raise ValueError('Tuning inputs changed')
    atomic_json(identity_path, inputs)
    trials = []
    prior_manifest = json.loads((old / 'manifest.json').read_text())
    for index, params in enumerate(search_space()):
        target = tuning / f'trial-{index:02d}.json'
        if target.exists():
            trial = json.loads(target.read_text())
            assert trial['parameters'] == params
        else:
            print(f'FISM tuning {index+1}/48: {params}', flush=True)
            started = time.monotonic()
            config = config_for(params, tuning / f'trial-{index:02d}-training')
            init_seed(config['seed'], config['reproducibility'])
            dataset = create_dataset(config)
            training, valid, testing = dataset.build()
            with tempfile.TemporaryDirectory() as tmp:
                generated = Path(tmp) / 'splits'
                split_id = _export_splits(dataset, [SimpleNamespace(dataset=d) for d in (training, valid, testing)], generated, config)
                assert split_id == prior_manifest['split_id']
                for part in ('train', 'valid', 'test'):
                    assert (generated / f'{part}.csv').read_bytes() == (old / 'splits' / f'{part}.csv').read_bytes()
            table = train_and_score(config, dataset, training, valid,
                                    tuning / f'trial-{index:02d}-training', split_id, 'fism')
            truth = read_pairs(old / 'splits/valid.csv')
            trial = {'parameters': params, 'validation': evaluate_accuracy(table.top_k(10), truth, 10)['aggregate'],
                     'seconds': time.monotonic() - started, 'split_id': split_id,
                     'test_evaluated': False, 'fixed_epochs': 20}
            atomic_json(target, trial)
            del table, dataset, training, valid, testing, config
            gc.collect()
        trials.append(trial)
        print(f"  NDCG@10={trial['validation']['NDCG@10']:.6f}; {trial['seconds']:.1f}s", flush=True)
    best = max(trials, key=lambda t: t['validation']['NDCG@10'])
    old_params = prior_manifest['component_parameters']['fism']
    baseline = next(t for t in trials if t['parameters'] == old_params)
    atomic_json(tuning / 'selection.json', {'best': best, 'old_parameters_control': baseline,
                'trials': len(trials), 'selection_metric': 'independent validation NDCG@10',
                'test_evaluated': False, 'hyperparameters_frozen': True})
    print('FISM selected: ' + json.dumps(best), flush=True)
    summary = root / 'component-selection.csv'
    with (old / 'tuned-summary.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row['model'].lower() == 'fism':
            row['hyperparameters'] = ', '.join(f'{key}:{value!r}' for key, value in best['parameters'].items())
    with summary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'hyperparameters'])
        writer.writeheader()
        writer.writerows(rows)
    return summary


def combine_and_fit(old, root):
    prepared = root / 'fism-prepared'
    output = root / 'final'
    output.mkdir(exist_ok=True)
    prep_manifest = json.loads((prepared / 'manifest.json').read_text())
    old_manifest = json.loads((old / 'manifest.json').read_text())
    assert prep_manifest['split_id'] == old_manifest['split_id']
    assert (prepared / 'folds.json').read_bytes() == (old / 'folds.json').read_bytes()
    for part in ('train', 'valid', 'test'):
        assert (prepared / 'splits' / f'{part}.csv').read_bytes() == (old / 'splits' / f'{part}.csv').read_bytes()
    shutil.copytree(old / 'splits', output / 'splits', dirs_exist_ok=True)
    for filename in ('folds.json', 'protocol.yaml'):
        shutil.copyfile(prepared / filename, output / filename)
    with np.load(old / 'oof-progress.npz', allow_pickle=False) as saved:
        features, target, folds = saved['features'].copy(), saved['targets'].copy(), saved['folds'].copy()
    with np.load(prepared / 'oof-progress.npz', allow_pickle=False) as saved:
        assert np.array_equal(folds, saved['folds'])
        np.testing.assert_allclose(target, 4 * saved['targets'] + 1, rtol=0, atol=1e-12)
        features[:, 5] = saved['features'][:, 0]
    assert features.shape == (60734, 6) and 1 <= target.min() < target.max() <= 5
    proof = {'old_run': str(old), 'old_oof_sha256': file_hash(old / 'oof-progress.npz'),
             'fism_prepared_oof_sha256': file_hash(prepared / 'oof-progress.npz'),
             'fism_selection_sha256': file_hash(root / 'tuning/selection.json'),
             'folds_sha256': file_hash(old / 'folds.json'), 'reused_components': {},
             'training_rows': len(target), 'test_evaluated': False,
             'source': json.loads((REPO_ROOT / 'source-provenance.json').read_text())}
    for column, name in enumerate(COMPONENTS):
        parent = prepared if name == 'fism' else old
        pointer = json.loads((parent / f'cache-{name}.json').read_text())
        component = Path(pointer['path'])
        for fold in range(5):
            held = np.flatnonzero(folds == fold)
            cached = load_fold(component / f'fold-{fold}.npz', held)
            assert cached is not None
            np.testing.assert_array_equal(features[held, column], cached)
        source = component / 'scores-valid.json.gz'
        assert file_hash(source) == json.loads((component / 'full.complete.json').read_text())['sha256']
        if name != 'fism':
            proof['reused_components'][name] = {'cache_key': pointer['key'], 'scores_sha256': file_hash(source)}
        atomic_json(output / f'cache-{name}.json', pointer)
    np.savez_compressed(output / 'oof-progress.npz', features=features, targets=target, folds=folds)
    atomic_json(output / 'inputs.json', proof)
    truth = read_pairs(output / 'splits/valid.csv')
    tables = [load_scores(Path(json.loads((output / f'cache-{name}.json').read_text())['path']) / 'scores-valid.json.gz') for name in COMPONENTS]
    pairs, design = candidate_features(tables)
    blocks = candidate_blocks(pairs)
    component_metrics = {table.model: evaluate_accuracy(table.top_k(10), truth, 10) for table in tables}
    del tables
    gc.collect()
    alphas = sorted(set([0.1, 1., 10., *DEFAULT_ALPHAS]))
    trials = []
    for alpha in alphas:
        fitted = fit_oof_regression(features, target, COMPONENTS, alpha)
        predictions = design @ np.asarray(fitted['coefficients']) + fitted['intercept']
        metrics = evaluate_accuracy(rank_predictions(blocks, predictions, 10), truth, 10)['aggregate']
        trials.append({'alpha': alpha, 'fitted': fitted, 'metrics': metrics})
        print(f"Ridge alpha={alpha:g}: validation NDCG@10={metrics['NDCG@10']:.6f}", flush=True)
    best = max(trials, key=lambda t: t['metrics']['NDCG@10'])
    atomic_json(output / 'trials.json', trials)
    atomic_json(output / 'hybrid.json', best['fitted'])
    atomic_json(output / 'selection.json', {'best': best, 'metric': 'validation NDCG@10',
                'test_evaluated': False, 'selected_highest_penalty': best['alpha'] == max(alphas),
                'provisional_fism': False})
    values = design @ np.asarray(best['fitted']['coefficients']) + best['fitted']['intercept']
    scores = {}
    for (user, item), value in zip(pairs, values):
        scores.setdefault(user, {})[item] = float(value)
    _write_scores(ScoreTable('ridge-hybrid', prep_manifest['split_id'], 'valid', scores), output / 'scores-valid-hybrid.json.gz')
    manifest = {**prep_manifest, 'components': list(COMPONENTS), 'status': 'validation_selected',
                'selected_alpha': best['alpha'], 'regressor_fitted': True, 'test_evaluated': False,
                'prepare_only': False, 'provisional_fism': False, 'target': 'raw 1-5 held-out rating',
                'unaffected_component_policy': 'Reuse original five cached components without retuning',
                'fism_parameter_policy': '48-setting validation NDCG search; fixed 20-epoch training'}
    atomic_json(output / 'manifest.json', manifest)
    del scores, design, pairs, values
    gc.collect()
    print('FISM and Ridge selection frozen before test evaluation', flush=True)
    evaluate_final(old, output, component_metrics, best, manifest)
    return output


def evaluate_final(old, output, component_metrics, best, manifest):
    from project.recsys.hybrids import predict_weighted_hybrid
    training, validation, testing = [read_pairs(output / 'splits' / f'{p}.csv') for p in ('train', 'valid', 'test')]
    catalogue = set().union(*training.values(), *validation.values(), *testing.values())
    metrics = {'valid': component_metrics, 'test': {}}
    metrics['valid']['regressor'] = evaluate_accuracy(load_scores(output / 'scores-valid-hybrid.json.gz').top_k(10), validation, 10)
    selected_hash = file_hash(output / 'hybrid.json')
    selection_hash = file_hash(output / 'selection.json')
    tables = []
    result = output / 'evaluation'
    result.mkdir(exist_ok=True)
    for name in COMPONENTS:
        pointer = json.loads((output / f'cache-{name}.json').read_text())
        table = load_scores(Path(pointer['path']) / 'scores-valid.json.gz')
        test = test_candidates(table, training, validation, testing, catalogue)
        metrics['test'][name] = evaluate_accuracy(test.top_k(10), testing, 10)
        _write_scores(test, result / f'scores-test-{name}.json.gz')
        tables.append(test)
    require_aligned(tables)
    test = predict_weighted_hybrid(tables, best['fitted'])
    metrics['test']['regressor'] = evaluate_accuracy(test.top_k(10), testing, 10)
    _write_scores(test, result / 'scores-test-hybrid.json.gz')
    assert selected_hash == file_hash(output / 'hybrid.json')
    assert selection_hash == file_hash(output / 'selection.json')
    prior = json.loads((old / 'ridge-extended/evaluation/metrics.json').read_text())
    for partition in ('valid', 'test'):
        for name in COMPONENTS[:-1]:
            assert metrics[partition][name]['aggregate'] == prior[partition][name]['aggregate']
    atomic_json(result / 'metrics.json', metrics)
    atomic_json(result / 'frozen-selection.json', {'fit_sha256': selected_hash, 'selection_sha256': selection_hash,
                'test_selection_performed': False, 'test_score_method': 'Fixed training histories; additional validation mask; renormalize candidates',
                'split_id': manifest['split_id'], 'users': len(testing)})
    columns = [f'{m}@10' for m in ('Precision', 'Recall', 'F1', 'MRR', 'NDCG', 'MAP')]
    lines = ['# Corrected FISM and five-fold OOF Ridge results', '',
             'MovieLens-100K; per-user 60/20/20 split; seed 42; 943 users; 60,734 OOF raw-rating targets. Candidate scores are z-scored per user.', '',
             'FISM excludes target self-contributions and was retuned over the existing 48-setting grid using independent validation NDCG@10. Training uses a fixed 20 epochs, matching the OOF budget. The other five components and all outer/inner splits are unchanged.', '',
             f"Ridge alpha {best['alpha']:g} was selected from 24 fixed penalties by validation NDCG@10 before test evaluation. Test candidates exclude training and validation histories; score normalization is recomputed on eligible test candidates.", '',
             'The test partition was already inspected in earlier experiments. These are corrected results on that same single split, not a new untouched generalization estimate. Five component hyperparameters still retain the previously approved transfer from the older tuning protocol.', '']
    for partition in ('valid', 'test'):
        lines += [f'## {partition.title()}', '', '| Model | ' + ' | '.join(columns) + ' |', '|---|' + '---:|' * len(columns)]
        with (result / f'metrics-{partition}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['Model', *columns]); writer.writeheader()
            for name in (*COMPONENTS, 'regressor'):
                m = metrics[partition][name]['aggregate']
                writer.writerow({'Model': name, **m})
                lines.append('| ' + name + ' | ' + ' | '.join(f'{m[c]:.6f}' for c in columns) + ' |')
        lines.append('')
    lines += ['## Coefficients', '', '| Component | Coefficient |', '|---|---:|']
    lines += [f'| {n} | {c:.9f} |' for n, c in zip(COMPONENTS, best['fitted']['coefficients'])]
    lines += ['', f"Intercept: {best['fitted']['intercept']:.9f}. Coefficients apply to correlated normalized scores and are not causal importance.", '',
              f"Previous provisional hybrid test NDCG@10: {prior['test']['regressor']['aggregate']['NDCG@10']:.6f}. Corrected hybrid: {metrics['test']['regressor']['aggregate']['NDCG@10']:.6f}."]
    (result / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest.update(status='complete', test_evaluated=True, independent_evaluation_complete=True)
    atomic_json(output / 'manifest.json', manifest)
    print(json.dumps({'selected_alpha': best['alpha'], 'test': {n: m['aggregate'] for n,m in metrics['test'].items()}}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    old, root = Path(args.old).resolve(), Path(args.output).resolve()
    root.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    logging.basicConfig(level=logging.ERROR)
    with run_lock(root), threadpool_limits(limits=2):
        summary = retune(old, root)
        prepared = root / 'fism-prepared'
        run(SimpleNamespace(output=str(prepared), resume=(prepared / 'cache-run.json').exists(),
                            components=['fism'], tuned_summary=str(summary), folds=5, seed=42,
                            smoke_epochs=None, prepare_only=True, alphas=[100.], k=10))
        print(combine_and_fit(old, root), flush=True)


if __name__ == '__main__':
    main()
