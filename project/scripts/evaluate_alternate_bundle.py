"""Replay committed fits on the exact shared split, without training or tuning."""

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from project.recsys.artifacts import REPO_ROOT, fingerprint, provenance, write_json
from project.recsys.component_cache import file_hash
from project.recsys.data import Interaction, prepare_metadata
from project.recsys.evaluation import evaluation_settings
from project.recsys.hybrid_bundle import HybridBundle
from project.recsys.metrics.accuracy import evaluate_accuracy
from project.recsys.metrics.beyond_accuracy import evaluate_beyond_accuracy


RESULTS = Path('hyperparameters/results/alternate-hybrids-60-20-20')


def evaluation_cases(bundle):
    cases = [(n, p, s) for n in ('bpr', 'itemknn', 'userknn', 'ease', 'slim')
             for p, s in (('valid', 'full'), ('test', 'full'), ('valid', 'warm'))]
    cases += [('switch', 'valid', 'warm'), ('meta', 'valid', 'full'),
              ('meta-control', 'valid', 'full'), ('bpr-inherited', 'valid', 'full')]
    cases += [('userknn', 'test', 'warm'), ('switch', 'test', 'warm'),
              ('meta', 'test', 'full'), ('meta-control', 'test', 'full')]
    cases += [(n, p, 'full') for n in ('cascade', 'paired-slim', 'aug-ease-weak',
                                     'aug-slim-weak', 'aug-slim-comparable')
              for p in ('valid', 'test')]
    cases += [(n, p, 'full') for n, spec in bundle.models.items() if spec['kind'] == 'mlp'
              for p in ('valid', 'test')]
    return cases


def summarize_evaluations(results=RESULTS):
    """Read the canonical Task 2.2 inputs; seed means are report summaries only."""
    compact = {}
    for path in sorted((results / 'evaluations').glob('*/*/*/manifest.json')):
        manifest = json.loads(path.read_text())
        if manifest['status'] != 'complete':
            raise ValueError(f'Incomplete evaluation: {path.parent}')
        key = f"{manifest['partition']}/{manifest['scope']}/{manifest['bundle_model']}"
        compact[key] = {'aggregate': json.loads((path.parent / 'metrics.json').read_text())}
    for family in ('mlp', 'residual'):
        for variant in ('mlp', 'mlp-content'):
            for part in ('valid', 'test'):
                key = f'{part}/full/{family}-{variant}'
                values = [compact[f'{key}-seed{s}']['aggregate'] for s in (42, 43, 44)]
                compact[key] = {
                    'aggregate': {k: float(np.mean([v[k] for v in values])) for k in values[0]},
                    'std': {k: float(np.std([v[k] for v in values], ddof=1)) for k in values[0]},
                    'seed_count': 3,
                }
    return compact


def write_evaluation(bundle, name, partition, scope, statistics, source, results=RESULTS):
    """Write exactly the four input files read by Task 2.2's compare function."""
    folder = results / 'evaluations' / partition / scope / name
    folder.mkdir(parents=True, exist_ok=True)
    spec = bundle.models[name]
    hybrid = spec['kind'] in {'switch', 'cascade', 'mlp', 'augmentation', 'meta'}
    hybrid &= name not in {'paired-slim', 'meta-control'}
    manifest = {
        **source, **evaluation_settings(partition, 10, scope), 'status': 'running',
        'model': f'hybrid-{name}' if hybrid else name,
        'bundle_model': name, 'scope': scope, 'dataset': 'ml-100k',
        'split_id': bundle.manifest['split_id'], 'seed': spec.get('seed', bundle.manifest['seed']),
        'split_seed': bundle.manifest['seed'], 'parameters': spec,
        'n_training_interactions': statistics.n_interactions,
        'n_catalogue_items': len(statistics.catalogue),
        'pending_work': statistics.pending,
    }
    write_json(folder / 'manifest.json', manifest)
    try:
        rankings = bundle.rankings(name, partition, scope)
        accuracy = evaluate_accuracy(rankings, bundle.truth(partition, scope), 10)
        beyond = evaluate_beyond_accuracy(rankings, statistics, accuracy['per_user'], 10)
        write_json(folder / 'metrics.json', accuracy['aggregate'])
        write_json(folder / 'per_user_metrics.json', accuracy['per_user'])
        write_json(folder / 'beyond_accuracy.json', beyond)
        manifest.update(status='complete', n_users=len(rankings))
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        write_json(folder / 'manifest.json', manifest)
    print(name, partition, scope, accuracy['aggregate'], flush=True)
    return accuracy


def evaluate():
    bundle = HybridBundle()
    package = json.loads((RESULTS / 'package.json').read_text())
    for file, expected in package['files_sha256'].items():
        if file_hash(bundle.root / file) != expected:
            raise ValueError(f'Bundle changed: {file}')
    train_rows = [Interaction(bundle.users[u], bundle.items[i]) for u, i in zip(*np.nonzero(bundle.train))]
    data_dir = REPO_ROOT / 'dataset/ml-100k'
    item_file, user_file = data_dir / 'ml-100k.item', data_dir / 'ml-100k.user'
    statistics = {scope: prepare_metadata(train_rows, item_file, user_file, catalogue)
                  for scope, catalogue in (
                      ('full', bundle.items),
                      ('warm', [i for i, count in zip(bundle.items, bundle.train.sum(axis=0)) if count > 0]))}
    source_commit = subprocess.run(
        ['git', '-C', str(REPO_ROOT), 'log', '-1', '--format=%H', '--', bundle.root.as_posix()],
        capture_output=True, text=True, check=True).stdout.strip()
    source = {
        **provenance(), 'source_run': bundle.root.as_posix(), 'source_git_commit': source_commit,
        'checkpoint': bundle.root.as_posix(), 'checkpoint_sha256': fingerprint(package['files_sha256']),
        'checkpoint_format': 'SHA-256 of canonical JSON mapping of portable bundle file hashes',
        'metadata_sources': {label: {'path': str(path), 'sha256': file_hash(path) if path.is_file() else None}
                             for label, path in (('items', item_file), ('users', user_file))},
        'retraining_performed': False, 'test_selection_performed': False,
    }
    cases = evaluation_cases(bundle)
    for name, partition, scope in cases:
        write_evaluation(bundle, name, partition, scope, statistics[scope], source)
    compact = summarize_evaluations()
    references = json.loads((RESULTS / 'reference-metrics.json').read_text())
    largest_error = 0.0
    for key, recorded in references.items():
        for metric, value in recorded.items():
            error = abs(compact[key]['aggregate'][metric] - value)
            largest_error = max(largest_error, error)
            if error > 1e-12:
                raise ValueError(f'Replay differs from frozen training evaluation: {key} {metric} {error}')
    rows = []
    for key, value in compact.items():
        part, scope, name = key.split('/')
        rows.append({'partition': part, 'scope': scope, 'model': name, **value['aggregate']})
    with (RESULTS / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    proof = {'status': 'complete', 'split_id': bundle.manifest['split_id'], 'users': len(bundle.users),
             'items': len(bundle.items), 'warm_items': int((bundle.train.sum(axis=0)>0).sum()),
             'training_pairs': int(bundle.train.sum()), 'model_evaluations': len(cases),
             'test_selection_performed': False, 'retraining_performed': False,
             'test_scope': 'All six missing report rows evaluated on their original validation scopes; frozen fits',
             'supplemental_test_evaluation': True,
             'bundle_hashes_verified': True,
             'source_metric_comparisons': len(references), 'maximum_metric_error': largest_error,
             'sha256': {'evaluation_source': file_hash(__file__), 'inference_source': file_hash('project/recsys/hybrid_bundle.py'),
                        'metrics_csv': file_hash(RESULTS / 'metrics.csv'),
                        'shared_evaluation_source': file_hash('project/recsys/evaluation.py'),
                        'metadata_source': file_hash('project/recsys/data.py'),
                        'beyond_accuracy_source': file_hash('project/recsys/metrics/beyond_accuracy.py')},
             'evaluation_files_sha256': {p.relative_to(RESULTS).as_posix(): file_hash(p)
                                         for p in sorted((RESULTS / 'evaluations').glob('*/*/*/*.json'))}}
    write_json(RESULTS / 'replay-verification.json', proof)
    return compact


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        evaluate()
