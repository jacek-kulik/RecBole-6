"""Replay committed fits on the exact shared split, without training or tuning."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from project.recsys.artifacts import write_json
from project.recsys.component_cache import file_hash
from project.recsys.hybrid_bundle import HybridBundle
from project.recsys.metrics.accuracy import evaluate_accuracy


RESULTS = Path('hyperparameters/results/alternate-hybrids-60-20-20')


def evaluate():
    bundle = HybridBundle()
    package = json.loads((RESULTS / 'package.json').read_text())
    for file, expected in package['files_sha256'].items():
        if file_hash(bundle.root / file) != expected:
            raise ValueError(f'Bundle changed: {file}')
    all_metrics = {}
    cases = [(n, p, s) for n in ('bpr', 'itemknn', 'userknn', 'ease', 'slim')
             for p, s in (('valid', 'full'), ('test', 'full'), ('valid', 'warm'))]
    cases += [('switch', 'valid', 'warm'), ('meta', 'valid', 'full'), ('meta-control', 'valid', 'full'), ('bpr-inherited', 'valid', 'full')]
    cases += [(n, p, 'full') for n in ('cascade', 'paired-slim', 'aug-ease-weak', 'aug-slim-weak', 'aug-slim-comparable')
              for p in ('valid', 'test')]
    for n, spec in bundle.models.items():
        if spec['kind'] == 'mlp':
            cases.append((n, 'valid', 'full'))
            if not spec['residual']:
                cases.append((n, 'test', 'full'))
    for name, partition, scope in cases:
        result = evaluate_accuracy(bundle.rankings(name, partition, scope), bundle.truth(partition, scope), 10)
        all_metrics[f'{partition}/{scope}/{name}'] = result
        print(name, partition, scope, result['aggregate'], flush=True)
    for family in ('mlp', 'residual'):
        for variant in ('mlp', 'mlp-content'):
            for part in (('valid', 'test') if family == 'mlp' else ('valid',)):
                values = [all_metrics[f'{part}/full/{family}-{variant}-seed{s}']['aggregate'] for s in (42, 43, 44)]
                means = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
                std = {k: float(np.std([v[k] for v in values], ddof=1)) for k in values[0]}
                all_metrics[f'{part}/full/{family}-{variant}'] = {'aggregate': means, 'std': std, 'seed_count': 3}
    compact = {k: {field: value for field, value in v.items() if field != 'per_user'} for k, v in all_metrics.items()}
    references = json.loads((RESULTS / 'reference-metrics.json').read_text())
    largest_error = 0.0
    for key, recorded in references.items():
        for metric, value in recorded.items():
            error = abs(compact[key]['aggregate'][metric] - value)
            largest_error = max(largest_error, error)
            if error > 1e-12:
                raise ValueError(f'Replay differs from frozen training evaluation: {key} {metric} {error}')
    write_json(RESULTS / 'metrics.json', compact)
    rows = []
    for key, value in compact.items():
        part, scope, name = key.split('/')
        rows.append({'partition': part, 'scope': scope, 'model': name, **value['aggregate']})
    with (RESULTS / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    output = Path('project/artifacts/alternate-bundle-replay')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'per-user-metrics.json', all_metrics)
    proof = {'status': 'complete', 'split_id': bundle.manifest['split_id'], 'users': len(bundle.users),
             'items': len(bundle.items), 'warm_items': int((bundle.train.sum(axis=0)>0).sum()),
             'training_pairs': int(bundle.train.sum()), 'model_evaluations': len(cases),
             'test_selection_performed': False, 'retraining_performed': False,
             'test_scope': 'Only hybrids with previously reported test results; frozen baseline choices',
             'bundle_hashes_verified': True,
             'source_metric_comparisons': len(references), 'maximum_metric_error': largest_error,
             'sha256': {'evaluation_source': file_hash(__file__), 'inference_source': file_hash('project/recsys/hybrid_bundle.py'),
                        'metrics': file_hash(RESULTS / 'metrics.json')}}
    write_json(RESULTS / 'replay-verification.json', proof)
    return compact


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        evaluate()
