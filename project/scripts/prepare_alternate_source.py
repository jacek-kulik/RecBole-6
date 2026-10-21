"""Recreate corrected inherited component caches from portable fitted baselines.

The jointly tuned cascade keeps its independent component selections. This
source is for hybrids whose sender representations/scores were inherited.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from recbole.data import create_dataset
from recbole.utils import init_seed
from project.recsys.artifacts import write_json, fingerprint
from project.recsys.component_cache import file_hash
from project.recsys.contracts import ScoreTable
from project.recsys.hybrid_bundle import HybridBundle
from project.recsys.recbole_adapter import _write_scores
from project.scripts.retune_ease_slim import config_for, export_splits, EXPECTED_SPLIT


def prepare(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bundle = HybridBundle()
    cfg = config_for('BPR', bundle.models['bpr']['parameters'])
    init_seed(42, True)
    dataset = create_dataset(cfg)
    parts = dataset.build()
    if export_splits(dataset, parts, output / 'splits', cfg) != EXPECTED_SPLIT:
        raise ValueError('Split differs from recorded experiments')
    params = {'bpr': bundle.models['bpr']['parameters'], 'userknn': bundle.models['userknn']['parameters'],
              'itemknn': bundle.models['itemknn']['parameters'], 'ease': bundle.models['ease']['parameters'],
              'slimelastic': bundle.models['slim']['parameters']}
    names = {'bpr': 'bpr', 'userknn': 'userknn', 'itemknn': 'itemknn', 'ease': 'ease', 'slimelastic': 'slim'}
    for name, model in names.items():
        key = fingerprint({'name': name, 'params': params[name], 'split_id': EXPECTED_SPLIT})
        directory = output / 'components' / name / key
        directory.mkdir(parents=True)
        write_json(directory / 'identity.json', {'name': name, 'parameters': params[name],
                   'split_id': EXPECTED_SPLIT, 'key': key})
        values, mask = bundle.scores(model), bundle.mask('valid')
        table = ScoreTable(name, EXPECTED_SPLIT, 'valid', {
            u: {i: float(values[n, j]) for j, i in enumerate(bundle.items) if mask[n, j]}
            for n, u in enumerate(bundle.users)})
        path = directory / 'scores-valid.json.gz'
        _write_scores(table, path)
        _write_scores(table, output / f'scores-valid-{name}.json.gz')
        write_json(directory / 'full.complete.json', {'sha256': file_hash(path)})
        write_json(output / f'cache-{name}.json', {'key': key, 'path': str(directory)})
        if name == 'bpr':
            checkpoint = directory / 'full-training'
            checkpoint.mkdir()
            with np.load(bundle.root / bundle.models['bpr']['file'], allow_pickle=False) as z:
                p = np.zeros((dataset.user_num, z['users'].shape[1]), dtype=np.float32)
                q = np.zeros((dataset.item_num, z['items'].shape[1]), dtype=np.float32)
                p[dataset.token2id(dataset.uid_field, bundle.users)] = z['users']
                q[dataset.token2id(dataset.iid_field, bundle.items)] = z['items']
            torch.save({'config': cfg, 'epoch': 49, 'state_dict': {
                'user_embedding.weight': torch.from_numpy(p), 'item_embedding.weight': torch.from_numpy(q)},
                'other_parameter': {}}, checkpoint / 'bpr.pth')
            testmask = bundle.mask('test')
            testtable = ScoreTable('bpr', EXPECTED_SPLIT, 'test', {
                u: {i: float(values[n, j]) for j, i in enumerate(bundle.items) if testmask[n, j]}
                for n, u in enumerate(bundle.users)})
            evaluation = output / 'evaluation'
            evaluation.mkdir(exist_ok=True)
            _write_scores(testtable, evaluation / 'scores-test-bpr.json.gz')
    write_json(output / 'manifest.json', {'status': 'complete', 'split_id': EXPECTED_SPLIT,
               'component_parameters': params, 'selection_metric': 'validation NDCG@10',
               'source': 'Retuned standalone portable baselines for inherited fits only',
               'test_metrics_computed': False, 'cascade_parameters_overridden': False})
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='project/artifacts/alternate-corrected-source')
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        prepare(args.output)
