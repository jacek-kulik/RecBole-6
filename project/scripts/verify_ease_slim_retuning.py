"""Audit original and freshly reproduced retuning without evaluating test scores."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed
from project.scripts.retune_ease_slim import EXPECTED_SPLIT, config_for, digest, export_splits, write_json


def verify(original, reproduced, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = config_for('EASE', {})
    init_seed(42, True)
    dataset = create_dataset(config)
    parts = dataset.build()
    actual_split = export_splits(dataset, parts, output / 'splits', config)
    if actual_split != EXPECTED_SPLIT:
        raise ValueError('Wrong split')
    uid, iid = dataset.uid_field, dataset.iid_field
    pair_sets, histories = [], []
    for part in parts:
        pairs = list(zip(part.inter_feat[uid].tolist(), part.inter_feat[iid].tolist()))
        if len(pairs) != len(set(pairs)):
            raise ValueError('Repeated user-item pair')
        pair_sets.append(set(pairs))
        histories.append(Counter(u for u, _ in pairs))
    if any(pair_sets[a] & pair_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Split leakage')
    all_users = set(histories[0])
    for u in all_users:
        total = sum(h[u] for h in histories)
        held_out = int(.2 * total)
        if [h[u] for h in histories] != [total - 2 * held_out, held_out, held_out]:
            raise ValueError('Per-user split ratios differ')
    train = parts[0].inter_matrix(form='csr').astype(np.float32)
    truth = parts[1].inter_matrix(form='csr')
    users = sorted(str(dataset.id2token(uid, u)) for u in all_users)
    items = sorted(str(i) for i in dataset.field2id_token[iid][1:])
    internal_users, internal_items = dataset.token2id(uid, users), dataset.token2id(iid, items)
    eligible = train[internal_users][:, internal_items].toarray() == 0
    truth = truth[internal_users][:, internal_items].toarray() > 0
    if np.any(truth & ~eligible):
        raise ValueError('Validation labels present in training')
    counts = truth.sum(axis=1)
    discounts = 1 / np.log2(np.arange(2, 12))
    ideal = np.array([discounts[:min(10, n)].sum() for n in counts])
    results = {}
    for label, directory in [('original', Path(original)), ('reproduced', Path(reproduced))]:
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['status'] != 'complete' or manifest['split_id'] != actual_split or manifest['test_evaluated']:
            raise ValueError('Incomplete/wrong-partition run')
        fits = {}
        for name, expected_count in [('EASE', 9), ('SLIMElastic', 25)]:
            trials = []
            paths = sorted((directory / name).glob('trial-*.json'))
            if len(paths) != expected_count:
                raise ValueError('Grid not exhaustive')
            for path in paths:
                trial = json.loads(path.read_text())
                if trial['status'] != 'complete' or trial['convergence_warnings']:
                    raise ValueError('Unconverged trial')
                if any(digest(trial[field]) != trial[field + '_sha256'] for field in ('checkpoint', 'scores')):
                    raise ValueError('Corrupt cache')
                checkpoint = torch.load(trial['checkpoint'], map_location='cpu', weights_only=False)
                cfg = checkpoint['config']
                if cfg['seed'] != 42 or cfg['eval_args']['split'] != {'RS': [.6, .2, .2]} or cfg['eval_args']['group_by'] != 'user':
                    raise ValueError('Checkpoint used wrong protocol')
                if cfg['valid_metric'] != 'MRR@10' or any(cfg[k] != v for k, v in trial['parameters'].items()):
                    raise ValueError('Checkpoint parameters/selection metric differ')
                other = checkpoint['other_parameter']
                fitted_history = csr_matrix(other['interaction_matrix'])
                if fitted_history.shape != train.shape or (fitted_history != train).nnz:
                    raise ValueError('Model fitted on data other than training interactions')
                weights = np.asarray(other['item_similarity'].toarray() if hasattr(other['item_similarity'], 'toarray') else other['item_similarity'])
                np.testing.assert_allclose(np.diag(weights), 0, rtol=0, atol=0)
                if name == 'SLIMElastic' and (np.any(weights < 0) or not cfg['hide_item'] or cfg['slim_max_iter'] != 1000):
                    raise ValueError('SLIM constraints/solver settings differ')
                with np.load(trial['scores'], allow_pickle=False) as saved:
                    values = saved['scores']
                    if list(saved['users']) != users or list(saved['items']) != items or not np.array_equal(saved['eligible'], eligible):
                        raise ValueError('Wrong token mapping/candidate mask')
                recomputed = (fitted_history[internal_users] @ weights)[:, internal_items]
                np.testing.assert_allclose(recomputed, values, rtol=1e-6, atol=1e-6)
                top = np.argsort(-np.where(eligible, values, -np.inf), axis=1, kind='stable')[:, :10]
                hits = np.take_along_axis(truth, top, axis=1)
                independent = {'MRR@10': np.max(hits / np.arange(1, 11), axis=1).mean(),
                               'Recall@10': (hits.sum(axis=1) / counts).mean(),
                               'NDCG@10': ((hits * discounts).sum(axis=1) / ideal).mean()}
                if any(abs(v - trial['validation'][m]) > 1e-12 for m, v in independent.items()):
                    raise ValueError('Independent validation metrics differ')
                trials.append(trial)
            best = max(trials, key=lambda t: t['validation']['MRR@10'])
            selection = json.loads((directory / name / 'selection.json').read_text())
            if best['parameters'] != selection['best']['parameters'] or best['validation'] != selection['best']['validation']:
                raise ValueError('Selection is not validation-MRR optimum')
            fits[name] = {json.dumps(t['parameters'], sort_keys=True): t['validation'] for t in trials}
            for file in (f'recbole/config/{name}/ml-100k.yaml',
                         f'project/configs/models/{"ease" if name=="EASE" else "slimelastic"}.yaml'):
                cfg_yaml = yaml.safe_load(Path(file).read_text(encoding='utf-8'))
                if any(cfg_yaml[k] != v for k, v in selection['parameters'].items()):
                    raise ValueError('Published parameters do not match selection')
            print(f'{label}: verified {expected_count} {name} training matrices, masks, checkpoints and independent metrics', flush=True)
        results[label] = fits
    if results['original'] != results['reproduced']:
        raise ValueError('Fresh fits failed to reproduce original trial metrics exactly')
    # Also exercise the published entry points, including the individual YAML
    # without a shared-protocol override. Its omitted float columns must not
    # change the actual user/item split membership.
    for name, short in [('EASE', 'ease'), ('SLIMElastic', 'slimelastic')]:
        for files in ([f'recbole/config/{name}/ml-100k.yaml'],
                      [f'project/configs/models/{short}.yaml', 'project/configs/protocol.yaml']):
            argv = sys.argv
            try:
                sys.argv = argv[:1]
                published = Config(model=name, config_file_list=files,
                                   config_dict={'data_path': str(Path('dataset').resolve()), 'use_gpu': False})
            finally:
                sys.argv = argv
            init_seed(42, True)
            loaded = create_dataset(published)
            built = loaded.build()
            for a, b in ((uid, loaded.uid_field), (iid, loaded.iid_field)):
                np.testing.assert_array_equal(dataset.field2id_token[a], loaded.field2id_token[b])
            for expected_pairs, part in zip(pair_sets, built):
                actual_pairs = set(zip(part.inter_feat[loaded.uid_field].tolist(), part.inter_feat[loaded.iid_field].tolist()))
                if actual_pairs != expected_pairs:
                    raise ValueError('Published YAML reconstructs different split membership')
    proof = {'status': 'passed', 'split_id': actual_split, 'split': [.6, .2, .2], 'seed': 42,
             'split_counts': [len(p) for p in pair_sets], 'users': len(all_users),
             'original_fits_verified': 34, 'fresh_fits_verified': 34,
             'per_user_split_verified': True, 'train_validation_test_disjoint': True,
             'all_four_published_config_splits_verified': True,
             'every_checkpoint_contains_training_interactions_only': True,
             'every_validation_mask_verified': True, 'independent_metrics': ['MRR@10', 'Recall@10', 'NDCG@10'],
             'all_trial_metrics_reproduced_exactly': True, 'selection_metric': 'validation MRR@10',
             'test_predictions_computed': False, 'test_metrics_computed': False,
             'original_manifest_sha256': digest(Path(original) / 'manifest.json'),
             'reproduced_manifest_sha256': digest(Path(reproduced) / 'manifest.json'),
             'verifier_sha256': digest(__file__)}
    write_json(output / 'verification.json', proof)
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', default='hyperparameters/results/individual-retuned-60-20-20')
    parser.add_argument('--reproduced', default='project/artifacts/individual-retuning/results')
    parser.add_argument('--output', default='project/artifacts/pr21-review-verification')
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        verify(args.original, args.reproduced, args.output)
