"""Exhaustive, resumable EASE/SLIM validation search on the shared 60/20/20 split.

Native RecBole constructors fit these non-iterative recommenders once. No dummy
training epochs are needed. Test membership is reconstructed for split identity;
test scores and metrics are never computed. Large checkpoints/scores stay local.
Defaults create a new local run under project/artifacts; committed result records
are evidence from the original run, not a cache containing downloadable models.
"""

import argparse
import ast
import csv
import gzip
import hashlib
import itertools
import json
import logging
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
import yaml
from threadpoolctl import threadpool_limits

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed
from project.recsys.contracts import ScoreTable
from project.recsys.metrics.accuracy import evaluate_accuracy


EXPECTED_SPLIT = '1c62b75f944bd6cea03626d8cc1bac05289d65ab4e04003d1f10a92d20edbea1'
MODELS = {'EASE': ('ease', {'reg_weight': 250.0}),
          'SLIMElastic': ('slimelastic', {'alpha': 0.2, 'l1_ratio': 0.01})}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    temp.replace(path)


def grid(path):
    space = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, kind, values = line.split(maxsplit=2)
        if kind != 'choice':
            raise ValueError('Only finite choice grids supported')
        space[name] = ast.literal_eval(values)
    return [dict(zip(space, values)) for values in itertools.product(*space.values())]


def export_splits(dataset, parts, output, config):
    output.mkdir(parents=True, exist_ok=True)
    split_hash = hashlib.sha256()
    for name, part in zip(('train', 'valid', 'test'), parts):
        interactions = part.inter_feat
        columns = {'user_id': dataset.id2token(dataset.uid_field, interactions[dataset.uid_field].numpy()),
                   'item_id': dataset.id2token(dataset.iid_field, interactions[dataset.iid_field].numpy())}
        for label, field in (('rating', config['RATING_FIELD']), ('timestamp', config['TIME_FIELD'])):
            if field in interactions:
                columns[label] = interactions[field].numpy()
        path = output / f'{name}.csv'
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(zip(*columns.values()))
        split_hash.update(name.encode())
        split_hash.update(path.read_bytes())
    return split_hash.hexdigest()


def rankings(values, eligible, users, items):
    if values.shape != eligible.shape or not np.isfinite(values).all():
        raise ValueError('Nonfinite or misaligned scores')
    if np.any(eligible.sum(axis=1) < 10):
        raise ValueError('Fewer than ten eligible candidates')
    order = np.argsort(-np.where(eligible, values, -np.inf), axis=1, kind='stable')[:, :10]
    return {user: [items[j] for j in row] for user, row in zip(users, order)}


def choose(trials):
    successful = [t for t in trials if t['status'] == 'complete' and not t['convergence_warnings']]
    if not successful:
        raise ValueError('No converged trial available')
    # max returns the first trial on a tie; grid order is declared and persisted.
    return max(successful, key=lambda t: t['validation']['NDCG@10'])


def config_for(name, params):
    argv = sys.argv
    try:
        sys.argv = argv[:1]
        return Config(model=name, config_file_list=['project/configs/protocol.yaml'], config_dict={
            'data_path': str(Path('dataset').resolve()), 'seed': 42, 'use_gpu': False,
            'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
            'slim_max_iter': 1000, 'positive_only': True, 'hide_item': True, **params})
    finally:
        sys.argv = argv


def run(args):
    from recbole.model.general_recommender.ease import EASE
    from recbole.model.general_recommender.slimelastic import SLIMElastic

    classes = {'EASE': EASE, 'SLIMElastic': SLIMElastic}
    output, artifacts = Path(args.output), Path(args.artifacts)
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    sources = ['project/scripts/retune_ease_slim.py', 'project/configs/protocol.yaml',
               'project/recsys/contracts.py', 'project/recsys/metrics/accuracy.py',
               'project/recsys/metrics/_validation.py', 'project/recsys/metrics/types.py',
               'recbole/model/general_recommender/ease.py', 'recbole/model/general_recommender/slimelastic.py',
               'recbole/model/abstract_recommender.py', 'recbole/data/dataset/dataset.py',
               'recbole/data/interaction.py', 'recbole/properties/overall.yaml',
               'recbole/properties/model/EASE.yaml', 'recbole/properties/model/SLIMElastic.yaml',
               'hyperparameters/hyper-EASE.test', 'hyperparameters/hyper-SLIMElastic.test',
               'dataset/ml-100k/ml-100k.inter']
    identity = {'sha256': {p: digest(p) for p in sources}, 'python': platform.python_version(),
                'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
                'torch': torch.__version__, 'threads': args.threads, 'seed': 42,
                'slim_max_iter': 1000, 'selection_metric': 'validation NDCG@10'}
    identity_path = output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Inputs changed; use new output and artifact directories')
    write_json(identity_path, identity)
    config = config_for('EASE', {})
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    parts = dataset.build()
    split_id = export_splits(dataset, parts, artifacts / 'splits', config)
    if split_id != EXPECTED_SPLIT:
        raise ValueError(f'Unexpected split identity: {split_id}')
    uid, iid = dataset.uid_field, dataset.iid_field
    relevant = {}
    for u, i in zip(parts[1].inter_feat[uid].numpy(), parts[1].inter_feat[iid].numpy()):
        relevant.setdefault(str(dataset.id2token(uid, int(u))), set()).add(str(dataset.id2token(iid, int(i))))
    users = sorted(relevant)
    items = sorted(str(i) for i in dataset.field2id_token[iid][1:])
    internal_users, internal_items = dataset.token2id(uid, users), dataset.token2id(iid, items)
    train_matrix = parts[0].inter_matrix(form='csr')
    seen = train_matrix[internal_users][:, internal_items].toarray() > 0
    eligible = ~seen
    item_indices = {i: j for j, i in enumerate(items)}
    if any(not all(eligible[n, item_indices[i]] for i in relevant[u]) for n, u in enumerate(users)):
        raise ValueError('Training/validation overlap')
    manifest = {'status': 'running', 'split_id': split_id, 'split': [0.6, 0.2, 0.2], 'seed': 42,
                'users': len(users), 'items': len(items), 'training_pairs': parts[0].inter_num,
                'validation_pairs': parts[1].inter_num, 'selection_metric': 'validation NDCG@10',
                'relevance': 'all held-out interactions', 'candidates': 'catalogue minus training history; no padding',
                'test_evaluated': False, 'test_access': 'membership only for split identity; no scores or metrics',
                'tie_break': 'external item token for scores; first declared grid trial for selection',
                'slim_max_iter': 1000, 'artifacts': str(artifacts),
                'identity_sha256': digest(identity_path)}
    write_json(output / 'manifest.json', manifest)
    selections = {}
    for name in args.models:
        trials = []
        model_output, model_artifacts = output / name, artifacts / name
        model_output.mkdir(exist_ok=True)
        model_artifacts.mkdir(exist_ok=True)
        for index, params in enumerate(grid(f'hyperparameters/hyper-{name}.test')):
            path = model_output / f'trial-{index:02d}.json'
            checkpoint_path = model_artifacts / f'trial-{index:02d}.pth'
            scores_path = model_artifacts / f'trial-{index:02d}.npz'
            if path.exists():
                trial = json.loads(path.read_text())
                if (trial['parameters'] != params or not checkpoint_path.exists() or not scores_path.exists()
                        or trial['checkpoint_sha256'] != digest(checkpoint_path)
                        or trial['scores_sha256'] != digest(scores_path)):
                    raise ValueError('Completed trial cache is missing or corrupt')
            else:
                print(f'{name} trial {index + 1}: {params}', flush=True)
                started = time.monotonic()
                trial_config = config_for(name, params)
                init_seed(42, True)
                model = classes[name](trial_config, parts[0])
                values = model.interaction_matrix[internal_users, :] @ model.item_similarity
                if hasattr(values, 'toarray'):
                    values = values.toarray()
                values = np.asarray(values)[:, internal_items]
                metrics = evaluate_accuracy(rankings(values, eligible, users, items), relevant, 10)
                torch.save({'config': trial_config, 'state_dict': model.state_dict(),
                            'other_parameter': model.other_parameter(), 'epoch': 0,
                            'fitting': 'native constructor; no dummy optimizer steps'}, checkpoint_path)
                np.savez_compressed(scores_path, scores=values, users=np.array(users), items=np.array(items), eligible=eligible)
                converged = int(getattr(model, 'convergence_warnings', 0))
                trial = {'parameters': params, 'status': 'complete' if not converged else 'nonconverged',
                         'convergence_warnings': converged, 'validation': metrics['aggregate'],
                         'seconds': round(time.monotonic() - started, 3),
                         'checkpoint': str(checkpoint_path), 'checkpoint_sha256': digest(checkpoint_path),
                         'scores': str(scores_path), 'scores_sha256': digest(scores_path)}
                write_json(path, trial)
            trials.append(trial)
            print(f"  MRR@10={trial['validation']['MRR@10']:.6f} NDCG@10={trial['validation']['NDCG@10']:.6f} "
                  f"convergence_warnings={trial['convergence_warnings']} seconds={trial['seconds']}", flush=True)
        best = choose(trials)
        baseline = next(t for t in trials if t['parameters'] == MODELS[name][1])
        selected_params = dict(best['parameters'])
        if name == 'SLIMElastic':
            selected_params.update(slim_max_iter=1000, positive_only=True, hide_item=True)
        selection = {'model': name, 'split_id': split_id, 'trials': len(trials),
                     'selection_metric': 'validation NDCG@10', 'best': best, 'baseline_same_solver': baseline,
                     'parameters': selected_params}
        write_json(model_output / 'selection.json', selection)
        (model_output / 'best-params.yaml').write_text(yaml.safe_dump(
            {'model': name, **selected_params}, sort_keys=False), encoding='utf-8', newline='\n')
        with np.load(best['scores'], allow_pickle=False) as saved:
            selected_scores = saved['scores'].copy()
        table = ScoreTable(name, split_id, 'valid', {
            u: {i: float(selected_scores[n, j]) for j, i in enumerate(items) if eligible[n, j]}
            for n, u in enumerate(users)})
        verified = evaluate_accuracy(table.top_k(10), relevant, 10)
        if verified['aggregate'] != best['validation']:
            raise ValueError('Independent exported ScoreTable metrics differ')
        with gzip.open(model_artifacts / 'scores-valid.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(asdict(table), stream, sort_keys=True, allow_nan=False)
        write_json(model_output / 'metrics-selected.json', verified)
        selections[name] = selection
    write_json(output / 'selection.json', selections)
    manifest.update(status='complete', completed_models=list(selections))
    write_json(output / 'manifest.json', manifest)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='project/artifacts/individual-retuning-ndcg/results')
    parser.add_argument('--artifacts', default='project/artifacts/individual-retuning-ndcg/models')
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    parser.add_argument('--threads', type=int, default=2)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.ERROR)
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        run(args)
