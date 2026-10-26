"""Compute one FISM fold with exactly the main runner's verified cache identity."""

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from project.recsys.artifacts import fingerprint
from project.recsys.component_cache import atomic_json, file_hash, load_fold, save_fold
from project.recsys.hybrids import candidate_features
from project.recsys.oof_hybrid import external_pairs, train_and_score, user_folds
from project.scripts.refresh_fism_oof import config_for


def compute(root, fold):
    from recbole.data import create_dataset
    from recbole.utils import init_seed
    selection = json.loads((root / 'tuning/selection.json').read_text())
    assert selection['hyperparameters_frozen'] and not selection['test_evaluated']
    prepared = root / 'fism-prepared'
    pointer = json.loads((prepared / 'cache-fism.json').read_text())
    directory = Path(pointer['path'])
    identity = json.loads((directory / 'identity.json').read_text())
    assert fingerprint(identity) == pointer['key']
    config = config_for(selection['best']['parameters'], directory / f'fold-{fold}-worker-training')
    effective = json.loads(json.dumps(config.final_config_dict, default=str))
    for key in ('checkpoint_dir', 'model_save_file', 'log_wandb'):
        effective.pop(key, None)
    assert effective == identity['config']
    for path, digest in identity['sources'].items():
        from project.recsys.artifacts import REPO_ROOT
        assert file_hash(REPO_ROOT / path) == digest
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    training, _, _ = dataset.build()
    pairs = external_pairs(dataset, training.inter_feat)
    assignment = user_folds([u for u, _ in pairs], [i for _, i in pairs], 5, 42)
    assert fingerprint(assignment.tolist()) == identity['fold_assignment']
    assert json.loads((prepared / 'folds.json').read_text()) == [
        {'user': user, 'item': item, 'fold': int(number)}
        for (user, item), number in zip(pairs, assignment)
    ]
    held = np.flatnonzero(assignment == fold)
    fit = np.flatnonzero(assignment != fold)
    target = directory / f'fold-{fold}.npz'
    if load_fold(target, held) is not None:
        print(f'Fold {fold+1}: already complete', flush=True)
        return
    if (directory / f'fold-{fold}-training').exists():
        print(f'Fold {fold+1}: main runner already training; leave its cache untouched', flush=True)
        return
    print(f'Fold {fold+1}: independent worker training', flush=True)
    table = train_and_score(config, dataset, training.copy(training.inter_feat[fit]),
                            training.copy(training.inter_feat[held]),
                            directory / f'fold-{fold}-worker-training',
                            f"{identity['split_id']}:fold-{fold}", 'fism')
    keys, values = candidate_features([table])
    lookup = {pair: value[0] for pair, value in zip(keys, values)}
    predictions = np.array([lookup[pairs[index]] for index in held])
    # Never publish into an active main-runner stage.
    if not (directory / f'fold-{fold}-training').exists():
        save_fold(target, held, predictions)
        atomic_json(directory / f'fold-{fold}.worker.json', {
            'source_sha256': file_hash(__file__), 'identity_key': pointer['key'],
            'selection_sha256': file_hash(root / 'tuning/selection.json'),
            'held_rows': len(held), 'fit_rows': len(fit), 'fold': fold,
            'test_evaluated': False})
        print(f'Fold {fold+1}: verified cache published', flush=True)
    del table, lookup
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--fold', required=True, type=int, choices=range(5))
    args = parser.parse_args()
    torch.set_num_threads(2)
    logging.basicConfig(level=logging.ERROR)
    with threadpool_limits(limits=2):
        compute(Path(args.root).resolve(), args.fold)


if __name__ == '__main__':
    main()
