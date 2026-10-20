"""Replay every validation checkpoint before allowing downstream reuse or test."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from project.recsys.artifacts import write_json
from project.recsys.component_cache import file_hash
from project.recsys.content_based import read_history
from project.recsys.feature_combination import PairMLP, evaluate
from project.recsys.residual_combination import ResidualMLP


def verify(run, source):
    run, source = Path(run), Path(source)
    read = lambda p: json.loads(p.read_text())
    manifest, mapping = read(run / 'manifest.json'), read(run / 'mapping.json')
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete training run')
    for part in ('train', 'valid'):
        if file_hash(source / f'splits/{part}.csv') != manifest['sha256'][part]:
            raise ValueError('Training or validation split changed')
    train, truth = [read_history(source / f'splits/{part}.csv') for part in ('train', 'valid')]
    mask = np.load(run / 'candidate-mask.npz')['eligible']
    expected = np.array([[i not in train[u] for i in mapping['items']] for u in mapping['users']])
    np.testing.assert_array_equal(mask, expected)
    checks = []
    for trial in read(run / 'trials.json'):
        folder = run / trial['directory']
        saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
        state = saved['state_dict']
        args = (state['users'], state['items'], saved['hidden'])
        model = ResidualMLP(*args, state['baseline']) if 'baseline' in state else PairMLP(*args)
        model.load_state_dict(state)
        model.eval()
        scores = model.full_scores()
        np.testing.assert_allclose(scores, np.load(folder / 'scores-valid.npz')['scores'], rtol=0, atol=1e-6)
        metrics = evaluate(scores, mask, mapping['users'], mapping['items'], truth)
        if metrics != read(folder / 'metrics.json'):
            raise ValueError('Checkpoint metrics differ')
        if saved['epoch'] != trial['selected_epoch'] or saved['hidden'] != trial['hidden']:
            raise ValueError('Checkpoint selection differs')
        history = read(folder / 'history.json')
        best = max(history, key=lambda row: row['NDCG@10'])
        if best['epoch'] != saved['epoch']:
            raise ValueError('Checkpoint was not selected by validation NDCG')
        checks.append({'trial': trial['directory'], 'checkpoint_sha256': file_hash(folder / 'model.pt'),
                       'scores_sha256': file_hash(folder / 'scores-valid.npz'), 'metrics_reproduced': True})
    write_json(run / 'verification.json', {'status': 'passed', 'trial_checks': checks,
               'mask_verified': True, 'test_accessed': False, 'split_id': mapping['split_id']})
    print(f'Verified {len(checks)} validation checkpoints: {run}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify(args.run, args.source)
