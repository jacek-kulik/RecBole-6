"""Frozen MF plus zero-initialized feature-combination corrections; validation only."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .artifacts import new_run, write_json
from .component_cache import file_hash
from .content_based import read_history
from .feature_combination import PairMLP, evaluate, sample_negatives


class ResidualMLP(PairMLP):
    """The fixed score matrix is the original, unstandardized MF dot product."""

    def __init__(self, users, items, hidden, baseline):
        super().__init__(users, items, hidden)
        self.register_buffer("baseline", baseline.clone())
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, users, items):
        return self.baseline[users, items] + super().forward(users, items)

    def full_scores(self, batch_size=16):
        return self.baseline.cpu().numpy() + super().full_scores(batch_size)


def train(users, items, baseline, pairs, seen, mask, mapping, truth, hidden, seed, epochs, folder):
    torch.manual_seed(seed)
    model = ResidualMLP(users, items, hidden, baseline)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    rng = np.random.default_rng(seed)
    initial_scores = model.full_scores()
    np.testing.assert_array_equal(initial_scores, baseline.numpy())
    metrics = evaluate(initial_scores, mask, mapping['users'], mapping['items'], truth)
    best = {'epoch': 0, 'metrics': metrics, 'state': copy.deepcopy(model.state_dict()), 'scores': initial_scores.copy()}
    history = [{'epoch': 0, 'loss': None, **metrics['aggregate']}]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(pairs))
        negatives = sample_negatives(rng, pairs[:, 0], seen)
        loss_sum = 0.
        model.train()
        for start in range(0, len(order), 2048):
            idx = order[start:start + 2048]
            u, positive = torch.from_numpy(pairs[idx].T.copy())
            negative = torch.from_numpy(negatives[idx])
            optimizer.zero_grad(set_to_none=True)
            loss = F.softplus(model(u, negative) - model(u, positive)).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)
        model.eval()
        scores = model.full_scores()
        metrics = evaluate(scores, mask, mapping['users'], mapping['items'], truth)
        history.append({'epoch': epoch, 'loss': loss_sum / len(pairs), **metrics['aggregate']})
        if metrics['aggregate']['NDCG@10'] > best['metrics']['aggregate']['NDCG@10']:
            best = {'epoch': epoch, 'metrics': metrics, 'state': copy.deepcopy(model.state_dict()), 'scores': scores.copy()}
        if epoch % 5 == 0:
            print(f"  epoch={epoch} NDCG={metrics['aggregate']['NDCG@10']:.6f} best_epoch={best['epoch']}", flush=True)
        write_json(folder/'history.json', history)
    torch.save({'state_dict': best.pop('state'), 'hidden': hidden, 'seed': seed, 'epoch': best['epoch']}, folder/'model.pt')
    np.savez_compressed(folder/'scores-valid.npz', scores=best.pop('scores'))
    write_json(folder/'metrics.json', best['metrics'])
    return {'seed': seed, 'hidden': hidden, 'selected_epoch': best['epoch'],
            'directory': folder.name, **best['metrics']['aggregate']}


def run(previous, source, output, epochs=40):
    started = time.monotonic()
    previous, source, output = Path(previous).resolve(), Path(source).resolve(), new_run(output)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    old = json.loads((previous/'manifest.json').read_text())
    mapping = json.loads((previous/'mapping.json').read_text())
    verification = json.loads((previous/'verification.json').read_text())
    checks = {t['trial']: t for t in verification['trial_checks']}
    if old['status'] != 'complete' or verification['status'] != 'passed':
        raise ValueError('Verified previous experiment required')
    for part in ('train', 'valid'):
        if file_hash(source/f'splits/{part}.csv') != old['sha256'][part]:
            raise ValueError('Split changed')
    training, truth = [read_history(source/f'splits/{part}.csv') for part in ('train', 'valid')]
    uid = {u: i for i, u in enumerate(mapping['users'])}
    iid = {item: i for i, item in enumerate(mapping['items'])}
    pairs = np.array([(uid[u], iid[i]) for u in mapping['users'] for i in sorted(training[u])], dtype=np.int64)
    mask = np.load(previous/'candidate-mask.npz')['eligible']
    seen = np.zeros_like(mask)
    seen[pairs[:, 0], pairs[:, 1]] = True
    if not np.array_equal(mask, ~seen) or any(training[u] & truth[u] for u in training):
        raise ValueError('Invalid mask or split overlap')
    baseline = torch.from_numpy(np.load(previous/'scores-valid-mf.npz')['scores'])
    baseline_metrics = evaluate(baseline.numpy(), mask, mapping['users'], mapping['items'], truth)
    if baseline_metrics != json.loads((previous/'metrics-mf.json').read_text()):
        raise ValueError('Original MF metrics changed')
    features, hashes = {}, {}
    for variant in ('mlp', 'mlp-content'):
        name = f'{variant}-h64-seed42'
        path = previous/name/'model.pt'
        if file_hash(path) != checks[name]['checkpoint_sha256']:
            raise ValueError('Saved frozen features changed')
        saved = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
        features[variant] = (saved['users'], saved['items'])
        hashes[variant] = file_hash(path)
    torch.testing.assert_close(features['mlp'][0], features['mlp-content'][0], rtol=0, atol=0)
    embedding_size = features['mlp'][1].shape[1]
    torch.testing.assert_close(features['mlp'][1], features['mlp-content'][1][:, :embedding_size], rtol=0, atol=0)
    for name in ('mapping.json', 'scalers.json', 'candidate-mask.npz', 'scores-valid-mf.npz', 'metrics-mf.json'):
        (output/name).write_bytes((previous/name).read_bytes())
    proof = {'status': 'running', 'split_id': old['split_id'], 'test_evaluated': False,
             'test_split_read': False, 'previous_test_results_used_for_selection': False,
             'previous_run': str(previous), 'frozen_embeddings': True,
             'architecture': 'fixed raw MF dot product + MLP(concat); MLP hidden widths h,h/2 with ReLU; zero output layer',
             'seeds': [42, 43, 44], 'widths': [64, 128], 'max_epochs': epochs,
             'selection': 'best validation NDCG checkpoint including epoch zero; evaluate every epoch; width by mean across seeds',
             'optimizer': old['optimizer'], 'loss': old['loss'], 'negative_sampling': old['negative_sampling'],
             'sha256': {'source': file_hash(__file__), 'feature_helpers': file_hash(Path(__file__).with_name('feature_combination.py')),
                        'previous_manifest': file_hash(previous/'manifest.json'), 'feature_checkpoints': hashes,
                        'mf_scores': file_hash(previous/'scores-valid-mf.npz'),
                        'train': file_hash(source/'splits/train.csv'), 'valid': file_hash(source/'splits/valid.csv')},
             'users': len(mapping['users']), 'items': len(mapping['items']), 'training_pairs': len(pairs)}
    write_json(output/'manifest.json', proof)
    trials = []
    for variant, (users, items) in features.items():
        for hidden in proof['widths']:
            for seed in proof['seeds']:
                folder = output/f'residual-{variant}-h{hidden}-seed{seed}'
                folder.mkdir()
                print(folder.name, flush=True)
                trial = train(users, items, baseline, pairs, seen, mask, mapping, truth, hidden, seed, epochs, folder)
                trials.append({'variant': variant, **trial})
                write_json(output/'trials.json', trials)
    results = {'mf': {'aggregate': baseline_metrics['aggregate'], 'seed_count': 1}}
    for variant in features:
        means = {h: float(np.mean([t['NDCG@10'] for t in trials if t['variant'] == variant and t['hidden'] == h])) for h in proof['widths']}
        width = max(proof['widths'], key=lambda h: means[h])
        chosen = [t for t in trials if t['variant'] == variant and t['hidden'] == width]
        keys = baseline_metrics['aggregate']
        results[variant] = {'selected_width': width, 'width_mean_ndcg': means, 'seed_count': len(chosen), 'trials': chosen,
                            'aggregate': {m: float(np.mean([t[m] for t in chosen])) for m in keys},
                            'std': {m: float(np.std([t[m] for t in chosen], ddof=1)) for m in keys}}
    results['paired_content_minus_mlp'] = [{'seed': a['seed'], **{m: b[m] - a[m] for m in baseline_metrics['aggregate']}}
        for a, b in zip(results['mlp']['trials'], results['mlp-content']['trials'])]
    write_json(output/'results.json', results)
    proof.update(status='complete', elapsed_seconds=time.monotonic()-started, epoch_zero_matches_mf_exactly=True)
    write_json(output/'manifest.json', proof)
    lines = ['# Residual feature combination: validation', '',
             'Frozen original MF scores plus a two-hidden-layer correction. Output layer initialized to zero. Epoch zero is eligible for selection and exactly reproduces MF.', '',
             'Same frozen features, training split, masks, negative sampling, Adam settings, widths {64,128}, and seeds {42,43,44} as the previous experiment. Up to 40 epochs; validation evaluated every epoch (previous experiment every five). Width selected by mean validation NDCG over seeds. No test evaluation.', '',
             '| Model | Recall@10 | MRR@10 | NDCG@10 |', '|---|---:|---:|---:|']
    for variant in ('mf', 'mlp', 'mlp-content'):
        row = results[variant]
        values = [f"{row['aggregate'][m]:.6f}" + (f" +/- {row['std'][m]:.6f}" if 'std' in row else '') for m in ('Recall@10','MRR@10','NDCG@10')]
        lines.append('| '+variant+' | '+' | '.join(values)+' |')
    lines += ['', '| Variant | Width | Seed | Selected epoch | NDCG@10 |', '|---|---:|---:|---:|---:|']
    for variant in features:
        lines += [f"| {variant} | {t['hidden']} | {t['seed']} | {t['selected_epoch']} | {t['NDCG@10']:.6f} |" for t in results[variant]['trials']]
    lines += ['', 'These validation-selected development results are conditional on one MF checkpoint. Including epoch zero guarantees no decrease in selected validation NDCG, not in other metrics or held-out performance. The original test set was inspected earlier in this research sequence; it should not be treated as a fresh independent holdout for subsequent design iterations.']
    (output/'validation.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({v: results[v]['aggregate'] for v in ('mf','mlp','mlp-content')}, indent=2), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', default='project/artifacts/feature-combination-mf-content')
    parser.add_argument('--source', default='project/artifacts/hybrid-60-20-20')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(run(args.previous, args.source, args.output))
