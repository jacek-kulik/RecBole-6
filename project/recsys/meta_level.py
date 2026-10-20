"""Validation-only BPR user embeddings -> neighbourhood collaborative filtering."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import new_run, write_json
from .component_cache import file_hash
from .contracts import ScoreTable, require_aligned
from .metrics.accuracy import evaluate_accuracy
from .recbole_adapter import load_scores, _write_scores


def read_history(path):
    history = {}
    with Path(path).open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            seen = history.setdefault(row['user_id'], set())
            if row['item_id'] in seen:
                raise ValueError('Duplicate training/relevance pair')
            seen.add(row['item_id'])
    return history


def cosine(features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError('Expected finite feature matrix')
    lengths = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = np.divide(features, lengths, out=np.zeros_like(features), where=lengths > 0)
    return normalized @ normalized.T


def neighbour_weights(similarities, k):
    """Positive cosine only; exclude self; ties follow sorted external user IDs."""
    if k < 1 or similarities.ndim != 2 or similarities.shape[0] != similarities.shape[1]:
        raise ValueError('Positive k and square similarities required')
    if not np.isfinite(similarities).all():
        raise ValueError('Nonfinite similarities')
    weights = np.maximum(similarities, 0).copy()
    np.fill_diagonal(weights, 0)
    order = np.argsort(-weights, axis=1, kind='stable')[:, :min(k, len(weights) - 1)]
    selected = np.zeros_like(weights)
    rows = np.arange(len(weights))[:, None]
    selected[rows, order] = weights[rows, order]
    return selected


def score_table(scores, reference, users, items, model):
    item_index = {item: n for n, item in enumerate(items)}
    table = ScoreTable(model, reference.split_id, 'valid', {
        user: {item: float(scores[n, item_index[item]]) for item in reference.scores[user]}
        for n, user in enumerate(users)})
    require_aligned([reference, table])
    return table


def run(source, output, ks=(10, 20, 40, 50, 80, 160, 320)):
    from recbole.data import create_dataset

    torch.set_num_threads(4)
    source, output = Path(source).resolve(), new_run(output)
    manifest = json.loads((source / 'manifest.json').read_text())
    training = read_history(source / 'splits/train.csv')
    relevant = read_history(source / 'splits/valid.csv')
    references, caches, provenance = {}, {}, {}
    for name in ('bpr', 'userknn'):
        pointer = json.loads((source / f'cache-{name}.json').read_text())
        cache = caches[name] = Path(pointer['path'])
        path = cache / 'scores-valid.json.gz'
        digest = file_hash(path)
        if digest != json.loads((cache / 'full.complete.json').read_text())['sha256']:
            raise ValueError(f'Corrupt {name} cache')
        references[name] = load_scores(path)
        if (references[name].model, references[name].partition, references[name].split_id) != (
                name, 'valid', manifest['split_id']):
            raise ValueError('Cache identity mismatch')
        provenance[name] = {'key': pointer['key'], 'path': str(path), 'sha256': digest}
    require_aligned(list(references.values()))
    reference = references['bpr']
    paths = list((caches['bpr'] / 'full-training').glob('*.pth'))
    if len(paths) != 1:
        raise ValueError('Expected one full BPR checkpoint')
    checkpoint = torch.load(paths[0], map_location='cpu', weights_only=False)
    # Reconstruct token mapping only; do not build/split or fit this dataset.
    dataset = create_dataset(checkpoint['config'])
    users = sorted(training)
    items = sorted(str(i) for i in dataset.field2id_token[dataset.iid_field][1:])
    if set(users) != set(relevant) or set(users) != set(reference.scores):
        raise ValueError('Training/validation/cache users differ')
    item_index = {item: n for n, item in enumerate(items)}
    interactions = np.zeros((len(users), len(items)), dtype=np.float64)
    for n, user in enumerate(users):
        if training[user] & relevant[user]:
            raise ValueError('Training and validation overlap')
        if set(reference.scores[user]) != set(items) - training[user]:
            raise ValueError('Candidate mask mismatch')
        if not relevant[user] <= reference.scores[user].keys():
            raise ValueError('Missing validation positives')
        interactions[n, [item_index[i] for i in training[user]]] = 1
    p = checkpoint['state_dict']['user_embedding.weight'][dataset.token2id(dataset.uid_field, users)].numpy()
    q = checkpoint['state_dict']['item_embedding.weight'][dataset.token2id(dataset.iid_field, items)].numpy()
    reconstructed = p @ q.T
    max_error = max(float(np.max(np.abs(
        reconstructed[n, [item_index[i] for i in reference.scores[u]]] -
        np.array(list(reference.scores[u].values()))))) for n, u in enumerate(users))
    if max_error > 1e-5:
        raise ValueError(f'Checkpoint/cache mapping mismatch: {max_error}')
    proof = {'status': 'running', 'split_id': reference.split_id, 'test_split_read': False,
             'test_evaluated': False, 'metadata_features_used': False,
             'dataset_access': 'Original dataset read only to reconstruct checkpoint token mapping; no build/split/fitting',
             'transferred_object': 'Frozen BPR user embeddings; item embeddings used only to verify checkpoint scores',
             'scorer': 'sum of positive cosine neighbour weights times binary training interactions; self excluded',
             'fallback': 'Zero weights produce zero scores, ties by external item ID; no popularity blending',
             'selection': 'validation NDCG@10; ties prefer smaller k', 'k_grid': list(ks),
             'scope': 'Full original validation users and catalogue minus training items',
             'users': len(users), 'items': len(items), 'training_pairs': int(interactions.sum()),
             'checkpoint': str(paths[0]), 'checkpoint_sha256': file_hash(paths[0]),
             'checkpoint_epoch': checkpoint['epoch'], 'checkpoint_cache_max_abs_error': max_error,
             'components': provenance,
             'sha256': {'source': file_hash(__file__), 'train': file_hash(source / 'splits/train.csv'),
                        'valid': file_hash(source / 'splits/valid.csv'),
                        'accuracy': file_hash(Path(__file__).parent / 'metrics/accuracy.py')}}
    write_json(output / 'manifest.json', proof)
    np.savez_compressed(output / 'representation.npz', user_embeddings=p, users=np.array(users), items=np.array(items))
    metrics = {name: evaluate_accuracy(table.top_k(10), relevant, 10) for name, table in references.items()}
    trials, selected = [], {}
    for name, features in (('bpr-userknn', p), ('interaction-cosine-control', interactions)):
        similarities = cosine(features)
        best = None
        for k in sorted(set(ks)):
            weights = neighbour_weights(similarities, k)
            table = score_table(weights @ interactions, reference, users, items, name)
            result = evaluate_accuracy(table.top_k(10), relevant, 10)
            row = {'model': name, 'k': k, **result['aggregate']}
            trials.append(row)
            print(json.dumps(row), flush=True)
            if best is None or row['NDCG@10'] > best[0]['NDCG@10']:
                best = (row, table, result, weights)
        row, table, result, weights = best
        selected[name] = row
        metrics[name] = result
        _write_scores(table, output / f'scores-valid-{name}.json.gz')
        write_json(output / f'neighbours-{name}.json', {
            u: [{'user': users[j], 'weight': float(weights[n, j])}
                for j in np.argsort(-weights[n], kind='stable') if weights[n, j] > 0]
            for n, u in enumerate(users)})
    write_json(output / 'trials.json', trials)
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'selected.json', selected)
    lines = ['# BPR embeddings -> UserKNN: validation', '',
             'Frozen BPR user vectors define cosine neighbours. Neighbours vote using binary training interactions. '
             'No metadata or BPR predictions enter the hybrid scorer. Only positive similarities are retained; self is excluded.', '',
             f'Scope: {len(users)} users, {len(items)} catalogue items, exact original candidate masks. '
             'All validation interactions are relevant under the existing protocol. Test split is not read or evaluated.', '',
             '| Model | Selected k | Recall@10 | MRR@10 | NDCG@10 |', '|---|---:|---:|---:|---:|']
    for name, result in metrics.items():
        k = selected[name]['k'] if name in selected else ('50' if name == 'userknn' else '-')
        lines.append(f'| {name} | {k} | ' + ' | '.join(f'{result["aggregate"][m+"@10"]:.6f}' for m in ('Recall', 'MRR', 'NDCG')) + ' |')
    lines += ['', 'The interaction-cosine control uses the identical scorer, tie handling and k grid; '
              'only the user representation changes. Cached UserKNN uses its original RecBole implementation. '
              'Each new model selects k independently using validation NDCG@10. These are development results, '
              'not independent evidence of generalization. No retraining or embedding-dimension search was performed.']
    (output / 'validation.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    proof.update(status='complete', candidate_alignment_verified=True, selected=selected)
    write_json(output / 'manifest.json', proof)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='project/artifacts/hybrid-60-20-20')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(run(args.source, args.output))
