"""Export existing selected fits as portable arrays and compact experiment records.

This packages frozen experiments; it never fits, tunes, or selects on test data.
Original source artifacts remain untouched. Run from the repository root.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix, save_npz

from project.recsys.artifacts import write_json
from project.recsys.component_cache import file_hash
from project.recsys.meta_level import cosine, neighbour_weights


ARTIFACTS = Path('project/artifacts')
RESULTS = Path('hyperparameters/results/alternate-hybrids-60-20-20')
MODELS = Path('project/models/alternate-hybrids-60-20-20')


def package():
    from recbole.data import create_dataset
    from recbole.utils import init_seed

    if (RESULTS / 'package.json').exists():
        raise FileExistsError('Completed package already exists; preserve it or use new output paths')
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    provenance, models = {}, {}

    def record(path):
        path = Path(path)
        provenance[str(path)] = file_hash(path)
        return path

    def read(path):
        return json.loads(record(path).read_text())

    source = ARTIFACTS / 'alternate-corrected-source'
    pointer = read(source / 'cache-bpr.json')
    old_bpr_dir = Path(pointer['path'])
    checkpoint_path = next((old_bpr_dir / 'full-training').glob('*.pth'))
    checkpoint = torch.load(record(checkpoint_path), map_location='cpu', weights_only=False)
    init_seed(42, True)
    dataset = create_dataset(checkpoint['config'])
    parts = dataset.build()
    mapping = read(ARTIFACTS / 'feature-combination-retuned-bpr/mapping.json')
    users, items = mapping['users'], mapping['items']
    ui, ii = dataset.token2id(dataset.uid_field, users), dataset.token2id(dataset.iid_field, items)
    ux, ix = {u: n for n, u in enumerate(users)}, {i: n for n, i in enumerate(items)}
    coordinates = {}
    for part in ('train', 'valid', 'test'):
        rows, cols = [], []
        with record(source / f'splits/{part}.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                rows.append(ux[row['user_id']]); cols.append(ix[row['item_id']])
        coordinates[part + '_users'], coordinates[part + '_items'] = np.array(rows), np.array(cols)
    train = csr_matrix((np.ones(len(coordinates['train_users']), dtype=np.float32),
                        (ui[coordinates['train_users']], ii[coordinates['train_items']])),
                       shape=(dataset.user_num, dataset.item_num))
    save_npz(MODELS / 'training.npz', train)
    np.savez_compressed(MODELS / 'catalogue.npz', users=np.array(users), items=np.array(items),
                        internal_users=ui, internal_items=ii, **coordinates)

    def native(name, directory, kind, parameters):
        directory = Path(directory)
        folder = directory / ('full-training' if (directory / 'full-training').exists() else 'training')
        files = list(folder.glob('*.pth'))
        if len(files) != 1:
            raise ValueError('Expected one selected checkpoint')
        saved = torch.load(record(files[0]), map_location='cpu', weights_only=False)
        if kind == 'bpr':
            p = saved['state_dict']['user_embedding.weight'][ui].numpy()
            q = saved['state_dict']['item_embedding.weight'][ii].numpy()
            np.savez_compressed(MODELS / f'{name}.npz', users=p, items=q)
        else:
            other = saved['other_parameter']
            if 'interaction_matrix' in other and (csr_matrix(other['interaction_matrix']) != train).nnz:
                raise ValueError('Selected component used another training matrix')
            weights = other['w'] if 'w' in other else other['item_similarity']
            if kind == 'linear':
                np.savez_compressed(MODELS / f'{name}.npz', weights=np.asarray(weights))
            else:
                save_npz(MODELS / f'{name}.npz', csr_matrix(weights))
        models[name] = {'kind': kind, 'file': f'{name}.npz', 'parameters': parameters}

    native('bpr-inherited', old_bpr_dir, 'bpr', read(source / 'manifest.json')['component_parameters']['bpr'])
    cascade = ARTIFACTS / 'cascade-itemknn-bpr'
    selected = read(cascade / 'frozen-selection.json')
    native('bpr', Path(selected['components']['bpr']['path']).parent, 'bpr', selected['bpr'])
    native('cascade-itemknn', Path(selected['components']['itemknn']['path']).parent, 'itemknn', selected['itemknn'])
    standalone, paths = read(cascade / 'standalone-selection.json'), read(cascade / 'component-paths.json')
    item_path = next(Path(p).parent for p in paths['itemknn'] if read(Path(p).parent / 'parameters.json') == standalone['itemknn'])
    native('itemknn', item_path, 'itemknn', standalone['itemknn'])
    for name, short in (('EASE', 'ease'), ('SLIMElastic', 'slim')):
        selection = read(f'hyperparameters/results/individual-retuned-60-20-20/{name}/selection.json')
        saved = torch.load(record(selection['best']['checkpoint']), map_location='cpu', weights_only=False)
        other = saved['other_parameter']
        if (csr_matrix(other['interaction_matrix']) != train).nnz:
            raise ValueError('Standalone training matrix differs')
        if short == 'ease':
            np.savez_compressed(MODELS / f'{short}.npz', weights=np.asarray(other['item_similarity']))
            kind = 'linear'
        else:
            save_npz(MODELS / f'{short}.npz', csr_matrix(other['item_similarity']))
            kind = 'itemknn'
        models[short] = {'kind': kind, 'file': f'{short}.npz', 'parameters': selection['parameters']}
    # Preserve the exact native UserKNN implementation for the retuned control.
    from recbole.model.general_recommender.itemknn import ItemKNN
    from project.scripts.retune_ease_slim import config_for
    cfg = config_for('ItemKNN', {'knn_method': 'user', 'k': 160, 'shrink': 0})
    if (parts[0].inter_matrix(form='csr') != train).nnz:
        raise ValueError('Reconstructed training split differs')
    user_model = ItemKNN(cfg, parts[0])
    save_npz(MODELS / 'userknn.npz', csr_matrix(user_model.w))
    models['userknn'] = {'kind': 'userknn', 'file': 'userknn.npz', 'parameters': {'k': 160, 'shrink': 0}}

    rule = read(ARTIFACTS / 'switch-retuned-components/selected-rule.json')
    routes = read(ARTIFACTS / 'switch-retuned-components/routes.json')
    models['switch'] = {'kind': 'switch', 'userknn': 'userknn', 'bpr': 'bpr-inherited',
                        'bpr_users': [n for n, u in enumerate(users) if routes[u] == 'bpr'], 'rule': rule}
    models['cascade'] = {'kind': 'cascade', 'itemknn': 'cascade-itemknn', 'bpr': 'bpr',
                         'shortlist': selected['selection']['shortlist_size']}
    with np.load(record(ARTIFACTS / 'augmentation-retuned-slim-025/baseline-model.npz'), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['users'], users)
        np.testing.assert_array_equal(z['items'], items)
        np.savez_compressed(MODELS / 'paired-slim.npz', weights=z['weights'])
    save_npz(MODELS / 'paired-slim-additions.npz', csr_matrix((len(users), len(items))))
    models['paired-slim'] = {'kind': 'augmentation', 'file': 'paired-slim.npz',
                             'additions': 'paired-slim-additions.npz',
                             'parameters': {'alpha': .2, 'l1_ratio': .005, 'max_iter': 1000}}
    for name, directory in [('aug-ease-weak', 'augmentation-retuned-ease-025'),
                            ('aug-slim-weak', 'augmentation-retuned-slim-025'),
                            ('aug-slim-comparable', 'augmentation-retuned-slim-100')]:
        directory = ARTIFACTS / directory
        manifest = read(directory / 'manifest.json')
        with np.load(record(directory / 'model.npz'), allow_pickle=False) as z:
            if list(z['users']) != users or list(z['items']) != items:
                raise ValueError('Augmentation token mapping differs')
            np.savez_compressed(MODELS / f'{name}.npz', weights=z['weights'])
            save_npz(MODELS / f'{name}-additions.npz', csr_matrix(z['additions']))
        models[name] = {'kind': 'augmentation', 'file': f'{name}.npz', 'additions': f'{name}-additions.npz',
                        'parameters': {k: manifest[k] for k in ('strength', 'top_n', 'reg', 'slim_alpha', 'slim_l1_ratio') if k in manifest},
                        'sender': 'userknn'}
    meta_k = read(ARTIFACTS / 'meta-retuned-bpr/selected.json')['bpr-userknn']['k']
    with np.load(record(ARTIFACTS / 'meta-retuned-bpr/representation.npz'), allow_pickle=False) as z:
        neighbours = neighbour_weights(cosine(z['user_embeddings']), meta_k)
    save_npz(MODELS / 'meta.npz', csr_matrix(neighbours))
    models['meta'] = {'kind': 'meta', 'file': 'meta.npz', 'parameters': {'k': meta_k, 'similarity': 'positive cosine'}}
    control_k = read(ARTIFACTS / 'meta-retuned-bpr/selected.json')['interaction-cosine-control']['k']
    canonical_train = train[ui][:, ii].toarray().astype(np.float64)
    save_npz(MODELS / 'meta-control.npz', csr_matrix(neighbour_weights(cosine(canonical_train), control_k)))
    models['meta-control'] = {'kind': 'meta', 'file': 'meta-control.npz',
                              'parameters': {'k': control_k, 'representation': 'binary training interactions'}}
    features = {}
    for family, folder in [('mlp', 'feature-combination-retuned-bpr'), ('residual', 'residual-retuned-bpr')]:
        directory = ARTIFACTS / folder
        results = read(directory / 'results.json')
        for variant in ('mlp', 'mlp-content'):
            feature_key = 'items_content' if variant == 'mlp-content' else 'items_cf'
            for trial in results[variant]['trials']:
                name = f'{family}-{variant}-seed{trial["seed"]}'
                saved = torch.load(record(directory / trial['directory'] / 'model.pt'), map_location='cpu', weights_only=False)
                state = saved['state_dict']
                for key, value in [('users', state['users']), (feature_key, state['items'])]:
                    if key in features:
                        np.testing.assert_array_equal(features[key], value.numpy())
                    features[key] = value.numpy()
                if 'baseline' in state:
                    features['baseline'] = state['baseline'].numpy()
                params = {k: v.numpy() for k, v in state.items() if k not in ('users', 'items', 'baseline')}
                np.savez_compressed(MODELS / f'{name}.npz', **params)
                models[name] = {'kind': 'mlp', 'file': f'{name}.npz', 'features': feature_key,
                                'hidden': trial['hidden'], 'residual': family == 'residual',
                                'seed': trial['seed'], 'selected_epoch': trial['selected_epoch']}
    np.savez_compressed(MODELS / 'mlp-features.npz', **features)

    # Compact original records preserve grids, every search trial, selection and
    # aggregate results; large score tables and per-user JSON are reproducible.
    record_files = {
        'switching': ('switch-retuned-components', ['manifest.json', 'selected-rule.json', 'best-nonconstant-rule.json', 'trials.json', 'diagnostics.json']),
        'cascade': ('cascade-itemknn-bpr', ['manifest.json', 'frozen-selection.json', 'standalone-selection.json', 'trials.json', 'retrieval.json']),
        'feature-combination': ('feature-combination-retuned-bpr', ['manifest.json', 'results.json', 'trials.json', 'verification.json', 'scalers.json']),
        'residual-combination': ('residual-retuned-bpr', ['manifest.json', 'results.json', 'trials.json', 'verification.json']),
        'meta-level': ('meta-retuned-bpr', ['manifest.json', 'selected.json', 'trials.json']),
    }
    for name, (folder, files) in record_files.items():
        target = RESULTS / name
        target.mkdir(exist_ok=True)
        for file in files:
            write_json(target / file, read(ARTIFACTS / folder / file))
        if name in ('feature-combination', 'residual-combination'):
            histories = {t['directory']: read(ARTIFACTS / folder / t['directory'] / 'history.json')
                         for t in read(ARTIFACTS / folder / 'trials.json')}
            write_json(target / 'histories.json', histories)
    write_json(RESULTS / 'userknn-trials.json', read(ARTIFACTS / 'alternate-hybrids-retuned-comparison-verified/userknn-trials.json'))
    target = RESULTS / 'augmentation'
    target.mkdir(exist_ok=True)
    for folder in ('augmentation-retuned-ease-025', 'augmentation-retuned-slim-025', 'augmentation-retuned-slim-100'):
        write_json(target / f'{folder}.json', read(ARTIFACTS / folder / 'manifest.json'))
    references = {}
    def reference(key, result):
        references[key] = result['aggregate'] if 'aggregate' in result else result
    for name, result in read(ARTIFACTS / 'switch-retuned-components/metrics.json').items():
        reference(f'valid/warm/{name}', result)
    reference('valid/full/meta', read(ARTIFACTS / 'meta-retuned-bpr/metrics.json')['bpr-userknn'])
    reference('valid/full/meta-control', read(ARTIFACTS / 'meta-retuned-bpr/metrics.json')['interaction-cosine-control'])
    for part, values in read(cascade / 'metrics.json').items():
        for name, recorded in [('cascade', 'cascade'), ('bpr', 'retuned-bpr'), ('itemknn', 'retuned-itemknn')]:
            reference(f'{part}/full/{name}', values[recorded])
    for name, folder in [('aug-ease-weak', 'augmentation-retuned-ease-025'),
                         ('aug-slim-weak', 'augmentation-retuned-slim-025'),
                         ('aug-slim-comparable', 'augmentation-retuned-slim-100')]:
        reference(f'valid/full/{name}', read(ARTIFACTS / folder / 'metrics-augmented.json'))
    reference('valid/full/paired-slim', read(ARTIFACTS / 'augmentation-retuned-slim-025/metrics-baseline.json'))
    test_aug = read(ARTIFACTS / 'augmentation-retuned-test-complete/metrics.json')
    for name, recorded in [('aug-ease-weak', 'ease-strength-0.25'), ('aug-slim-weak', 'slimelastic-strength-0.25'),
                           ('aug-slim-comparable', 'slimelastic-strength-1'), ('paired-slim', 'slimelastic-baseline'),
                           ('ease', 'ease-baseline'), ('userknn', 'userknn')]:
        reference(f'test/full/{name}', test_aug[recorded])
    for family, folder in [('mlp', 'feature-combination-retuned-bpr'), ('residual', 'residual-retuned-bpr')]:
        for part in (('valid', 'test') if family == 'mlp' else ('valid',)):
            root = ARTIFACTS / folder / ('test' if part == 'test' else '')
            for variant in ('mlp', 'mlp-content'):
                result = read(root / 'results.json')[variant]
                reference(f'{part}/full/{family}-{variant}', result)
                for trial in result['trials']:
                    reference(f'{part}/full/{family}-{variant}-seed{trial["seed"]}',
                              {k: trial[k] for k in references[f'{part}/full/{family}-{variant}']})
    for name, short in [('EASE', 'ease'), ('SLIMElastic', 'slim')]:
        reference(f'valid/full/{short}', read(Path('hyperparameters/results/individual-retuned-60-20-20') / name / 'selection.json')['best']['validation'])
    write_json(RESULTS / 'reference-metrics.json', references)
    manifest = {'split_id': read(ARTIFACTS / 'meta-retuned-bpr/manifest.json')['split_id'],
                'seed': 42, 'scope': 'Existing training users and catalogue only; no cold-start claim',
                'models': models, 'source_sha256': provenance}
    write_json(MODELS / 'models.json', manifest)
    write_json(RESULTS / 'package.json', {'status': 'complete', 'split_id': manifest['split_id'],
               'model_count': len(models), 'models': str(MODELS),
               'files_sha256': {p.name: file_hash(p) for p in MODELS.iterdir()}})
    print(f'Packaged {len(models)} model entries; {sum(p.stat().st_size for p in MODELS.iterdir())/1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        package()
