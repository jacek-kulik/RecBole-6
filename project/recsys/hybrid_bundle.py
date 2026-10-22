"""Portable inference for the committed fitted alternate-hybrid experiment bundle.

Only NumPy arrays, sparse matrices and JSON are loaded; no machine-specific
RecBole Config pickles or original artifact paths are needed for inference.
"""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz


class HybridBundle:
    def __init__(self, root='project/models/alternate-hybrids-60-20-20'):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'models.json').read_text())
        self.models = self.manifest['models']
        with np.load(self.root / 'catalogue.npz', allow_pickle=False) as data:
            self.users, self.items = data['users'].tolist(), data['items'].tolist()
            self.ui, self.ii = data['internal_users'], data['internal_items']
            self.positives = {part: {u: set() for u in self.users} for part in ('valid', 'test')}
            for part in ('valid', 'test'):
                for u, i in zip(data[part + '_users'], data[part + '_items']):
                    self.positives[part][self.users[u]].add(self.items[i])
        self.internal_train = load_npz(self.root / 'training.npz')
        self.train = self.internal_train[self.ui][:, self.ii].toarray()
        self.cache = {}

    def scores(self, name):
        if name in self.cache:
            return self.cache[name]
        spec = self.models[name]
        kind = spec['kind']
        if kind == 'bpr':
            with np.load(self.root / spec['file'], allow_pickle=False) as z:
                values = z['users'] @ z['items'].T
        elif kind in ('userknn', 'itemknn', 'linear'):
            if kind == 'linear':
                with np.load(self.root / spec['file'], allow_pickle=False) as z:
                    weights = z['weights']
            else:
                weights = load_npz(self.root / spec['file'])
            values = (weights @ self.internal_train if kind == 'userknn'
                      else self.internal_train @ weights)
            if hasattr(values, 'toarray'):
                values = values.toarray()
            values = np.asarray(values)[self.ui][:, self.ii]
        elif kind == 'switch':
            values = self.scores(spec['userknn']).copy()
            selected = np.array(spec['bpr_users'], dtype=int)
            values[selected] = self.scores(spec['bpr'])[selected]
        elif kind == 'augmentation':
            additions = load_npz(self.root / spec['additions']).toarray()
            with np.load(self.root / spec['file'], allow_pickle=False) as z:
                values = (self.train.astype(np.float64) + additions) @ z['weights']
        elif kind == 'meta':
            values = load_npz(self.root / spec['file']).toarray() @ self.train.astype(np.float64)
        elif kind == 'mlp':
            import torch
            from .feature_combination import PairMLP
            from .residual_combination import ResidualMLP

            with np.load(self.root / 'mlp-features.npz', allow_pickle=False) as z:
                users = torch.from_numpy(z['users'])
                items = torch.from_numpy(z[spec['features']])
                model = (ResidualMLP(users, items, spec['hidden'], torch.from_numpy(z['baseline']))
                         if spec['residual'] else PairMLP(users, items, spec['hidden']))
            state = model.state_dict()
            with np.load(self.root / spec['file'], allow_pickle=False) as z:
                state.update({key: torch.from_numpy(z[key]) for key in z.files})
            model.load_state_dict(state)
            model.eval()
            values = model.full_scores()
        elif kind == 'cascade':
            values = self.scores(spec['bpr'])
        else:
            raise ValueError(f'Unknown model kind: {kind}')
        if values.shape != self.train.shape or not np.isfinite(values).all():
            raise ValueError('Invalid score matrix')
        self.cache[name] = values
        return values

    def mask(self, partition, scope='full'):
        if partition not in ('valid', 'test'):
            raise ValueError('Unknown evaluation partition')
        eligible = self.train == 0
        if partition == 'test':
            index = {i: n for n, i in enumerate(self.items)}
            for n, u in enumerate(self.users):
                eligible[n, [index[i] for i in self.positives['valid'][u]]] = False
        if scope == 'warm':
            eligible &= (self.train.sum(axis=0) > 0)[None]
        return eligible

    def truth(self, partition, scope='full'):
        if scope == 'full':
            return self.positives[partition]
        supported = {i for i, n in zip(self.items, self.train.sum(axis=0)) if n > 0}
        return {u: positives & supported for u, positives in self.positives[partition].items()}

    def rankings(self, name, partition='valid', scope='full'):
        from project.scripts.retune_ease_slim import rankings

        eligible = self.mask(partition, scope)
        spec = self.models[name]
        if spec['kind'] == 'cascade':
            retrieval = np.where(eligible, self.scores(spec['itemknn']), -np.inf)
            top = np.argsort(-retrieval, axis=1, kind='stable')[:, :spec['shortlist']]
            selected = np.zeros_like(eligible)
            np.put_along_axis(selected, top, True, axis=1)
            eligible &= selected
        return rankings(self.scores(name), eligible, self.users, self.items)
