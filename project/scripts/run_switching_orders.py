"""Matched switching reranker orders, with the original warm scope and routes."""

import argparse
from pathlib import Path
import json

import numpy as np
from threadpoolctl import threadpool_limits

from project.recsys.artifacts import fingerprint, write_json
from project.recsys.component_cache import file_hash
from project.recsys.contracts import RerankContext, ScoreTable
from project.recsys.data import Interaction, prepare_metadata
from project.recsys.evaluation import evaluation_settings, write_ranking_results
from project.recsys.experiments.switching_orders import rerank_then_switch, switch_then_rerank, switch_scores
from project.recsys.hybrid_bundle import HybridBundle
from project.recsys.rerank import select_strength
from project.scripts.analyze_hybrid_coefficients import verify_metrics
from project.scripts.run_reranking_orders import METHODS, GRID, write_csv


ORDERS = {'pre': rerank_then_switch, 'post': switch_then_rerank}


def read(path):
    return json.loads(Path(path).read_text())


def prepare(bundle):
    package = read('hyperparameters/results/alternate-hybrids-60-20-20/package.json')
    for name, expected in package['files_sha256'].items():
        if file_hash(bundle.root / name) != expected:
            raise ValueError(f'Portable source changed: {name}')
    spec = bundle.models['switch']
    names = [spec['userknn'], spec['bpr']]
    selected = set(spec['bpr_users'])
    if len(selected) != 210 or len(bundle.users) != 943 or not selected <= set(range(len(bundle.users))):
        raise ValueError('Frozen routing identity differs')
    routes = {u: names[int(n in selected)] for n, u in enumerate(bundle.users)}
    supported = [i for i, mass in zip(bundle.items, bundle.train.sum(axis=0)) if mass > 0]
    if len(supported) != 1615:
        raise ValueError('Warm catalogue differs')
    training = [Interaction(bundle.users[u], bundle.items[i]) for u, i in zip(*np.nonzero(bundle.train))]
    stats = prepare_metadata(training, 'dataset/ml-100k/ml-100k.item',
                             'dataset/ml-100k/ml-100k.user', supported)
    context = RerankContext(item_features=stats.item_features, user_profiles=stats.genre_profiles,
                            popularity_profiles=stats.popularity_profiles,
                            user_groups=stats.gender_groups, item_groups=stats.item_groups)
    return names, routes, stats, context


def pooled_components(bundle, names, routes, part):
    mask = bundle.mask(part, 'warm')
    values = bundle.scores('switch')
    indices = np.argsort(-np.where(mask, values, -np.inf), axis=1, kind='stable')[:, :50]
    if any(not mask[n, row].all() for n, row in enumerate(indices)):
        raise ValueError('Insufficient warm candidates')
    tables = []
    for name in names:
        values = bundle.scores(name)
        tables.append(ScoreTable(name, bundle.manifest['split_id'], part,
                                {u: {bundle.items[i]: float(values[n, i]) for i in indices[n]}
                                 for n, u in enumerate(bundle.users)}))
    baseline = switch_scores(tables, routes).top_k(10)
    if baseline != bundle.rankings('switch', part, 'warm'):
        raise ValueError('Pooling changed the original switch top ten')
    return tables, baseline


def evaluate(out, bundle, stats, identity, part, name, ranked, method='original', order='baseline', strength=0., base=None):
    folder = out / part / name
    folder.mkdir(parents=True)
    manifest = {**evaluation_settings(part, 10, 'warm'), 'status': 'running',
                'model': 'hybrid-switch-'+name, 'dataset': 'ml-100k', 'split_id': bundle.manifest['split_id'],
                'seed': 42, 'input_identity': identity, 'n_catalogue_items': len(stats.catalogue),
                'n_training_interactions': stats.n_interactions, 'reranking_candidate_count': 50,
                'method': method, 'order': order, 'strength': strength, 'scope': 'warm'}
    try:
        accuracy, beyond, _ = write_ranking_results(folder, ranked, bundle.truth(part, 'warm'), stats, 10)
        manifest.update(status='complete', n_users=len(ranked), project_evaluation_complete=False)
    except Exception as error:
        manifest.update(status='failed', error=str(error)); raise
    finally:
        write_json(folder / 'manifest.json', manifest)
    row = {'partition': part, 'name': name, 'method': method, 'order': order, 'strength': strength,
           'ndcg': accuracy['aggregate']['NDCG@10'], **accuracy['aggregate'], **beyond['aggregate']}
    row['relative_ndcg_loss'] = 0. if base is None else 1 - row['ndcg'] / base
    row['target'] = None if method == 'original' else row[METHODS[method][1]]
    print(f"{part}/{name}: NDCG={row['ndcg']:.6f}; KL={row['GenreMiscalibration@10']:.4f}; "
          f"JS={row['UPD@10']:.4f}; Gini={row['ExposureGini@10']:.4f}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--results', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.results.exists(): raise ValueError('Use a fresh results directory')
    out = args.output
    bundle = HybridBundle()
    names, routes, stats, context = prepare(bundle)
    code = [Path(__file__), Path('project/recsys/experiments/switching_orders.py'),
            Path('project/recsys/hybrid_bundle.py'), Path('project/recsys/contracts.py'),
            Path('project/recsys/data.py'), Path('project/recsys/evaluation.py'),
            Path('project/recsys/metrics/accuracy.py'), Path('project/recsys/metrics/beyond_accuracy.py'),
            Path('project/recsys/rerank.py'), Path('project/scripts/run_reranking_orders.py'),
            *sorted(Path('project/recsys/rerankers').glob('*.py'))]
    protocol = {'hybrid': 'switching', 'split_id': bundle.manifest['split_id'], 'seed': 42,
                'scope': 'warm', 'users': 943, 'catalogue_items': 1615, 'k': 10, 'candidate_count': 50,
                'components': names, 'bpr_users': 210, 'userknn_users': 733, 'routing_frozen': True,
                'routing_rule': bundle.models['switch']['rule'],
                'candidate_retrieval': 'original switched score top50 warm eligible candidates; same pool for both orders',
                'pre_exposure_cohort': 'each component reranks all 943 users independently before frozen routing',
                'post_exposure_cohort': 'one counter across the 943 final switched user rows',
                'strengths': GRID, 'shared_component_strength': True, 'max_relative_validation_ndcg_loss': .05,
                'test_previously_inspected': True, 'test_selection_performed': False,
                'package_sha256': file_hash('hyperparameters/results/alternate-hybrids-60-20-20/package.json'),
                'routes_sha256': fingerprint(routes), 'code_sha256': {str(p): file_hash(p) for p in code},
                'metadata_sha256': {n: file_hash('dataset/ml-100k/ml-100k.'+n) for n in ('item','user')}}
    identity = fingerprint(protocol)
    write_json(out / 'protocol.json', protocol)
    write_json(out / 'routes.json', routes)
    rows, choices, invariance = [], {}, {}
    with threadpool_limits(limits=2):
        tables, baseline = pooled_components(bundle, names, routes, 'valid')
        write_json(out/'valid-pool.json', {u: sorted(row) for u,row in tables[0].scores.items()})
        original = evaluate(out,bundle,stats,identity,'valid','original',baseline)
        expected = read('hyperparameters/results/alternate-hybrids-60-20-20/evaluations/valid/warm/switch/metrics.json')
        verify_metrics({k:original[k] for k in expected},expected,'Switch validation replay')
        rows.append(original)
        for method, (reranker, target) in METHODS.items():
            trials = {order: [] for order in ORDERS}
            for strength in GRID:
                outputs = {}
                for order, pipeline in ORDERS.items():
                    print(f'Running valid {method}/{order} strength={strength:g}',flush=True)
                    ranked = pipeline(tables,reranker,routes,context,10,strength)
                    if strength == 0 and ranked != baseline: raise ValueError('Zero strength changed the switch')
                    outputs[order] = ranked
                    row = evaluate(out,bundle,stats,identity,'valid',f'{method}-{order}-{strength:g}',
                                   ranked,method,order,strength,original['ndcg'])
                    trials[order].append(row); rows.append(row)
                identical = outputs['pre'] == outputs['post']
                invariance[f'valid/{method}/{strength:g}'] = identical
                if method != 'exposure' and not identical: raise ValueError('Local calibration failed to commute')
                write_json(out/'progress.json',rows)
            for order in ORDERS:
                chosen = select_strength(trials[order],original['ndcg'])
                choices[f'{method}/{order}'] = {k:chosen[k] for k in ('strength','ndcg','target','name')}
        selection = {'status':'complete','partition':'valid','identity':identity,'choices':choices,
                     'test_evaluated':False,'target_metrics':{m:t for m,(_,t) in METHODS.items()}}
        write_json(out/'selection.json',selection)
        frozen_hash = file_hash(out/'selection.json')
        print('All six switching selections frozen. Starting test.',flush=True)
        tables,baseline = pooled_components(bundle,names,routes,'test')
        write_json(out/'test-pool.json',{u:sorted(row) for u,row in tables[0].scores.items()})
        original = evaluate(out,bundle,stats,identity,'test','original',baseline)
        expected = read('hyperparameters/results/alternate-hybrids-60-20-20/evaluations/test/warm/switch/metrics.json')
        verify_metrics({k:original[k] for k in expected},expected,'Switch test replay')
        rows.append(original)
        for method,(reranker,target) in METHODS.items():
            outputs = {}
            for order,pipeline in ORDERS.items():
                strength = choices[f'{method}/{order}']['strength']
                ranked = pipeline(tables,reranker,routes,context,10,strength)
                outputs[order] = ranked
                if strength == 0 and ranked != baseline: raise ValueError('Zero strength changed test baseline')
                rows.append(evaluate(out,bundle,stats,identity,'test',f'{method}-{order}',ranked,
                                     method,order,strength,original['ndcg']))
            same_strength = choices[f'{method}/pre']['strength'] == choices[f'{method}/post']['strength']
            invariance[f'test/{method}/selected'] = outputs['pre'] == outputs['post']
            if method != 'exposure' and (not same_strength or outputs['pre'] != outputs['post']):
                raise ValueError('Frozen local calibration orders differ')
        if file_hash(out/'selection.json') != frozen_hash: raise ValueError('Selection changed on test')
    write_csv(out/'metrics.csv',rows)
    write_json(out/'metrics.json',rows)
    write_json(out/'invariance.json',invariance)
    manifest = {'status':'complete','identity':identity,'protocol':protocol,'evaluations':len(rows),
                'selection_sha256':frozen_hash,'metrics_sha256':file_hash(out/'metrics.csv'),
                'local_run':str(out.resolve()),'test_selection_performed':False}
    write_json(out/'manifest.json',manifest)
    args.results.mkdir(parents=True)
    for name in ('protocol.json','manifest.json','routes.json','selection.json','metrics.csv','metrics.json','invariance.json'):
        (args.results/name).write_bytes((out/name).read_bytes())
    print('Completed 38 switching evaluations.',flush=True)


if __name__ == '__main__': main()
