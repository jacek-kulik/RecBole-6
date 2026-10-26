"""Task 3.3: matched candidate pools and frozen Ridge, both reranking orders."""

import argparse
import csv
from functools import partial
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from project.recsys.analysis.coefficients import candidate_zscores
from project.recsys.artifacts import fingerprint, write_json
from project.recsys.component_cache import file_hash
from project.recsys.contracts import RerankContext, ScoreTable
from project.recsys.data import prepare_metadata, read_training_interactions
from project.recsys.evaluation import evaluation_settings, write_ranking_results
from project.recsys.experiments.task3 import combine_standardized_scores, combine_then_rerank, rerank_then_combine
from project.recsys.hybrid_bundle import HybridBundle
from project.recsys.rerank import select_strength
from project.recsys.rerankers import calibration, item_fairness
from project.scripts.analyze_hybrid_coefficients import WEIGHTED, verify_weighted, verify_metrics
from project.scripts.retune_ease_slim import rankings


METHODS = {
    'genre': (calibration.rerank, 'GenreMiscalibration@10'),
    'popularity': (partial(calibration.rerank, attribute='popularity'), 'UPD@10'),
    'exposure': (item_fairness.rerank, 'ExposureGini@10'),
}
ORDERS = {'pre': rerank_then_combine, 'post': combine_then_rerank}
GRID = [0., .25, .5, .75, 1.]


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def pooled_components(bundle, raw, fitted, partition):
    mask = bundle.mask(partition)
    features = candidate_zscores(raw, mask)
    values = features @ np.asarray(fitted['coefficients']) + fitted['intercept']
    baseline = rankings(values, mask, bundle.users, bundle.items)
    selected = np.argsort(-np.where(mask, values, -np.inf), axis=1, kind='stable')[:, :50]
    tables = []
    for j, name in enumerate(fitted['components']):
        scores = {u: {bundle.items[i]: float(features[n, i, j]) for i in selected[n]}
                  for n, u in enumerate(bundle.users)}
        tables.append(ScoreTable(name, bundle.manifest['split_id'], partition, scores))
    pooled_baseline = combine_standardized_scores(tables, fitted).top_k(10)
    if pooled_baseline != baseline:
        raise ValueError('Restricting the hybrid to its top 50 changed its top 10')
    return tables, baseline


def evaluate(out, partition, name, ranked, bundle, statistics, identity, settings, baseline_ndcg=None):
    directory = out / partition / name
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {**evaluation_settings(partition, 10), 'status': 'running', 'model': 'hybrid-'+name,
                'dataset': 'ml-100k', 'split_id': bundle.manifest['split_id'], 'seed': 42,
                'input_identity': identity, 'n_catalogue_items': len(bundle.items),
                'n_training_interactions': statistics.n_interactions, 'reranking_candidate_count': 50,
                **settings}
    try:
        accuracy, beyond, _ = write_ranking_results(directory, ranked, bundle.truth(partition), statistics, 10)
        manifest.update(status='complete', n_users=len(ranked), project_evaluation_complete=False)
    except Exception as error:
        manifest.update(status='failed', error=str(error)); raise
    finally:
        write_json(directory / 'manifest.json', manifest)
    row = {'partition': partition, 'name': name, 'order': settings.get('order', 'baseline'),
           'method': settings.get('method', 'original'), 'strength': settings.get('strength', 0.),
           'ndcg': accuracy['aggregate']['NDCG@10'], **accuracy['aggregate'], **beyond['aggregate']}
    row['relative_ndcg_loss'] = 0. if baseline_ndcg is None else 1 - row['ndcg'] / baseline_ndcg
    print(f"{partition}/{name}: NDCG={row['ndcg']:.6f}; KL={row['GenreMiscalibration@10']:.4f}; "
          f"UPD={row['UPD@10']:.4f}; Gini={row['ExposureGini@10']:.4f}", flush=True)
    return row


def render_report(results, selection):
    baseline = next(r for r in results if r['partition'] == 'test' and r['order'] == 'baseline')
    lines = ['# Task 3.3: rerank then combine versus combine then rerank', '',
             '| Method | Order | Strength | NDCG@10 | NDCG loss | Genre KL | Popularity JS | Exposure Gini |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in [r for r in results if r['partition'] == 'test']:
        lines.append(f"| {row['method']} | {row['order']} | {row['strength']:g} | {row['ndcg']:.6f} | "
                     f"{100*row['relative_ndcg_loss']:+.2f}% | {row['GenreMiscalibration@10']:.4f} | "
                     f"{row['UPD@10']:.4f} | {row['ExposureGini@10']:.4f} |")
    lines += ['', 'Pre = rerank each component then combine; post = combine then rerank. '
              'Baseline = frozen six-component OOF Ridge hybrid.', '',
              'Each pipeline selects a shared component strength on validation from 0, 0.25, 0.5, 0.75, 1, '
              'minimizing its designated beyond-accuracy metric subject to at most 5% NDCG loss. '
              'All six choices are frozen before any test evaluation. Candidate pools are the original hybrid\'s '
              'top 50 eligible items, rebuilt separately after validation/test history masking. Both orders use '
              'identical pools, components, profiles, coefficients and final K=10. Pre-order reranks each '
              'component\'s top 10, appends unselected candidates in original order, and assigns its original '
              'descending standardized scores to those positions before fixed-weight combination. '
              'This explicit score-preserving conversion is one operational definition of rerank-then-combine; '
              'results depend on it and the fixed pool. Strength zero exactly reproduces the original hybrid. '
              'No component or combiner is retrained. These are descriptive results on the previously inspected '
              'seed-42 split; the validation budget is not a guarantee of test accuracy.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    if args.results.exists(): raise ValueError('Use a fresh results directory')
    bundle = HybridBundle()
    package_path = Path('hyperparameters/results/alternate-hybrids-60-20-20/package.json')
    for path, digest in read(package_path)['files_sha256'].items():
        if file_hash(bundle.root / path) != digest: raise ValueError('Portable source changed')
    code = [Path(__file__), Path('project/recsys/experiments/task3.py'), Path('project/recsys/rerank.py'),
            *sorted(Path('project/recsys/rerankers').glob('*.py')), Path('project/recsys/data.py'),
            Path('project/recsys/analysis/coefficients.py'), Path('project/scripts/analyze_hybrid_coefficients.py'),
            Path('project/recsys/contracts.py'), Path('project/recsys/evaluation.py'),
            Path('project/recsys/metrics/accuracy.py'), Path('project/recsys/metrics/beyond_accuracy.py')]
    protocol = {'split_id': bundle.manifest['split_id'], 'seed': 42, 'k': 10, 'candidate_count': 50,
                'strengths': GRID, 'shared_strength_across_components': True,
                'selection': 'minimize target on validation subject to NDCG >= 0.95*baseline; tie higher NDCG then lower strength',
                'candidate_retrieval': 'frozen original Ridge top50 on eligible candidates; same pool for both orders',
                'rank_to_score': 'reranked top10 + original-order remainder; assign original sorted z-scores to positions',
                'score_normalization': 'component z-scores over full eligible catalogue, before truncation; reranker minmax over pool',
                'test_previously_inspected': True, 'test_selection_performed': False,
                'source': str(WEIGHTED), 'code_sha256': {str(p): file_hash(p) for p in code},
                'metadata_sha256': {name: file_hash(Path('dataset/ml-100k')/('ml-100k.'+name)) for name in ('item','user')},
                'reranker_parent': '580bfaa63de4b176b2018e6e2a5108e8f9e390c7'}
    write_json(out / 'protocol.json', protocol)
    with threadpool_limits(limits=2):
        fitted, _, _, raw = verify_weighted(bundle, out)
        protocol['weighted_inputs_sha256'] = file_hash(out / 'weighted-inputs.json')
        identity = fingerprint(protocol)
        write_json(out / 'frozen-protocol.json', protocol)
        stats = prepare_metadata(read_training_interactions(WEIGHTED/'splits/train.csv'),
                                 'dataset/ml-100k/ml-100k.item', 'dataset/ml-100k/ml-100k.user', bundle.items)
        context = RerankContext(item_features=stats.item_features, user_profiles=stats.genre_profiles,
                                popularity_profiles=stats.popularity_profiles,
                                user_groups=stats.gender_groups, item_groups=stats.item_groups)
        write_json(out / 'hybrid.json', fitted)
        rows, choices = [], {}
        tables, baseline = pooled_components(bundle, raw, fitted, 'valid')
        write_json(out/'valid-pool.json', {u: sorted(row) for u,row in tables[0].scores.items()})
        original = evaluate(out,'valid','original',baseline,bundle,stats,identity,{})
        verify_metrics({k:original[k] for k in read(WEIGHTED/'evaluation/metrics.json')['valid']['regressor']['aggregate']},
                       read(WEIGHTED/'evaluation/metrics.json')['valid']['regressor']['aggregate'],'Ridge validation replay')
        rows.append(original)
        for method,(reranker,target) in METHODS.items():
            for order, pipeline in ORDERS.items():
                trials=[]
                for strength in GRID:
                    print(f'Running validation {method}/{order} strength={strength:g}',flush=True)
                    ranked=pipeline(tables,reranker,fitted,context,{'k':10,'strength':strength})
                    if strength==0 and ranked!=baseline: raise ValueError('Zero strength changes original rankings')
                    row=evaluate(out,'valid',f'{method}-{order}-{strength:g}',ranked,bundle,stats,identity,
                                 {'method':method,'order':order,'strength':strength},original['ndcg'])
                    row['target']=row[target]; trials.append(row); rows.append(row)
                    write_json(out/'progress.json',rows)
                chosen=select_strength(trials,original['ndcg'])
                choices[f'{method}/{order}']={k:chosen[k] for k in ('strength','ndcg','target','name')}
        selection={'status':'complete','partition':'valid','identity':identity,'choices':choices,
                   'test_evaluated':False,'target_metrics':{m:t for m,(_,t) in METHODS.items()}}
        write_json(out/'selection.json',selection)
        frozen_hash=file_hash(out/'selection.json')
        print('All six pipeline selections frozen; starting test evaluation.',flush=True)
        tables,baseline=pooled_components(bundle,raw,fitted,'test')
        write_json(out/'test-pool.json',{u:sorted(row) for u,row in tables[0].scores.items()})
        original=evaluate(out,'test','original',baseline,bundle,stats,identity,{})
        expected=read(WEIGHTED/'evaluation/metrics.json')['test']['regressor']['aggregate']
        verify_metrics({k:original[k] for k in expected},expected,'Ridge test replay')
        rows.append(original)
        for method,(reranker,target) in METHODS.items():
            for order,pipeline in ORDERS.items():
                strength=choices[f'{method}/{order}']['strength']
                print(f'Running test {method}/{order} strength={strength:g}',flush=True)
                ranked=pipeline(tables,reranker,fitted,context,{'k':10,'strength':strength})
                if strength==0 and ranked!=baseline: raise ValueError('Zero strength changes test baseline')
                row=evaluate(out,'test',f'{method}-{order}',ranked,bundle,stats,identity,
                             {'method':method,'order':order,'strength':strength},original['ndcg'])
                row['target']=row[target]; rows.append(row)
        if frozen_hash!=file_hash(out/'selection.json'):raise ValueError('Selection changed during test')
    # Uniform CSV schema includes the baseline and every validation trial.
    for row in rows: row.setdefault('target',None)
    write_csv(out/'metrics.csv',rows)
    write_json(out/'metrics.json',rows)
    text=render_report(rows,selection)
    (out/'report.md').write_text(text,encoding='utf-8')
    Path('project/reports/task-3.3-reranking-orders.md').write_text(text,encoding='utf-8')
    manifest={'status':'complete','identity':identity,'protocol':protocol,'evaluation_count':len(rows),
              'selection_sha256':frozen_hash,'test_selection_performed':False,
              'metrics_sha256':file_hash(out/'metrics.csv'),'local_run':str(out.resolve())}
    write_json(out/'manifest.json',manifest)
    args.results.mkdir(parents=True)
    for name in ('frozen-protocol.json','weighted-inputs.json','hybrid.json','selection.json',
                 'metrics.json','metrics.csv','manifest.json'):
        (args.results/name).write_bytes((out/name).read_bytes())
    print('Completed all 38 evaluations.',flush=True)


if __name__=='__main__':main()
