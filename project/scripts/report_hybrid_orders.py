"""Render the combined weighted/switching Task 3.3 report from frozen results."""

import argparse
import json
from pathlib import Path

from project.recsys.artifacts import write_json
from project.recsys.component_cache import file_hash


def read(path):
    return json.loads(Path(path).read_text())


def render(weighted, switching, destination):
    sources = {'Weighted': weighted, 'Switch': switching}
    targets = {'genre': 'GenreMiscalibration@10', 'popularity': 'UPD@10', 'exposure': 'ExposureGini@10'}
    lines = ['# Task 3.3: reranking before or after hybrid combination', '',
             '| Hybrid | Target (strength) | Order | Test NDCG | NDCG loss | Target reduction |',
             '|---|---|---|---:|---:|---:|']
    baselines = []
    for label, folder in sources.items():
        if read(folder/'manifest.json')['status'] != 'complete' or read(folder/'verification.json')['status'] != 'passed':
            raise ValueError('Report requires completed, verified results')
        rows = [r for r in read(folder/'metrics.json') if r['partition'] == 'test']
        baseline = next(r for r in rows if r['order'] == 'baseline')
        scope = 'full, 1,682 items' if label == 'Weighted' else 'warm, 1,615 items'
        baselines.append(f"{label}: {baseline['ndcg']:.6f} ({scope})")
        for method, target in targets.items():
            pre = next(r for r in rows if r['method'] == method and r['order'] == 'pre')
            post = next(r for r in rows if r['method'] == method and r['order'] == 'post')
            collapse = label == 'Switch' and method != 'exposure'
            if collapse and pre['strength'] != post['strength']:
                raise ValueError('Cannot collapse different switching strengths')
            for r in ([pre] if collapse else [pre, post]):
                order = 'pre = post' if collapse else r['order']
                reduction = 100 * (1 - r[target] / baseline[target])
                lines.append(f"| {label} | {method} ({r['strength']:g}) | {order} | {r['ndcg']:.6f} | "
                             f"{100*r['relative_ndcg_loss']:.2f}% | {reduction:.2f}% |")
    lines += ['', 'Baseline NDCG: ' + '; '.join(baselines) + '.', '',
              'Pre = rerank components then combine/route; post = combine/route then rerank. '
              'Changes use each hybrid\'s own baseline. Targets: genre KL, popularity JS and exposure Gini.', '',
              '**Discussion**', '']
    discussion = (
        'Both hybrids retain their fitted components, coefficients/routing and original evaluation scopes. '
        'Each uses identical top-50 eligible candidates in both orders and returns ten items. Weighted '
        'pre-reranking transfers the reranked prefix through the documented score-preserving permutation '
        'before fixed Ridge combination. Each method/order selects strength on validation under a 5% '
        'NDCG-loss budget; all choices are frozen before test. '
        'For the weighted hybrid, post-reranking improves each target more but incurs greater accuracy loss. '
        'For switching, genre and popularity calibration produce identical lists in both orders at every '
        'validation strength and their frozen test choices: per-user reranking commutes with fixed routing. '
        'Exposure fairness is cohort-dependent. Pre-reranking maintains separate counters for each component\'s '
        'entire cohort, including lists later discarded by routing; post-reranking allocates exposure across '
        'the final switched output. Its stronger Gini reduction therefore comes with a larger NDCG loss. '
        'These findings depend on the stated conversion, pool and exposure policy. The validation budget '
        'does not guarantee test accuracy. Results are descriptive on one previously inspected split, '
        'and warm/full absolute scores should not be treated as a controlled cross-hybrid comparison.'
    )
    words = len(discussion.split())
    if words > 200: raise ValueError('Discussion exceeds 200 words')
    lines.append(discussion)
    destination.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    proof = {'discussion_words': words, 'report_sha256': file_hash(destination),
             'renderer_sha256': file_hash(__file__),
             'source_sha256': {str(p): file_hash(p) for folder in sources.values()
                              for p in (folder/'metrics.json',folder/'selection.json',folder/'verification.json')}}
    for folder in sources.values(): write_json(folder/'combined-report-provenance.json',proof)
    print(f'{destination}: {words}-word discussion')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weighted',type=Path,default=Path('hyperparameters/results/task-3.3-reranking-orders'))
    parser.add_argument('--switching',type=Path,default=Path('hyperparameters/results/task-3.3-switching-orders'))
    parser.add_argument('--report',type=Path,default=Path('project/reports/task-3.3-reranking-orders.md'))
    args=parser.parse_args();render(args.weighted,args.switching,args.report)
