"""Audit switching order experiments without fitting or selecting on test."""

import argparse
import json
from pathlib import Path

from project.recsys.artifacts import fingerprint, write_json
from project.recsys.component_cache import file_hash
from project.recsys.comparisons import compare_runs
from project.recsys.hybrid_bundle import HybridBundle
from project.recsys.metrics.accuracy import evaluate_accuracy
from project.recsys.metrics.beyond_accuracy import evaluate_beyond_accuracy
from project.recsys.rerank import select_strength
from project.scripts.run_switching_orders import prepare, read
from project.scripts.verify_reranking_orders import independent_targets


def verify(out, results):
    manifest, protocol = read(out/'manifest.json'), read(out/'protocol.json')
    if manifest['status']!='complete' or fingerprint(protocol)!=manifest['identity']:
        raise ValueError('Incomplete or changed switching run')
    for path,digest in protocol['code_sha256'].items():
        if file_hash(path)!=digest: raise ValueError(f'Changed numerical source: {path}')
    if file_hash(out/'selection.json')!=manifest['selection_sha256'] or file_hash(out/'metrics.csv')!=manifest['metrics_sha256']:
        raise ValueError('Selection or results changed')
    bundle=HybridBundle()
    _,routes,stats,_=prepare(bundle)
    if routes!=read(out/'routes.json') or fingerprint(routes)!=protocol['routes_sha256']:
        raise ValueError('Routing changed')
    selection=read(out/'selection.json'); rows=read(out/'metrics.json')
    if len(rows)!=38 or len({(r['partition'],r['name']) for r in rows})!=38:
        raise ValueError('Expected 38 distinct evaluations')
    pools={p:read(out/f'{p}-pool.json') for p in ('valid','test')}
    masks={p:bundle.mask(p,'warm') for p in pools}
    ui={u:n for n,u in enumerate(bundle.users)}; ii={i:n for n,i in enumerate(bundle.items)}
    originals={p:read(out/p/'original/rankings.json') for p in pools}
    stored={}; max_error=0.; list_count=0
    for row in rows:
        part=row['partition']; folder=out/part/row['name']
        ranked=read(folder/'rankings.json'); child=read(folder/'manifest.json')
        if child['status']!='complete' or child['input_identity']!=manifest['identity'] or child['n_catalogue_items']!=1615:
            raise ValueError('Evaluation identity differs')
        if set(ranked)!=set(bundle.users): raise ValueError('Cohort changed')
        relevant={u:set(items) for u,items in read(folder/'relevant.json').items()}
        if relevant!=bundle.truth(part,'warm'): raise ValueError('Warm relevance changed')
        for user,items in ranked.items():
            if len(items)!=10 or len(set(items))!=10 or not set(items)<=set(pools[part][user]):
                raise ValueError('Invalid list or pool escape')
            if not all(masks[part][ui[user],ii[i]] for i in items): raise ValueError('History/unsupported item recommended')
        acc=evaluate_accuracy(ranked,relevant,10)
        if acc['per_user']!=read(folder/'per_user_metrics.json'): raise ValueError('Per-user metrics changed')
        beyond=evaluate_beyond_accuracy(ranked,stats,acc['per_user'],10)
        for k,v in {**acc['aggregate'],**beyond['aggregate'],**independent_targets(ranked,stats)}.items():
            error=abs(v-row[k]);max_error=max(max_error,error)
            if error>1e-12: raise ValueError(f'Metric mismatch {row["name"]}/{k}')
        if row['strength']==0 and ranked!=originals[part]: raise ValueError('Zero strength changes switch')
        stored[(part,row['name'])]=ranked;list_count+=len(ranked)
    for label,chosen in selection['choices'].items():
        method,order=label.split('/')
        trials=[r for r in rows if r['partition']=='valid' and r['method']==method and r['order']==order]
        if sorted(r['strength'] for r in trials)!=protocol['strengths']: raise ValueError('Incomplete strength grid')
        base=next(r['ndcg'] for r in rows if r['partition']=='valid' and r['order']=='baseline')
        best=select_strength(trials,base)
        if chosen!={k:best[k] for k in chosen}: raise ValueError('Validation selection differs')
        test=next(r for r in rows if r['partition']=='test' and r['method']==method and r['order']==order)
        if test['strength']!=chosen['strength']: raise ValueError('Test reselection')
    local_checks=0
    for method in ('genre','popularity'):
        for strength in protocol['strengths']:
            if stored[('valid',f'{method}-pre-{strength:g}')]!=stored[('valid',f'{method}-post-{strength:g}')]:
                raise ValueError('Local calibration validation orders differ')
            local_checks+=1
        if selection['choices'][f'{method}/pre']['strength']!=selection['choices'][f'{method}/post']['strength']:
            raise ValueError('Local calibration selections differ')
        if stored[('test',f'{method}-pre')]!=stored[('test',f'{method}-post')]:
            raise ValueError('Local calibration test lists differ')
        local_checks+=1
    frozen=(out/'selection.json').stat().st_mtime_ns
    if max((out/'valid'/r['name']/'manifest.json').stat().st_mtime_ns for r in rows if r['partition']=='valid')>frozen:
        raise ValueError('Validation written after freeze')
    if min((out/'test'/r['name']/'manifest.json').stat().st_mtime_ns for r in rows if r['partition']=='test')<frozen:
        raise ValueError('Test written before freeze')
    comparison=out/'comparison'
    if not comparison.exists():
        compare_runs([f"{r['name']}={out/'test'/r['name']}" for r in rows if r['partition']=='test'],comparison,10)
    exposure_pre,exposure_post=stored[('test','exposure-pre')],stored[('test','exposure-post')]
    exposure_different_users=sum(exposure_pre[u]!=exposure_post[u] for u in bundle.users)
    proof={'status':'passed','evaluations':38,'recommendation_lists':list_count,'max_metric_error':max_error,
           'local_order_equalities_checked':local_checks,'local_calibration_lists_identical':True,
           'exposure_test_users_with_different_lists':exposure_different_users,
           'all_six_choices_reproduced_from_validation':True,'all_validation_precedes_test':True,
           'routes_unchanged':True,'full_warm_relevance_and_history_masks_checked':True,
           'warm_items':len(stats.catalogue),'relevant_positives':{p:sum(map(len,bundle.truth(p,'warm').values())) for p in pools},
           'verifier_sha256':file_hash(__file__)}
    write_json(out/'verification.json',proof);write_json(results/'verification.json',proof)
    print(json.dumps(proof,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path);parser.add_argument('--results',required=True,type=Path)
    args=parser.parse_args();verify(args.run,args.results)
