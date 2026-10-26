"""Task 1.3: cross-fitted rating regression over the Task 1.1 recommenders.

Run with ``python -m project.recsys.oof_hybrid --help``. Test labels are never
scored. Component training uses fixed epochs, not fold-target early stopping.
"""

import argparse
import ast
import csv
import json
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .artifacts import REPO_ROOT, fingerprint, new_run, provenance, write_json
from .component_cache import atomic_json, file_hash, load_fold, run_lock, save_fold
from .hybrids import candidate_features, fit_oof_regression, predict_weighted_hybrid
from .recbole_adapter import _export_splits, _write_scores, export_scores, load_scores

COMPONENTS = ("itemknn", "userknn", "bpr", "slimelastic", "ease", "fism")


def user_folds(users, items, folds, seed):
    """Keep repeated pairs together; require training evidence for every user."""
    if folds < 2:
        raise ValueError("At least two folds are required")
    groups = {}
    for index, (user, item) in enumerate(zip(users, items)):
        groups.setdefault(user, {}).setdefault(item, []).append(index)
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(users), dtype=int)
    for user in sorted(groups):
        pairs = list(groups[user].values())
        if len(pairs) < 2:
            raise ValueError(f"User {user} needs at least two distinct training items")
        order = rng.permutation(len(pairs))
        offset = int(rng.integers(folds))
        for number, group in enumerate(order):
            assignment[pairs[group]] = (number + offset) % folds
    if set(assignment) != set(range(folds)):
        raise ValueError("Some folds are empty; reduce folds or supply more training data")
    return assignment


def tuned_parameters(path):
    """Read the existing Task 1.2 summary, including list-valued parameters."""
    result = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            values = {}
            for part in re.split(r",(?=\s*[A-Za-z_]\w*\s*:)", row["hyperparameters"]):
                name, value = part.split(":", 1)
                values[name.strip()] = ast.literal_eval(value.strip())
            result[row["model"].lower()] = values
    return result


def ndcg_at_k(table, relevant, k):
    """Binary relevance: every held-out interaction is positive, per protocol."""
    scores = []
    for user, items in table.top_k(k).items():
        truth = relevant[user]
        dcg = sum(1 / np.log2(rank + 2) for rank, item in enumerate(items) if item in truth)
        ideal = sum(1 / np.log2(rank + 2) for rank in range(min(k, len(truth))))
        scores.append(dcg / ideal if ideal else 0.0)
    return float(np.mean(scores))


def external_pairs(dataset, interactions):
    users = dataset.id2token(dataset.uid_field, interactions[dataset.uid_field].cpu().numpy())
    items = dataset.id2token(dataset.iid_field, interactions[dataset.iid_field].cpu().numpy())
    return [(str(u), str(i)) for u, i in zip(users, items)]


def train_and_score(config, dataset, training, evaluation, directory, split_id, label):
    """Reuse Task 1.1 model factories/exporter with explicit interaction subsets."""
    from recbole.data.utils import create_samplers, get_dataloader
    from recbole.utils import get_model, get_trainer, init_seed

    directory.mkdir(parents=True, exist_ok=True)
    config["checkpoint_dir"] = str(directory)
    # Third partition is deliberately empty. Neither sampling nor training sees
    # outer validation/test interactions when producing OOF features.
    empty = dataset.copy(training.inter_feat[:0])
    samplers = create_samplers(config, dataset, [training, evaluation, empty])
    train_loader = get_dataloader(config, "train")(config, training, samplers[0], shuffle=True)
    eval_loader = get_dataloader(config, "valid")(config, evaluation, samplers[1], shuffle=False)
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, training).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    try:
        trainer.fit(train_loader, valid_data=None, saved=True, verbose=False, show_progress=False)
        model.eval()
        return export_scores(model, dataset, eval_loader, split_id, "valid", label)
    finally:
        if hasattr(trainer, "tensorboard"):
            trainer.tensorboard.close()


def run(args):
    output = Path(args.output).resolve()
    if args.resume:
        if not (output / 'cache-run.json').is_file():
            raise ValueError('Resume requires a run created by the resumable runner')
    else:
        new_run(output)
    with run_lock(output):
        return _run(args, output)


def _run(args, output):
    from recbole.config import Config
    from recbole.data import create_dataset
    from recbole.utils import init_seed

    names = args.components
    if len(set(names)) != len(names):
        raise ValueError("Duplicate components")
    params = tuned_parameters(args.tuned_summary)
    if any(name not in params for name in names):
        raise ValueError("Tuning summary is missing a requested component")
    protocol = REPO_ROOT / "project/configs/protocol.yaml"
    run_identity = {'cache_version': 1, 'protocol': file_hash(protocol),
                    'folds': args.folds, 'seed': args.seed,
                    'dataset': {p.name: file_hash(p) for p in sorted((REPO_ROOT / 'dataset/ml-100k').glob('*.inter'))}}
    identity_path = output / 'cache-run.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != run_identity:
        raise ValueError('Split, dataset, seed, or folds changed: use a new output directory')
    atomic_json(identity_path, run_identity)
    manifest = {**provenance(), "status": "running", "components": names,
                "folds": args.folds, "test_evaluated": False,
                "component_selection_source": str(Path(args.tuned_summary).resolve()),
                "component_parameters": {n: params[n] for n in names},
                "selection_metric": f"NDCG@{args.k}", "fixed_epochs": True,
                "smoke_only": args.smoke_epochs is not None, "prepare_only": args.prepare_only,
                "parameter_transfer": "Existing 80/10/10 tuning settings accepted for 60/20/20 by user"}
    write_json(output / "manifest.json", manifest)
    config_files = [REPO_ROOT / f"project/configs/models/{name}.yaml" for name in names]
    (output / "protocol.yaml").write_bytes(protocol.read_bytes())
    (output / "tuned-summary.csv").write_bytes(Path(args.tuned_summary).read_bytes())
    try:
        oof, final_tables, signature, base_pairs = None, [], None, None
        for column, (name, path) in enumerate(zip(names, config_files)):
            overrides = {**params[name], "data_path": str(REPO_ROOT / "dataset"),
                         "checkpoint_dir": str(output / name), "seed": args.seed,
                         "save_dataset": False, "save_dataloaders": False}
            if args.smoke_epochs is not None:
                overrides["epochs"] = args.smoke_epochs
            argv = sys.argv
            try:
                sys.argv = argv[:1]
                config = Config(config_file_list=[str(path), str(protocol)], config_dict=overrides)
            finally:
                sys.argv = argv
            write_json(output / f"config-{name}.json", json.loads(json.dumps(config.final_config_dict, default=str)))
            init_seed(config["seed"], config["reproducibility"])
            dataset = create_dataset(config)
            training, validation, test = dataset.build()
            pairs = external_pairs(dataset, training.inter_feat)
            valid_pairs = external_pairs(dataset, validation.inter_feat)
            test_pairs = external_pairs(dataset, test.inter_feat)
            if set(pairs) & (set(valid_pairs) | set(test_pairs)) or set(valid_pairs) & set(test_pairs):
                raise ValueError("Outer split contains overlapping user-item pairs")
            identity = fingerprint([pairs, valid_pairs, test_pairs])
            if column == 0:
                signature, base_pairs = identity, pairs
                targets = training.inter_feat[config["RATING_FIELD"]].cpu().numpy()
                assignment = user_folds([p[0] for p in pairs], [p[1] for p in pairs], args.folds, args.seed)
                oof = np.full((len(pairs), len(names)), np.nan)
                with tempfile.TemporaryDirectory() as temporary:
                    generated = Path(temporary) / 'splits'
                    split_id = _export_splits(dataset, [SimpleNamespace(dataset=d) for d in (training, validation, test)], generated, config)
                    stored = output / 'splits'
                    stored.mkdir(exist_ok=True)
                    for source in generated.iterdir():
                        target = stored / source.name
                        if target.exists() and target.read_bytes() != source.read_bytes():
                            raise ValueError('Cached outer split differs from reconstructed split')
                        target.write_bytes(source.read_bytes())
                write_json(output / "folds.json", [{"user": u, "item": i, "fold": int(f)} for (u, i), f in zip(pairs, assignment)])
                relevant = {}
                for user, item in valid_pairs:
                    relevant.setdefault(user, set()).add(item)
            elif identity != signature or pairs != base_pairs:
                raise ValueError("Component outer splits differ")
            # Hash only this component's model source, so fixing FISM does not
            # invalidate the five unaffected models. Common training/scoring
            # code and effective configuration invalidate caches when changed.
            effective = json.loads(json.dumps(config.final_config_dict, default=str))
            for ignored in ('checkpoint_dir', 'model_save_file', 'log_wandb'):
                effective.pop(ignored, None)
            sources = [Path(__file__), REPO_ROOT / 'project/recsys/hybrids.py',
                       REPO_ROOT / 'project/recsys/recbole_adapter.py',
                       REPO_ROOT / 'recbole/model/general_recommender' / (config['model'].lower() + '.py'),
                       REPO_ROOT / 'recbole/model/abstract_recommender.py',
                       REPO_ROOT / 'recbole/trainer/trainer.py',
                       REPO_ROOT / 'recbole/data/dataset/dataset.py',
                       REPO_ROOT / 'recbole/data/utils.py',
                       REPO_ROOT / 'recbole/utils/case_study.py']
            cache_spec = {'config': effective, 'split_id': split_id,
                          'fold_assignment': fingerprint(assignment.tolist()),
                          'sources': {str(p.relative_to(REPO_ROOT)): file_hash(p) for p in sources}}
            cache_key = fingerprint(cache_spec)
            component_dir = output / 'components' / name / cache_key
            component_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(component_dir / 'identity.json', cache_spec)
            atomic_json(output / f'cache-{name}.json', {'key': cache_key, 'path': str(component_dir)})
            for fold in range(args.folds):
                held = np.flatnonzero(assignment == fold)
                fit = np.flatnonzero(assignment != fold)
                fold_path = component_dir / f'fold-{fold}.npz'
                cached = load_fold(fold_path, held)
                if cached is not None:
                    print(f'{name}: reuse fold {fold + 1}/{args.folds}', flush=True)
                    oof[held, column] = cached
                    continue
                print(f"{name}: fold {fold + 1}/{args.folds}", flush=True)
                table = train_and_score(config, dataset, training.copy(training.inter_feat[fit]),
                                        training.copy(training.inter_feat[held]),
                                        component_dir / f"fold-{fold}-training", f"{split_id}:fold-{fold}", name)
                keys, values = candidate_features([table])
                lookup = {pair: value[0] for pair, value in zip(keys, values)}
                oof[held, column] = [lookup[pairs[index]] for index in held]
                save_fold(fold_path, held, oof[held, column])
                np.savez_compressed(output / "oof-progress.npz", features=oof, targets=targets, folds=assignment)
            full_path = component_dir / 'scores-valid.json.gz'
            marker = component_dir / 'full.complete.json'
            if marker.exists():
                if not full_path.exists() or file_hash(full_path) != json.loads(marker.read_text())['sha256']:
                    raise ValueError(f'Corrupt completed validation cache: {full_path}')
                table = load_scores(full_path)
                if (table.model, table.split_id, table.partition) != (name, split_id, 'valid'):
                    raise ValueError('Cached validation metadata differs')
                print(f'{name}: reuse full-training validation scores', flush=True)
            else:
                print(f"{name}: refit on full training partition", flush=True)
                table = train_and_score(config, dataset, training, validation,
                                        component_dir / "full-training", split_id, name)
                temporary = component_dir / 'scores-valid.tmp.gz'
                _write_scores(table, temporary)
                temporary.replace(full_path)
                atomic_json(marker, {'sha256': file_hash(full_path)})
            _write_scores(table, output / f"scores-valid-{name}.json.gz")
            final_tables.append(table)
            np.savez_compressed(output / 'oof-progress.npz', features=oof, targets=targets, folds=assignment)
            print(f'{name}: cached all folds and validation scores', flush=True)
        if args.prepare_only:
            manifest.update(status='prepared', split_id=split_id, regressor_fitted=False)
            return output
        trials = []
        for alpha in args.alphas:
            fitted = fit_oof_regression(oof, targets, names, alpha)
            prediction = predict_weighted_hybrid(final_tables, fitted)
            trials.append((ndcg_at_k(prediction, relevant, args.k), fitted))
        best_score, best = max(trials, key=lambda trial: trial[0])
        write_json(output / "hybrid.json", best)
        write_json(output / "validation.json", {
            "metric": f"NDCG@{args.k}", "relevance": "all observed interactions",
            "trials": [{"alpha": fit["alpha"], "score": score} for score, fit in trials],
            "selected_score": best_score,
            "components": {table.model: ndcg_at_k(table, relevant, args.k) for table in final_tables}})
        _write_scores(predict_weighted_hybrid(final_tables, best), output / "scores-valid-hybrid.json.gz")
        manifest.update(status="complete", split_id=split_id, selected_alpha=best["alpha"])
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse matching completed stages in an existing run")
    parser.add_argument("--prepare-only", action="store_true", help="Cache component predictions without fitting the regressor")
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=list(COMPONENTS))
    parser.add_argument("--tuned-summary", default=str(REPO_ROOT / "hyperparameters/results/best-run-summary.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1., 10., 100.])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--smoke-epochs", type=int, help="Override neural training budget; never use for reported results")
    args = parser.parse_args()
    if args.k < 1 or args.folds < 2 or any(not np.isfinite(a) or a < 0 for a in args.alphas):
        parser.error("Invalid cutoff, folds, or ridge penalties")
    if args.smoke_epochs is not None and args.smoke_epochs < 1:
        parser.error("smoke epochs must be positive")
    print(run(args))


if __name__ == "__main__":
    main()
