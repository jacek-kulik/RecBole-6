"""Caio and Gabriel: boundary between this project and the existing RecBole fork."""

import csv
from dataclasses import asdict
import gzip
import hashlib
import json
from math import isfinite
from pathlib import Path
import shutil
import sys

from .artifacts import REPO_ROOT, new_run, provenance, write_json
from .contracts import ScoreTable


def _export_splits(dataset, loaders, output, config):
    """Export the actual in-memory splits from this training run, using raw IDs."""
    output.mkdir()
    digest = hashlib.sha256()
    for partition, loader in zip(("train", "valid", "test"), loaders):
        interactions = loader.dataset.inter_feat
        users = dataset.id2token(dataset.uid_field, interactions[dataset.uid_field].cpu().numpy())
        items = dataset.id2token(dataset.iid_field, interactions[dataset.iid_field].cpu().numpy())
        columns = {"user_id": users, "item_id": items}
        for name, field in (("rating", config["RATING_FIELD"]), ("timestamp", config["TIME_FIELD"])):
            if field in interactions:
                columns[name] = interactions[field].cpu().numpy()
        path = output / f"{partition}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(zip(*columns.values()))
        digest.update(partition.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def train(model_config, output):
    """Train one configured model and save validation diagnostics and exact splits.

    RecBole imports stay here so the demo and independent evaluation code need
    only Python. This entry point does not evaluate test data. Full project
    evaluation still needs export_scores and the independent metrics below.
    """
    model_config = Path(model_config).resolve()
    protocol = REPO_ROOT / "project/configs/protocol.yaml"
    if not model_config.is_file():
        raise FileNotFoundError(model_config)
    output = new_run(output)
    manifest = {**provenance(), "status": "running", "kind": "recbole-training"}
    write_json(output / "manifest.json", manifest)
    trainer = None
    try:
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import get_model, get_trainer, init_logger, init_seed

        shutil.copyfile(model_config, output / "model.yaml")
        shutil.copyfile(protocol, output / "protocol.yaml")
        # RecBole also parses sys.argv. Our CLI has already parsed its arguments;
        # the two explicit YAML files are the configuration source for this run.
        command = sys.argv
        try:
            sys.argv = command[:1]
            config = Config(
                config_file_list=[str(model_config), str(protocol)],
                config_dict={
                    "data_path": str(REPO_ROOT / "dataset"),
                    "checkpoint_dir": str(output / "checkpoints"),
                },
            )
        finally:
            sys.argv = command
        # Record the effective configuration too. Enum/device objects are saved
        # as strings for inspection; reload runs from the copied YAMLs instead.
        write_json(output / "resolved-config.json", json.loads(json.dumps(config.final_config_dict, default=str)))
        init_seed(config["seed"], config["reproducibility"])
        init_logger(config)
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        split_id = _export_splits(dataset, (train_data, valid_data, test_data), output / "splits", config)
        init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
        model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        best_score, best_metrics = trainer.fit(train_data, valid_data, saved=True, show_progress=False)

        import torch
        checkpoint = torch.load(trainer.saved_model_file, map_location=config["device"], weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))

        label = model_config.stem
        score_files, candidate_counts = {}, {}

        for partition, loader in (("valid", valid_data), ("test", test_data)):
            table = export_scores(model, dataset, loader, split_id, partition, label)
            score_files[partition] = f"scores-{partition}.json.gz"
            candidate_counts[partition] = sum(len(items) for items in table.scores.values())
            _write_scores(table, output / score_files[partition])
        write_json(output / "validation.json", {
            "source": "RecBole diagnostics; independent project evaluation is still TODO",
            "selection_metric": config["valid_metric"],
            "best_score": float(best_score),
            "metrics": {key: float(value) for key, value in best_metrics.items()},
        })
        manifest.update({
            "status": "complete", "model": config["model"], "label": label, "dataset": config["dataset"],
            "score_files": score_files, "candidate_counts": candidate_counts,
            "split_id": split_id, "seed": config["seed"],
            "checkpoint": str(Path(trainer.saved_model_file).relative_to(output)),
            "test_evaluated": False, "project_evaluation_complete": False,
        })
    except Exception as error:
        manifest.update({"status": "failed", "error": str(error)})
        raise
    finally:
        if trainer is not None and hasattr(trainer, "tensorboard"):
            trainer.tensorboard.close()
        write_json(output / "manifest.json", manifest)
    return output


def _scores_to_table(label, split_id, partition, user_tokens, item_tokens, rows, eligible) -> ScoreTable:
    """Build a ScoreTable from per-user score rows, keeping only eligible item indices.

    rows[n][i] is the score of item index i for user_tokens[n]; eligible[n] lists the
    item indices that user may be recommended. Masking is decided by eligible.
    """
    scores = {}
    for user, row, items in zip(user_tokens, rows, eligible):
        values = {}
        for index in items:
            value = float(row[index])
            if not isfinite(value):
                raise ValueError(f"{label}: nonfinite score for user {user}, item {item_tokens[index]}.")
            values[str(item_tokens[index])] = value
        scores[str(user)] = values
    table = ScoreTable(label, split_id, partition, scores)
    table.validate()
    return table


def export_scores(model, dataset, loader, split_id, partition, label=None, batch_size=256) -> ScoreTable:
    """Score every eligible candidate for each user evaluated in loader.

    Users are those with at least one interaction in this partition. Candidates are
    all items except padding and the loader's history, which is RecBole's own full-sort masking.
    The loader must belong to the same exported split as split_id.
    """
    import torch
    from recbole.utils.case_study import full_sort_scores

    item_tokens = dataset.id2token(dataset.iid_field, list(range(dataset.item_num)))
    uids = loader.uid_list.tolist()
    users, rows, eligible = [], [], []
    for start in range(0, len(uids), batch_size):
        batch = uids[start:start + batch_size]
        with torch.no_grad():
            batch_scores = full_sort_scores(batch, model, loader).cpu().numpy()
        for uid, row in zip(batch, batch_scores):
            masked = set(loader.uid2history_item[uid].tolist()) | {0}
            users.append(dataset.id2token(dataset.uid_field, uid))
            rows.append(row)
            eligible.append([i for i in range(1, dataset.item_num) if i not in masked])
    return _scores_to_table(label or model.__class__.__name__, split_id, partition,
                            users, item_tokens, rows, eligible)


def _write_scores(table, path):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(asdict(table), stream, sort_keys=True, allow_nan=False)


def load_scores(path) -> ScoreTable:
    """Read a ScoreTable written by train (scores-<partition>.json.gz)."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        table = ScoreTable(**json.load(stream))
    table.validate()
    return table
