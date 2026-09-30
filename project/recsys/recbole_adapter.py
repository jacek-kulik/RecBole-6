"""Caio and Gabriel: boundary between this project and the existing RecBole fork."""

import csv
import hashlib
import json
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
        # TODO Caio and Gabriel: load the selected checkpoint and call export_scores
        # for the agreed evaluation partition. Do not export last-epoch weights
        # while labelling them as the best validation checkpoint.
        write_json(output / "validation.json", {
            "source": "RecBole diagnostics; independent project evaluation is still TODO",
            "selection_metric": config["valid_metric"],
            "best_score": float(best_score),
            "metrics": {key: float(value) for key, value in best_metrics.items()},
        })
        manifest.update({
            "status": "complete", "model": config["model"], "dataset": config["dataset"],
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


def export_scores(model, dataset, loader, split_id, partition) -> ScoreTable:
    """TODO Caio and Gabriel: connect a selected RecBole model to the score contract.

    Use recbole.utils.case_study.full_sort_scores in user batches. Translate BOTH
    axes using dataset.id2token; exclude padding and agreed seen/history items.
    Retain every eligible candidate score, not just top-K items. Reject unexpected
    NaNs/infinities rather than silently dropping different items for each model.
    Record and compare the actual candidate sets using require_aligned.
    The provided loader must belong to the same exported split as split_id.
    """
    raise NotImplementedError("Caio and Gabriel: implement aligned full-candidate score export.")
