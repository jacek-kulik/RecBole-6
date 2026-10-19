"""Retune corrected FISM on the shared protocol; select exclusively on validation.

Run from the repository root: .venv/Scripts/python.exe -m project.scripts.retune_fism
Completed trials are saved individually and reused only with matching input hashes.
Historical results are preserved. Test is evaluated for the selected model and the
historical parameter baseline only, after the validation search has completed.
"""

import argparse
import ast
import hashlib
import itertools
import json
import logging
import platform
import time
from pathlib import Path

import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender.fism import FISM
from recbole.trainer import Trainer
from recbole.utils import init_seed
from project.scripts.summarize_hyperparameter_results import parse_trials, choose_trial


CONFIGS = ["recbole/config/FISM/ml-100k.yaml", "project/configs/protocol.yaml"]
OUTPUT = Path("hyperparameters/results/fism-corrected-60-20-20")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def search_space():
    space = {}
    for line in Path("hyperparameters/hyper-FISM.test").read_text().splitlines():
        name, kind, values = line.split(maxsplit=2)
        if kind != "choice":
            raise ValueError("This runner supports finite choice grids only")
        space[name] = [ast.literal_eval(v) if isinstance(v, str) else v
                       for v in ast.literal_eval(values)]
    return [dict(zip(space, values)) for values in itertools.product(*space.values())]


def train(params, directory):
    config = Config(model="FISM", config_file_list=CONFIGS, config_dict={
        **params, "checkpoint_dir": str(directory),
        "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision"],
    })
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = FISM(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)
    score, valid = trainer.fit(train_data, valid_data, verbose=False, saved=True)
    trainer.tensorboard.close()
    return trainer, test_data, {
        "parameters": params, "validation_mrr": score, "validation": valid,
        "checkpoint": trainer.saved_model_file,
    }


def evaluate(params, checkpoint):
    config = Config(model="FISM", config_file_list=CONFIGS, config_dict={
        **params, "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision"],
    })
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, _, test_data = data_preparation(config, dataset)
    model = FISM(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)
    result = trainer.evaluate(test_data, model_file=checkpoint)
    trainer.tensorboard.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.ERROR)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inputs = [*CONFIGS, "hyperparameters/hyper-FISM.test",
              "hyperparameters/results/hyper-FISM.result", "project/scripts/retune_fism.py",
              "recbole/model/general_recommender/fism.py",
              "recbole/trainer/trainer.py", "recbole/data/dataset/dataset.py",
              "recbole/data/interaction.py", "recbole/properties/overall.yaml",
              "recbole/properties/model/FISM.yaml",
              *map(str, Path("dataset/ml-100k").glob("*.inter"))]
    identity = {"sha256": {str(p): digest(p) for p in inputs},
                "torch": torch.__version__, "python": platform.python_version(),
                "threads": args.threads}
    manifest = OUTPUT / "identity.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise RuntimeError("Run inputs changed; use a new output directory")
    write_json(manifest, identity)
    historical = choose_trial(parse_trials(Path("hyperparameters/results/hyper-FISM.result")), "best", 10)
    old_params = dict(part.split(":", 1) for part in historical.parameters.split(", "))
    old_params = {k: ast.literal_eval(v) for k, v in old_params.items()}
    trials = []
    grid = search_space()
    for index, params in enumerate(grid):
        result_file = OUTPUT / f"trial-{index:02d}.json"
        if result_file.exists():
            result = json.loads(result_file.read_text())
            if result["parameters"] != params or not Path(result["checkpoint"]).exists():
                raise RuntimeError("Saved trial parameters or checkpoint do not match")
        else:
            started = time.monotonic()
            print(f"Trial {index + 1}/{len(grid)}: {params}", flush=True)
            _, _, result = train(params, Path("saved/fism-corrected-60-20-20") / f"trial-{index:02d}")
            result["seconds"] = round(time.monotonic() - started, 2)
            write_json(result_file, result)
        trials.append(result)
        print(f"  validation MRR@10={result['validation_mrr']:.4f}; {result['seconds']}s", flush=True)
    best = max(trials, key=lambda trial: trial["validation_mrr"])
    baseline = next(t for t in trials if t["parameters"] == old_params)
    # Freeze selection before inspecting either test result.
    write_json(OUTPUT / "selection.json", {"best": best, "baseline": baseline})
    for trial in (baseline, best):
        if "test" not in trial:
            trial["test"] = evaluate(trial["parameters"], trial["checkpoint"])
    summary = {"selection_metric": "validation MRR@10", "trials": len(trials),
               "historical_80_10_10": {"parameters": old_params, "validation": historical.valid, "test": historical.test},
               "corrected_old_parameters_60_20_20": baseline,
               "corrected_retuned_60_20_20": best}
    write_json(OUTPUT / "summary.json", summary)
    (OUTPUT / "best-params.yaml").write_text(
        "# Load after recbole/config/FISM/ml-100k.yaml and project/configs/protocol.yaml.\n"
        + "\n".join(f"{key}: {value}" for key, value in best["parameters"].items()) + "\n",
        encoding="utf-8",
    )
    lines = ["# FISM logits fix and retuning", "",
             "The historical results accompany an 80/10/10 configuration (default seed 2020); their runtime provenance was not recorded. The corrected runs use the shared per-user random 60/20/20 protocol, seed 42, full-sort evaluation, 20 epochs, validation every 10 epochs, and validation MRR@10 selection.", "",
             "All 48 existing grid combinations were evaluated without search-level early stopping. Test was evaluated only after selection, for the historical parameter configuration and the selected configuration. The original historical files are preserved.", "",
             "Historical versus corrected results also change the protocol; they do not isolate the bug fix. The two corrected rows isolate parameter retuning on the same split, seed, training budget, and scoring code. History vectors are summed before candidate dot products to avoid large repeated tensors; regression tests check equivalent scores and gradients.", "",
             "| Run | Embedding size | Regularization (source, destination) | Alpha | Validation MRR@10 | Test MRR@10 | Test NDCG@10 | Test Recall@10 |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for name, params, valid, test in [
        ("Historical (buggy, 80/10/10)", old_params, historical.valid, historical.test),
        ("Corrected, old parameters (60/20/20)", baseline["parameters"], baseline["validation"], baseline["test"]),
        ("Corrected, retuned (60/20/20)", best["parameters"], best["validation"], best["test"]),
    ]:
        lines.append(f"| {name} | {params['embedding_size']} | {params['reg_weights']} | {params['alpha']} | {valid['mrr@10']:.4f} | {test['mrr@10']:.4f} | {test['ndcg@10']:.4f} | {test['recall@10']:.4f} |")
    lines += ["", "These are RecBole training diagnostics; the shared protocol calls for independent metrics for final reporting.", "",
              "Reproduce: `.venv/Scripts/python.exe -m project.scripts.retune_fism --threads 2`.", "",
              "`summary.json` contains all metrics and selected checkpoint paths; `trial-*.json` contains every validation result and runtime. `identity.json` records input hashes and runtime versions. Checkpoints remain local in ignored `saved/`. Load `best-params.yaml` after the model and shared protocol YAMLs to reuse the selected hyperparameters."]
    (OUTPUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
