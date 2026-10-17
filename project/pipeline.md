## Running a single model

Run these commands from the repository root. This example uses BPR.
Use new output directory names for each run; existing directories are rejected.

### 1. Activate the environment and check the code

```sh
source .venv/bin/activate
ruff check project/recsys project/tests
python -W error -m unittest discover -s project/tests -v
```

### 2. Train the model

Check the settings in `project/configs/models/bpr.yaml` first.
The current BPR configuration is a small five-epoch starting run.

```sh
python -m project.recsys train \
  --model-config project/configs/models/bpr.yaml \
  --output project/artifacts/bpr-001
```

The shared protocol supplies the per-user 60/20/20 split and seed 42.
Training saves the checkpoint selected by validation MRR@10, the exact splits,
and the configuration. It does not evaluate the test set.

### 3. Calculate independent validation metrics

```sh
python -m project.recsys validate \
  --run project/artifacts/bpr-001 \
  --output project/artifacts/bpr-valid-001 \
  --k 10 --batch-size 1024
```

Inspect the accuracy results, comparison with RecBole, and beyond-accuracy results:

```sh
python -m json.tool project/artifacts/bpr-valid-001/metrics.json
python -m json.tool project/artifacts/bpr-valid-001/comparison.json
python -m json.tool project/artifacts/bpr-valid-001/beyond_accuracy.json
```

To try other hyperparameters, update the model configuration and repeat training
and validation with new directory names, such as `bpr-002` and `bpr-valid-002`.
Choose settings using validation results, then freeze the chosen configuration.

### 4. Evaluate the selected model on the test set

Run this after model settings have been frozen. Replace `bpr-001` with the
selected training run. Both `validate` and `evaluate` take the training directory
as `--run`.

```sh
python -m project.recsys evaluate \
  --run project/artifacts/bpr-001 \
  --output project/artifacts/bpr-test-001 \
  --k 10 --batch-size 1024
```

Test evaluation excludes both training and validation history from recommendations.

### 5. Inspect the final results and pending work

```sh
python -m json.tool project/artifacts/bpr-test-001/metrics.json
python -m json.tool project/artifacts/bpr-test-001/beyond_accuracy.json
python -m json.tool project/artifacts/bpr-test-001/groups.json
python -m json.tool project/artifacts/bpr-test-001/group_metrics.json
python -m json.tool project/artifacts/bpr-test-001/manifest.json
```

Keep the training, validation, and test directory names with any reported results.

Metadata preparation and group reporting still contain placeholders. Check
unavailable metric reasons and `pending_work` before using results in the report.
Full score export, hybrids, and rerankers also remain unfinished; there are no
working shell commands for those stages yet.