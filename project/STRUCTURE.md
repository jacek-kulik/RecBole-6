# Project starter structure

Put the group's implementation under `project/`. Keep `recbole/` as the framework
code and change it only when a documented extension cannot live in the project
adapter. This gives each part a clear home without moving existing coursework
scripts or mixing generated results into source directories.

## Start here

Run commands from the repository root with Python 3.9 or newer. The demo and
starter tests use only the standard library:

```sh
python -m project.recsys --help
python -m project.recsys demo --output project/artifacts/demo-001
python -m unittest discover -s project/tests -v
```

The demo runs random and popularity baselines on a tiny synthetic fixture,
filters training-seen items, ranks candidates, calculates independent Recall,
and saves JSON scores, rankings, metrics, the split, and a run manifest. It is
not a MovieLens experiment and does not complete any assignment subtask.
Choose a new output directory for each run; existing directories are rejected.

For real model training, use the repository's RecBole environment. If you need
to create one, the existing project installation is:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m project.recsys train \
  --model-config project/configs/models/bpr.yaml \
  --output project/artifacts/bpr-001
```

Use the PyTorch build appropriate for your machine when setting up that
environment. The starter defaults to CPU. In a separate worktree you may invoke
an existing repository environment's Python by absolute path while keeping the
working directory at this worktree's root.

Training runs the local RecBole implementation, selects a checkpoint using
validation, and exports the actual train/validation/test interaction splits with
external IDs, ratings, and timestamps. It saves the supplied YAMLs, resolved
configuration, split hash, checkpoint path, and validation diagnostics. It does
not yet export model scores or perform the complete independent evaluation.

The `task1`, `task2`, and `task3` commands deliberately stop with a descriptive
`NotImplementedError` until the corresponding orchestration is filled in. The
same applies to the unfinished algorithms; there are no fabricated zero metrics
or silent pass-through implementations.

## File layout

```text
project/
  STRUCTURE.md                 # This guide and task-to-file map
  PROTOCOL.md                  # Decisions the team must settle before comparison
  work_distribution.md         # Ownership, reviewers, and schedule
  configs/
    protocol.yaml              # Shared provisional split/evaluation settings
    experiments.yaml           # TODO model/search/metric/reranker/analysis plan
    models/bpr.yaml             # First individual-model config; add the lecture set
  examples/toy.json             # Tiny synthetic example, committed with the code
  recsys/
    __main__.py                # CLI: demo, train, task1, task2, task3
    contracts.py               # ScoreTable, alignment checks, ranking, metadata types
    data.py                    # Candidate helper and metadata-preparation placeholder
    artifacts.py               # Run directories, JSON, hashes, Git provenance
    baselines.py               # Random and training-popularity scoring examples
    demo.py                    # Small runnable example joining the shared pieces
    recbole_adapter.py         # Real training/split export; full-score export TODO
    hybrids.py                 # Regression hybrid and alternative-hybrid placeholders
    tuning.py                  # Individual/hybrid search and selection placeholder
    metrics/
      accuracy.py              # Recall reference and remaining accuracy metrics TODO
      beyond_accuracy.py       # Diversity, novelty, calibration, fairness, bias TODO
    rerankers/
      diversification.py       # PERSON-2
      calibration.py           # PERSON-4
      item_fairness.py          # PERSON-4
      user_fairness.py          # PERSON-5
    analysis/
      coefficients.py          # Contribution/ablation analysis: PERSON-3
      groups.py                # User/item comparisons: PERSON-5
      explanations.py          # Mechanisms and improvement evidence: PERSON-5
    experiments/
      task1.py                 # Model training, tuning, and hybrid orchestration
      task2.py                 # Independent evaluation and analysis orchestration
      task3.py                 # Reranking sweeps and both required pipeline orders
  tests/test_starter.py         # Focused checks for the working shared pieces
  scripts/                     # Existing batch/tuning-summary helpers
  artifacts/                   # Generated run directories; ignored by Git
```

Keep reusable algorithms in their own modules; experiment files should call
them and record the experiment settings. `configs/experiments.yaml` supplies
commented placeholders for those settings; the unfinished task commands do not
consume it yet. Put one-off exploration in an optional
`project/notebooks/` directory, then move reusable logic into `recsys/`. Put
selected report figures under an optional `project/report/figures/` directory,
with their generating command and source run ID recorded. This starter does not
add the separately prepared Overleaf ZIP to the repository.

## How data moves through the project

1. Agree `PROTOCOL.md` and update `configs/protocol.yaml`. The shared YAML is
   loaded after the model YAML so a model file cannot silently change the split.
2. Train individuals through `recbole_adapter.py`. Export the actual splits
   from that run; do not reconstruct them independently without comparing IDs
   and hashes. The saved CSVs preserve relevance-related fields.
3. Implement `export_scores` in the adapter. Emit a `ScoreTable` for every model,
   using original user/item tokens as strings. Export all eligible candidates
   per user so a hybrid can align scores across models.
4. Fit and tune models/hybrids on their assigned development data. Call
   `require_aligned` before combining scores. Save fitted normalization and
   regression coefficients with the component run IDs.
5. Produce rankings, run the group's independent metrics, and save both detailed
   and aggregate measurements. The existing Recall function demonstrates one
   definition; it does not settle the lecture metric set or cohort policy.
6. Apply the four rerankers and compare strengths. Task 3.3 needs both rerank then
   combine and combine then rerank. A reranker returns an ordered list; combining
   those lists requires an explicit fusion/rank-to-score and truncation policy.
7. Feed recorded results into coefficient, explanatory, and group analyses.
   Export tables alongside figures and cite their run IDs in the report.

The score contract checks identical split ID, partition, users, and candidate
sets. It cannot prove that an algorithm avoided leakage: fitting data, relevance,
masking, and group definitions still need the scheduled peer reviews. Padding or
history-masked scores must be removed according to one agreed candidate policy;
unexpected nonfinite scores are errors, not missing candidates to quietly drop.

## Where each numbered subtask belongs

| Subtask | Main files | Owner / first reviewer |
| --- | --- | --- |
| 1.1 Individual recommenders | `configs/models/`, `recsys/recbole_adapter.py` | PERSON-2 / PERSON-3 |
| 1.2 Individual tuning | `recsys/tuning.py`, `recsys/experiments/task1.py` | PERSON-2 / PERSON-4 |
| 1.3 Weighted regression hybrid | `recsys/hybrids.py` | PERSON-3 / PERSON-2 |
| 1.4 Other hybrids | `recsys/hybrids.py` (split into modules as needed) | PERSON-3 / PERSON-5 |
| 1.5 Hybrid tuning | `recsys/tuning.py`, `recsys/experiments/task1.py` | PERSON-3 / PERSON-1 |
| 2.1 Independent metrics | `recsys/metrics/` | PERSON-4 / PERSON-5 |
| 2.2 Model/baseline comparison | `recsys/baselines.py`, `recsys/experiments/task2.py` | PERSON-1 / PERSON-2 |
| 2.3 Coefficient analysis | `recsys/analysis/coefficients.py` | PERSON-3 / PERSON-5 |
| 2.4 Explanatory analysis | `recsys/analysis/explanations.py` | PERSON-5 / PERSON-3 |
| 2.5 User/item groups | `recsys/analysis/groups.py`, `recsys/data.py` | PERSON-5 / PERSON-1 |
| 2.6 Insights and improvement | `recsys/analysis/explanations.py`, report discussion | PERSON-5 / PERSON-4 |
| 3.1 Four rerankers | `recsys/rerankers/` | PERSON-4, with PERSON-2/5 / PERSON-2 |
| 3.2 Reranking trade-offs | `recsys/experiments/task3.py`, `recsys/metrics/` | PERSON-4 / PERSON-1 |
| 3.3 Both pipeline orders | `recsys/experiments/task3.py`, `recsys/hybrids.py` | PERSON-3 / PERSON-4 |
| 3.4 Reranker group effects | `recsys/analysis/groups.py`, `recsys/experiments/task3.py` | PERSON-5 / PERSON-3 |

## First contributions

- PERSON-1: settle the split/candidate contract with PERSON-4, then finish the
  adapter's batched full-score export with PERSON-2. Reuse the working artifact
  helpers so model runs can be compared and traced.
- PERSON-2: add the lecture model configurations and establish a bounded tuning
  search. Start the diversification method once metadata is agreed.
- PERSON-3: choose coefficient-fitting and selection data, implement the
  regression hybrid, then add the other hybrids and both reranking orders.
- PERSON-4: implement the agreed metrics with known-answer cases, then calibration
  and item fairness. Review score masking before the larger experiment runs.
- PERSON-5: prepare training-derived groups/profiles with PERSON-1, implement user
  fairness, and draft the explanatory comparisons and report figures.

Search for `TODO` and `NotImplementedError` under `project/recsys/` to find the
remaining implementation points. Update the matching tests when adding actual
behavior, especially alignment, relevance, group aggregation, and leakage checks.

## Existing tools and framework changes

`run_recbole.py` and `run_hyper.py` remain available. The project wrapper is a
starting point for shared configuration and artifacts; it does not replace every
RecBole feature. Existing root-level `save_split.py`, `save_recommendations.py`,
and `score_from_saved.py` are useful references, but their output formats differ:
the existing split export omits ratings, and the recommendation export keeps only
top-K items. Adapt them deliberately if reusing them for the full project.

Generated project artifacts go under `project/artifacts/<run-id>/`. RecBole also
uses its existing root `log/` and `log_tensorboard/` locations. Keep source/configs
in Git; save selected evidence and reproduction instructions for the final code
ZIP. If a framework edit is necessary, name the changed `recbole/` files and the
reason in the project documentation so reviewers can find it.
