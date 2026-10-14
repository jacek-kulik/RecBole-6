# Project starter structure

Put the group's implementation under `project/`. Keep `recbole/` as the framework
code and change it only when a documented extension cannot live in the project
adapter. This gives each part a clear home without moving existing coursework
scripts or mixing generated results into source directories.

This is a map of ownership and interfaces, not a list of 36 files to finish.
The task files show where work can go; the team can use existing repository
scripts when they meet the agreed protocol. Only build shared helpers when a
required experiment needs them.

## Start here

Run commands from the repository root with Python 3.10 or newer. The demo and
starter tests use only the standard library:

```sh
python -m project.recsys --help
python -m project.recsys demo --output project/artifacts/demo-001
python -m unittest discover -s project/tests -v
```

The demo runs random and popularity baselines on a tiny synthetic fixture,
filters training-seen items, ranks candidates, calculates all the independent accuracy evaluation metrics,
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
  --output project/artifacts/bpr-60-001
```

Use the PyTorch build appropriate for your machine when setting up that
environment. The starter defaults to CPU. In a separate worktree you may invoke
an existing repository environment's Python by absolute path while keeping the
working directory at this worktree's root.

Training runs the local RecBole implementation with the shared 60/20/20 protocol,
selects a checkpoint using validation, and exports the actual splits with
external IDs, ratings, and timestamps. It saves the supplied YAMLs, resolved
configuration, split hash, checkpoint path, and validation diagnostics. It does
not yet export full candidate scores for hybrids. Run independent evaluation
separately on the selected checkpoint.

For independent development metrics, use the saved training run and a new output
directory. Adjust the paths to your actual run:

```sh
python -m project.recsys validate \
  --run project/artifacts/bpr-60-001 \
  --output project/artifacts/bpr-60-valid-001 \
  --k 10 --batch-size 1024
```

This loads the checkpoint on CPU and verifies its saved split before scoring.
It preserves rankings, relevance, aggregate/per-user accuracy, comparisons and
a worked example. It also saves all eight beyond-accuracy measurements and
activity reports, with F/M gender fairness within each cohort.

When model settings are frozen, final test evaluation uses the same arguments:

```sh
python -m project.recsys evaluate \
  --run project/artifacts/bpr-60-001 \
  --output project/artifacts/bpr-60-test-001 \
  --k 10 --batch-size 128
```

`validate` scores validation only. `evaluate` scores test only and does not
compare test metrics against validation diagnostics. The test loader masks both
training and validation history. Older saved runs reconstruct their original
splits; changing the shared YAML does not change their data.

The output files are:

| Artifact | Contents |
| --- | --- |
| `metrics.json`, `per_user_metrics.json` | Existing six accuracy metrics, preserving their format. |
| `beyond_accuracy.json` | Per-user and aggregate measurements, availability counts/reasons, gender gaps, and movie-group exposure. |
| `groups.json` | Training-derived activity/popularity memberships, tercile boundaries, achieved interaction shares, and recorded gender membership. |
| `group_metrics.json` | Overall and low/medium/high reports; cohort exposure is recomputed and F/M unfairness is calculated inside each cohort. |
| `rankings.json`, `relevant.json`, `example.json` | Ordered predictions, observed positives and a worked accuracy example. |
| `comparison.json` | Matching RecBole diagnostics, on validation only. |
| `manifest.json` | Checkpoint/split identity, partition, source metadata hashes, formulas, smoothing, group rules and run status. |

Use `groups.json` to see the inclusion criteria and sample sizes. Activity groups
serve reporting; gender supplies user-fairness groups. Missing measurements are
`null` with reasons. The formula choices are in `PROTOCOL.md`.

The reusable functions are `prepare_metadata`, `evaluate_accuracy`,
`evaluate_beyond_accuracy` and `compare_groups`. They use external IDs and typed
Python data structures. A teammate can feed aligned rankings and relevance to
these functions without training another model.

The `task1`, `task2`, and `task3` commands deliberately stop with a descriptive
`NotImplementedError`. They are optional places to connect completed work, not
three extra deliverables. The same applies to the unfinished algorithms; there
are no fabricated zero metrics or silent pass-through implementations.

## What the team needs first

| Handoff                       | Owner                         | Consumer          | Minimum shared result                                                                                     |
|-------------------------------|-------------------------------|-------------------|-----------------------------------------------------------------------------------------------------------|
| Protocol and candidate policy | Caio with Jacek               | Everyone          | Agreed split, relevance rule, cutoff, external IDs, and candidate set; record decisions in `PROTOCOL.md`. |
| Individual predictions        | Gabriel with Caio             | Bogdan and Jacek  | Model/config/run ID and comparable `ScoreTable` values on the agreed development split.                   |
| First weighted hybrid         | Bogdan                        | Caio and Jacek    | Fitting rule, component run IDs, coefficients, and predictions for the same candidates.                   |
| Independent metric pilot      | Jacek                         | Caio and Victor   | Metric definition, known-answer example, and result on the first real model output.                       |
| Reranker inputs               | Jacek with Gabriel and Victor | Bogdan and Victor | Agreed `RerankContext`, method objective, strength, and ordered output.                                   |

The first four rows support the 6 October end-to-end checkpoint in
`work_distribution.md`; the reranker interface follows for Task 3. The
synthetic demo only illustrates the interfaces. Real score export, hybrid
fitting, and rerankers remain team work; independent metrics are implemented.

The generic `experiments/task*.py` commands, `tuning.py`, and
`configs/experiments.yaml` are optional coordination placeholders. Keep the
working algorithm files separate, but use `run_hyper.py` or a small script if it
already covers an experiment. The artifact helpers are available for recording
runs; do not build a larger run framework before a required comparison needs it.

## File layout

```text
project/
  STRUCTURE.md                 # This guide and task-to-file map
  PROTOCOL.md                  # Decisions the team must settle before comparison
  work_distribution.md         # Ownership, reviewers, and schedule
  configs/
    protocol.yaml              # Shared 60/20/20 split/evaluation settings
    experiments.yaml           # TODO model/search/metric/reranker/analysis plan
    models/bpr.yaml             # First individual-model config; add the lecture set
  examples/toy.json             # Tiny synthetic example, committed with the code
  notebooks/
    movielens_analysis.ipynb   # Dataset analysis for data cleanliness and split decision
  recsys/
    __main__.py                # CLI: demo, train, validate, evaluate, task1, task2, task3
    contracts.py               # ScoreTable, alignment checks, ranking, metadata types
    data.py                    # Metadata, training profiles and separate activity/gender groups
    artifacts.py               # Small run-record helpers; extend only if needed
    baselines.py               # Random and training-popularity scoring examples
    demo.py                    # Small runnable example joining the shared pieces
    recbole_adapter.py         # Real training/split export; full-score export TODO
    hybrids.py                 # Regression hybrid and alternative-hybrid placeholders
    tuning.py                  # Optional search wrapper; existing scripts may suffice
    validate.py                # Development validation entrypoint
    evaluate.py                # Final test entrypoint
    evaluation.py              # Shared saved-run metrics/artifact runner
    metrics/
      _validation.py           # External ID and ranking list validation for metric computation
      accuracy.py              # Computes Precision@k, Recall@k, F1@k, MRR@k, NDCG@k, MAP@k, returning in aggregate and per-user
      beyond_accuracy.py       # Eight independent beyond-accuracy measurements
      types.py                 # Shared types used for the metrics computation
    rerankers/
      diversification.py       # Gabriel
      calibration.py           # Jacek
      item_fairness.py          # Jacek
      user_fairness.py          # Victor
    analysis/
      coefficients.py          # Contribution/ablation analysis: Bogdan
      groups.py                # Cohort reports for Victor; Jacek supplies reusable helpers
      explanations.py          # Mechanisms and improvement evidence: Victor
    experiments/
      task1.py                 # Optional Task 1 orchestration placeholder
      task2.py                 # Optional Task 2 orchestration placeholder
      task3.py                 # Required order interfaces; optional overall runner
  tests/                       # Contains the test files for all the modules
  scripts/                     # Existing batch/tuning-summary helpers
  artifacts/                   # Generated run directories; ignored by Git
```

Keep reusable algorithms in their own modules; experiment files should call
them and record the experiment settings. `configs/experiments.yaml` supplies
commented placeholders for those settings; the unfinished task commands do not
consume it yet, and the team need not fill unused fields. Put one-off exploration
in an optional `project/notebooks/` directory, then move reusable logic into
`recsys/`. Put selected report figures under an optional
`project/report/figures/` directory,
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
   and aggregate measurements. Accuracy and beyond-accuracy metrics are implemented;
   the reranker algorithms remain TODO.
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

| Subtask                        | Main files                                                                        | Owner / first reviewer                   |
|--------------------------------|-----------------------------------------------------------------------------------|------------------------------------------|
| 1.1 Individual recommenders    | `configs/models/`, `recsys/recbole_adapter.py`                                    | Gabriel / Bogdan                         |
| 1.2 Individual tuning          | `run_hyper.py` or `recsys/tuning.py`; optional `recsys/experiments/task1.py`      | Gabriel / Caio                           |
| 1.3 Weighted regression hybrid | `recsys/hybrids.py`                                                               | Bogdan / Gabriel                         |
| 1.4 Other hybrids              | `recsys/hybrids.py` (split into modules as needed)                                | Bogdan / Caio                            |
| 1.5 Hybrid tuning              | `recsys/tuning.py` or a small script; optional `recsys/experiments/task1.py`      | Bogdan / Caio                            |
| 2.1 Independent metrics        | `recsys/metrics/`                                                                 | Jacek / Caio                             |
| 2.2 Model/baseline comparison  | `recsys/baselines.py`; optional `recsys/experiments/task2.py`                     | Caio / Gabriel                           |
| 2.3 Coefficient analysis       | `recsys/analysis/coefficients.py`                                                 | Bogdan / Victor                          |
| 2.4 Explanatory analysis       | `recsys/analysis/explanations.py`                                                 | Victor / Bogdan                          |
| 2.5 User/item groups           | `recsys/analysis/groups.py`, `recsys/data.py`                                     | Victor / Caio                            |
| 2.6 Insights and improvement   | `recsys/analysis/explanations.py`, report discussion                              | Victor / Jacek                           |
| 3.1 Four rerankers             | `recsys/rerankers/`                                                               | Jacek, with Gabriel and Victor / Gabriel |
| 3.2 Reranking trade-offs       | `recsys/metrics/` and `recsys/rerankers/`; optional `recsys/experiments/task3.py` | Jacek / Caio                             |
| 3.3 Both pipeline orders       | `recsys/experiments/task3.py`, `recsys/hybrids.py`                                | Bogdan / Jacek                           |
| 3.4 Reranker group effects     | `recsys/analysis/groups.py`; optional `recsys/experiments/task3.py`               | Victor / Caio                            |

Use `work_distribution.md` as the source for assignments and dates. A consumer
of a result is not necessarily its reviewer. Workers supply runnable evidence
and the matching report section, make corrections, and record their resolution.
Caio checks closure for the seven tasks assigned to them for review;
reranker contribution reviews follow the separate table in the work plan.

## First contributions

- Caio: settle the split/candidate contract with Jacek, then finish the
  adapter's batched full-score export with Gabriel. Reuse the working artifact
  helpers so model runs can be compared and traced. Review Tasks 1.2, 1.4, 1.5,
  2.1, 2.5, 3.2, and 3.4, including their evidence and report claims; check that
  workers resolve the feedback.
- Gabriel: add the lecture model configurations and establish a bounded tuning
  search. Start the diversification method once metadata is agreed.
- Bogdan: choose coefficient-fitting and selection data, implement the
  regression hybrid, then add the other hybrids and both reranking orders.
- Jacek: metrics, activity reports and gender fairness are implemented. Next,
  implement calibration and item fairness and integrate the other contributions.
  Review score masking before the larger experiment runs.
- Victor: use the shared activity/gender and movie-group reports, implement user
  fairness reranking, and draft explanatory comparisons and report figures.

Search for `TODO` and `NotImplementedError` under `project/recsys/` to find the
remaining implementation points. Update the matching tests when adding actual
behaviour, especially alignment, relevance, group aggregation, and leakage checks.

## Existing tools and framework changes

The metrics/groups branch also changes `recbole/data/interaction.py`:
`Interaction.sort` converts its Torch key to a CPU NumPy array before stable
sorting, and copies reversed indices before creating a descending-sort tensor.
The previous NumPy-on-Torch call emits a NumPy 2 deprecation warning
when evaluation loaders sort users. This conversion preserves the sort order
and lets saved-run evaluation work with warnings treated as errors.

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


## Verification on 14 October

Implementation commit: `b492059a1`. The following checks were run in the task
worktree with the repository's Python 3.11 environment:

- All 59 project tests passed with warnings treated as errors, including the
  real full-sort history mask and interaction sorting checks. The full RecBole
  pytest suite was not run; pytest is not installed in this environment.
- All nine notebook code cells ran with warnings treated as errors. Saved
  outputs include both splits and the training-derived activity/movie groups.
- `metrics-old-valid-002` evaluated the existing `bpr-001` checkpoint using its
  original 80/10/10 split. The six accuracy artifact files match the previous
  `bpr-valid-005` output exactly, including per-user results and rankings.
- A new five-epoch BPR run, `bpr-60-001`, trained with a clean checkout of the
  implementation commit. `bpr-60-valid-001` independently validated it with
  batch size 128. Split counts were 60,734 / 19,633 / 19,633; activity boundaries
  were 26 and 69, with 321 / 310 / 312 users. All three saved RecBole diagnostics
  agreed at their reported precision. Validation MRR@10 was 0.29505336.

The BPR run verifies the pipeline; it is not a tuned final comparison. Its
training and validation manifests both record `test_evaluated: false`.
Final test routing was checked on fixtures only. Generated checkpoints and
metric JSON stay under the ignored `project/artifacts/` directory.

After activating the repository environment, reproduce the checks from the
repository root, choosing new output directories if these already exist:

```sh
python -W error -m unittest discover -s project/tests -q
python -W error -m project.recsys train \
  --model-config project/configs/models/bpr.yaml \
  --output project/artifacts/bpr-60-002
python -W error -m project.recsys validate \
  --run project/artifacts/bpr-60-002 \
  --output project/artifacts/bpr-60-valid-002 \
  --k 10 --batch-size 128
```

Open the notebook with this same environment and run all cells. Its code checks
that RecBole comes from the selected checkout. The source metadata files remain
unchanged, including their original placeholders.
