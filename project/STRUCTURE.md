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
`NotImplementedError`. They are optional places to connect completed work, not
three extra deliverables. The same applies to the unfinished algorithms; there
are no fabricated zero metrics or silent pass-through implementations.

## What the team needs first

| Handoff | Owner | Consumer | Minimum shared result |
| --- | --- | --- | --- |
| Protocol and candidate policy | Caio with Jacek | Everyone | Agreed split, relevance rule, cutoff, external IDs, and candidate set; record decisions in `PROTOCOL.md`. |
| Individual predictions | Gabriel with Caio | Bogdan and Jacek | Model/config/run ID and comparable `ScoreTable` values on the agreed development split. |
| First weighted hybrid | Bogdan | Caio and Jacek | Fitting rule, component run IDs, coefficients, and predictions for the same candidates. |
| Independent metric pilot | Jacek | Caio and Victor | Metric definition, known-answer example, and result on the first real model output. |
| Reranker inputs | Jacek with Gabriel and Victor | Bogdan and Victor | Agreed `RerankContext`, method objective, strength, and ordered output. |

The first four rows support the 6 October end-to-end checkpoint in
`work_distribution.md`; the reranker interface follows for Task 3. The
synthetic demo only illustrates the interfaces. Real score export, hybrid
fitting, metrics, and rerankers remain team work.

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
    protocol.yaml              # Shared provisional split/evaluation settings
    experiments.yaml           # TODO model/search/metric/reranker/analysis plan
    models/bpr.yaml             # First individual-model config; add the lecture set
  examples/toy.json             # Tiny synthetic example, committed with the code
  recsys/
    __main__.py                # CLI: demo, train, task1, task2, task3
    contracts.py               # ScoreTable, alignment checks, ranking, metadata types
    data.py                    # Candidate helper and metadata-preparation placeholder
    artifacts.py               # Small run-record helpers; extend only if needed
    baselines.py               # Random and training-popularity scoring examples
    demo.py                    # Small runnable example joining the shared pieces
    recbole_adapter.py         # Real training/split export; full-score export TODO
    hybrids.py                 # Regression hybrid and alternative-hybrid placeholders
    tuning.py                  # Optional search wrapper; existing scripts may suffice
    metrics/
      accuracy.py              # Recall reference and remaining accuracy metrics TODO
      beyond_accuracy.py       # Diversity, novelty, calibration, fairness, bias TODO
    rerankers/
      diversification.py       # Gabriel
      calibration.py           # Jacek
      item_fairness.py          # Jacek
      user_fairness.py          # Victor
    analysis/
      coefficients.py          # Contribution/ablation analysis: Bogdan
      groups.py                # User/item comparisons: Victor
      explanations.py          # Mechanisms and improvement evidence: Victor
    experiments/
      task1.py                 # Optional Task 1 orchestration placeholder
      task2.py                 # Optional Task 2 orchestration placeholder
      task3.py                 # Required order interfaces; optional overall runner
  tests/test_starter.py         # Focused checks for the working shared pieces
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
| 1.1 Individual recommenders | `configs/models/`, `recsys/recbole_adapter.py` | Gabriel / Bogdan |
| 1.2 Individual tuning | `run_hyper.py` or `recsys/tuning.py`; optional `recsys/experiments/task1.py` | Gabriel / Caio |
| 1.3 Weighted regression hybrid | `recsys/hybrids.py` | Bogdan / Gabriel |
| 1.4 Other hybrids | `recsys/hybrids.py` (split into modules as needed) | Bogdan / Caio |
| 1.5 Hybrid tuning | `recsys/tuning.py` or a small script; optional `recsys/experiments/task1.py` | Bogdan / Caio |
| 2.1 Independent metrics | `recsys/metrics/` | Jacek / Caio |
| 2.2 Model/baseline comparison | `recsys/baselines.py`; optional `recsys/experiments/task2.py` | Caio / Gabriel |
| 2.3 Coefficient analysis | `recsys/analysis/coefficients.py` | Bogdan / Victor |
| 2.4 Explanatory analysis | `recsys/analysis/explanations.py` | Victor / Bogdan |
| 2.5 User/item groups | `recsys/analysis/groups.py`, `recsys/data.py` | Victor / Caio |
| 2.6 Insights and improvement | `recsys/analysis/explanations.py`, report discussion | Victor / Jacek |
| 3.1 Four rerankers | `recsys/rerankers/` | Jacek, with Gabriel and Victor / Gabriel |
| 3.2 Reranking trade-offs | `recsys/metrics/` and `recsys/rerankers/`; optional `recsys/experiments/task3.py` | Jacek / Caio |
| 3.3 Both pipeline orders | `recsys/experiments/task3.py`, `recsys/hybrids.py` | Bogdan / Jacek |
| 3.4 Reranker group effects | `recsys/analysis/groups.py`; optional `recsys/experiments/task3.py` | Victor / Caio |

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
- Jacek: implement the agreed metrics with known-answer cases, then calibration
  and item fairness. Review score masking before the larger experiment runs.
- Victor: prepare training-derived groups/profiles with Caio, implement user
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
