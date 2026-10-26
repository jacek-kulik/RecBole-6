# Reranking and strength selection

Jacek's Task 3.1 contribution implements genre calibration, popularity calibration,
and item-side exposure fairness. The runner also registers Gabriel's
diversification and Victor's user-side fairness functions. Those functions still
raise `NotImplementedError` on the current base and are reported as pending.
They are not replaced with unchanged rankings.

## Methods and inputs

The methods retain `rerank(scores, context, k, strength) -> Rankings`.
Calibration accepts the keyword-only `attribute="genre"` or `"popularity"`.
`RerankContext.user_profiles` contains training genre distributions;
`popularity_profiles` contains training head/middle/tail distributions.
The context is built from the existing `TrainingStatistics` metadata helpers.

The runner takes each user's top 50 eligible candidates and returns 10 items.
When fewer than 50 candidates exist, it uses all of them, requiring at least K.
Relevance scores are min-max normalized within each user's candidate pool;
constant scores map to zero. Every tie is resolved by original score, then
external item ID. Strength zero exactly reproduces the original top K.

| Method | Greedy objective | Selection target |
| --- | --- | --- |
| Genre calibration | `(1-strength) * relevance - strength * KL(p, 0.99*q + 0.01*p)` | GenreMiscalibration@10 |
| Popularity calibration | `(1-strength) * relevance - strength * JS(p, q)` | UPD@10 |
| Item fairness | `(1-strength) * relevance + strength / (1 + item_exposure)` | ExposureGini@10 |

Genre distributions give each movie one unit of mass, divided equally among
its genres. KL uses natural logs; JS is base-2 divergence, matching the existing
UPD metric. Item groups retain the existing 20/60/20 cumulative training
interaction-share construction, including its count-tie policy.

Item fairness visits users in sorted external-ID order for each rank, then
increments the selected movie's exposure by `1/log2(rank+1)`. Exposure starts
at zero for every experiment. This is a deterministic heuristic, not a guarantee
of equal exposure or minimum Gini. Its outcome depends on the supplied cohort
and the declared user order. Gini includes the full checkpoint catalogue,
including movies absent from every top-50 candidate pool.

Missing required metadata makes the affected method unavailable for the cohort.
The runner preserves the cohort and continues other methods. If a new test
candidate lacks metadata, the affected method retains its frozen choice and
is recorded as unavailable; the comparison includes the methods that ran.
Profiles use only training interactions. Validation masks training history;
test masks training and validation history.

## Validation and final test

Sweep `[0, 0.25, 0.5, 0.75, 1]` independently for each model and method.
Choose the lowest target metric among strengths satisfying:

```text
validation NDCG@10 >= 0.95 * original validation NDCG@10
```

Break ties by higher NDCG, then lower strength. Strength zero remains an eligible
fallback. Registered diversification maximizes ILD; registered user fairness
minimizes the existing absolute F/M NDCG gap when those methods become available.

Test evaluation requires the completed validation `selection.json`. It binds
the checkpoint, both score exports, split files, dataset files, relevant source
code, K and candidate count. Changed inputs/settings require a new validation
sweep. Hashing the test export verifies identity; its outcomes do not determine
the validation choice. A test command cannot supply a strength grid.

Methods absent from an otherwise valid frozen selection are reported as
unavailable and skipped on test. They do not receive a default strength or a
new test-based selection. The source-code identity check still applies; a
selection created with different code requires a new validation sweep.

The 5% limit applies to validation. Test losses can be larger: they are recorded
as `exceeds_5_percent_loss`, without changing the frozen strength.

## Commands

Run from the repository root using the RecBole environment. A separate worktree
can use the original checkout's Python by absolute path. Use new output names;
existing artifact directories are rejected.

For one completed training run with full score exports:

```sh
python -m project.recsys rerank \
  --run project/artifacts/ease-002 \
  --output project/artifacts/ease-rerank-valid-002

python -m project.recsys rerank \
  --run project/artifacts/ease-002 \
  --partition test \
  --selection project/artifacts/ease-rerank-valid-002/selection.json \
  --output project/artifacts/ease-rerank-test-002
```

Options include `--k`, `--candidate-count`, `--batch-size` and validation-only
`--strengths`. Defaults are K=10, 50 candidates and the five strengths above.
Old training runs without full score exports must be trained with the current
adapter. The source score table's model label distinguishes UserKNN from ItemKNN.

For all six checked-in configurations, without additional model tuning:

```sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
sh project/scripts/train_individuals.sh rerankers-002
python -m project.scripts.rerank_individuals \
  --run-suffix rerankers-002 \
  --output project/artifacts/reranking-002
python -m project.scripts.verify_reranking \
  --run project/artifacts/reranking-002 \
  --output project/artifacts/reranking-002/verification.json
```

The training script also accepts `PYTHON=/absolute/path/to/python`. The experiment
script accepts `--source-root` for completed training artifacts elsewhere.
It completes all six validation sweeps before beginning test evaluation.

Plotting additionally requires Matplotlib and does not load checkpoints:

```sh
python -m project.scripts.plot_reranking \
  --run project/artifacts/reranking-002 \
  --output project/artifacts/reranking-002/figures
```

## Results and handoff

Each model has `valid/summary.csv`, `valid/selection.json`, and
`test/summary.csv`. Each evaluated point contains the standard rankings,
relevance, aggregate/per-user accuracy, beyond-accuracy metrics, group interfaces,
and manifest. Test folders work with Caio's `compare` command. The six-model
runner creates a combined 24-row comparison and per-user CSV.

The verifier independently recalculates all six accuracy metrics, discounted
exposure Gini and coverage from saved lists. It also checks candidate/history
exclusion, original rankings, source hashes, validation choices, frozen test
strengths, and agreement with the final comparison table.

[The experiment record and report drafts](reports/task-3.1-3.2-rerankers.md)
document the first completed run. Checkpoints, score exports, detailed metrics
and figures remain under ignored `project/artifacts/` directories.

Victor retains group interpretation, Bogdan retains both hybrid/reranker
orders, and their existing placeholders remain explicit. Finishing a runnable
reranker evaluation does not mark those tasks or the whole project complete.
