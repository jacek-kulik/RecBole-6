# EASE and SLIMElastic retuning on 60/20/20

Retune only EASE and SLIMElastic on MovieLens-100K's shared per-user random
60/20/20 split, seed 42. UserKNN and FISM are outside this experiment. During
PR review, the user confirmed **validation NDCG@10** as the primary selection
metric for this experiment and the shared protocol. MRR@10 and Recall@10 are
secondary diagnostics. The superseded MRR selection is preserved in
`mrr-selection-before-review.json` and commit `046bc186`.

## Protocol and search

Exact split hash:
`1c62b75f944bd6cea03626d8cc1bac05289d65ab4e04003d1f10a92d20edbea1`.
There are 60,734 training, 19,633 validation and 19,633 test interactions.
RecBole rounds within each user's history: validation/test each receive
floor(0.2 times history length), with the remainder in training. All three
partitions are disjoint. All 943 users and 1,682 catalogue items are retained.
Every validation interaction is relevant; candidates exclude training history
and padding. Ranking ties follow external item tokens.

The existing grids are evaluated exhaustively:

- EASE: `reg_weight` in `[1, 10, 25, 50, 100, 250, 500, 1000, 2000]` (9 fits).
- SLIMElastic: `alpha` in `[0.005, 0.01, 0.05, 0.2, 0.5]` crossed with
  `l1_ratio` in `[0.005, 0.01, 0.02, 0.1, 0.5]` (25 fits).

Both native RecBole constructors fit directly on training data; no dummy
optimizer epochs are needed. Select the highest validation NDCG@10, retaining
the first declared configuration on exact ties. SLIM retains positive
coefficients and target-item hiding. Its configurable iteration cap is 1,000
for every trial, compared with the old hard-coded 100. Nonconverged fits cannot
win selection; all 25 SLIM fits converged. Old parameters are refitted with the
same solver budget to isolate the hyperparameter change.

Test membership is reconstructed only to verify split identity. No test
predictions or metrics are computed by tuning or verification. These are
single-seed development results, without a significance claim.

## Validation results

| Model/configuration | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| EASE old: reg_weight=250 | 0.186311 | 0.521718 | 0.297866 |
| EASE selected: reg_weight=500 | 0.187217 | 0.521165 | 0.298099 |
| SLIM old: alpha=0.2, l1_ratio=0.01 | 0.185749 | 0.525217 | 0.300154 |
| SLIM selected: alpha=0.2, l1_ratio=0.005 | 0.184538 | 0.526636 | 0.301125 |

EASE's NDCG-selected regularization changes from 250 to 500: NDCG increases by
0.000233 and Recall by 0.000905, while MRR decreases by 0.000553. The earlier
MRR-based selection retained 250. SLIM selects the same parameters under either
metric: NDCG increases by 0.000971 (0.32% relative), MRR by 0.001419 (0.27%),
and Recall decreases by 0.001211. These gains are small and metric-specific.

## Reproduction and handoff

From the repository root using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m project.scripts.retune_ease_slim
.\.venv\Scripts\python.exe -m unittest project.tests.test_retune_ease_slim project.tests.test_accuracy project.tests.test_artifacts
```

Defaults create local results under
`project/artifacts/individual-retuning-ndcg/results` and models/scores under
`project/artifacts/individual-retuning-ndcg/models`. Completed trials resume
only with matching source/config/data identity and checkpoint/score hashes.
Use new `--output` and `--artifacts` directories when inputs change. Defaults
avoid the committed JSON evidence because its large checkpoints are not in Git.
JSON evidence uses stable LF line endings.

Committed results under
`hyperparameters/results/individual-retuned-60-20-20/` contain all 34 trials,
selected and baseline parameters, aggregate metrics and input/artifact hashes.
Large checkpoints, full scores and selected per-user metrics remain local and
reproducible. Both parameter YAML formats match the NDCG selection. The shared
protocol and both individual-model YAMLs select by NDCG@10. Historical 80/10/10
result files and existing hybrid caches are not rewritten.

## PR review and independent reproduction

After fixing output defaults and aligning the metric, all 34 configurations
were fitted afresh with the NDCG protocol. Every trial's validation metrics
exactly reproduce the corresponding original fit: the metric changes which
EASE fit is selected, not either constructor's mathematical training objective.

The independent verifier checks 34 original and 34 reproduced fits:

- Correct per-user split counts and disjoint training/validation/test pairs.
- Every saved interaction matrix equals training data exactly.
- Checkpoint hyperparameters, zero coefficient diagonals, SLIM constraints,
  scores, external-ID mapping and training-history candidate masks.
- Independent validation MRR@10, Recall@10 and NDCG@10 for every trial.
- Selection maximizes each run's declared metric; the final run uses NDCG.
- Both individual YAMLs alone and both project model YAMLs with the shared
  protocol reconstruct the same user/item split membership.

Twenty-one focused tests pass. Compact evidence is in `review-verification.json`.
The original runner source hash was checked against commit `c27ab257`; the
MRR-selected records remain in Git history. Repeat the audit when both sets
of local checkpoints are available, using a new output directory:

```powershell
.\.venv\Scripts\python.exe -m project.scripts.verify_ease_slim_retuning --original project/artifacts/pr21-mrr-records
```
