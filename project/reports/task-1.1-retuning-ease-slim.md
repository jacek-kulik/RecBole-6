# EASE and SLIMElastic retuning on 60/20/20

The existing EASE and SLIMElastic parameters were selected under the earlier
80/10/10 protocol. This experiment retunes only those two recommenders on the
shared MovieLens-100K per-user 60/20/20 split, seed 42. UserKNN and FISM are
outside its scope. Other recommenders and hybrid implementations are unchanged.

## Protocol and search

The reconstructed split hash is
`1c62b75f944bd6cea03626d8cc1bac05289d65ab4e04003d1f10a92d20edbea1`,
matching the saved hybrid experiment. There are 60,734 training interactions,
19,633 validation interactions, 943 evaluation users and 1,682 catalogue items.
Every validation interaction is relevant; candidates exclude training history
and padding. Ranking ties follow the external item token.

Selection uses **validation MRR@10**, the primary metric in the shared protocol.
Exact ties retain the first configuration in the declared grid. NDCG@10 and
Recall@10 are reported but do not select parameters. The test partition is
constructed only to verify split membership; no test scores or metrics are
computed. Results are development evidence from one seed.

The existing finite search spaces were evaluated exhaustively:

- EASE: `reg_weight` in `[1, 10, 25, 50, 100, 250, 500, 1000, 2000]` (9 fits).
- SLIMElastic: `alpha` in `[0.005, 0.01, 0.05, 0.2, 0.5]` crossed with
  `l1_ratio` in `[0.005, 0.01, 0.02, 0.1, 0.5]` (25 fits).

Both models fit in their native RecBole constructors, so no dummy optimizer
epochs are needed. SLIM retains positive coefficients and target-item hiding.
Its configurable iteration cap is 1,000 for every trial, compared with the
previous hard-coded 100. Convergence warnings are recorded; nonconverged fits
cannot win selection. All 25 SLIM fits completed with zero convergence warnings.
The old parameter baseline is refitted with that same solver budget to isolate
the effect of the parameter change.

## Validation results

| Model/configuration | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| EASE old and selected: reg_weight=250 | 0.186311 | 0.521718 | 0.297866 |
| SLIM old: alpha=0.2, l1_ratio=0.01 | 0.185749 | 0.525217 | 0.300154 |
| SLIM selected: alpha=0.2, l1_ratio=0.005 | 0.184538 | 0.526636 | 0.301125 |

EASE's existing regularization remains best by MRR@10. SLIM gains 0.001419
MRR@10 (0.27% relative) and 0.000971 NDCG@10 (0.32% relative), while Recall@10
decreases by 0.001211. These small validation gains do not establish a general
improvement. EASE reg_weight=500 has slightly higher NDCG@10 (0.298099), but
lower MRR@10 (0.521165), so it is not the selected configuration.

## Reproduction and handoff

From the repository root, using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m project.scripts.retune_ease_slim
.\.venv\Scripts\python.exe -m unittest project.tests.test_retune_ease_slim project.tests.test_accuracy
```

Completed trials resume only when source/config/data identity, checkpoint hashes
and dense-score hashes match. The default command creates local results under
`project/artifacts/individual-retuning/results` and fitted models under
`project/artifacts/individual-retuning/models`; it does not attempt to resume
the committed historical JSON records, whose large checkpoints are not in Git.
Use new `--output` and `--artifacts` directories when changing inputs.
The original committed result JSON includes all 34 trials, the selected parameters,
and the old configurations evaluated on the same split:
[retuning results](../../hyperparameters/results/individual-retuned-60-20-20/).

Selected parameters are also applied to `recbole/config/EASE/ml-100k.yaml` and
`recbole/config/SLIMElastic/ml-100k.yaml`, including seed 42 and the corrected
60/20/20 split. Lightweight project model YAMLs contain the same parameters.
Historical 80/10/10 result files are preserved. Large checkpoints, complete
validation score tables and split exports remain local under
`project/artifacts/individual-retuned-60-20-20/` and can be regenerated.
Selected per-user metrics are regenerated under each model's result directory
and excluded from Git; aggregate metrics are present in the committed JSON.

Six focused retuning tests, ten independent accuracy tests and five artifact tests pass. A post-run
check verified all 34 checkpoint/score hashes, unchanged input hashes, selected
checkpoint scores, exported candidate masks, exact per-user metrics and matching
parameters in both config formats. The EASE baseline reproduces the original
cached scores exactly. A resume check reused all 34 completed fits.

Existing hybrid caches still refer to their original configurations. Their
manifests are not rewritten to imply that these new parameters were used.

## PR review and fresh reproduction

The review found and fixed a reproduction issue: the default runner previously
tried to resume the committed result JSON even though its checkpoints were
excluded from Git. Defaults now create local run directories, and JSON evidence
uses stable LF line endings. The mathematical fitting and selection are unchanged.

After the fix, all 34 configurations were fitted again from scratch. Every
trial reproduced its original validation metrics exactly. The independent audit
in `project/scripts/verify_ease_slim_retuning.py` checked both sets of 34 fits:

- Each user's validation/test count is floor(0.2 times their history length),
  with rounding remainder assigned to training: totals 60,734 / 19,633 / 19,633.
- All train/validation/test user-item pairs are disjoint.
- Every saved model's interaction matrix equals the training partition exactly.
- Checkpoint parameters, zero coefficient diagonals, SLIM constraints, score
  arrays, external-ID ordering and training-history masks are consistent.
- Independently computed validation MRR@10, Recall@10 and NDCG@10 match all
  trials; selected configurations maximize validation MRR@10.
- Both individual YAMLs alone and both project model YAMLs with the shared
  protocol reproduce the same user/item split membership.

The original run's script hash was checked against commit `c27ab257`, preserving
its provenance after the CLI-only fix. No test predictions or test metrics were
computed in this review. Compact evidence is committed as
`hyperparameters/results/individual-retuned-60-20-20/review-verification.json`.
The audit can be repeated when the original and freshly reproduced local
checkpoints are available:

```powershell
.\.venv\Scripts\python.exe -m project.scripts.verify_ease_slim_retuning
```
