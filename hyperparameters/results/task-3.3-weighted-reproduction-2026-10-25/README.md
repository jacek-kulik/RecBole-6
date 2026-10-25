# Full weighted-hybrid reconstruction and audit

Completed an independent rebuild of the weighted experiment from PR #27. The original Windows run and OOF caches were unavailable, so all six components were fitted again using the recorded source and selected settings. No original cached predictions or weights were substituted into this reconstruction.

## Work performed

- Reconstructed the frozen Task 1.3 source and dataset identities from Git history.
- Rebuilt all 30 out-of-fold fits and six full-training fits, using the recorded component hyperparameters and fixed training budgets.
- Matched all train/validation/test memberships: 60,734 / 19,633 / 19,633 interactions. The five-fold map matches the recorded SHA256 after Windows line-ending normalization.
- Recovered raw 1-5 rating targets by external user/item IDs and refitted Ridge. The 24-penalty validation check independently selected the original alpha of 30,000.
- Completed 30 validation reranking points plus the original baseline, froze all six choices, then completed the six selected test pipelines plus their baseline.
- Audited all 38 evaluations and 35,834 user recommendation lists, including the input/source hashes, masks, full held-out relevance, pool membership, aggregate/per-user metrics, CSV/JSON consistency, report aliases, independent target calculations and zero-strength controls.

Audit status: **passed**. Maximum metric recomputation error: `1.67e-15`. All six reranker strengths match the original selections. The 48-setting FISM tuning search was not repeated; the original selected FISM configuration was frozen for this reproduction.

## Test results

| Pipeline | Reported NDCG@10 | Rebuilt NDCG@10 | Rebuilt NDCG loss | Rebuilt target reduction |
|---|---:|---:|---:|---:|
| Original weighted hybrid | 0.409095 | 0.409041 | 0.00% | - |
| genre-pre | 0.386796 | 0.386784 | 5.44% | 27.41% |
| genre-post | 0.379720 | 0.379901 | 7.12% | 37.17% |
| popularity-pre | 0.395378 | 0.395515 | 3.31% | 31.65% |
| popularity-post | 0.391661 | 0.391544 | 4.28% | 37.85% |
| exposure-pre | 0.399308 | 0.399004 | 2.45% | 3.96% |
| exposure-post | 0.388479 | 0.386825 | 5.43% | 5.87% |

Post-reranking reduces each designated target more and incurs a larger NDCG loss than pre-reranking. Both genre orders and post-exposure exceed 5% accuracy loss on test. These are the same qualitative conclusions as the original report. All validation choices remain frozen; test outcomes were not used to revise them.

## Reproduction limits

The reconstruction is close but not numerically identical to the original Windows run. Baseline test NDCG differs by -0.000053813. The largest selected-test NDCG discrepancy is post-exposure: -0.001653490. ItemKNN and UserKNN validation NDCG differ slightly; BPR, SLIMElastic, EASE and FISM validation NDCG match the recorded aggregate values exactly. The KNN implementation uses argpartition without an explicit tie-breaking policy for choosing neighbors; this is a possible contributor to cross-environment differences, not a proven explanation.

Current runtime: Python 3.11.16, NumPy 2.4.6, Torch 2.13.0+rocm7.2; CPU, two numerical threads. The original final-correction record reports Torch 2.5.1+cpu. This audit certifies the new complete run; it does not establish byte identity with the unavailable original recommendation lists.

## Committed evidence

- `metrics.csv`: all 38 evaluations and their accuracy/beyond-accuracy metrics.
- `selection.json`: six validation choices frozen before test.
- `hybrid.json`: coefficients fitted to the freshly rebuilt OOF predictions.
- `verification.json`: the hardened verifier's outcome for all saved lists.
- `split-and-fold-verification.json`: original split and fold identity checks.
- `component-validation-comparison.json`: per-component validation discrepancies.
- `provenance.json`: source identities, runtime, checks and compact-file hashes.

The full local run is `project/artifacts/weighted-audit-reproduction-20261025/` in the main checkout. It retains all fold payloads, checkpoints, full component scores, canonical evaluation folders, source snapshots, commands and the orchestration script. `verification.json.report_path` is relative to that run's `audit-workspace/` directory. This compact evidence directory does not contain those large inputs and is not a portable training bundle.

The audit fixes were committed as `e4f45cb7f`. The original reported results remain separate in `hyperparameters/results/task-3.3-reranking-orders/`.
