# Task 2.3 experiment record

All predefined experiments completed: 13 weighted variants (original, six frozen
removals, six fixed-penalty combiner refits) and six augmentation configurations
(two receivers times three strengths), each evaluated on validation and test.
There was no new hyperparameter selection. `weighted.csv` and `augmentation.csv`
contain all six accuracy metrics, paired changes, top-10 overlap, user win/loss
counts and descriptive paired-bootstrap intervals. `weighted-fits.json` contains
the original and all twelve ablated coefficient vectors and intercepts.

## Controlled comparisons

- Same MovieLens-100K per-user 60/20/20 seed-42 split throughout: 943 users,
  1,682 items, 60,734 training and 19,633 validation/test interactions each.
  All held-out interactions count as relevant. Validation masks training;
  test masks training plus validation. No profiles are extended with validation.
  Ties use lexicographic external item IDs.
- Weighted source is the corrected FISM/OOF Ridge fit in
  `project/artifacts/hybrid-60-20-20-fism-self-excluded/final`. Its six components
  are ItemKNN, UserKNN, BPR, SLIMElastic, EASE and FISM, in that order. Five original
  component settings retain their previously approved transfer; FISM uses the
  corrected, retuned fit. The newer standalone component settings are not
  substituted into this existing hybrid.
- Per-user component scores are z-scored across eligible candidates, recomputed
  independently for validation and test. Constant columns map to zero. The
  intercept is global and does not change rankings. Coefficients are signed
  regression coefficients, not percentages or causal importance measures.
- Frozen removal zeros one coefficient, retaining all other coefficients,
  intercept and transforms. Refit removal deletes that OOF feature column and
  fits the other five coefficients plus intercept on the same 60,734 raw rating
  targets, retaining Ridge alpha=30,000. No base recommender is retrained and no
  Ridge penalty is reselected. These are conditional ablations of the existing fit.
- Augmentation uses corrected UserKNN k=160, shrink=0. Its positive unseen top-50
  entries are selected once with external-token tie handling. Predictions are
  promoted to float64 before normalization by the per-user positive unseen
  maximum, exactly matching the original JSON score-table pipeline. All existing
  saved additions match exactly. Observed inputs remain 1; targets remain the
  unchanged binary training matrix (including zeros).
- Retrain each receiver at strengths 0, 0.25 and 1, holding its regularization
  fixed: EASE lambda=500; SLIM alpha=0.2, l1_ratio=0.005, nonnegative coefficients,
  hidden target-item input column, no intercept, random coordinate selection,
  random_state=42, tolerance=1e-4, max_iter=1,000. Any SLIM convergence warning
  fails the run. All columns converged. There are no equal-valued additions.
- Compare augmentation to paired refits at strength zero, not a mismatched native
  SLIM implementation. The native float32 EASE and float64 paired EASE have a
  maximum eligible-score difference of 7.995e-7 but reproduce every metric exactly.
  Other existing augmentation fits replay to floating-point precision. EASE
  strength 1 is newly fitted; the other saved reference metrics replay exactly.

The two hybrid families use different historical component configurations.
Interpret changes within each family; an absolute cross-family comparison does
not isolate hybrid architecture.

## Scope of inference

This is one existing split, whose test results were previously inspected. All
ablation/strength comparisons were specified before this analysis ran, and all
are retained irrespective of outcome. No removal or strength is selected as a
new final model. Bootstrap intervals use 2,000 paired resamples of users, seed 42,
conditional on this split and these fitted models. They exclude training/split
variation and have no multiple-comparison adjustment. Treat them as descriptive.

## Provenance and verification

`protocol.json` records predefined settings; `manifest.json` records completion,
source and result hashes. `weighted-inputs.json` identifies the original caches,
splits and OOF inputs. All thirty OOF fold cache hashes, indices and feature
values were checked; raw rating targets and corrected FISM column were verified;
the original six-component Ridge fit was reconstructed exactly. The original
weighted validation/test metrics replay with zero error. Saved augmentation
metric replay errors are in `augmentation-replay.json`.

The first local attempt stopped at an augmentation precision identity check.
A subsequent run completed all weighted evaluations and stopped when a score
comparison included observed items that are never eligible for ranking. The
completed weighted stage was verified against source hashes, reconstructed
coefficients, saved predictions and every per-user comparison before resuming.
Only the augmentation stage was then fitted. These checks did not change any
experiment configuration or select among outcomes. The completed local run is
`project/artifacts/task-2.3-coefficient-analysis-2026-10-23-verified`.

Full per-user metrics, rankings, candidate masks, score matrices, augmentation
inputs and six trained receiver matrices are retained in that local run.
Numerical source snapshots preserve the exact scripts used before later report
formatting edits. The compact report discussion is below 200 words.

## Reproduce

From the repository root, with the original verified weighted caches present:

```powershell
.\.venv\Scripts\python.exe -m project.scripts.analyze_hybrid_coefficients `
  --output project/artifacts/task-2.3-reproduction `
  --results hyperparameters/results/task-2.3-reproduction `
  --report project/reports/task-2.3-reproduction.md
```

The numerical runner requires local original OOF caches; this compact results
directory alone is not a portable retraining bundle. The alternate-hybrid sender
and receiver reference weights are committed in the existing portable bundle.
Use fresh output paths to preserve all previous evidence. `--resume` verifies a
completed weighted stage before reusing it. `--report-only` only renders existing
CSVs and performs no fitting or evaluation.

```powershell
.\.venv\Scripts\python.exe -m unittest project.tests.test_coefficient_analysis project.tests.test_feature_augmentation project.tests.test_accuracy -v
```
