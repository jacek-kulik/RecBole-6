# Task 1.3: out-of-fold weighted hybrid

This branch depends on Gabriel's Task 1.1 PR (#14). It reuses its six model
YAMLs, RecBole model/trainer factories, full-candidate exporter, external-ID
ScoreTable contract, and score-file format. Tasks 1.4 and final test evaluation
are outside this command. A small ridge grid supplies the Task 1.3 validation
selection; it does not claim completion of all Task 1.5 experiments.

## Run on the main PC

From the repository root, with the existing project environment:

```powershell
.venv/Scripts/python.exe -m project.recsys.oof_hybrid --output project/artifacts/hybrid-1.3
```

The default run uses all six components, five folds, seed 42, and ridge alpha
values 0.1, 1, 10, 100. New runs require a new output directory. Use `--resume`
to reuse completed stages. An interrupted stage restarts its training; completed
folds and full-training validation exports are reused. Keep the machine awake.

## Agreed split and staged FISM handoff

Bogdan confirmed the team's 60/20/20 split on October 15. The shared protocol
now uses that split. Existing component hyperparameters tuned under 80/10/10
are intentionally carried over for now; this assumption is recorded in each
run manifest. No 80/10/10 prediction cache can be reused.

Prepare the five unaffected models without fitting any regressor:

```powershell
.venv/Scripts/python.exe -m project.recsys.oof_hybrid --output project/artifacts/hybrid-60-20-20 --components itemknn userknn bpr slimelastic ease --prepare-only
```

Repeat that command with `--resume` after an interruption. Once FISM is fixed
and its selected parameters are updated in the summary, add FISM and fit:

```powershell
.venv/Scripts/python.exe -m project.recsys.oof_hybrid --output project/artifacts/hybrid-60-20-20 --resume
```

The default six-component command reuses the other five caches and trains only
FISM if their identities still match. `--tuned-summary` can point to a separate
updated CSV. Component caches include the effective settings, split hash, fold
assignment, and relevant source hashes. Changing FISM's source or settings does
not invalidate other component caches. Changing common training code does.
Fold payloads and validation scores have atomic completion markers and SHA256
checksums. Incompatible versions get separate component cache directories;
they never silently overwrite old evidence. `cache-<model>.json` identifies
the currently used version. `run.lock` prevents concurrent writers; after a
hard interruption, verify its recorded PID has stopped before removing it.
The aggregate `oof-progress.npz` is diagnostic; individual verified caches are
the resume authority. Alpha and NDCG cutoff changes require only recombination.

The tuned component parameters come from
`hyperparameters/results/best-run-summary.csv` (override with `--tuned-summary`).
The existing summary generator selects configurations by validation MRR@10;
its displayed test metrics are not read by this runner. Hyperparameters stay
fixed during cross-fitting. This is not nested component hyperparameter tuning.
Check that the selected configurations are approved for the shared protocol.

A short integration check (not report evidence):

```powershell
.venv/Scripts/python.exe -m project.recsys.oof_hybrid --output project/artifacts/hybrid-smoke --components itemknn bpr --folds 2 --alphas 1 --smoke-epochs 1
```

## Statistical design

1. Reproduce the shared outer train/validation/test split and verify identical
   external user-item pairs across components. Reject overlapping pairs.
2. Partition only outer-training interactions into user-stratified folds.
   Repeated pairs stay together; each held-out user retains training history.
   Catalogue IDs remain known across folds (transductive evaluation). Rare items
   may have no fold-training history; no interaction targets are imputed.
3. Train each component on the other folds. Use the configured fixed epochs
   without validation/early stopping, so the fold labels cannot select a
   checkpoint. Model construction and training see only fold-training history.
4. Score all eligible items for each held-out user, excluding training history
   and padding. Standardize each component across that user's candidate scores.
   This transform uses predictions, not target ratings; constant columns map to
   zero. Use the identical rule during full-training validation inference.
5. Collect one feature row per outer-training interaction from its held-out
   fold. Fit one Ridge regressor to raw 1–5 ratings with intercept and
   unconstrained coefficients. Targets are not binary interaction labels.
6. Refit each component on all outer-training interactions, again at its fixed
   budget. Apply the fitted combiner to every eligible validation candidate.
   Select ridge alpha by binary NDCG@10 (all validation interactions relevant).
   This selection metric is the design agreed for this hybrid, while the
   existing component tuning uses MRR@10. Confirm this distinction with the team.

The final test partition is exported for split provenance but never trained on,
scored, or evaluated. Validation is selection evidence, not an unbiased final
performance estimate. Component selection may already use this same validation
partition; the untouched test is required for final reporting. Fold-trained and
full-trained scores can differ despite candidate normalization; inspect this
distribution shift rather than assuming cross-fitting removes every source of
error. Small nonzero score variation can also be amplified by standardization.

## Outputs and interpretation

`manifest.json` records status, parameters, split ID, and smoke-only status.
`folds.json` records every pair's fold; `splits/` stores actual outer splits.
`oof-progress.npz` contains OOF features, raw targets, and fold assignments.
`config-*.json` records effective RecBole configs; model directories contain
checkpoints for each fold and final fit. `hybrid.json` holds coefficient order,
transform definition, intercept, alpha, and target definition.
`validation.json` compares every alpha and individual component NDCG on the same
candidates. Compressed full-candidate score tables support downstream reranking.

Coefficient magnitude is not causal model importance: components are correlated,
and candidate-relative scaling affects interpretation. Rating MSE is a surrogate
for ranking, not a guarantee of NDCG improvement. The inherited FISM sigmoid /
BCEWithLogitsLoss concern from PR #14 is unchanged and must be resolved before
interpreting its full experiment. This implementation does not silently patch
individual models.
