> Historical experiment notes. Corrected fits and current comparisons are documented in [task-1.4-corrected-hybrid-comparison.md](reports/task-1.4-corrected-hybrid-comparison.md).

# UserKNN feature augmentation with observed targets

The experiment tests whether inferred preferences help a linear item model
predict recorded interactions. It keeps binary training interactions as targets,
including zeros for missing interactions. Only the input matrix is augmented.
All comparisons use the existing seed-42, per-user 60/20/20 split, full validation
candidates and the independent accuracy metrics. Validation runs do not evaluate
test data; a separate frozen-model test report is described below.

UserKNN's verified cached full-training scores supply predictions for unseen
items. For each user, select the 50 largest positive scores, resolving ties by
external item token. Divide these by that user's maximum unseen score. Add
the resulting strengths to previously zero entries; observed inputs remain 1.
This is a starting configuration, not a selected optimum. The sender uses 50
neighbors and zero shrinkage. Its own-user similarity is zero in RecBole.

Variants supported by `project.recsys.feature_augmentation`:

- Weak: `--strength 0.25`; added entries are at most 0.25.
- Comparable: `--strength 1`; added entries are normalized scores up to 1.
- Equal: `--strength 1 --equal`; every selected added entry is 1.

The same top-50 entries are selected for all three. No prediction is used as a
target. Predicted features can indirectly contain their target's training
interaction; held-out validation measures whether this helps generalization.

## Receivers

EASE solves `min ||R-XB||^2 + lambda*||B||^2`, with zero diagonal. Since input
and targets differ, its usual inverse formula is replaced by the constrained
Ridge optimum. The unaugmented solution reproduces cached EASE scores within
7.1e-7 absolute error at lambda 250.

SLIM solves a separate nonnegative elastic-net regression for each column of
the unchanged target R, hiding that column of X. It retains alpha 0.2 and
l1_ratio 0.01. Both fresh baseline and augmented fits use identical solver
settings: random coordinate selection, seed 42, tolerance 1e-4 and at most
1000 iterations. Unconverged columns cause a failed run. The original cache
used 100 iterations and a global RNG, so the new baseline is refitted for a
controlled comparison; it differs slightly in NDCG from the cached baseline.

## First runs, 19 October 2026

| Receiver | Input | NDCG@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| EASE | Observed only | 0.297866 | 0.186311 | 0.521718 |
| EASE | Weak augmentation | 0.298627 | 0.187710 | 0.518341 |
| SLIM | Observed only, refitted | 0.300200 | 0.185749 | 0.525217 |
| SLIM | Weak augmentation | 0.302813 | 0.188958 | 0.523908 |
| SLIM | Comparable augmentation | 0.303557 | 0.188287 | 0.520299 |

Weak SLIM augmentation improves NDCG by about 0.87% and Recall by 1.73%,
while MRR decreases about 0.25%. These are single-split exploratory results;
they do not establish statistical significance. Each user receives 50 added
entries with mean total added mass 7.44, compared with 64.41 observed entries.
Comparable augmentation improves NDCG by 1.12% over the refitted baseline,
and by 0.25% over weak augmentation. Recall is slightly lower than weak
augmentation, and MRR falls further. Thus stronger input weights help NDCG
on this split, with a metric trade-off. Mean added mass rises to 29.76 per
user. Verification confirmed identical input hashes, token order, selected
entries and baseline metrics; every comparable added weight is exactly four
times its weak counterpart. All SLIM columns converged. The equal variant
has not been run yet.

Reproduction from the repository root (use a new output directory):

```powershell
.\.venv\Scripts\python.exe -m unittest project.tests.test_feature_augmentation -v
.\.venv\Scripts\python.exe -m project.recsys.feature_augmentation --receiver slimelastic --strength 0.25 --top-n 50 --output project/artifacts/augmentation-userknn-slim-weak-025
.\.venv\Scripts\python.exe -m project.recsys.feature_augmentation --receiver slimelastic --strength 1 --top-n 50 --output project/artifacts/augmentation-userknn-slim-comparable-100
```

The run directory contains full candidate scores, per-user and aggregate
metrics, fitted coefficients, the exact added-entry matrix, token order and
a manifest with settings and input/output hashes. Source caches are read-only.
The implementation lives entirely under `project/`; framework files are unchanged.

## Frozen-model test evaluation

At the user's request, all three completed augmentation runs and their baselines
are evaluated on test without retraining or changing parameters. Fixed scores
are masked to exclude training and validation history, following the existing
project test protocol. Augmented input profiles are unchanged: validation
interactions are not added as observed features. The equal-entry-weight variant
has not been fitted and is excluded from this report.

```powershell
.\.venv\Scripts\python.exe -m project.recsys.feature_augmentation_report --runs project/artifacts/augmentation-userknn-ease-weak-025 project/artifacts/augmentation-userknn-slim-weak-025 project/artifacts/augmentation-userknn-slim-comparable-100 --output project/artifacts/augmentation-test-2026-10-19
```

The separate output contains `comparison.md`, `comparison.csv`, full test
score tables, all six per-user and aggregate accuracy metrics, and a manifest
with frozen source manifests and artifact hashes. Original validation artifacts
are preserved. Test results describe these fixed configurations; no test-based
configuration selection is performed.
