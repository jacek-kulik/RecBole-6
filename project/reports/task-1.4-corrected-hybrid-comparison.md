# Alternate hybrid comparison: corrected inherited fits

Per-user random 60/20/20, seed 42; 60,734 training, 19,633 validation and 19,633 test interactions. 943 users and 1,682 catalogue items. All held-out interactions are relevant. Validation candidates exclude training history; test candidates exclude training and validation history. External item-ID ties; NDCG@10 selects settings/checkpoints using validation only.

The cascade retains its independently fitted joint grid and selected ItemKNN k=200/shrink=10 plus BPR 128 dimensions/lr=0.0005/50 epochs, shortlist 10. Standalone ItemKNN instead selects shrink=100. Individual settings are not substituted into a jointly tuned hybrid. Switching, augmentation, feature combination and meta-level originally inherited older component fits; these dependencies were replaced by BPR 128 dimensions and/or UserKNN k=160 and retrained/reselected downstream. Augmentation receivers use EASE reg=500 or SLIM alpha=0.2/l1_ratio=0.005; their original receiver settings were inherited rather than jointly searched.

| Model | Scope | Validation NDCG@10 | Test NDCG@10 |
|---|---|---:|---:|
| BPR | full | 0.274439 | 0.373665 |
| ItemKNN | full | 0.275934 | 0.363588 |
| UserKNN | full | 0.283854 | 0.368763 |
| EASE | full | 0.298099 | 0.404872 |
| SLIM native | full | 0.301125 | 0.403266 |
| SLIM paired receiver | full | 0.301112 | 0.403254 |
| UserKNN warm control | warm | 0.283897 | 0.368763 |
| Switching | warm | 0.284915 | 0.371960 |
| Cascade | full | 0.278699 | 0.384593 |
| Feature combination, CF | full | 0.263231 +/- 0.006986 | 0.345623 +/- 0.009039 |
| Feature combination, metadata control | full | 0.262957 +/- 0.001532 | 0.346077 +/- 0.003032 |
| Residual combination, CF | full | 0.275614 +/- 0.001106 | 0.373233 +/- 0.002061 |
| Residual combination, metadata control | full | 0.276642 +/- 0.001818 | 0.373613 +/- 0.001870 |
| UserKNN -> EASE, strength 0.25 | full | 0.298156 | 0.403747 |
| UserKNN -> SLIM, strength 0.25 | full | 0.304670 | 0.411904 |
| UserKNN -> SLIM, strength 1.0 | full | 0.303055 | 0.407374 |
| BPR user vectors -> UserKNN | full | 0.282216 | 0.375364 |
| Interaction-cosine control | full | 0.283823 | 0.369014 |

Switching uses the original warm-item scope: 1,615 supported items and 19,590 validation positives. Compare it to the warm control, not the full-catalogue number. The same training-supported item mask is applied to test candidates and relevance, retaining 19,588 of 19,633 test positives and all 943 users. Other rows use the identical full-catalogue candidates.

SLIM augmentation at strength 0.25 improves test NDCG by 2.15% against its paired refit; strength 1.0 improves it by 1.02%. The native SLIM and custom paired receiver differ by about 0.000012 test NDCG because their numerical implementations differ. EASE augmentation is now slightly worse on test (-0.28%). Cascade improves over its BPR control (+2.92%) but stays below EASE and SLIM. Plain feature combination remains below BPR. Residual corrections produce small validation gains; both residual test means remain slightly below frozen BPR. Switching exceeds its warm UserKNN control on test. The meta-level hybrid exceeds its interaction-cosine control and full-catalogue UserKNN on test, despite being below both on validation. These additional test results do not change the validation-selected fits or routing.

MLP rows report mean +/- sample standard deviation over three scorer seeds, conditional on one frozen BPR fit. Width and checkpoint selection were repeated after correcting that fit. Metadata variants are retained negative controls from the earlier feature-combination experiment, not a proposed content-based solution. The final five styles are switching, cascade, feature combination, feature augmentation and meta-level.

These are single-split development comparisons. The test set was already inspected during earlier iterations; re-evaluation is a protocol correction, not a fresh independent holdout. No test metrics were used to select these reruns. The six previously validation-only report rows now include supplemental test evaluations using the frozen fits, routing and original validation scopes; no retraining or selection on test was performed. Residual test means and sample standard deviations use all three previously selected scorer seeds.

## Reproduction

Run `python -m project.scripts.evaluate_alternate_bundle` from the repository root. Committed NumPy/sparse trained weights, mappings, masks and labels reproduce the reported metrics without original artifact directories or machine-specific checkpoint pickles. `reference-metrics.json` checks replay against frozen training evaluations; `replay-verification.json` records hashes and exact agreement. The package includes selected fits, all search trials, MLP epoch histories, scaler values, routing decisions and hyperparameters. Canonical evaluation folders are under `hyperparameters/results/alternate-hybrids-60-20-20/evaluations/<partition>/<scope>/<model>/`. Each contains the four Task 2.2 inputs: `manifest.json`, flat `metrics.json`, `per_user_metrics.json`, and `beyond_accuracy.json`. Beyond-accuracy results use the shared evaluator with the shared genre/gender metadata loader. Each selected MLP seed has its own folder; seed means/std and `metrics.csv` are generated summaries, not additional fitted models. Historical style notes describe the earlier fits; this report and the committed manifests describe the corrected comparison.

With Task 2.2 available, pass the individual test folders directly to `python -m project.recsys compare`. For example, compare `evaluations/test/warm/switch` with `evaluations/test/warm/userknn` (paths relative to the results directory above). Compare full-catalogue models using `evaluations/test/full/<model>` folders. Warm/full scopes must stay in separate comparisons; validation folders are for inspection and cannot be passed as test evaluations.

To repeat fitting, run `project.scripts.prepare_alternate_source` to reconstruct the inherited component cache from committed baseline weights, then invoke each style module with `--source project/artifacts/alternate-corrected-source` and a new output directory. Use `project.scripts.verify_feature_fits` before downstream residual fitting or MLP test evaluation. The cascade independently fits its own grid via `project.recsys.cascade`.
