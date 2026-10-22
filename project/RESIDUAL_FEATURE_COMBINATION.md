> Historical experiment notes. Corrected fits and current comparisons are documented in [task-1.4-corrected-hybrid-comparison.md](reports/task-1.4-corrected-hybrid-comparison.md).

# Residual feature combination

The original MF scorer was stronger than replacing its dot product with a plain
MLP. This experiment instead preserves that exact scorer and learns a correction:

`score(u,i) = original_MF_score(u,i) + MLP([MF_user; MF_item; optional genres/year])`.

Compare MF alone, MF plus a collaborative-only correction, and MF plus a correction
that also receives 18 genre indicators and standardized year. The MF score uses
the original unstandardized dot product. MLP inputs reuse the exact frozen,
training-standardized features from the previous validated experiment. No explicit
content user profile or content recommender score is included.

The correction has two ReLU hidden layers of widths h and h/2. Initialize its final
linear layer to zero, making the initial scorer exactly MF. Include this epoch-zero
checkpoint in selection. Use h in {64,128} and seeds {42,43,44}, one shared frozen
MF checkpoint, 40 epochs, Adam lr=0.001, weight decay=0.0001, batch size 2048, and
pairwise BPR softplus loss. One uniform negative per training positive per epoch
excludes training positives only. Order and negatives are paired across variants.

Evaluate validation every epoch, including zero, instead of the previous
experiment's five-epoch interval. Select the highest validation NDCG@10 checkpoint
per trial, retaining the earliest on ties. Select width by mean validation NDCG
across seeds, preferring the smaller width on ties. Epoch zero ensures the selected
validation NDCG cannot decrease from MF; it does not guarantee gains in other
metrics or held-out results.

Keep the existing 60/20/20 split, training interactions, candidate mask, relevance
definition, and external-token tie policy. No test labels or metrics are loaded by
this runner. The original test results have already been viewed earlier in this
research sequence, so a subsequent evaluation on that same set would not be a fresh
independent holdout for architecture development.

```powershell
.venv/Scripts/python.exe -m project.recsys.residual_combination --output project/artifacts/residual-feature-combination-new
```

Use a new output directory. The completed run is saved under
`project/artifacts/residual-feature-combination/`, with `validation.md`,
`results.json`, `manifest.json`, per-trial histories, selected checkpoints, full
dense candidate scores, per-user metrics, common eligibility masks, and the reused
external-ID mapping and feature scalers. Source and input hashes record provenance.

Six focused scorer tests cover frozen buffers, negative sampling, training-only
factor scaling, masking/ties, exact MF initialization, nonzero correction scoring,
gradient flow, checkpoint reload, and pairwise/full-catalogue scoring equivalence.

## Validation results, 18 October 2026

MLP values are means over three seeds; NDCG uncertainty is sample standard deviation.

| Model | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| Original MF | 0.167718 | 0.485968 | 0.269292 |
| MF + collaborative MLP correction | 0.167749 | 0.487160 | 0.270213 +/- 0.001595 |
| MF + genres/year MLP correction | 0.167585 | 0.487913 | 0.270204 +/- 0.001519 |

Both correction families select width 128. Selected epochs for seeds 42,43,44 are
0,0,3 for the collaborative correction and 0,1,3 for the content correction. Epoch
zero is unchanged MF. Relative mean NDCG gains over MF are 0.342% and 0.339%,
respectively. The content correction's mean NDCG is 0.00000888 below the
collaborative correction, providing no evidence of an incremental content benefit
in this experiment. Content also slightly lowers mean recall despite raising MRR.

The residual architecture avoids the large validation degradation observed when
replacing MF with an MLP. Its observed gains over MF are small and selected on
validation, with some seeds choosing no correction at all. Later epochs generally
reduce validation quality. These results do not support a claim that additional
training alone would improve this architecture's generalization.

`verification.json` records independent ranking/metric checks for all 12 trials,
2,400 direct checkpoint pair predictions checked against the saved dense scores,
the selected epoch checked against each complete 41-point history, and source/input
hashes. Six focused scorer tests passed. Test was not evaluated for these models.
