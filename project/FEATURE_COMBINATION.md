> Historical experiment notes. Corrected fits and current comparisons are documented in [task-1.4-corrected-hybrid-comparison.md](reports/task-1.4-corrected-hybrid-comparison.md).

# MF and metadata feature combination

## Experiment, 18 October 2026

Compare three scorers on the existing per-user random 60/20/20 split (seed 42):

1. Original BPR-MF dot product, reloaded from the existing 20-epoch checkpoint.
2. MLP over concatenated frozen MF user and item embeddings.
3. The same MLP family with 18 raw multi-hot genre indicators and one standardized
   release-year feature appended to the item embedding. No explicit content user
   profile and no content recommender score.

User/item MF embeddings have 64 dimensions. Standardize each latent dimension
using training users/items only, identically for both MLP variants. Fit release-year
mean and standard deviation on distinct training items with known years; impute
missing/malformed years with that mean. Genres remain binary; omit `unknown`.

Both MLP variants use `concat -> Linear(h) -> ReLU -> Linear(h/2) -> ReLU -> Linear(1)`.
Search h in {64, 128}; run seeds 42, 43, 44 for each width and variant (12 trials).
Use Adam, learning rate 0.001, weight decay 0.0001, batch size 2048, and 40 epochs.
Optimize BPR pairwise softplus loss with one resampled uniform-catalogue negative
per positive per epoch, excluding training positives only. Training order and
negative draws are paired across variants. Neither validation nor test labels
enter negative sampling, feature scaling, or gradient updates.

Evaluate every five epochs. Retain each trial's highest validation NDCG@10
checkpoint, then select width by mean NDCG@10 over its three seeds. Both variants
selected h=64. The MF baseline is one shared checkpoint; the three seeds quantify
MLP variability conditional on those fixed embeddings, not MF or split variability.

Score exactly the original full-catalogue validation candidates for all 943 users
and 1,682 movies, masking training history. Treat every validation interaction as
relevant and break ties by external item-ID token. The checkpoint's MF scores match
the existing BPR score cache exactly. Test split files were not opened or evaluated.
RecBole's original dataset loader reconstructs checkpoint token mappings from the
raw catalogue/interactions; its interaction values are not used to fit these models.

## Validation results

MLP values are mean across three seeds; NDCG uncertainty is sample standard deviation.

| Model | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| Original BPR-MF | 0.167718 | 0.485968 | 0.269292 |
| MF embeddings + MLP | 0.158859 | 0.472680 | 0.258721 +/- 0.003309 |
| MF embeddings + genres/year + MLP | 0.161645 | 0.472214 | 0.262734 +/- 0.006189 |

Original MF is best on all three metrics. Content raises mean NDCG by 0.004013
(1.55% relative) and mean recall by 0.002786 over the CF-only MLP, but mean MRR
decreases slightly. Paired content-minus-MLP NDCG differences are +0.006913,
+0.011629, and -0.006504 for seeds 42, 43, and 44 respectively. The metadata effect
is mixed across seeds. Its mean NDCG remains 2.44% below original MF.

This experiment does not support replacing the MF dot product with these MLP
scorers. It also does not establish that metadata cannot help: it tests one
architecture family, a small width search, and frozen MF representations.
Checkpoints and widths were selected on validation, so these are development
results, not an independent final estimate or a significance claim. Comparing the
MLP against MF changes model capacity and training stage as well as nonlinearity.

## Reproduction and evidence

```powershell
.venv/Scripts/python.exe -m project.recsys.feature_combination --output project/artifacts/feature-combination-mf-content-new
```

The output must be a new directory. Existing component caches remain unchanged.
The completed run is in `project/artifacts/feature-combination-mf-content/`:

- `validation.md`, `results.json`, `trials.json`: aggregate results and selection.
- `manifest.json`, `scalers.json`, `mapping.json`: settings, input/source hashes,
  feature transformations, and external user/item ordering.
- `candidate-mask.npz`: common eligible candidates; apply this mask to dense scores.
- Each trial directory: selected checkpoint, full dense validation score matrix,
  per-user metrics, and evaluation history.
- `verification.json`: all 12 saved score matrices independently reranked with
  external-ID ties and metrics recomputed exactly; 200 direct checkpoint pair
  predictions per trial checked against stored dense scores (2,400 total).

Twenty-one targeted feature-combination, content, and genre/year tests passed,
including frozen embeddings, optimized full-catalogue scoring equivalence,
training-only scaling, negative sampling, masking, and tie handling.

## Frozen test evaluation, 18 October 2026

After validation, the user authorized test evaluation. Evaluate the exact selected
widths and per-seed checkpoints above, with no retraining or test-based selection.
Mask both training and validation interactions, matching the existing component
test protocol. Verify test masks and MF predictions against the existing saved
BPR test reference. Every selected MLP checkpoint reproduces its saved predictions.

| Model | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| Original BPR-MF | 0.193918 | 0.576779 | 0.354497 |
| MF embeddings + MLP | 0.187021 | 0.557083 | 0.336847 +/- 0.003401 |
| MF embeddings + genres/year + MLP | 0.187413 | 0.565112 | 0.342649 +/- 0.000703 |

Content improves mean test NDCG by 1.72% over the CF-only MLP, with positive paired
NDCG differences in all three seeds. It remains 3.34% below original MF, which
leads on Recall, MRR, and NDCG. This supports a modest incremental content benefit
within this MLP design, without a gain over the original MF scorer. No significance
claim is made. These test observations must not select further settings on this
same holdout. Test and validation absolute values use different candidate masks
and relevance sets, so their magnitudes are not directly comparable.

Reproduce once into the new `test/` subdirectory of a completed validation run:

```powershell
.venv/Scripts/python.exe -m project.recsys.feature_combination_test --run project/artifacts/feature-combination-mf-content
```

The completed `test/` directory retains `comparison.md`, `results.json`, all
per-user metrics, the test eligibility mask, frozen selection hashes, and a
verification record. An independent lexicographic ranking pass reproduces all six
MLP test metric files; MF metrics exactly match the existing test reference.
The original validation artifacts and their manifest remain unchanged.
