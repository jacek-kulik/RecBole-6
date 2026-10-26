# Task 1.3 experiment audit

Status: the FISM correction and required rerun are complete. Final evidence is
in [task-1.3-final-results.md](task-1.3-final-results.md).

PR #14 corrected target self-exclusion and normalization, with regression tests
for scores and gradients. The corrected model was retuned over all 48 settings,
its five OOF folds and full-training scores regenerated, and Ridge refitted on
60,734 raw-rating targets. Validation selection over 24 penalties chose alpha
30000 before test scoring. The five unaffected columns and per-user metrics
match the original run exactly. Test NDCG@10 is 0.372549 for FISM and 0.409095
for the hybrid. The historical audit below records why earlier results were
provisional; it does not describe a defect remaining in the final run.

## Explicit course requirements

- Project brief, page 5: Task 1.3 builds a weighted hybrid and learns individual recommender coefficients using a regression model. It does not prescribe OOF, Ridge, normalization, or a regression target.
- Project brief, pages 4 and 6: use RecBole on MovieLens-100K, focus on ranking, evaluate independently, and compare against popularity and random baselines for Task 2.2.
- Hybrid lecture, page 6: combine component scores linearly with weights summing to one; weights may be learned and user-specific. No explicit nonnegativity constraint is printed on that slide.
- Neighborhood-model lecture, page 49: low-rank item-item FISM has a zero-diagonal constraint.

## Historical FISM implementation defect (resolved)

`recbole/model/general_recommender/fism.py`, `inter_forward`, aggregates the complete training history without removing the candidate item. For positive training interactions, that item therefore contributes its own source embedding to its prediction. This violates the intended exclusion of self-contributions. The sigmoid/loss fix did not address this additional inherited defect.

An isolated diagnostic used history [1, 2] and target item 2, zero biases and all-zero embeddings except item 2's source and destination embeddings, both [1, 0]. The current score was 1.0; zeroing only the target's source embedding reduced it to 0.0. All of the original score therefore came from its self-contribution. No saved model parameters were modified.

The existing regression tests checked logits, probabilities, and algebraic equivalence, but did not test candidate items already present in training history. Passing those tests did not establish this model constraint.

## Weighted-regression implementation

The core Task 1.3 regression requirement is implemented. Five-fold OOF component features, per-user candidate z-score normalization, Ridge with an intercept, and 1-5 rating targets were our design choices, not explicit lecture requirements. The loader's target normalization was corrected and verified as an affine transformation preserving rankings.

The historical alpha-100 fit had a negative UserKNN coefficient and coefficient sum 0.2011817347235848. The final alpha-30000 fit has positive coefficients summing to approximately 0.171301256. The sum-one formula is not literally enforced during regression. Dividing the final coefficients by their positive sum gives a sum-one parameterization with identical rankings; the global intercept also has no ranking effect.

## Experiment limits

- Five components retain the previously approved 80/10/10-tuned hyperparameters; their actual cached predictions and comparisons use the agreed common 60/20/20 split.
- The initial Ridge grid (0.1, 1, 10, 100) selected its upper boundary. A subsequent 21-penalty validation-only search from 100 through 1e12 selected 100000; this resolves that boundary concern, without establishing exhaustive tuning.
- The seven-model table does not complete Tasks 2.1-2.6: random/popularity baselines, beyond-accuracy measures, group analyses, and evidence-based coefficient analysis remain outside this comparison.
- A rating regression target is allowed by the brief's unspecified target, but alignment with ranking evaluation must be motivated and empirically tested.

## Required correction (completed)

Correct FISM target self-exclusion, including its history-length normalization; test present and absent candidates; retune FISM; regenerate only FISM's OOF folds and full-training scores; then refit/select the regressor on validation. Keep the unaffected five verified caches. The current six-model hybrid result does not establish failure of weighted hybrids generally.

## Primary sources inspected

- project/Project-RecSys.pdf, pages 4-8 and 15-16.
- project/RecBole Instruction.pdf, pages 2-4.
- C:/Users/radud/Downloads/W4S2-Hybrid (1).pdf, page 6.
- C:/Users/radud/Downloads/W2S1-CF Neighborhood Models.pdf, page 49.
