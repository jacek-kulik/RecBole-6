# Task 1.3 experiment audit

Status: provisional results for FISM and the six-component regressor.

This audit originally examined the alpha-100 fit. The subsequent validation-only
21-penalty search selected alpha 100000, an interior grid value. Its frozen test
evaluation now leads the seven-model comparison; see
`task-1.3-ridge-100k-comparison.md`. The FISM defect and required correction below
remain unresolved.

## Explicit course requirements

- Project brief, page 5: Task 1.3 builds a weighted hybrid and learns individual recommender coefficients using a regression model. It does not prescribe OOF, Ridge, normalization, or a regression target.
- Project brief, pages 4 and 6: use RecBole on MovieLens-100K, focus on ranking, evaluate independently, and compare against popularity and random baselines for Task 2.2.
- Hybrid lecture, page 6: combine component scores linearly with weights summing to one; weights may be learned and user-specific. No explicit nonnegativity constraint is printed on that slide.
- Neighborhood-model lecture, page 49: low-rank item-item FISM has a zero-diagonal constraint.

## Confirmed FISM implementation defect

`recbole/model/general_recommender/fism.py`, `inter_forward`, aggregates the complete training history without removing the candidate item. For positive training interactions, that item therefore contributes its own source embedding to its prediction. This violates the intended exclusion of self-contributions. The sigmoid/loss fix did not address this additional inherited defect.

An isolated diagnostic used history [1, 2] and target item 2, zero biases and all-zero embeddings except item 2's source and destination embeddings, both [1, 0]. The current score was 1.0; zeroing only the target's source embedding reduced it to 0.0. All of the original score therefore came from its self-contribution. No saved model parameters were modified.

The existing regression tests checked logits, probabilities, and algebraic equivalence, but did not test candidate items already present in training history. Passing those tests did not establish this model constraint.

## Weighted-regression implementation

The core Task 1.3 regression requirement is implemented. Five-fold OOF component features, per-user candidate z-score normalization, Ridge with an intercept, and 1-5 rating targets were our design choices, not explicit lecture requirements. The loader's target normalization was corrected and verified as an affine transformation preserving rankings.

The fitted coefficients sum to 0.2011817347235848, and UserKNN has a negative coefficient. The sum-one formula is not literally enforced. Because the sum is positive, dividing every coefficient by it produces a sum-one parameterization with identical rankings; the global intercept also has no ranking effect. This presentation issue cannot explain the measured shortfall.

## Experiment limits

- Five components retain the previously approved 80/10/10-tuned hyperparameters; their actual cached predictions and comparisons use the agreed common 60/20/20 split.
- The initial Ridge grid (0.1, 1, 10, 100) selected its upper boundary. A subsequent 21-penalty validation-only search from 100 through 1e12 selected 100000; this resolves that boundary concern, without establishing exhaustive tuning.
- The seven-model table does not complete Tasks 2.1-2.6: random/popularity baselines, beyond-accuracy measures, group analyses, and evidence-based coefficient analysis remain outside this comparison.
- A rating regression target is allowed by the brief's unspecified target, but alignment with ranking evaluation must be motivated and empirically tested.

## Required correction before final interpretation

Correct FISM target self-exclusion, including its history-length normalization; test present and absent candidates; retune FISM; regenerate only FISM's OOF folds and full-training scores; then refit/select the regressor on validation. Keep the unaffected five verified caches. The current six-model hybrid result does not establish failure of weighted hybrids generally.

## Primary sources inspected

- project/Project-RecSys.pdf, pages 4-8 and 15-16.
- project/RecBole Instruction.pdf, pages 2-4.
- C:/Users/radud/Downloads/W4S2-Hybrid (1).pdf, page 6.
- C:/Users/radud/Downloads/W2S1-CF Neighborhood Models.pdf, page 49.
