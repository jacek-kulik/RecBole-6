# Self-assessment: answer key and scoring rubric

Attempt `self-test.html` before reading this file. This key also contains every prompt so it can be used as a printable assessment. Allow 75–90 minutes.

Each half is worth 50 points: ten objective questions worth 2.5 each and five written explanations worth 5 each. Exact-set multiple choice earns either 0 or 2.5. Numeric tolerance is 0.0001. Each demonstrated explanation criterion earns one point; unsupported claims or material contradictions do not earn that criterion. Written scoring is self-assessment, not automatic semantic grading.

Interpretation: 85–100 strong on this diagnostic; 70–84 good working understanding with gaps; 50–69 partial; below 50 rebuild foundations. These are study bands, not statistically validated proficiency thresholds. Missing any marked readiness check blocks a claim of readiness regardless of the numerical total. A complete explanation rubric is required before using the combined score.

## A — Your metrics and rerankers

### A01 · A3 · Input contracts · 2.5 points

One user has ten recommendations, but two entries are the same item. What does the independent metric layer do?

1. Count both occurrences as separate hits.
2. Reject the ranking because the top-K items must be distinct.
3. Drop the duplicate and average over nine items.
4. Fill the last slot with the most popular unseen movie.

**Answer:** 2. Reject the ranking because the top-K items must be distinct.

validate_rankings requires K distinct external IDs. Changing the list silently would change the experiment.

### A02 · A2 · Accuracy arithmetic · 2.5 points

K=3; ranking [a,b,c]; relevant={a,c,d,e,f}. What is this code’s AP@3? Enter a fraction or decimal.

**Answer:** 0.333333 (equivalent fractions accepted)

Hits occur at ranks 1 and 3. AP=(1+2/3)/5=1/3. The denominator is the full relevant set, not min(K,R).

### A03 · A2 · Macro aggregation · 2.5 points

User 1 has precision=1 and recall=0.2. User 2 has precision=0.2 and recall=1. What is the project’s aggregate F1? Enter a fraction or decimal.

**Answer:** 0.333333 (equivalent fractions accepted)

Each user has F1=2*1*0.2/1.2=1/3, so their mean is 1/3. The harmonic mean of the aggregate P=R=0.6 would be 0.6 and is not the implemented quantity.

### A04 · A4 · Calibration definitions · 2.5 points

Which pairing matches the implemented metrics?

1. Genre: unsmoothed JS; popularity: KL with natural logs.
2. Both use base-two JS distance, the square root of divergence.
3. Genre: KL(p || 0.99q+0.01p) with natural logs; popularity: base-two JS divergence.
4. Both compare recommendation distributions with global test popularity.

**Answer:** 3. Genre: KL(p || 0.99q+0.01p) with natural logs; popularity: base-two JS divergence.

Genre miscalibration uses smoothed natural-log KL; UPD uses base-two Jensen–Shannon divergence. Their numerical scales differ.

### A05 · A4 · Catalogue exposure · 2.5 points

Four catalogue items have total discounted exposures [3,1,0,0]. What is the Exposure Gini? Enter a decimal.

**Answer:** 0.625000 (equivalent fractions accepted)

Sort to [0,0,1,3]. The weighted numerator is 1*1+3*3=10; the denominator is 4*4=16. Gini=0.625. The zero-exposure items remain in the catalogue.

### A06 · A5 · Training-only groups · 2.5 points

How are head/middle/tail groups constructed?

1. Equal numbers of catalogue items after sorting by test popularity.
2. Cumulative training-interaction shares around 20/60/20, preserving complete count-tie buckets.
3. User-specific quantiles of predicted relevance.
4. Fixed item-ID ranges so every run has the same item counts.

**Answer:** 2. Cumulative training-interaction shares around 20/60/20, preserving complete count-tie buckets.

The bands describe interaction shares, not equal item counts. Tied buckets can overshoot a boundary.

### A07 · A7 · Exposure algorithm · 2.5 points

Select every true statement about the implemented item-fairness reranker.

1. Exposure state is shared across users within one call.
2. Each user independently gets a list without affecting the next user.
3. Users are visited in sorted external-ID order within each rank round.
4. Exposure resets for a new experiment.
5. The objective is the exact marginal decrease in catalogue Gini.

**Answer:** 1. Exposure state is shared across users within one call.; 3. Users are visited in sorted external-ID order within each rank round.; 4. Exposure resets for a new experiment.

The algorithm shares local cohort state, iterates rank rounds then sorted users, and resets each call. Its inverse-exposure bonus is a heuristic, not an exact Gini optimizer.

### A08 · A8 · Validation selection · 2.5 points

Original validation NDCG is 0.300. Strengths 0, 0.25, 0.5, 0.75 have (NDCG,target) values (0.300,0.80), (0.292,0.55), (0.286,0.35), (0.282,0.20). The target is minimized. Which strength is selected?

**Answer:** 0.500000 (equivalent fractions accepted)

The relative 5% threshold is 0.285. Strength 0.75 fails it; among feasible strengths, 0.5 has the smallest target.

### A09 · A8–A9 · Frozen test choices · 2.5 points

A validation-selected strength loses 6.7% test NDCG. What should happen?

1. Reselect the strongest point that stays below 5% on test.
2. Average validation and test metrics, then select again.
3. Keep the frozen strength, report the loss, and flag that the validation budget did not transfer.
4. Drop this model from the final comparison.

**Answer:** 3. Keep the frozen strength, report the loss, and flag that the validation budget did not transfer.

The 5% constraint applies to validation selection. Test-based reselection contaminates the evaluation. **Readiness check: do not select on test.**

### A10 · A1 · Current implementation status · 2.5 points

Select the reranking variants implemented and evaluated in PR #24 at the inspected head.

1. Genre calibration.
2. Popularity calibration.
3. Item-exposure fairness.
4. Gabriel’s diversification.
5. Victor’s user-side fairness.

**Answer:** 1. Genre calibration.; 2. Popularity calibration.; 3. Item-exposure fairness.

The first three are implemented. The last two are registered placeholders with explicit pending status.

### A11 · A3, A5, A9 · Evaluation handoff · 5 points

A teammate sends a completed checkpoint and asks you for a final accuracy/calibration report. Describe the path from that handoff to comparable saved test results, including what information each stage is allowed to use.

Award one point for each of the following:

- Verify a completed training manifest, checkpoint identity, and reconstructed saved split membership/hash.
- Use test candidates excluding training plus validation history; preserve the evaluation users and external IDs.
- Build popularity, profiles, and groups from training interactions only; never use held-out outcomes to build reranker context. **Readiness check.**
- Use independent accuracy and beyond-accuracy definitions, and record missing metadata/availability explicitly.
- Apply validation-frozen settings, save rankings/per-user results/provenance, and check comparison scope/settings.

Model explanation: Trace the manifest to its checkpoint and exact split exports. Reconstruct the checkpoint configuration and verify identity. Score the test partition with training+validation history masked, then convert IDs. Training interactions and approved static metadata supply profiles/counts/groups. Apply settings already selected on validation, evaluate with the independent metrics, retain nulls with reasons where needed, and save rankings, relevance, detailed metrics and provenance for the comparison tool. Test labels measure outcomes; they do not create context or select strengths.

### A12 · A4, A6 · Greedy genre calibration · 5 points

A user’s profile is p=(0.75 A,0.25 B), and the growing list currently contains one A-only movie. Candidate x is A-only with normalized relevance 1.0; candidate y is B-only with relevance 0.8. At strength 0.5, which candidate does the next greedy step choose? Show the distributions and objective, and explain one limitation.

Award one point for each of the following:

- Correctly form q_x=(1,0) and q_y=(0.5,0.5) after addition, treating movie genre mass correctly.
- Apply KL(p || 0.99q+0.01p) with natural logs rather than unsmoothed KL or JS.
- Compute approximately KL_x=0.9374 and KL_y=0.1283, or enough correct arithmetic to establish the ordering.
- Compare 0.5*relevance-0.5*KL and select y despite its lower relevance.
- Explain greediness, candidate-pool limits, historical-profile assumptions, or the lack of a global optimum guarantee.

Model explanation: Adding x gives q=(1,0), smoothed to (0.9975,0.0025); KL is about 0.9374. Adding y gives q=(0.5,0.5), smoothed to (0.5025,0.4975); KL is about 0.1283. Objectives are about 0.0313 for x and 0.3358 for y, so y wins. The method evaluates the growing list and makes a local greedy choice; it neither searches all lists nor introduces an item absent from the candidate pool.

### A13 · A7 · Cohort exposure walkthrough · 5 points

Two users with external IDs "1" and "2" each have candidates a (relevance 1.0) and b (relevance 0.9). Set K=1 and strength=1. Trace the exposure reranker. Contrast with strength zero and explain why resetting state matters.

Award one point for each of the following:

- User "1" picks a: initial bonuses tie and original relevance breaks the tie.
- After the first choice, a receives rank-one exposure 1; user "2" picks b because its bonus is 1 versus a’s 1/2.
- Strength zero returns a for both users, exactly following the original top-K.
- Explain shared cohort state and deterministic user/rank ordering; another cohort/order can change a user’s result.
- Explain reset per call, rank discount, and why the reciprocal bonus does not guarantee minimum Gini.

Model explanation: With zero initial exposure, both bonuses are one. User "1" takes a by original score and gives it exposure 1. User "2" sees bonus 1/2 for a and 1 for b, so takes b. Strength zero instead gives a to both. State must reset so a prior model or strength cannot change the next experiment’s starting condition. At later ranks the update is 1/log2(r+1). The method is cohort/order dependent and heuristic.

### A14 · A8 · Experimental design criticism · 5 points

Someone says: “Top fifty and a 5% budget were required by the course, and this one reranker optimizes diversity, calibration and all fairness notions together.” Correct this claim and propose a bounded follow-up experiment.

Award one point for each of the following:

- State that top fifty is a chosen cost/candidate policy, not a demonstrated optimum or a course requirement.
- State that the relative 5% validation-loss budget is a chosen operating point, not a test guarantee.
- Explain that genre, popularity and item-exposure variants optimize separate objectives; side effects are only measured.
- Propose a small declared validation-only depth/budget sensitivity study with a fixed compute budget.
- Freeze the resulting choices before later evaluation, and acknowledge that the existing test has already been inspected. **Readiness check.**

Model explanation: The brief requires analysis of trade-offs; it does not supply the chosen candidate depth or budget. Each implemented variant balances relevance against one objective. A bounded follow-up might compare depths 20/50/100 at budgets 0/5/10%, reusing fixed component scores and applying the same validation selection policy. Report selection sensitivity and runtime, freeze the policy, and avoid choosing from the existing test results. Robust generalization would need an appropriately new evaluation design.

### A15 · A10 · Reading the actual results · 5 points

ItemKNN exposure reranking changes test NDCG 0.363396→0.362948, coverage 11.59%→25.98%, and Gini 0.966588→0.938365. Give a defensible interpretation and explain what this does not establish.

Award one point for each of the following:

- Calculate or identify the small relative NDCG loss of approximately 0.12%.
- Explain that coverage increases the union of recommended items and is distinct from equal exposure.
- Explain the Gini reduction while noting that exposure remains strongly concentrated.
- Restrict the claim to the given model, top-fifty pools, cohort and single-seed offline experiment.
- Do not infer better user-group fairness, online satisfaction or significance without additional evidence.

Model explanation: This operating point broadens catalogue reach substantially at roughly 0.12% measured accuracy loss. Exposure becomes somewhat less concentrated, but Gini remains high; coverage and equality measure different properties. The result is conditional on the model, cohort and top-fifty pools. It does not establish significance, increased user satisfaction, or improved F/M quality gaps, and it cannot substitute for subgroup analysis.

## B — Shared system and other contributions

### B01 · B1 · Adopted protocol · 2.5 points

Which configuration describes current merged master?

1. MovieLens-100K; random per-user 60/20/20; seed 42; full-sort K=10; validation NDCG@10.
2. MovieLens-100K; temporal 80/10/10; seed 42; sampled candidates; validation MRR@10.
3. MovieLens-1M; random 70/15/15; seed 1; test NDCG selection.
4. MovieLens-100K; random 60/20/20; only ratings ≥4 relevant; validation RMSE.

**Answer:** 1. MovieLens-100K; random per-user 60/20/20; seed 42; full-sort K=10; validation NDCG@10.

Current protocol.yaml uses the first setting. Old MRR and 80/10/10 records remain historical evidence.

### B02 · B2–B3 · Adapter/model identity · 2.5 points

Why must score export preserve the UserKNN configuration label?

1. UserKNN has no external movie IDs.
2. Both user- and item-neighborhood variants use the ItemKNN class in this fork.
3. UserKNN scores are always identical to ItemKNN scores.
4. Only UserKNN is evaluated on validation.

**Answer:** 2. Both user- and item-neighborhood variants use the ItemKNN class in this fork.

knn_method selects the user or item mode of the same framework class; its class name alone cannot distinguish the experiments.

### B03 · B3 · FISM correctness · 2.5 points

Select every correction required by the inspected FISM implementation.

1. Pass raw logits to BCEWithLogitsLoss.
2. Apply sigmoid twice before the training loss.
3. Remove the target’s source-history contribution, including repeated occurrences.
4. Adjust the history-length normalization for those removed occurrences.
5. Use test positives to build the history vector.

**Answer:** 1. Pass raw logits to BCEWithLogitsLoss.; 3. Remove the target’s source-history contribution, including repeated occurrences.; 4. Adjust the history-length normalization for those removed occurrences.

The loss applies sigmoid internally. FISM also excludes the candidate from its own source sum and effective history size; test positives never build history.

### B04 · B5 · Weighted-hybrid target · 2.5 points

What are the final Ridge combiner’s training rows and targets?

1. One row per test user, with test NDCG as the target.
2. All unobserved candidates with zero target ratings.
3. Out-of-fold component predictions for outer-training interactions, paired with raw 1–5 ratings.
4. In-sample predictions of outer-validation interactions with binary targets.

**Answer:** 3. Out-of-fold component predictions for outer-training interactions, paired with raw 1–5 ratings.

Cross-fitting supplies held-out component features for observed outer-training ratings. Unrated candidates are not imputed zero-rating targets.

### B05 · B2 · Score alignment · 2.5 points

Which must agree before combining ScoreTables? Select all required by require_aligned.

1. Split ID and evaluation partition.
2. Evaluation-user IDs.
3. Each user’s eligible candidate item IDs.
4. The numerical range of every model’s raw scores.
5. Every model’s class name.

**Answer:** 1. Split ID and evaluation partition.; 2. Evaluation-user IDs.; 3. Each user’s eligible candidate item IDs.

The contract validates finite scores and matching identities/candidates. Different models can have different score ranges and class names; normalization is a later design choice.

### B06 · B6 · Feature augmentation · 2.5 points

What changes in UserKNN→EASE/SLIM feature augmentation?

1. Test outcomes are added to the training interaction targets.
2. Predicted unseen entries augment input X; observed binary target R stays unchanged and the receiver is refitted.
3. The receiver’s top ten are concatenated with the sender’s top ten.
4. Only an intercept is refitted, with no input-matrix change.

**Answer:** 2. Predicted unseen entries augment input X; observed binary target R stays unchanged and the receiver is refitted.

Inferred values are features, not observed targets. X≠R requires the constrained generalized EASE fit and matched receiver controls.

### B07 · B6 · Cascade evaluation · 2.5 points

ItemKNN misses a relevant movie when retrieving a shortlist for BPR. How is that movie treated in accuracy evaluation?

1. Remove it from the relevant set so BPR is evaluated only on retrievable positives.
2. Credit it as half a hit because the first-stage model failed.
3. Keep it in the relevance denominator; retrieve only after applying the partition’s history mask.
4. Move it to the training set and rerun the cascade.

**Answer:** 3. Keep it in the relevance denominator; retrieve only after applying the partition’s history mask.

Retrieval failure is part of cascade performance. Masking after truncation would also produce the wrong candidate policy.

### B08 · B8 · Coefficient interpretation · 2.5 points

ItemKNN has the largest fitted coefficient, but zeroing it improves test NDCG. Which interpretation is defensible?

1. The report must be arithmetically impossible.
2. The coefficient is not a causal ranking-contribution percentage; correlated rating-regression features can have different ranking utility.
3. ItemKNN should be removed immediately because test has selected the improvement.
4. All weighted hybrids are worse than all individual models.

**Answer:** 2. The coefficient is not a causal ranking-contribution percentage; correlated rating-regression features can have different ranking utility.

Coefficient size depends on correlated standardized features and a rating-error objective. Ablation evaluates a different quantity and must not become test-based selection.

### B09 · B7 · Warm-scope control · 2.5 points

Which comparison isolates switching behavior most fairly?

1. Warm switching against full-catalogue UserKNN with all 19,633 test positives.
2. Warm switching against warm UserKNN with matching users, candidates and supported relevance.
3. Warm switching against popularity from another split.
4. Switching test NDCG against UserKNN validation NDCG.

**Answer:** 2. Warm switching against warm UserKNN with matching users, candidates and supported relevance.

Switching retains 1,615 supported items and 19,588 test positives; scope must match candidates and relevance for the control too.

### B10 · B9 · Subgroup meaning · 2.5 points

In PR #26, what does an item group’s audience_accuracy primarily describe?

1. Precision computed only from hits in that item group.
2. Mean accuracy of randomly sampled catalogue items.
3. User whole-list quality weighted by recommendation occurrences in that group.
4. The probability that the group is causally responsible for a hit.

**Answer:** 3. User whole-list quality weighted by recommendation occurrences in that group.

Audience metrics repeat whole-list user metrics per group occurrence. They are distinct from the report’s item-group hits/recommended precision.

### B11 · B5 · Cross-fitting design · 5 points

Design the weighted-hybrid learning procedure from the outer split onward. Explain why in-sample features would be problematic and distinguish training objective, selection objective and test evaluation.

Award one point for each of the following:

- Divide only outer-training interactions into folds, keeping repeated user-item pairs together and retaining user training support.
- Train each fold component without the held-out-fold labels or early stopping on them; produce genuinely held-out features. **Readiness check.**
- Normalize component scores per user over eligible candidates; fit raw observed 1–5 rating targets, not zeros for missing items.
- Fit Ridge on concatenated OOF rows; select alpha by outer-validation NDCG while distinguishing it from squared-rating-error training.
- Refit components at the declared full-training budget, freeze settings/coefficients, and evaluate masked test without reselection.

Model explanation: Create user-stratified folds inside outer training. For each fold, train components on the rest at fixed budgets and score eligible held-out interactions without using their labels for checkpoint selection. Standardize per-user candidate scores and combine the resulting OOF rows with raw ratings. Ridge minimizes squared rating error plus coefficient penalty; outer-validation NDCG selects alpha. Full-training components then feed the saved combiner and its declared transform. Test uses frozen choices and the test history mask. In-sample features could teach the combiner to exploit a component’s fit to its own training answers.

### B12 · B3–B4 · FISM debugging and dependency repair · 5 points

A FISM implementation applies sigmoid before BCEWithLogitsLoss and includes the target in the source history. Explain both bugs, the required implementation changes, and which downstream evidence must be regenerated.

Award one point for each of the following:

- Explain that BCEWithLogitsLoss already applies sigmoid and expects raw logits; the extra sigmoid changes training loss/gradients.
- Remove all target occurrences from the source embedding sum and effective history length, with padding excluded.
- Keep pair scoring/full-sort behavior consistent and distinguish prediction probabilities from training logits.
- Retrain affected FISM fits and regenerate affected OOF features, predictions and downstream combiner selection/results.
- Preserve unaffected verified components, record source/config identity, and avoid attributing a combined before/after change solely to one fix.

Model explanation: Return raw logits to the loss; apply a sigmoid only where probability prediction is explicitly required. Remove the candidate’s source contribution and normalize by the remaining history count in both scoring paths, including repeated occurrences. Refit affected FISM models, reconstruct its fold features and full predictions, and redo dependent hybrid selection/evaluation. Verify unaffected columns against their saved hashes. If retuning and metric-selection policy also change, the overall metric difference does not isolate just the code correction.

### B13 · B6–B7 · Hybrid design reasoning · 5 points

Explain how cascade, feature combination, feature augmentation and meta-level transfer different information. For a claim that UserKNN→SLIM helps, specify a useful control and the evidence needed to compare it fairly.

Award one point for each of the following:

- Cascade transfers a retrieved candidate set to another scorer; it does not blend the two score vectors.
- Feature combination trains a scorer over combined features such as frozen user/item embeddings.
- Feature augmentation adds inferred input entries while targets stay observed and the receiver is refitted.
- Meta-level transfers a learned representation, here BPR user embeddings, into a neighborhood model.
- Use a paired unaugmented SLIM receiver with matched solver/settings, split, masking, scope, validation selection and frozen test evaluation.

Model explanation: Cascade transfers a shortlist; feature combination transfers feature vectors into a learned scorer; augmentation transfers inferred inputs to a refitted receiver; meta-level transfers a learned representation to another learning/scoring mechanism. For augmentation, compare against the same receiver implementation at strength zero with identical solver and regularization, not just any historical SLIM row. Keep source predictions and observed targets fixed, select strength on validation, and compare frozen test outcomes on matching candidates and relevance.

### B14 · B8 · Component contribution analysis · 5 points

The original hybrid has test NDCG 0.409095. Removing ItemKNN without refitting changes it by +0.004194; removing FISM changes it by −0.004608. Explain the results, distinguish frozen and refitted removal, and interpret a user-bootstrap interval.

Award one point for each of the following:

- Correctly state that frozen ItemKNN removal improves this fitted ranking and frozen FISM removal harms it.
- Distinguish zeroing a coefficient with other parameters fixed from relearning remaining coefficients on the same OOF rows and fixed alpha.
- Explain correlated standardized scores and the rating-regression versus ranking-objective mismatch; avoid causal coefficient claims.
- Interpret a paired user bootstrap as conditional on this split/trained fits rather than complete training/data/selection uncertainty.
- Do not use the favorable test ablation to select a new model; a new model choice requires development-only selection. **Readiness check.**

Model explanation: Under frozen removal, ItemKNN’s contribution is detrimental to this ranking outcome while FISM’s is beneficial. Refit removal tests how remaining features compensate at the same penalty; it can yield a different delta. These are correlated surrogate-regression features, so coefficient size does not identify causal utility. The bootstrap resamples users with paired predictions, conditional on the split and trained systems. Its interval omits new training and model-selection variation. Report the ablation descriptively; do not select the reduced model on test.

### B15 · B9–B10 · Reviewing the open group-analysis PR · 5 points

You review PR #26 and its integration with PR #24. Identify five concrete interpretation or integration issues to check, using the actual code rather than assuming the PR title means all group analysis is finished.

Award one point for each of the following:

- Check actual activity boundaries (26/69 here) against stale hard-coded 34/92 descriptions.
- Explain top-three overlapping user-genre cohorts and multi-label item genres; their counts/shares are not a disjoint partition.
- Distinguish audience whole-list averages from conditional item-group precision/recall; note small-group flags do not suppress estimates.
- Identify pending_inputs carrying a catalogue instead of pending reasons, and/or the misleading top-one item-genre description.
- Identify the evaluator conflict and changed compare_groups signature/report schema; adapt the shared writer and reranker consumers before claiming integrated operation.

Model explanation: Inspect the emitted definitions, not just the numbers. Activity memberships use training terciles despite a stale 34/92 description. User genre takes the top three and overlaps; items can belong to multiple genres. Audience metrics and item-conditional metrics have different denominators, and n<30 is merely flagged. pending_inputs currently holds the catalogue, and item_genre’s top-one description does not match its all-genre members. PR #24 and #26 conflict in evaluation.py and disagree about the group-report API, so a coherent shared writer and reranker manifest handoff is still needed.

## Retest method

For each missed objective item, explain the rule without looking at the options and locate its implementation using the report’s code map. For each missed reasoning criterion, write a concrete worked example or trace the relevant data flow. A repeat of the same questions mainly measures learning and memory; a better transfer test is to change the numbers, candidate scope, or failure condition and explain the new result.
