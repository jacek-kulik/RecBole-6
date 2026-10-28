# Self-assessment: answer key and scoring rubric

An assessment of the project’s concepts, implementation decisions and measured trade-offs.

Attempt [self-test.html](self-test.html) before reading this key. Every prompt is included for printing. Allow 75–90 minutes.

Each half is worth 50 points: ten objective questions worth 2.5 each and five explanations worth 5 each. Multiple choice uses exact-set scoring. Numeric tolerance is 0.0001; simple fractions are accepted. A blank explanation scores zero.

For explanations, award one point per criterion only when the locked answer demonstrates it without a material contradiction. These points are self-assessed. Finish all rubrics before interpreting the combined score.

Study bands: 85–100 strong on this diagnostic; 70–84 working understanding with gaps; 50–69 partial; below 50 revisit the foundations. These are study guidance, not validated proficiency thresholds. Resolve every marked readiness check even if your total is high.

## A — Your metrics and rerankers

### A01 · 3 · Ranking contracts · 2.5 points

One user has ten recommendations, but two entries are the same item. What does the independent metric layer do?

1. Count both occurrences as separate hits.
2. Reject the ranking because the top-K items must be distinct.
3. Drop the duplicate and average over nine items.
4. Fill the last slot with the most popular unseen movie.

**Answer:** 2: Reject the ranking because the top-K items must be distinct.

validate_rankings requires K distinct external IDs. Changing the list silently would change the experiment.

### A02 · 3 · Accuracy arithmetic · 2.5 points

K=3; ranking [a,b,c]; relevant={a,c,d,e,f}. What is this code’s AP@3? Enter a fraction or decimal.

**Answer:** 0.333333

Hits occur at ranks 1 and 3. AP=(1+2/3)/5=1/3. The denominator is the full relevant set, not min(K,R).

### A03 · 3 · Macro aggregation · 2.5 points

User 1 has precision=1 and recall=0.2. User 2 has precision=0.2 and recall=1. What is the project’s aggregate F1? Enter a fraction or decimal.

**Answer:** 0.333333

Each user has F1=2*1*0.2/1.2=1/3, so their mean is 1/3. The harmonic mean of the aggregate P=R=0.6 would be 0.6 and is not the implemented quantity.

### A04 · 3 · Calibration definitions · 2.5 points

Which pairing matches the implemented metrics?

1. Genre: unsmoothed JS; popularity: KL with natural logs.
2. Both use base-two JS distance, the square root of divergence.
3. Genre: KL(p || 0.99q+0.01p) with natural logs; popularity: base-two JS divergence.
4. Both compare recommendation distributions with global test popularity.

**Answer:** 3: Genre: KL(p || 0.99q+0.01p) with natural logs; popularity: base-two JS divergence.

Genre miscalibration uses smoothed natural-log KL; UPD uses base-two Jensen–Shannon divergence. Their numerical scales differ.

### A05 · 3 · Catalogue exposure · 2.5 points

Four catalogue items have total discounted exposures [3,1,0,0]. What is the Exposure Gini? Enter a decimal.

**Answer:** 0.625

Sort to [0,0,1,3]. The weighted numerator is 1*1+3*3=10; the denominator is 4*4=16. Gini=0.625. The zero-exposure items remain in the catalogue.

### A06 · 5 · Training-only profiles · 2.5 points

How are head/middle/tail groups constructed?

1. Equal numbers of catalogue items after sorting by test popularity.
2. Cumulative training-interaction shares around 20/60/20, preserving complete count-tie buckets.
3. User-specific quantiles of predicted relevance.
4. Fixed item-ID ranges so every run has the same item counts.

**Answer:** 2: Cumulative training-interaction shares around 20/60/20, preserving complete count-tie buckets.

The bands describe interaction shares, not equal item counts. Tied buckets can overshoot a boundary.

### A07 · 6 · Exposure allocation · 2.5 points

Select every true statement about the implemented item-fairness reranker.

1. Exposure state is shared across users within one call.
2. Each user independently gets a list without affecting the next user.
3. Users are visited in sorted external-ID order within each rank round.
4. Exposure resets for a new experiment.
5. The objective is the exact marginal decrease in catalogue Gini.

**Answer:** 1: Exposure state is shared across users within one call.; 3: Users are visited in sorted external-ID order within each rank round.; 4: Exposure resets for a new experiment.

The algorithm shares local cohort state, iterates rank rounds then sorted users, and resets each call. Its inverse-exposure bonus is a heuristic, not an exact Gini optimizer.

### A08 · 7 · Constrained selection · 2.5 points

Original validation NDCG is 0.300. Strengths 0, 0.25, 0.5, 0.75 have (NDCG,target) values (0.300,0.80), (0.292,0.55), (0.286,0.35), (0.282,0.20). The target is minimized. Which strength is selected?

**Answer:** 0.5

The relative 5% threshold is 0.285. Strength 0.75 fails it; among feasible strengths, 0.5 has the smallest target.

### A09 · 7 · Fixed test choices · 2.5 points

A validation-selected strength loses 6.7% test NDCG. What should happen?

1. Reselect the strongest point that stays below 5% on test.
2. Average validation and test metrics, then select again.
3. Keep the frozen strength, report the loss, and flag that the validation budget did not transfer.
4. Drop this model from the final comparison.

**Answer:** 3: Keep the frozen strength, report the loss, and flag that the validation budget did not transfer.

The 5% constraint applies to validation selection. Test-based reselection contaminates the evaluation.

**Readiness check.**

### A10 · 6 · Objectives and measurements · 2.5 points

Select every correct distinction between a reranker’s objective and the measurements of its output.

1. The exposure reranker uses an inverse-exposure bonus; Gini measures its resulting lists.
2. Popularity calibration matches a user profile; novelty promotes rare training items.
3. Higher coverage guarantees that Exposure Gini becomes zero.
4. Genre calibration reads held-out relevance labels to choose each next item.
5. The same strength gives the same utility trade-off for every model and objective.

**Answer:** 1: The exposure reranker uses an inverse-exposure bonus; Gini measures its resulting lists.; 2: Popularity calibration matches a user profile; novelty promotes rare training items.

Objectives, measured targets and their units remain distinct. Coverage is breadth rather than equality, training context excludes held-out answers, and strength is selected empirically.

### A11 · 7 · Evaluation design · 5 points

Design a comparable accuracy and calibration evaluation for a fitted model. Explain what training, validation and test may influence, and how candidates and relevance must be handled.

Award one point for each demonstrated criterion:

- State the same users, split, K and full relevant sets for original and reranked comparisons; relevant items outside a shortlist stay in the denominators.
- Mask training history for validation, and training plus validation history for test.
- Build profiles, counts and category memberships from training interactions and static metadata; held-out outcomes never build reranker context. **Readiness check.**
- Use independent metric definitions and distinguish macro user means from cohort coverage/Gini; preserve unavailable values with counts.
- Choose settings on validation, fix them before test and report accuracy/target/group effects without test-based reselection.

**Model explanation:** Keep the population, split, K, relevance definition and candidate policy fixed between the original and intervention. Validation masks training history; test masks training and validation. Build preference profiles and popularity from training only, with known static attributes. Use validation outcomes to select a strength under the declared target direction and loss budget. Apply the fixed setting on test, retaining all relevant positives even outside the top-fifty pool, then calculate independent accuracy, calibration, cohort exposure and group effects. Missing measurements stay explicit.

### A12 · 6 · Greedy calibration · 5 points

A user’s profile is p=(0.75 A,0.25 B), and the growing list currently contains one A-only movie. Candidate x is A-only with normalized relevance 1.0; candidate y is B-only with relevance 0.8. At strength 0.5, which candidate does the next greedy step choose? Show the distributions and objective, and explain one limitation.

Award one point for each demonstrated criterion:

- Correctly form q_x=(1,0) and q_y=(0.5,0.5) after addition, treating movie genre mass correctly.
- Apply KL(p || 0.99q+0.01p) with natural logs rather than unsmoothed KL or JS.
- Compute approximately KL_x=0.9374 and KL_y=0.1283, or enough correct arithmetic to establish the ordering.
- Compare 0.5*relevance-0.5*KL and select y despite its lower relevance.
- Explain greediness, candidate-pool limits, historical-profile assumptions, or the lack of a global optimum guarantee.

**Model explanation:** Adding x gives q=(1,0), smoothed to (0.9975,0.0025); KL is about 0.9374. Adding y gives q=(0.5,0.5), smoothed to (0.5025,0.4975); KL is about 0.1283. Objectives are about 0.0313 for x and 0.3358 for y, so y wins. The method evaluates the growing list and makes a local greedy choice; it neither searches all lists nor introduces an item absent from the candidate pool.

### A13 · 6 · Shared exposure state · 5 points

Two users with external IDs "1" and "2" each have candidates a (relevance 1.0) and b (relevance 0.9). Set K=1 and strength=1. Trace the exposure reranker. Contrast with strength zero and explain why resetting state matters.

Award one point for each demonstrated criterion:

- User "1" picks a: initial bonuses tie and original relevance breaks the tie.
- After the first choice, a receives rank-one exposure 1; user "2" picks b because its bonus is 1 versus a’s 1/2.
- Strength zero returns a for both users, exactly following the original top-K.
- Explain shared cohort state and deterministic user/rank ordering; another cohort/order can change a user’s result.
- Explain reset per call, rank discount, and why the reciprocal bonus does not guarantee minimum Gini.

**Model explanation:** With zero initial exposure, both bonuses are one. User "1" takes a by original score and gives it exposure 1. User "2" sees bonus 1/2 for a and 1 for b, so takes b. Strength zero instead gives a to both. State must reset so a prior model or strength cannot change the next experiment’s starting condition. At later ranks the update is 1/log2(r+1). The method is cohort/order dependent and heuristic.

### A14 · 7 · Experimental choices · 5 points

Someone says: “Top fifty and a 5% budget were required by the course, and this one reranker optimizes diversity, calibration and all fairness notions together.” Correct this claim and propose a bounded follow-up experiment.

Award one point for each demonstrated criterion:

- State that top fifty is a chosen cost/candidate policy, not a demonstrated optimum or a course requirement.
- State that the relative 5% validation-loss budget is a chosen operating point, not a test guarantee.
- Explain that genre, popularity and item-exposure variants optimize separate objectives; side effects are only measured.
- Propose a small declared validation-only depth/budget sensitivity study with a fixed compute budget.
- Freeze the resulting choices before later evaluation, and acknowledge that the existing test has already been inspected. **Readiness check.**

**Model explanation:** The brief requires analysis of trade-offs; it does not supply the chosen candidate depth or budget. Each implemented variant balances relevance against one objective. A bounded follow-up might compare depths 20/50/100 at budgets 0/5/10%, reusing fixed component scores and applying the same validation selection policy. Report selection sensitivity and runtime, freeze the policy, and avoid choosing from the existing test results. Robust generalization would need an appropriately new evaluation design.

### A15 · 7 · Interpreting the results · 5 points

ItemKNN exposure reranking changes test NDCG 0.363396→0.362948, coverage 11.59%→25.98%, and Gini 0.966588→0.938365. Give a defensible interpretation and explain what this does not establish.

Award one point for each demonstrated criterion:

- Calculate or identify the small relative NDCG loss of approximately 0.12%.
- Explain that coverage increases the union of recommended items and is distinct from equal exposure.
- Explain the Gini reduction while noting that exposure remains strongly concentrated.
- Restrict the claim to the given model, top-fifty pools, cohort and single-seed offline experiment.
- Do not infer better user-group fairness, online satisfaction or significance without additional evidence.

**Model explanation:** This operating point broadens catalogue reach substantially at roughly 0.12% measured accuracy loss. Exposure becomes somewhat less concentrated, but Gini remains high; coverage and equality measure different properties. The result is conditional on the model, cohort and top-fifty pools. It does not establish significance, increased user satisfaction, or improved F/M quality gaps, and it cannot substitute for subgroup analysis.

## B — The shared system and other contributions

### B01 · 1 · Prediction task · 2.5 points

Which configuration defines the project’s ranking task?

1. MovieLens-100K; random per-user 60/20/20; seed 42; full-sort K=10; validation NDCG@10.
2. MovieLens-100K; temporal 80/10/10; seed 42; sampled candidates; validation MRR@10.
3. MovieLens-1M; random 70/15/15; seed 1; test NDCG selection.
4. MovieLens-100K; random 60/20/20; only ratings ≥4 relevant; validation RMSE.

**Answer:** 1: MovieLens-100K; random per-user 60/20/20; seed 42; full-sort K=10; validation NDCG@10.

Current protocol.yaml uses the first setting. Old MRR and 80/10/10 records remain historical evidence.

### B02 · 2 · Model identity · 2.5 points

Why must score export preserve the UserKNN configuration label?

1. UserKNN has no external movie IDs.
2. Both user- and item-neighborhood variants use the ItemKNN class in this fork.
3. UserKNN scores are always identical to ItemKNN scores.
4. Only UserKNN is evaluated on validation.

**Answer:** 2: Both user- and item-neighborhood variants use the ItemKNN class in this fork.

knn_method selects the user or item mode of the same framework class; its class name alone cannot distinguish the experiments.

### B03 · 2 · FISM scoring · 2.5 points

Select every statement that describes the implemented FISM scoring and training loss.

1. Pass raw logits to BCEWithLogitsLoss.
2. Apply sigmoid twice before the training loss.
3. Remove the target’s source-history contribution, including repeated occurrences.
4. Adjust the history-length normalization for those removed occurrences.
5. Use test positives to build the history vector.

**Answer:** 1: Pass raw logits to BCEWithLogitsLoss.; 3: Remove the target’s source-history contribution, including repeated occurrences.; 4: Adjust the history-length normalization for those removed occurrences.

The loss applies sigmoid internally. FISM also excludes the candidate from its own source sum and effective history size; test positives never build history.

### B04 · 8 · Ridge training targets · 2.5 points

What are the final Ridge combiner’s training rows and targets?

1. One row per test user, with test NDCG as the target.
2. All unobserved candidates with zero target ratings.
3. Out-of-fold component predictions for outer-training interactions, paired with raw 1–5 ratings.
4. In-sample predictions of outer-validation interactions with binary targets.

**Answer:** 3: Out-of-fold component predictions for outer-training interactions, paired with raw 1–5 ratings.

Cross-fitting supplies held-out component features for observed outer-training ratings. Unrated candidates are not imputed zero-rating targets.

### B05 · 1 · Score alignment · 2.5 points

Which must agree before combining ScoreTables? Select all required by require_aligned.

1. Split ID and evaluation partition.
2. Evaluation-user IDs.
3. Each user’s eligible candidate item IDs.
4. The numerical range of every model’s raw scores.
5. Every model’s class name.

**Answer:** 1: Split ID and evaluation partition.; 2: Evaluation-user IDs.; 3: Each user’s eligible candidate item IDs.

The contract validates finite scores and matching identities/candidates. Different models can have different score ranges and class names; normalization is a later design choice.

### B06 · 8 · Feature augmentation · 2.5 points

What changes in UserKNN→EASE/SLIM feature augmentation?

1. Test outcomes are added to the training interaction targets.
2. Predicted unseen entries augment input X; observed binary target R stays unchanged and the receiver is refitted.
3. The receiver’s top ten are concatenated with the sender’s top ten.
4. Only an intercept is refitted, with no input-matrix change.

**Answer:** 2: Predicted unseen entries augment input X; observed binary target R stays unchanged and the receiver is refitted.

Inferred values are features, not observed targets. X≠R requires the constrained generalized EASE fit and matched receiver controls.

### B07 · 9 · Score-preserving conversion · 2.5 points

In the weighted pre-reranking path, a component has scores {a:3,b:2,c:1}. Its reranked prefix is [c]. Which score table reaches the frozen combiner?

1. {a:3,b:2,c:1}; reranked prefixes are discarded.
2. {c:3,a:2,b:1}; the original score multiset is reassigned to the promoted order.
3. {c:1,a:0.5,b:0.333}; scores become reciprocal ranks.
4. {c:1,a:0,b:-1}; normalize the prefix again and remove the rest.

**Answer:** 2: {c:3,a:2,b:1}; the original score multiset is reassigned to the promoted order.

Promote the reranked prefix, append the remainder in original score order, then assign the original descending standardized values. Frozen Ridge combines these values without a second normalization.

### B08 · 8 · Coefficient interpretation · 2.5 points

ItemKNN has the largest fitted coefficient, but zeroing it improves test NDCG. Which interpretation is defensible?

1. The report must be arithmetically impossible.
2. The coefficient is not a causal ranking-contribution percentage; correlated rating-regression features can have different ranking utility.
3. ItemKNN should be removed immediately because test has selected the improvement.
4. All weighted hybrids are worse than all individual models.

**Answer:** 2: The coefficient is not a causal ranking-contribution percentage; correlated rating-regression features can have different ranking utility.

Coefficient size depends on correlated standardized features and a rating-error objective. Ablation evaluates a different quantity and must not become test-based selection.

### B09 · 8 · Warm scope · 2.5 points

Which comparison isolates switching behavior most fairly?

1. Warm switching against full-catalogue UserKNN with all 19,633 test positives.
2. Warm switching against warm UserKNN with matching users, candidates and supported relevance.
3. Warm switching against popularity from another split.
4. Switching test NDCG against UserKNN validation NDCG.

**Answer:** 2: Warm switching against warm UserKNN with matching users, candidates and supported relevance.

Switching retains 1,615 supported items and 19,588 test positives; scope must match candidates and relevance for the control too.

### B10 · 11 · Audience measurements · 2.5 points

What does an item group’s audience_accuracy primarily describe?

1. Precision computed only from hits in that item group.
2. Mean accuracy of randomly sampled catalogue items.
3. User whole-list quality weighted by recommendation occurrences in that group.
4. The probability that the group is causally responsible for a hit.

**Answer:** 3: User whole-list quality weighted by recommendation occurrences in that group.

Audience metrics repeat whole-list user metrics per group occurrence. They are distinct from the report’s item-group hits/recommended precision.

### B11 · 8 · Out-of-fold learning · 5 points

Design the weighted-hybrid learning procedure from the outer split onward. Explain why in-sample features would be problematic and distinguish training objective, selection objective and test evaluation.

Award one point for each demonstrated criterion:

- Divide only outer-training interactions into folds, keeping repeated user-item pairs together and retaining user training support.
- Train each fold component without the held-out-fold labels or early stopping on them; produce genuinely held-out features. **Readiness check.**
- Normalize component scores per user over eligible candidates; fit raw observed 1–5 rating targets, not zeros for missing items.
- Fit Ridge on concatenated OOF rows; select alpha by outer-validation NDCG while distinguishing it from squared-rating-error training.
- Refit components at the declared full-training budget, freeze settings/coefficients, and evaluate masked test without reselection.

**Model explanation:** Create user-stratified folds inside outer training. For each fold, train components on the rest at fixed budgets and score eligible held-out interactions without using their labels for checkpoint selection. Standardize per-user candidate scores and combine the resulting OOF rows with raw ratings. Ridge minimizes squared rating error plus coefficient penalty; outer-validation NDCG selects alpha. Full-training components then feed the saved combiner and its declared transform. Test uses frozen choices and the test history mask. In-sample features could teach the combiner to exploit a component’s fit to its own training answers.

### B12 · 2 · FISM representation and loss · 5 points

Explain how FISM constructs a score from a user’s history. Why is the target excluded, how does history-length normalization matter, and why does BCEWithLogitsLoss receive raw logits? Contrast the user representation with BPR.

Award one point for each demonstrated criterion:

- Sum source embeddings of training-history items, then compare the normalized sum with the target embedding and add biases.
- Exclude the target contribution from both the source sum and effective history length so it cannot explain itself.
- The history-length exponent controls how the magnitude depends on the number of observations.
- BCEWithLogitsLoss expects raw logits and applies sigmoid internally; applying sigmoid beforehand changes the intended objective.
- FISM derives the user representation from history items; BPR learns a free user vector. Both evaluation paths respect the same history mask.

**Model explanation:** FISM sums source-item vectors from training history, excluding the candidate if present, and scales by the remaining history length raised to a negative exponent. It takes the dot product with a target-item vector and adds biases. The exclusion prevents target self-copying; the exponent controls the contribution of longer histories. A raw logit is passed to the combined logistic cross-entropy loss, which already performs sigmoid internally. BPR instead learns an independent vector per user. Full-sort and individual scoring must represent the same calculation, with observed-history masking applied for evaluation.

### B13 · 9 · Order and shared state · 5 points

Why do the two switching orders give exactly equal genre/popularity-calibration lists under the recorded conditions, while exposure fairness differs? Contrast catalogue coverage with exposure, and explain why novelty promotion is different from popularity calibration.

Award one point for each demonstrated criterion:

- Fixed routing and identical user-specific scores, pools, profiles and strength let local calibration commute with the route.
- The recorded experiment checks exact list equality, not only equal aggregate metrics; routing remains frozen.
- Pre-exposure uses separate whole-component cohort counters, including later-discarded lists; post-exposure uses one final mixed cohort.
- Coverage tracks an ever-recommended set; exposure accumulates discounted counts. Coverage-order equality or inequality was not measured by the calibration/exposure order experiment.
- Novelty favors training self-information, whereas popularity calibration matches the user’s head/middle/tail profile; all settings still require validation selection and frozen test.

**Model explanation:** A fixed route selects the same user-local component input before or after calibration, so the deterministic local operation produces the same list. The saved experiment establishes exact list equality across validation strengths and frozen test choices. Exposure depends on other users: separate pre-component counters include lists the switch later discards, unlike the final mixed post cohort, and 747 test users differ. Coverage also has cohort state but tracks first appearance rather than discounted repetition; the order study did not test this new variant. Novelty promotes low training-frequency items; popularity calibration matches a particular user profile, which can be head-heavy. These mechanisms do not replace validation selection, masks or frozen test evaluation.

### B14 · 8 · Ablation and uncertainty · 5 points

The original hybrid has test NDCG 0.409095. Removing ItemKNN without refitting changes it by +0.004194; removing FISM changes it by −0.004608. Explain the results, distinguish frozen and refitted removal, and interpret a user-bootstrap interval.

Award one point for each demonstrated criterion:

- Correctly state that frozen ItemKNN removal improves this fitted ranking and frozen FISM removal harms it.
- Distinguish zeroing a coefficient with other parameters fixed from relearning remaining coefficients on the same OOF rows and fixed alpha.
- Explain correlated standardized scores and the rating-regression versus ranking-objective mismatch; avoid causal coefficient claims.
- Interpret a paired user bootstrap as conditional on this split/trained fits rather than complete training/data/selection uncertainty.
- Do not use the favorable test ablation to select a new model; a new model choice requires development-only selection. **Readiness check.**

**Model explanation:** Under frozen removal, ItemKNN’s contribution is detrimental to this ranking outcome while FISM’s is beneficial. Refit removal tests how remaining features compensate at the same penalty; it can yield a different delta. These are correlated surrogate-regression features, so coefficient size does not identify causal utility. The bootstrap resamples users with paired predictions, conditional on the split and trained systems. Its interval omits new training and model-selection variation. Report the ablation descriptively; do not select the reduced model on test.

### B15 · 11 · Group interpretation · 5 points

A group report shows a narrower F/M NDCG gap after reranking, while both group means fall. It also reports two tail hits from five tail recommendations and 3,891 relevant tail interactions. Explain what can be concluded, including the difference between item and audience metrics and the effect of overlapping groups.

Award one point for each demonstrated criterion:

- A narrower gap can result from the higher-scoring group losing more; it does not establish improvement for the lower-scoring group.
- Tail conditional precision is 2/5=40%, while tail recall is 2/3891≈0.0514%; the precision estimate uses only five recommendations.
- Audience accuracy weights recipients’ whole-list metrics by group-item occurrences; it is different from conditional item-group precision.
- Top-three user-genre cohorts and multi-genre item memberships overlap; genre membership counts differ from fractional calibration mass.
- A minimum-size flag does not suppress or validate a small estimate; recompute cohort coverage and Gini rather than averaging subgroup values.

**Model explanation:** Report both absolute group means and their gap: narrowing caused by losses is different from helping the lower-performing group. Tail precision is 40% but is supported by only five slots, while tail recall is about 0.0514%, so the group is barely recovered. Audience metrics describe whole-list quality of recipients, repeated per group occurrence. Genre cohorts and item memberships overlap; their counts are not a partition and do not use calibration’s fractional movie mass. Read counts, availability and small-group flags, and recalculate pooled catalogue statistics for each cohort.

## Use the result

Keep the first attempt and its export. Revisit the linked report sections, trace the relevant code, and explain one example aloud. Retake after a delay using changed numbers or a different user; recalling an answer is weaker evidence than explaining why it follows from the implementation.
