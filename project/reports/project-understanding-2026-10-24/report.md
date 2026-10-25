# Understanding our recommender-system project

Study report for Jacek · refreshed 25 October 2026

This report explains what the project does, how the implementation works, why its decisions matter, and what the evidence supports. It is a learning document, rather than the space-limited course submission. Read it alongside the scored assessment in `self-test.html`; keep `answer-key.md` closed until you have attempted the questions.

The substantive report has two equally weighted halves. **Part A covers your assigned work: independent metrics, reranker integration, calibration, item fairness, and reranker evaluation. Part B covers the other contributions and the shared system.** The ownership source is the current [work distribution][assignments], including its scope correction: implementing metric formulas does not transfer ownership of metadata preparation or subgroup interpretation to you.

The refreshed source snapshot is merged `master` at **1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf**. The clean local checkout was fast-forwarded from `b0eb9781a` and matched `origin/master`. Your reranker [PR #24][pr24], Bogdan's order-comparison [PR #27][pr27], and [Gabriel's diversification, #28][pr28] are now merged. PR #28 merged during this refresh, so its code and results are included on master. The remaining open work is [Victor's group analysis, #26][pr26], inspected at **a3a55153f** without merging it into the report checkout.

The twelve annotated figures follow the same balance: A1–A6 explain your work; B1–B6 explain the other contributions. Four empirical plots are copied unchanged from saved project results. Other figures show recorded numerical evidence or explicitly labeled conceptual examples. Select any image to view it at full size.

| Change since the previous report | Current interpretation |
| --- | --- |
| Calibration and item-exposure rerankers merged | Their implementation and native evaluation commands are available on master. |
| Weighted/switching order comparison merged | Task 3.3 has real implementations and measured results, despite the generic task runner retaining an old placeholder message. |
| New weighted reconstruction recorded | The same split/folds and qualitative conclusions are reproduced closely; the rebuilt predictions are not numerically identical. |
| Diversification merged with six-model evidence | Genre variety, novelty, and catalogue reach are separate implemented objectives on master through PR #28. |
| Group PR updated | Some definition text and tests changed; the shared evaluator integration still needs resolution. |

“Merged,” “implemented in an open PR,” and “supported by saved evidence” are separate claims throughout this report. Historical result tables keep their original experiment identities instead of being relabeled as measurements of today's source.

## Part A — Your work: measuring and changing recommendations

### A1. Your responsibility and its place in the system

Your main assignments are Task 2.1, independent accuracy and beyond-accuracy metrics; Task 3.1, integration of four reranker families; and Task 3.2, evaluation of reranking trade-offs. Within Task 3.1, you implement calibration and item-side fairness, Gabriel supplies diversification, and Victor supplies user-side fairness. You also review diversification, user fairness, Bogdan's pipeline-order comparison, and Victor's improvement proposal.

The common thread is turning model scores into defensible statements about recommendations. A model can produce plausible numbers while using the wrong candidate set, an unsuitable metric denominator, or information unavailable at recommendation time. Your code supplies a second measurement layer and the machinery for evaluating controlled changes to the resulting lists.

There are three distinct responsibilities. A **metric** describes an already produced ranking. A **reranker** constructs another ranking using model scores and permitted context. An **experiment runner** chooses settings on validation and records their later test performance. Keeping those roles separate makes the implementation understandable: changing a metric does not silently change the algorithm, and evaluation relevance labels do not become reranker inputs.

Merged master now contains your independent metrics, shared validation/test writer, genre/popularity calibration, and item-exposure reranking. These are the three evaluated variants delivered through PR #24. Bogdan's two-order experiments and Gabriel's three diversification variants with six-model evidence are also merged. Master therefore has six implemented reranking variants. User-side fairness remains an explicit placeholder. Four assigned method families do not translate into four finished functions: calibration and diversification each have multiple distinct variants, while another family is still pending.

The important interface is `rerank(scores, context, k, strength) -> Rankings`. The input `ScoreTable` contains eligible item scores; `RerankContext` supplies training-derived profiles and metadata; the output is an ordered list of external item IDs for every evaluation user. That contract lets the same independent metric functions score original and modified rankings. [Sources: assignments][assignments], [metric runner][evaluation], [PR implementation][rerank].

### A2. Accuracy metrics: the exact quantities your code calculates

`metrics/accuracy.py:evaluate_accuracy` makes one pass through each user's top-K list and returns both per-user results and equal-user means. Let `h_r` equal one when rank `r` contains a held-out relevant item, `H = sum(h_r)`, and `R` be the number of that user's relevant items.

| Metric | Implemented per-user definition | What it rewards |
| --- | --- | --- |
| Precision@K | `H / K` | A high fraction of relevant recommendations. |
| Recall@K | `H / R` | Recovering a high fraction of the user's relevant set. |
| F1@K | `2PR/(P+R)`, or zero without hits | Balance between that user's precision and recall. |
| MRR@K | Reciprocal rank of the first hit, or zero | Getting at least one relevant item very early. |
| NDCG@K | `sum(h_r/log2(r+1)) / IDCG` | Multiple relevant items, with earlier ranks worth more. |
| AP@K, reported as per-user MAP@K | `sum(h_r * hits_through_r/r) / R` | Early relevant items and recovery of the relevant set. |

`IDCG` sums the same discounts over the first `min(K,R)` ranks. Relevance is binary: a five-star held-out interaction contributes the same gain as a one-star held-out interaction under the adopted all-interaction protocol. That is an evaluation choice, not a claim that the two ratings express equal satisfaction.

The AP denominator deserves particular attention. Your code divides by **all relevant items**, rather than `min(K,R)` or the number retrieved. When `R > K`, even a perfect top-K list can have AP below one. Other implementations use different truncated-AP conventions. A metric called MAP@10 is therefore comparable only after checking its definition. Within this repository, the verifier intentionally reproduces the same full-relevant denominator.

Consider K=3, ranking `[a,b,c]`, and relevant set `{a,c,d,e,f}`. The hit pattern is `[1,0,1]`. Precision is `2/3`, recall `2/5`, F1 `1/2`, and reciprocal rank `1`. DCG is `1 + 1/log2(4) = 1.5`; ideal DCG is approximately `2.13093`, giving NDCG approximately `0.70392`. AP is `(1 + 2/3)/5 = 1/3`. This example separates membership, order, and relevant-set size in one calculation.

Moving the same two hits later leaves precision and recall unchanged but lowers MRR, NDCG, and AP. This is why the project is a ranking task, and why ranking quality cannot be reduced to classification accuracy. Likewise, a model that predicts raw ratings well need not place the right items near the top.

The aggregate F1 is the mean of individual F1 values. It is not computed by inserting aggregate precision and aggregate recall into the harmonic-mean formula. Each user gets equal weight, so an active user with many held-out interactions does not automatically dominate the report. This macro convention must remain the same across models, baselines, hybrids, and rerankers. [Source: accuracy implementation][accuracy].

### A3. Validation contracts and why invalid inputs stop evaluation

`metrics/_validation.py:validate_rankings` checks that K is a positive integer, users have nonempty external string IDs, and each list supplies K distinct valid item IDs. It evaluates the first K entries of a longer ranking. A too-short list or duplicate item is an error rather than a recommendation to silently pad the list. When a catalogue is supplied, recommended items must belong to it.

Accuracy additionally requires the relevance mapping to contain exactly the evaluation users and a nonempty valid relevant set for each. It converts each user's positives to a set, so repeated positive IDs do not enlarge the denominator. Beyond-accuracy evaluation checks that its users match the per-user accuracy users. These checks prevent seemingly harmless differences in coverage from becoming misleading model comparisons.

External IDs matter because RecBole assigns internal integer indices. The integer at a matrix position is not sufficient provenance across separately created datasets or checkpoints. Your metric layer sees original dataset tokens as strings. The adapters handle index conversion before evaluation, and `ScoreTable` carries split and partition identity along with scores.

There is a deliberate distinction between a **bad experiment input** and an **unavailable quantity**. Repeated recommendations, mismatched users, or nonfinite eligible scores should stop the experiment. Missing genre information can instead make a genre-dependent metric unavailable, with a reason and usable-user count. Returning zero in that case would falsely imply perfect calibration or no diversity. An empty subgroup similarly needs explicit unavailable values, rather than a fabricated mean.

These checks are useful but finite in scope. A valid string can still identify the wrong movie, and aligned score tables can still have been trained with leaked labels. Contract validation establishes structural consistency; provenance checks and experimental review establish whether the values were legitimately produced. Understanding that boundary is more useful than treating a passing validation method as a complete proof of correctness. Review must establish that each saved result answers the intended task using evidence available when those recommendations would actually be produced. [Sources: input checks][metric-validation], [score contracts][contracts].

### A4. Beyond accuracy: eight families with different units and meanings

`evaluate_beyond_accuracy` produces per-user metrics where meaningful, cohort-wide statistics where necessary, availability records, recorded-gender comparisons, and item-group exposure summaries. The eight families answer different questions; increasing one does not establish improvement in all the others.

**Intra-list diversity, ILD**, averages Jaccard distance over unordered pairs of recommended movie genre sets. For sets A and B, distance is `1 - |A intersect B|/|A union B|`. At K=10 there are 45 pairs. Identical genre sets have distance zero; disjoint sets have distance one. ILD measures genre variety inside a list, rather than novelty, accuracy, or fairness. It is undefined for K=1 and unavailable when required genres are missing.

**Mean self-information** averages `-log2((training_count_i+1)/(training_interactions+catalogue_size))` across recommended items. The added one is Laplace smoothing, giving zero-count catalogue items finite novelty. Less popular training items receive larger values. This is a popularity-based novelty proxy: it does not know whether a particular user has heard of the movie, and it does not measure surprise conditional on individual taste.

**Genre miscalibration** compares the user's training genre profile `p` with recommendation distribution `q` using `KL(p || 0.99q+0.01p)`, with natural logarithms. Each movie contributes one total unit of genre mass, divided across its genres. A movie with four labels therefore does not contribute four times as much profile evidence as a single-genre movie. The mixture avoids infinite divergence when the list lacks a genre present in the profile. Lower values mean closer genre proportions, rather than better relevance.

**Recorded-gender unfairness, GRU**, reports F and M user counts, group means for all six accuracy metrics, absolute gaps, and signed F-minus-M gaps. A zero gap says the groups have equal measured mean quality under this definition. Both groups could still receive poor recommendations. Unknown recorded gender is counted explicitly and is not silently reassigned. This dataset-specific binary comparison is narrower than a general account of user fairness.

**Catalogue coverage** is the number of distinct recommended movies divided by the evaluation catalogue size. It is a union across users' lists, not the average fraction of a catalogue in an individual top ten. Recommending a new item once can raise coverage while barely changing the overall concentration of exposure.

**Exposure Gini** measures inequality of accumulated discounted item exposure across the entire catalogue, including movies receiving zero exposure. Each appearance at rank `r` contributes `1/log2(r+1)`. Sorting the resulting exposures and applying the Gini formula gives a cohort statistic. Excluding zero-exposure items would hide part of the concentration the project is trying to measure. Lower Gini means more even exposure under this definition, not necessarily exposure proportional to relevance or item quality.

**User popularity deviation, UPD**, is base-two Jensen–Shannon divergence between the user's training and recommendation head/middle/tail distributions. It uses unweighted item occurrences in those profiles, rather than rank discounts. It is the divergence, not its square root. Lower values mean popularity proportions better match the user's history. A user who mostly consumes popular movies can be well calibrated by another popular list; forcing every user toward tail items is a different objective.

**Item-group exposure** reports each popularity band's item count, total discounted exposure, share of all exposure, and exposure per item. These views cannot be substituted for one another. A large group can receive substantial total exposure but very little per movie. Equal total exposure across differently sized groups would not mean equal exposure per movie.

The code averages available per-user values and records how many users contributed. Coverage, Gini, and exposure are recomputed for the supplied cohort. They cannot be reconstructed by averaging subgroup Ginis or subgroup coverages. Always read availability beside a metric: a high value over a small metadata-complete subset may answer a different question from the full-cohort accuracy result. [Sources: formulas][beyond], [statistics helpers][data].

### A5. Training profiles, grouping decisions, and leakage boundaries

Your committed training-statistics helpers build histories, item counts, genre distributions, popularity distributions, activity terciles, and popularity bands. Caio and Victor retain ownership of metadata preparation, including reading the item `class` and user `gender` columns. Those loaders are now merged; older documentation describing metadata as wholly absent is stale.

Activity groups use linear one-third and two-third quantiles of training interaction counts. Equal counts stay together, so equal-sized groups are not guaranteed. On the inspected saved run, boundaries are **26 and 69**, giving **321 low-, 310 medium-, and 312 high-activity users**. A zero-history user is assigned low by the reporting interface. These are data-derived boundaries, rather than universal constants.

Popularity bands are based on cumulative **training interaction shares**, not equal numbers of movies. Items are sorted by count; a complete tied-count bucket receives the band indicated by the cumulative total before that bucket. Crossing a 20% or 80% boundary can overshoot because ties are preserved. On this run, the catalogue contains **60 head, 485 middle, and 1,137 tail movies**. Calling that an equal-item tercile partition would be wrong.

All profiles, counts, and activity memberships use training interactions only. Item metadata itself can be known for a catalogue item without observing a held-out interaction with it. Using that static metadata is different from estimating popularity from validation/test occurrences. The crucial question is what information is fitted from outcomes, rather than whether a file happens to contain catalogue entries later encountered during evaluation.

The same boundary applies to ratings. The independent ranking protocol treats all held-out interactions as positives, while the regression hybrid uses raw ratings as training targets. A reranker cannot read a user's held-out ratings to improve their calibration profile. That would change the task from recommending with existing evidence to arranging recommendations using the answers.

Missing metadata is handled explicitly. The loader records pending reasons, metric availability reflects incomplete inputs, and reranking can mark a method unavailable for the cohort. In PR #24, required metadata is checked for the whole supplied candidate pool. Silently dropping users or inconvenient items would change the experiment and make apparent gains difficult to interpret. [Source: training statistics][data].

#### Figure A1. What the training popularity groups actually contain

<figure class="report-figure" id="figure-a1"><a href="figures/training-popularity.svg"><img src="figures/training-popularity.svg" alt="Cumulative training interaction share by sorted catalogue share, beside item and interaction shares for head, middle and tail groups."></a></figure>

**Read the figure.** The left curve sorts movies from most to least frequent in training. Its horizontal axis is the fraction of catalogue items encountered, and its vertical axis is the fraction of training interactions accounted for. The shaded regions are the existing head, middle, and tail memberships. Dashed lines mark the intended 20% and 80% cumulative boundaries. Complete count ties explain why the actual boundaries cross those targets slightly.

The right panel makes the denominator difference visible. The 60 head movies are only **3.6% of the catalogue**, yet account for **20.6% of training interactions**. The middle contains 485 movies and 60.1% of interactions; the tail contains **67.6% of movies but only 19.3% of interactions**. These are neither equal movie groups nor exactly equal interaction percentages.

**Interpretation.** A popularity-oriented recommender has much more historical evidence for a small part of the catalogue. This motivates reporting novelty, coverage, and exposure alongside accuracy. It also explains why equal total exposure for head and tail would give very different exposure per movie: the tail contains many more movies. The curve describes training observations; it does not show user satisfaction, causal popularity bias, or any reranker's effect. A recommendation-exposure curve would be a different measurement.

**Origin.** This is a new visualization of the already saved training counts and popularity memberships from the 22 October run. No groups were fitted using held-out interactions. Exact inputs and plotted values are in the [figure provenance record](figures/manifest.json).

### A6. Calibration reranking: a greedy construction of a balanced list

The merged implementation from PR #24 provides genre and popularity calibration in `rerankers/calibration.py`. A keyword-only `attribute` chooses which profile and divergence to use. The default is genres. Both variants build a list one position at a time and select the remaining candidate maximizing a combination of relevance and profile agreement.

For genre calibration, the objective is `(1-lambda)*normalized_relevance - lambda*smoothed_KL(p,q_after_addition)`. Popularity calibration uses the same shape with base-two Jensen–Shannon divergence. The algorithm evaluates the distribution of the **growing list including the candidate**, rather than rewarding each item independently. An individually relevant drama can become less desirable when the list already overrepresents drama relative to the user's history.

Relevance is min-max normalized separately within each user's candidate pool. This puts the relevance term on a bounded scale despite different models' raw score ranges. Constant score vectors map to zero. It does not make a unit of KL equivalent to a unit of JS, or guarantee equal trade-offs at the same lambda across models. The strength remains an empirical operating parameter.

The implementation precomputes each candidate's category distribution, maintains counts for the growing list, and tests every remaining candidate at each rank. This is a greedy search, approximately proportional to users times K times pool depth times the category work. It does not enumerate all possible ordered top-ten lists and cannot claim a globally optimal calibration-relevance solution.

Ties are resolved by the original model score and then external item ID. A set supplies remaining items, but the explicit tuple comparison removes dependence on set iteration order. At strength zero, the function directly returns `scores.top_k(k)`, reproducing the original ranking exactly, including ties. This supplies an interpretable control and avoids introducing changes merely from invoking a reranker.

At strength one, divergence dominates the greedy objective, with original score still resolving exact objective ties. Even then, calibration is limited by the candidate pool and the attainable discrete genre proportions in a ten-item list. A profile asking for 30% of a rare genre cannot be satisfied if that genre never enters the pool. Calibration also preserves historical preferences, which may themselves reflect previous exposure constraints. Matching history is a specified measurement objective, rather than proof that the history is ideal. [Sources: calibration][calibration], [normalization and checks][rerank-common].

### A7. Item fairness: exposure allocation across users

The item-side method in PR #24 differs fundamentally from independent per-user calibration. It maintains one exposure map across the supplied evaluation cohort. At each rank, it visits users in sorted external-ID order and chooses an eligible remaining item maximizing `(1-lambda)*normalized_relevance + lambda/(1+accumulated_item_exposure)`.

After selecting an item at rank `r`, it increments exposure by `1/log2(r+1)`. Rank one for one user can therefore influence another user's first recommendation. The traversal runs in rank rounds, rather than finishing all ten positions for the first user before moving to the next. That choice limits one user's opportunity to consume low-exposure items before anyone else receives a recommendation.

Exposure begins at zero for each call. Otherwise, results would depend on which model or strength happened to run earlier. The state is local to the experiment and does not represent an operational history of real impressions. Fixed user ordering makes the experiment deterministic, but ordering can affect allocation; a different cohort or ordering can yield different lists for an otherwise identical user.

The reciprocal bonus favors items with less accumulated exposure. It is **not the marginal change in catalogue Gini**, and it does not mathematically guarantee equal exposure, minimum inequality, or a specified head/tail allocation. Gini is a downstream measurement used to choose a strength. Separating the heuristic from its evaluation prevents an attractive formula from being mistaken for an optimization guarantee.

The context parameter is retained to fit the common interface, although this method's current objective does not consume genre profiles or user groups. It is item fairness under a particular exposure-inequality definition, whereas user fairness asks about quality across groups of people. A redistribution that broadens item reach could help, harm, or leave unchanged the measured F/M quality gap. Both outcomes need evaluation. [Source: item fairness][item-fairness].

### A8. Candidate depth and strength selection are experimental decisions

The reranking runner first verifies the complete score table against the reconstructed checkpoint split and the exact eligible catalogue. Only then does it retain the user's highest-scoring **50** candidates and return K=10. Fewer than fifty is allowed when every eligible item is used, but fewer than ten is rejected. Validation excludes training history; test excludes both training and validation history.

Depth fifty is a bounded-cost, common-pool design choice. It was not prescribed by the brief and was not established as optimal by a depth-sensitivity experiment. It places an upper bound on what any reranker can retrieve. Exposure improvements inside this restricted pool do not demonstrate redistribution over all catalogue possibilities. Candidate depth must accompany every interpretation of the result.

Each model/method sweeps strengths `{0,0.25,0.5,0.75,1}` on validation. For the three implemented variants, choose the lowest target metric among points satisfying `validation_NDCG >= 0.95 * original_validation_NDCG`. Tie-breaking prefers higher NDCG and then lower strength. Zero is always inserted into the grid, providing the unchanged-ranking fallback when the target is available.

The **5% relative validation-loss budget** is also a chosen operating point. If original validation NDCG is 0.30, the threshold is 0.285, not 0.25. It is relative loss, not five NDCG percentage points. The policy expresses a preference for limited measured accuracy loss in exchange for a lower mismatch or inequality metric; it does not prove that 5% is the best trade-off for users.

Selection is per model and per objective. Genre, popularity, and exposure runs are separate alternatives. They are not chained and do not jointly optimize all eight beyond-accuracy families. A combined method would need an explicit objective or ordering policy, additional choices, and another validation process. Task 3.3's hybrid/reranker order comparison is a different question from combining several reranking objectives.

Once selected, the strength remains frozen on test. A validation constraint does not guarantee the same test loss, and a test outcome exceeding 5% is retained and flagged. Choosing another strength after seeing that outcome would turn test into another development set. [Source: candidate and selection runner][rerank].

#### Figure A2. Reading the original validation trade-off curves

<figure class="report-figure" id="figure-a2"><a href="figures/validation-tradeoffs.png"><img src="figures/validation-tradeoffs.png" alt="Six model panels showing validation NDCG change against reduction of genre mismatch, popularity deviation, or exposure Gini; stars identify selected strengths and a dashed vertical line marks minus five percent NDCG."></a></figure>

**Read the figure.** Each panel is one base model. A point is an evaluated strength, and the connecting lines follow the strength sweep. Horizontal position is `100*(reranked_NDCG/original_NDCG-1)`, so negative values mean accuracy loss. Vertical position is the percentage reduction in that method's target metric. Blue means genre mismatch, orange popularity deviation, and green exposure Gini. The stars identify the validation-selected settings.

The dashed vertical line marks **−5%** NDCG change. Feasible choices lie on or to its right; the largest target improvement outside that region cannot win under the declared policy. For a given method, movement upward and rightward is preferable, but the colors represent different quantities. An orange reduction of forty percent and a green reduction of five percent do not quantify comparable amounts of user benefit.

**Interpretation.** Stronger genre/popularity calibration can reduce mismatch much further than the selected stars, but often only at substantial accuracy cost. The experiment therefore reports a constrained operating point rather than the largest attractive percentage. ItemKNN's green star lies slightly to the right of zero: its selected exposure variant improves validation NDCG a little, while redistributing exposure. That is a result of this measured combination, not a general guarantee from the heuristic.

The panels use different horizontal ranges, so compare tick values rather than apparent distances. These curves show development choices on one split; they are not training-loss curves, confidence intervals, or evidence that the validation trade-off will transfer unchanged to test.

Use the selected strengths recorded in the stored files, since visual proximity alone cannot replace the explicit validation selection rule.

**Origin.** This is the existing `validation-tradeoffs.png`, copied without alteration from the saved reranking bundle. The [original plotting implementation][rerank-plots] defines both percentage formulas and the selection stars.

### A9. Provenance, saved outputs, and what the safeguards establish

`rerank.py:source_identity` binds the choice to the checkpoint hash, both score exports, saved split identity, dataset files, relevant source code, K, candidate count, and accuracy-loss budget. Test evaluation requires a completed validation `selection.json` with matching identity and settings. It rejects a fresh strength grid. Hashing test scores establishes input identity; it is not itself test-based model selection.

The six-model script completes every validation sweep before starting test evaluation. Each point saves rankings, relevance, aggregate and per-user accuracy, beyond-accuracy measures, group interfaces, and a manifest. The final test folders follow the shared comparison format, so Caio's comparison command can read original and reranked outputs without implementing another set of formulas.

This is a practical separation of algorithm, orchestration, and evidence. The pure reranker returns lists. The runner checks inputs and writes one experiment. The batch script coordinates models. The plotting script visualizes saved summaries without loading checkpoints, while `verify_reranking.py` independently recalculates accuracy, Gini, and coverage from saved lists and checks selections and masks.

The earlier 24 October audit of the 22 October bundle passed **120 evaluations, 113,160 recommendation lists, 18 validation selections, 24 final comparison rows, and 22,632 per-user comparison rows**. All **1,237 artifact checksums** matched. These remain historical checks of that specific bundle. The current refresh additionally rechecks the locally saved weighted reconstruction and the new diversification selections, with the exact scope recorded in `verification.json`. Updating a document does not retroactively rerun its old experiments.

These checks support the integrity and arithmetic of the saved experiment. They do not establish online user satisfaction, statistical significance, fairness under another definition, or an untouched test design. Similarly, `status: complete` for one evaluation folder means that process finished; `project_evaluation_complete: false` correctly preserves outstanding project work.

There is still an integration boundary around open PR #26. Master now uses `write_ranking_results` from the merged reranker work, while the group branch introduces `compare_groups(rankings,relevant,statistics,k)` through an older evaluator layout. Its complete report lacks the pending report's top-level `reason`, which existing reranker consumers expect. A read-only merge-tree check confirms an evaluator conflict. Resolving it requires updating the shared writer and its consumers consistently; passing tests on each branch separately does not prove the combined system works. The maximizing directions and expanded context merged through PR #28 must survive that integration. [Sources: runner][rerank], [shared writer][evaluation], [PR #26][pr26], [PR #28][pr28].

### A10. The recorded six-model calibration/exposure results

The saved 22 October experiment uses seed 42, 943 users, 1,682 items, and 60,734/19,633/19,633 train/validation/test interactions. It retrains the six checked-in configurations without a new model-tuning search. Those configurations differ from some OOF and alternate-hybrid component fits discussed in Part B. Each reranked row must be compared with its own original model.

| Model | Original test NDCG | Genre strength / test loss | Popularity strength / test loss | Exposure strength / test loss |
| --- | ---: | ---: | ---: | ---: |
| ItemKNN | 0.363396 | 0.50 / 5.80% | 0.75 / 4.32% | 0.50 / 0.12% |
| UserKNN | 0.368776 | 0.50 / 5.27% | 0.75 / 4.00% | 0.75 / 6.45% |
| BPR | 0.354497 | 0.50 / 3.77% | 0.50 / 2.81% | 0.25 / 2.44% |
| SLIMElastic | 0.403266 | 0.50 / 5.54% | 0.75 / 3.79% | 0.50 / 4.49% |
| EASE | 0.404872 | 0.50 / 6.73% | 0.75 / 4.31% | 0.50 / 5.60% |
| FISM | 0.348765 | 0.50 / 4.80% | 0.50 / 3.47% | 0.25 / 2.74% |

Genre mismatch fell by **35.00–38.70%**, popularity deviation by **20.10–42.44%**, and exposure Gini by **2.48–7.30%**. These reductions use different metrics and cannot be compared as a common unit of benefit. All eighteen choices met the validation budget; six exceeded it on test, with a maximum test loss of 6.73%.

ItemKNN exposure reranking is an instructive example: test NDCG changes from 0.363396 to 0.362948, approximately 0.12% loss, while catalogue coverage rises from 11.59% to 25.98%. Yet Gini falls only from about 0.966588 to 0.938365. Reaching many more distinct movies and achieving roughly equal exposure are separate properties.

UserKNN exposure obtains the largest relative Gini reduction, 7.30%, but loses 6.45% test NDCG. EASE genre calibration substantially improves the targeted mismatch measure while suffering the largest accuracy loss. Neither result is an implementation failure merely because it reveals a cost; the purpose of Task 3.2 is to measure and explain such costs.

The defensible claim is that these specific greedy methods can improve their chosen beyond-accuracy objectives within common top-fifty pools, at model-dependent accuracy costs. It is a single-seed offline experiment, with no demonstrated significance or uniform subgroup benefit. Your explanation should end at that evidence boundary, while identifying sensitivity studies and group analysis as the next ways to test robustness. [Source: fresh PR experiment and local bundle][pr24].

#### Figure A3. What happened to the frozen choices on test

<figure class="report-figure" id="figure-a3"><a href="figures/test-effects.png"><img src="figures/test-effects.png" alt="Six model panels comparing grey bars for test NDCG change with colored bars for target metric reduction, with selected strengths annotated and a minus five percent reference line."></a></figure>

**Read the figure.** Each group of two bars is one method applied to one model. The grey bar is the relative change in test NDCG: below zero means worse ranking accuracy. The colored bar is the reduction in that method's test target: above zero means less mismatch or less exposure inequality. Text above the bars records the strength already chosen on validation. Nothing in this figure chooses a new strength.

**Interpretation.** The large positive blue/orange bars alongside negative grey bars make the central trade-off explicit. Genre calibration reduces its target substantially, yet EASE loses approximately **6.73%** NDCG. UserKNN exposure reduces Gini by approximately **7.30%**, at **6.45%** NDCG loss. ItemKNN exposure has a much smaller grey loss, approximately **0.12%**, showing that the cost depends on the base model.

The dashed horizontal line at −5% is a reference to the validation budget, not a test feasibility rule. Six of the eighteen frozen choices fall below it on test. Those outcomes are retained and reported; moving them back above the line by test-based reselection would invalidate the procedure. The figure therefore shows a limit of validation transfer as well as successful target reductions.

Panel heights and tick ranges differ. Colored reductions use different target metrics, and the bars have no uncertainty intervals. Their size cannot establish significance, equal benefits for user groups, or improved satisfaction. The accompanying coverage example also matters: reaching more distinct movies can coexist with a high remaining Gini.

**Origin.** This is the existing `test-effects.png` from the same fresh experiment, copied without alteration. Its underlying lists and selections passed the saved-result audit described in A9.

### A11. Follow the information flow before judging a result

<figure class="report-figure" id="figure-a4"><a href="figures/evaluation-flow.svg"><img src="figures/evaluation-flow.svg" alt="Training data builds models and profiles; validation measures and selects settings; a freeze separates development from final test measurement, with no test-label feedback arrow."></a></figure>

**Figure A4 — A map of permitted information.** The green training lane supplies fitted model parameters, training histories, popularity counts, and user profiles. The validation lane measures candidate rankings and selects checkpoints or operating parameters. The final lane uses frozen choices to measure test outcomes. The picture describes responsibilities and information use; it does not claim that every stage is one Python function or that all algorithms train in the same way.

Think through one user. Their training interactions help the model learn scores and provide evidence for the calibration profile. Their validation interactions tell us how well a proposed setting ranks held-out items. They may select a checkpoint or strength, but they do not get added to that training profile in the recorded experiments. When evaluating test, those validation items join the history mask so they cannot be recommended again. That masking change is not a train-plus-validation model refit.

The same distinction explains why exporting test scores can be acceptable while selecting a setting using test NDCG is not. A score file contains model predictions. Selection needs outcome information and a decision rule. The report must therefore establish which labels influenced a decision, rather than infer leakage merely from a filename or timestamp. Conversely, a manifest flag saying `test_evaluated: false` is not proof if another stage already used test outcomes.

The freeze is concrete. For native reranking it is the completed validation selection bound to input/settings identity. For Task 3.3 it is the saved method/order choice made after all validation points and before test outputs. Hashes catch changed inputs; they cannot establish that a person never inspected earlier test results. That is why the reused-split limitation remains even when every file matches.

Your integration review should follow the entire chain: source checkpoint and split, complete eligible candidates, training-only context, declared selection direction, frozen choice, and saved final lists. A metric value is trustworthy only for the task that chain actually implements. This is also why the shared writer is useful: native, reranked, and portable-hybrid rankings can use the same definitions while retaining different declared candidate scopes.

### A12. A numerical calibration step makes the objective concrete

<figure class="report-figure" id="figure-a5"><a href="figures/calibration-step.svg"><img src="figures/calibration-step.svg" alt="A toy user profile with 75 percent genre A and 25 percent genre B, comparing another A recommendation with a B recommendation; smoothed KL and the combined objective favor B at strength one half."></a></figure>

**Figure A5 — A worked example, not a MovieLens result.** Suppose the user's training profile is 75% genre A and 25% genre B. The growing list already contains an A-only movie. We now compare another A-only candidate, x, with normalized relevance 1.0, against a B-only candidate, y, with relevance 0.8. Those relevance values are already normalized within the candidate pool; do not normalize just the two displayed alternatives again.

Appending x gives recommendation proportions `(1,0)`. Appending y gives `(0.5,0.5)`. Neither exactly matches the profile, but the second avoids entirely omitting the minority genre. The smoothed recommendation distributions become `(0.9975,0.0025)` and `(0.5025,0.4975)` respectively. Using natural-log KL from the profile to those mixtures gives approximately **0.937408** for x and **0.128325** for y.

At strength 0.5, x's objective is `0.5*1.0 - 0.5*0.937408 ≈ 0.031296`. Candidate y's is `0.5*0.8 - 0.5*0.128325 ≈ 0.335838`. The greedy step chooses y. It sacrifices some model relevance to make the list's composition closer to the user's history. This is the mechanism behind the aggregate mismatch reductions in A10, rather than a claim that genre B is always the right second recommendation.

The calculation also clarifies several implementation choices. The divergence is computed for the growing list after adding a candidate. It is not an independent genre bonus attached permanently to each movie. The 0.01 profile mixture keeps missing profile genres from causing infinite KL. A multi-genre movie would divide its one unit of mass across its genres, so its contribution would have to be computed before forming the new list distribution.

At zero strength, the explicit baseline branch returns the original top K. At greater strength, a changed order is intentional. Correctness therefore means reproducing the stated objective, tie rules, eligibility, and profile policy; it does not mean guaranteeing that NDCG rises. Finally, the arithmetic is one local greedy decision. It neither proves that the completed list is globally optimal nor establishes that preserving a historical profile is always desirable.

### A13. Exposure fairness changes one user's answer through other users

<figure class="report-figure" id="figure-a6"><a href="figures/exposure-round.svg"><img src="figures/exposure-round.svg" alt="Two users both prefer item a to b. With pure exposure bonus, user one picks a; accumulated exposure then makes user two pick b. Strength zero would recommend a to both."></a></figure>

**Figure A6 — Trace the state, rather than just the formula.** Consider users `1` and `2`, each with candidates a and b, with a ranked higher by the original model. Set K=1 and strength=1. At the start, both items have exposure zero, so both have bonus one. User `1` chooses a through the original-score tie-break. Rank-one exposure adds one to a's counter.

User `2` now sees bonus `1/(1+1)=0.5` for a and `1/(1+0)=1` for b, and chooses b. The final exposure vector is `(1,1)`. With strength zero, both users would receive a. The example shows how a cohort-level mechanism can change a recommendation even when that user's own scores and profile have not changed.

Real runs use rank rounds: every user receives a rank-one choice before rank-two choices begin. Later exposure increments are discounted by `1/log2(rank+1)`. Original model scores and external item IDs resolve objective ties. Sorting external IDs means lexical string order, so identifiers such as `10` need not follow `2` numerically. Determinism makes the chosen policy reproducible; it does not make user ordering inconsequential.

This state dependence is the key to understanding the new switching-order experiment in B11. Reranking two components independently maintains two exposure histories, including lists later discarded by routing. Reranking the final switched cohort maintains one history. Both can faithfully implement the exposure heuristic and still produce different lists. User-local calibration has no such cross-user counter, which permits exact equality under the recorded fixed-routing conditions.

The same reasoning matters when reviewing Gabriel's merged coverage diversification. That method tracks whether an item has ever been recommended, rather than accumulating discounted exposure. Once a movie is in the covered set, another appearance no longer adds coverage; it can still add considerable exposure. A breadth objective and an equality objective therefore need separate implementations, selection targets, and interpretations.

As the integration owner, check the operation being optimized as well as its direction. Your mismatch and Gini targets are minimized. PR #28's ILD, self-information, and coverage targets are maximized. A sign error could select a perfectly valid file with the wrong scientific meaning. The frozen-test path must also mark a newly introduced method unavailable when an older selection does not contain it; adding methods must never silently initiate test-time tuning. Review these contracts whenever a teammate extends the registry.

## Part B — The other contributions and the shared system

### B1. Project objective, ownership, and the adopted protocol

The project studies offline top-ten recommendation on MovieLens-100K. It compares individual collaborative recommenders, several forms of hybrid recommendation, and post-processing methods that alter accuracy and beyond-accuracy properties. The central scientific question is how these design choices change recommendation behavior under a common evaluation procedure, rather than simply which file prints the highest score.

The planned division remains Gabriel for individual models and diversification; Bogdan for hybrid methods, their tuning, coefficients, and order experiments; Caio for shared comparisons, baselines, metadata collaboration, and extensive review; Victor for explanatory/group analysis, user fairness, and report coordination; and you for the work in Part A. Ownership is an assigned responsibility, not proof that a particular student alone wrote every supporting helper.

The current `protocol.yaml` specifies MovieLens-100K, seed 42, a random per-user 60/20/20 split, full-sort evaluation, K=10, and **validation NDCG@10** for selection. The primary selection metric changed from historical MRR@10 during the EASE/SLIM retuning review. Older narrative notes still mention MRR; they do not override the current configuration or a frozen experiment's explicitly recorded selection rule.

Actual split sizes are **60,734 training, 19,633 validation, and 19,633 test interactions**. RecBole allocates held-out fractions within each user's history, with rounding leaving the remainder in training. This is why nominal 60% does not give exactly 60,000 training rows. There are 943 users and 1,682 nonpadding catalogue items.

Random order, `RO`, shuffles rather than respecting time. The resulting task is held-out interaction recovery under that split; it is not a simulation of predicting each user's future from their past. Replacing it with timestamp order would change the experiment and require new training and artifacts. A timestamp column in an exported CSV does not itself make an evaluation temporal.

Every held-out interaction is relevant, regardless of rating. That supplies enough positive evidence for ranking evaluation and matches the adopted implicit-feedback interpretation. It also means low-rated observed movies are positives. Missing interactions are unknown outcomes, even though training algorithms may sample them as negatives. Those sampled labels are a modeling approximation rather than observed dislike.

Validation masks training interactions; test masks both training and validation interactions. Most comparisons retain the full catalogue, including items with no training interactions. Switching uses a separate warm-item scope, so it requires a matching warm control. Split, scope, relevance, K, and aggregation together define the task; a model name and dataset name do not. [Sources: current protocol][protocol], [assignments][assignments].

### B2. Code structure and the path from a dataset to a report

Reusable project code lives under `project/recsys/`; RecBole remains the model-training framework. YAMLs define experiments, scripts coordinate particular runs, and report files explain their evidence. Framework changes are limited to cases such as FISM correctness and SLIM solver configuration that cannot be solved solely in a project wrapper.

```text
dataset + model YAML + shared protocol
                 |
       recbole_adapter.train
                 |
  selected checkpoint + exact splits + full score tables
                 |
       individual / hybrid scores
                 |
  original rankings or PR #24 reranking
                 |
  independent accuracy + beyond-accuracy + group reporting
                 |
  comparable test folders -> tables / figures / interpretation
```

The shared `ScoreTable` holds `model`, `split_id`, `partition`, and `scores[user][item]`. Scores cover every eligible candidate, rather than just each component's top ten. `require_aligned` checks split/partition, evaluation users, and each user's candidate IDs before fusion. Equal array dimensions would not detect swapped users or different candidate catalogues.

`recbole_adapter.train` loads the model YAML first and shared protocol second, preventing a model file from silently replacing shared split settings. It exports actual in-memory train/validation/test rows with external IDs, ratings, and timestamps. A SHA-256 digest includes partition names and CSV bytes. Copied inputs, resolved configuration, checkpoint location, and a manifest explain what produced a run.

The selected checkpoint is reloaded before full score export. Both validation and test scores can be stored without calculating test performance. Keeping test labels out of selection is the experimental boundary; the mere existence of a test score file is not equivalent to choosing a configuration on test.

Independent checkpoint evaluation loads on CPU, reconstructs data from the checkpoint's configuration, and verifies reconstructed split bytes and digest against saved exports. It batches full-sort prediction and converts internal indices back to original tokens. The checkpoint's frozen configuration governs reconstruction; today's changed YAML cannot relabel an old model as a new experiment.

There are two inference paths. Native training artifacts use RecBole checkpoints and the adapter. The alternate-hybrid package uses portable NumPy/sparse weights and JSON through `HybridBundle`, avoiding machine-specific checkpoint pickles for replay. Both use shared evaluation settings so candidate-policy differences remain visible to the comparison tool.

`new_run` refuses an existing destination, protecting previous evidence from overwrite. Manifests distinguish running, failed, and complete processes, and JSON writing rejects NaN. These are small explicit helpers rather than a large experiment framework. The optional `task1`, `task2`, and `task3` orchestrators can remain placeholders while working task-specific scripts exist. Conversely, an unimplemented required algorithm cannot be excused merely because an orchestration placeholder was optional. [Sources: adapter][adapter], [contracts][contracts], [artifact helpers][artifacts], [portable bundle][bundle].

### B3. Individual recommenders: six models, different assumptions

Gabriel's individual-model layer exposes ItemKNN, UserKNN, BPR, SLIMElastic, EASE, and FISM through a common score export. Sharing outputs lets the rest of the project compare and combine models without depending on their internal training APIs.

**ItemKNN** constructs item similarities from the interaction matrix and scores candidates using a user's history. The neighborhood size limits contributing similar items; shrinkage regularizes weak similarity evidence. **UserKNN** instead uses similar users' interactions. In this fork, both are instances of the `ItemKNN` class, selected by `knn_method: item` or `user`. The exported label must therefore preserve their distinct identities. A table reporting both merely as the class name would be ambiguous.

**BPR** learns user and item embeddings with pairwise training. A positive observed item should score above a sampled unobserved item for that user; the loss penalizes failures of that ordering. The score is a user-item dot product. Embedding size, learning rate, training budget, negative sampling, and checkpoint selection all influence the fit. A ranking-oriented training loss is useful but still does not directly optimize the finite top-ten NDCG statistic.

**EASE** is a linear item model with a closed-form regularized solution and a zero diagonal. It reconstructs interactions from other item interactions. The diagonal restriction prevents simply copying an item's own input. Regularization controls coefficient magnitude and sensitivity to sparse evidence. This constructor fits the model directly, so repeated neural-style epochs would not represent repeated independent learning progress.

**SLIMElastic** fits a sparse linear item model through per-target elastic-net regression. The inspected configurations constrain coefficients to be nonnegative and hide the target item's input while fitting its column. Alpha and the L1 ratio control total regularization and sparsity. Solver convergence matters: a poorly converged fit is not a fair representative of its hyperparameters. The project exposes a 1,000-iteration budget for retuning rather than relying on the earlier hard-coded 100.

**FISM** learns source and destination item embeddings, builds a user representation by summing history embeddings, scales by history length raised to a negative exponent, and scores a target with the destination embedding plus biases. The candidate must be removed from its own source history, including repeated occurrences, before both the sum and length normalization are used.

Two FISM corrections are central implementation decisions. `BCEWithLogitsLoss` receives raw logits because it applies the sigmoid internally; adding sigmoid beforehand changes the loss and gradients. Target self-contributions are also removed in both pairwise-item and full-sort computation. Excluding seen items only at evaluation would not repair training that learned to exploit the target itself. Tests compare scoring and gradients across paths, making the correction stronger than a superficially improved benchmark number.

Checked-in configuration files are a reproducible starting set, not a universal best-model list. For example, the generic FISM YAML and the final OOF experiment's separately tuned FISM settings differ. Reading a YAML today does not identify every historical fit named FISM. [Sources: individual configurations][model-configs], [framework models][framework], [FISM correction][fism].

### B4. Tuning, frozen experiments, and why similarly named results differ

Current EASE/SLIM retuning uses nine EASE regularization values and a 25-setting SLIM alpha/L1-ratio grid. Selection maximizes independent validation NDCG@10; MRR and recall remain diagnostics. Nonconverged SLIM fits cannot win. Old parameters are refitted with the same solver budget, providing a controlled comparison of the hyperparameter change.

EASE selects regularization **500**, compared with old 250. Validation NDCG changes from **0.297866 to 0.298099** while MRR decreases slightly. SLIM selects **alpha 0.2 and L1 ratio 0.005**, with validation NDCG **0.301125**, compared with **0.300154** for the old ratio 0.01. These small, metric-specific changes illustrate why declaring the selection metric before looking at results matters.

The project retains older 80/10/10 tuning records and transfers some approved hyperparameters into later 60/20/20 experiments. Transferring a parameter is different from transferring predictions: scores and training must still belong to the new split. Retaining the history makes the limitation visible, but does not turn transferred settings into an exhaustive current-protocol optimum.

The OOF hybrid uses fixed-budget full-training components; PR #24 uses selected checkpoints from the checked-in individual configurations; the alternate-hybrid bundle contains additional corrected inherited fits and jointly tuned cascade components. Thus, three tables can show different BPR or FISM numbers without any arithmetic contradiction. A meaningful comparison needs the source run, parameter values, training budget, checkpoint rule, normalization, candidate scope, and held-out membership.

Source corrections also require redoing dependent calculations. The final OOF update regenerates FISM's folds and full fit while verifying the five unchanged component columns. Alternate-hybrid corrections replace inherited BPR/UserKNN fits and redo downstream selection. Reusing cached outputs after changing the generating code would preserve numbers while losing their meaning.

Cached stages use source/config/data identity and hashes, atomic completion markers, and run locks. Resume is allowed only for compatible completed work. This makes expensive cross-fitting reproducible and prevents a partial artifact from masquerading as a completed stage. It also means an exact historical rerun may require original local caches that are not committed. [Sources: retuning report][retuning], [OOF reproduction notes][oof-doc].

#### Figure B1. What the recorded tuning search says about regularization

<figure class="report-figure" id="figure-b1"><a href="figures/individual-tuning.svg"><img src="figures/individual-tuning.svg" alt="EASE validation NDCG against logarithmic regularization, with previous and selected settings marked, beside a five by five SLIM alpha and L1 ratio heatmap."></a></figure>

**Read the figure.** The EASE curve shows all nine declared regularization settings, using a logarithmic horizontal axis. The square marks the previous setting, 250; the star marks the NDCG-selected setting, 500. The SLIM heatmap shows all twenty-five alpha/L1-ratio combinations. Brighter cells have higher validation NDCG, and the outlined cell is the recorded selection. Rows and columns are the tested categorical grid positions, not a continuous optimization surface.

**Interpretation.** EASE improves markedly from very weak regularization, then flattens around the better settings. Increasing regularization indefinitely does not continue improving the score. The small separation between 250 and 500 is real but modest: NDCG changes from about 0.297866 to 0.298099. That small gain should not be described with the same emphasis as the large difference from the weakest settings.

SLIM selects alpha 0.2 and L1 ratio 0.005. The very poor lower-right cell demonstrates that plausible hyperparameter ranges can also include a badly performing combination. It does not prove that every increase in either parameter harms performance: the rows have different patterns. Rounded cell labels and saturated colors can hide small differences, so exact selection uses the stored full-precision values.

All SLIM fits converged, and both searches share the declared protocol. These are validation measurements used for selection, not independent evidence of test gains. The figure visualizes existing trial records; neither models nor hyperparameters were fitted again for this report.

**Origin.** The chart reads the nine EASE and twenty-five SLIM JSON trial records plus their NDCG-based selections under `individual-retuned-60-20-20/`.

### B5. The weighted hybrid: out-of-fold rating regression

Bogdan's weighted hybrid learns how to combine the six component score columns using Ridge regression. The target is each observed training interaction's **raw 1–5 rating**. Unrated candidates are not assigned zero ratings. The final prediction is an intercept plus a weighted sum of normalized component scores, subsequently sorted to obtain a ranking.

To avoid training the combiner on in-sample component predictions, outer-training interactions are divided into five user-stratified folds. For each fold, components train on the other folds and predict the held-out interactions. Repeated user-item pairs stay together, and users retain training evidence. Concatenating those predictions gives one out-of-fold feature row per outer-training interaction: 60,734 rows in the final experiment.

Fold components use fixed training budgets without held-out-fold early stopping. Otherwise, the labels meant to provide out-of-fold targets would also select the component checkpoint. Outer validation chooses Ridge's penalty; outer test is reserved for the frozen combination's later evaluation. This is cross-fitting, rather than fully nested tuning of every component hyperparameter within each fold.

Each component is z-scored separately over the user's eligible candidates. Constant columns map to zero. That removes arbitrary score location and scale before combination while using predictions rather than target labels. The transform is recomputed on eligible test candidates after validation-history masking. It is a declared candidate-relative transform, not a scaler fitted using test ratings.

The Ridge objective trades squared rating error against squared coefficient magnitude; the intercept is unpenalized and coefficients are unconstrained. Alpha **30,000** is selected from the final 24-penalty validation search. Its numerical size is meaningful only with the target units, feature scaling, row count, and solver objective.

An earlier data-loader behavior normalized ratings to 0–1 despite raw-rating descriptions. The correction recovers raw ratings by external user-item pairs. Multiplying normalized targets by four and adding one correspondingly scales coefficients and adjusts the intercept; the recorded correction verifies unchanged rankings for that affine unit transformation. The final report supersedes earlier alpha-100 and alpha-100000 provisional snapshots.

The original final corrected hybrid has validation NDCG **0.299155** and test NDCG **0.409095**; its retained SLIM component has test NDCG **0.401882** and corrected FISM **0.372549**. The test difference over that SLIM fit is about 1.80% relative. These are the original frozen results, retained in the historical comparison figures. The newly committed independent reconstruction reaches **0.409041** test NDCG, discussed in B11. Close reconstruction is different from an exact replay. The hybrid does not lead SLIM on validation, and no significance is claimed. The test split had already been inspected during earlier iterations, so neither table is a fresh untouched generalization estimate.

Known catalogue identities across folds make the design transductive. A rare item's identity can be known even when its fold-training interactions are absent. That is different from observing its held-out target, and different again from solving deployment cold start. [Sources: hybrid implementation][hybrids], [cross-fitting runner][oof], [final results][oof-results].

### B6. Five alternative hybrid styles and the questions they test

The implemented alternative styles live in dedicated modules; the generic `predict_other_hybrid` placeholder is not a complete map of available work. The portable package exposes their frozen fits through a common inference layer.

**Switching** chooses a whole UserKNN or BPR ranking per user. Training-only features describe history length and neighborhood support. A validation grid searches overlap, reliable-neighbor count, and history thresholds, including constant-model controls. It avoids cross-model score calibration because it does not blend scores. Its warm scope removes unsupported items from candidates and relevance together; this is an explicit change in task.

**Cascade** uses ItemKNN to retrieve a shortlist and BPR to order it. It jointly searches component settings and shortlist depth by final validation NDCG, rather than substituting independently optimal standalone settings. Twenty ItemKNN fits, twelve BPR fits, and eight depths yield 1,920 combinations with cached component scores. Shortlisting must follow the current partition's history mask. Missed retrieval positives remain in the relevance denominator. Candidate fraction estimates potential scoring volume; cached full-score experiments do not establish measured latency savings.

**Feature combination** concatenates frozen BPR user and item features and learns a two-hidden-layer ReLU MLP with pairwise ranking loss. It standardizes using training-supported entities, samples negatives excluding training positives, and selects checkpoints and width on validation. A metadata-input variant supplies a control. Three scorer seeds measure some training variability conditional on the same frozen BPR fit, rather than three independent dataset splits.

The **residual extension** adds an MLP correction to the frozen raw BPR score. Its output layer begins at zero, so epoch zero exactly reproduces the baseline and is eligible for validation selection. This makes it possible to retain the original model when learned corrections do not help. It is an extension of feature combination, not a sixth required hybrid taxonomy.

**Feature augmentation** adds normalized positive UserKNN predictions to unseen entries in the receiver's input matrix, while leaving binary observed targets unchanged. Strength controls inferred input mass. EASE and SLIM are retrained as receivers, with zero diagonal or target-column hiding retained. Because augmented input X differs from target R, EASE uses the constrained solution to `||R-XB||² + lambda||B||²`, rather than blindly applying the ordinary same-input-and-target shortcut. Adding inferred features is not declaring them observed target interactions.

**Meta-level** transfers BPR user embeddings into a user-neighborhood recommender. Positive cosine similarities define neighbors, self-similarity is excluded, and neighborhood size is selected on validation. Multiplying neighbor weights by the original training interaction matrix yields item scores. An interaction-cosine control changes only the representation, helping isolate whether learned embeddings improve the neighborhood signal. This transfers a learned representation rather than predictions as augmentation targets. [Sources: alternative implementations][alternatives], [portable replay][bundle].

### B7. What the hybrid evidence supports

The corrected alternative-hybrid report gives the following representative frozen test results. These rows explain particular experiment comparisons; they are not one globally interchangeable leaderboard with Part A's reranking table.

| Frozen model or variant | Scope | Validation NDCG | Test NDCG |
| --- | --- | ---: | ---: |
| EASE | full | 0.298099 | 0.404872 |
| Native SLIM | full | 0.301125 | 0.403266 |
| Paired SLIM receiver | full | 0.301112 | 0.403254 |
| UserKNN → SLIM, strength 0.25 | full | 0.304670 | 0.411904 |
| UserKNN → EASE, strength 0.25 | full | 0.298156 | 0.403747 |
| Jointly selected cascade | full | 0.278699 | 0.384593 |
| Switching | warm | 0.284915 | 0.371960 |
| Matching warm UserKNN control | warm | 0.283897 | 0.368763 |
| BPR-user-vector meta-level | full | 0.282216 | 0.375364 |
| Interaction-cosine control | full | 0.283823 | 0.369014 |

SLIM weak augmentation improves test NDCG by approximately **2.15% against its paired receiver**. Strength one improves less, so more transferred mass is not automatically better. EASE's slight validation improvement becomes a small test decline. Plain feature-combination test NDCG is about **0.345623 ± 0.009039**, below the corrected BPR control; residual CF reaches about **0.373233 ± 0.002061**, still slightly below that control's **0.373665**. These mean/sample-standard-deviation rows are summaries of three scorer seeds.

Switching's warm catalogue contains **1,615 items**, with **19,590 validation and 19,588 test positives**. Full-catalogue test has 19,633 positives. Comparing switching directly to a full-catalogue control would confound routing with removal of difficult unsupported items. The same scope rule applies when presenting coverage and novelty, whose catalogue denominators can also change.

Cascade gains over its own BPR control but remains below EASE and SLIM in the reported comparison. The meta-level model underperforms its interaction-cosine control on validation yet exceeds it on test. That pattern must be reported without retrospectively selecting by test. It illustrates ordinary validation/test variation, rather than permission to rewrite the selection policy.

The bundle stores trained arrays, mappings, masks, labels, search evidence, and reference metrics. `HybridBundle.rankings` reconstructs eligible lists and applies special cascade masking where needed. That portable inference is stronger reproducibility evidence than a report table alone, although it does not independently regenerate every historical training fit. The preparation of this report replayed the available evaluation cases against their committed accuracy results. [Source: corrected results and reproduction][alternate-results].

#### Figure B2. Compare hybrid results with the right component controls

<figure class="report-figure" id="figure-b2"><a href="figures/hybrid-comparisons.svg"><img src="figures/hybrid-comparisons.svg" alt="Validation and test NDCG markers for the OOF hybrid and its six components on the left, and separately for the corrected full catalogue alternative hybrids and their controls on the right."></a></figure>

**Read the figure.** Blue and orange markers are validation and test NDCG for the same recorded system. The left panel contains the OOF experiment's six retained components and final Ridge combination. The right panel contains the corrected alternative-hybrid fits and their controls. Their source configurations differ, so the panels deliberately preserve the experiment families instead of treating every repeated model name as one interchangeable fit.

**Interpretation.** In the left panel, Ridge's validation NDCG is slightly below the retained SLIM component, while its test score is higher: approximately **0.409095 versus 0.401882**. This is a measured validation/test ordering change. It does not mean that the combination was selected on test, or that the validation result should be rewritten to make the story simpler.

In the right panel, weak UserKNN-to-SLIM augmentation reaches approximately **0.411904** test NDCG against the paired receiver's **0.403254**. Strength one gives a smaller gain. The EASE augmentation row, in contrast, is slightly worse than its control on test. Feature combination and residual correction also do not automatically improve the base model. These controlled counterexamples are central to the project's explanation of when additional model structure helps.

Every displayed row uses full-catalogue scope. Warm switching is excluded because it changes candidates and relevance. Asterisks mark means over three scorer seeds conditional on one frozen BPR fit; no uncertainty bars are drawn, so those means must not be mistaken for noiseless estimates. Higher test markers are not an improvement from validation to test: those are different held-out partitions with different masks.

**Origin.** This new view uses the committed final OOF aggregate metrics and corrected alternative-hybrid comparison CSV, retaining the same frozen outcomes and selection rules.

### B8. Coefficients, ablation, and explanatory claims

Task 2.3 compares two forms of component removal. A frozen removal sets one Ridge coefficient to zero while holding the others and intercept fixed. A refit removes that feature and relearns remaining coefficients on the same OOF targets, at the same alpha 30,000. The first measures dependence of this particular fitted combination; the second measures whether other components can compensate under that fixed design.

ItemKNN has the largest coefficient, about **0.046954**, yet removing it improves test NDCG by **0.004194** without refitting and **0.002524** with refitting. Removing FISM lowers test NDCG by **0.004608** in the frozen comparison, and removing SLIM lowers it by **0.004289**. These facts directly contradict interpreting coefficient magnitude as a percentage contribution to ranking success.

The components are correlated; standardization changes coefficient units; the regression target is rating error while the reported outcome is NDCG. A feature can help a regression surrogate without improving the final ranking. Refitting also changes the remaining weights, so frozen and refit deltas answer related but different questions.

`paired_summary` records per-user differences, counts of improved/worse/tied users, top-ten overlap, and a seeded 2,000-resample user bootstrap. Such intervals are conditional on this split and these trained fits. They do not include variation from new training seeds, new datasets, repeated model selection, or a fresh test holdout. An interval should therefore accompany a precise estimand rather than a broad claim of universal model importance.

Task 2.3's result is explanatory evidence, not authorization to select a new reduced hybrid from test improvements. A deployable removal choice would need validation selection before a later evaluation. This is also the standard you apply when reviewing Victor's Task 2.6 improvement: distinguish an observed mechanism, a proposed change, and an independently evaluated improvement. [Sources: ablation code][coefficients], [coefficient report][coefficient-results].

#### Figure B3. Why the largest coefficient is not the largest contribution

<figure class="report-figure" id="figure-b3"><a href="figures/coefficient-ablation.svg"><img src="figures/coefficient-ablation.svg" alt="Fitted regression coefficients beside test NDCG changes from frozen and refitted component removal, with recorded paired user bootstrap intervals and a zero reference line."></a></figure>

**Read the figure.** The left bars are coefficients multiplying per-user standardized component scores. Their units belong to the raw-rating regression. The right points ask a different question: how test NDCG changes when that component is removed. Orange holds the remaining fit fixed; green relearns remaining coefficients at the same Ridge penalty. A point to the right of zero means that removing the component improved this recorded ranking.

**Interpretation.** ItemKNN receives the largest coefficient, yet its removal has a positive point estimate in both comparisons. FISM has a relatively large coefficient too, but removing it hurts. SLIM also causes a substantial loss on removal despite its smaller coefficient. The paired panels make the failure of a coefficient-as-importance interpretation visible: correlated score features and a rating-error training objective are not equivalent to independent contributions to ranking accuracy.

Error bars show the recorded 95% paired user-bootstrap intervals. Several cross zero, including ItemKNN's refit interval, while its frozen-removal interval stays positive in this conditional analysis. This distinguishes uncertainty in a particular controlled comparison from a broad claim that the component is always harmful. The intervals hold the split and trained systems fixed; they omit new training fits, new data splits, and uncertainty from the model-selection process.

The graph is descriptive evidence for Task 2.3. Choosing a reduced hybrid because its test removal looks favorable would reuse test for selection. It therefore motivates a development-only follow-up rather than declaring a newly selected best model.

**Origin.** Coefficients come from the final frozen `hybrid.json`; removal deltas and intervals come from the existing Task 2.3 `weighted.csv`. Both are preserved at their original precision in the figure provenance record.

### B9. Baselines, comparisons, metadata, and subgroup interpretation

Caio's reference popularity baseline counts training interactions only; unseen-count catalogue items receive zero. The random baseline uses a local seeded generator and sorted users/items so dictionary insertion order does not change the sequence of scores. Both return `ScoreTable`s and use the same candidate contract as other models. They provide essential context: complexity is not useful if it fails to improve on simple popularity under the same task.

`comparisons.py` reads completed test folders rather than retraining models. It requires matching split identity, dataset, partition, cutoff, relevance, candidate policy, metric definitions, aggregation, smoothing, group construction, catalogue/training sizes, and evaluation users. It rejects validation folders and duplicate labels. It writes aggregate CSV/JSON, per-user CSV, and source provenance. It does not establish that a human-selected folder was chosen without looking at test results.

Victor's PR #26 replaces pending group reports with activity, recorded-gender, and top-three-training-genre user cohorts, plus popularity and genre item groups. For each user cohort it recomputes the shared metrics, including cohort-wide exposure statistics. A minimum group size of thirty is reported as a flag; smaller groups are still calculated. Genre cohorts overlap, so their user counts must not be added as if they formed a partition.

The item report distinguishes recommendation slots, relevant hits, held-out relevant occurrences, precision/recall, and audience metrics. Audience quality weights a user's whole-list metric by their recommendation occurrences in a group. It is not precision restricted to that group's items. Multi-genre movies count in multiple genre memberships, so genre shares describe membership occurrences rather than mutually exclusive movie slots. For example, an Action/Comedy movie contributes one occurrence to each genre group. That differs from the fractional per-movie genre mass used by calibration: the two reports answer different questions and should not be combined without checking denominators.

A smoke run of the updated PR #26 on saved original ItemKNN test lists completes with activity counts 321/310/312 and recorded-gender counts **273 F and 670 M**. That establishes operability on this input, not completed cross-model subgroup interpretation. The current PR changes five files, including updates to the existing validation and beyond-accuracy tests. In particular, missing-gender groups and unavailable metadata retain explicit counts and nulls in the revised test expectations.

Definition text is partially corrected, not fully resolved. The emitted activity description now says 26/69, but uses strict inequalities that do not describe boundary users correctly; the helper actually uses `<=26` and `<=69`. The function docstring still mentions high activity above 92. User-genre membership is top three, while item genres are all supplied genres despite a top-one description. `pending_inputs` still contains the catalogue tuple instead of pending-reason metadata. These defects do not invalidate every numeric calculation, but make unqualified interpretation of emitted definitions unsafe. The evaluator still conflicts with current master and needs a consistent shared-writer API before combined use. [Sources: comparisons][comparisons], [updated group implementation][groups-pr].

### B10. Exploratory analysis, unfinished work, and the project's claim boundary

The repository also contains content-based experiments: genre cosine profiles, item-vector normalization, training-item IDF, raw-rating profile weights, and per-user Ridge predictors using genres with an optional release-year feature. These are useful exploratory evidence and negative controls; their existence does not complete every assigned explanatory task.

The genre-only experiments perform poorly against popularity on their recorded validation comparison. IDF reaches roughly **0.0509 NDCG**, compared with popularity around **0.1807**. Per-user genre Ridge reaches roughly **0.0409**; adding the tested linear year feature falls to about **0.0244** despite reducing ties. These observations challenge those specific feature/objective choices, rather than establishing that content-based recommendation is impossible. The small zero-training-history item cohort is inadequate for a strong cold-start conclusion.

At the refreshed snapshot, user-side fairness and the explanatory entrypoint remain placeholders. Group reporting is implemented in open PR #26. Gabriel's diversification and its numerical evidence are now merged through PR #28. The two hybrid/reranker orders are also implemented on master through PR #27. A stale message in the generic `task3.run()` does not override those real entrypoints and scripts. Task 3.3 cannot be inferred from the older cascade alone: it requires the explicit order comparison described next.

The strongest current conclusion is methodological: comparable candidates, explicit objectives, training-only context, validation selection, frozen tests, and inspectable saved predictions make the results defensible within their scope. The remaining uncertainty concerns robustness, pending algorithms and interpretation, integration of open work, and reuse of an already inspected single test split. Those limits should remain visible in the final project narrative. [Sources: exploratory code][content], [order interfaces][orders], [explanatory placeholder][explanations].

### B11. The newly merged comparison of hybrid/reranker orders

Task 3.3 is now concrete code and evidence in merged PR #27. It compares rerank-then-combine, called **pre**, with combine-then-rerank, called **post**, for the frozen weighted Ridge hybrid. A second experiment compares rerank-then-route with route-then-rerank for the existing UserKNN/BPR switch. Bogdan owns these implementations; your responsibility includes reviewing the interface, candidate controls, and interpretation.

<figure class="report-figure" id="figure-b4"><a href="figures/pipeline-orders.svg"><img src="figures/pipeline-orders.svg" alt="Two controlled paths through the same frozen weighted hybrid: rerank each component then convert ranks back to scores and combine, or combine original component scores then rerank; both use the same hybrid-selected top fifty pool."></a></figure>

**Figure B4 — What changes when order changes.** Both weighted paths begin with scores standardized over each user's full eligible candidates, then use the **same unchanged-hybrid-selected top fifty**. Profiles, coefficients, base fits, K, and validation policy stay fixed. Positives outside the pool remain in the relevance denominator. This prevents the result from quietly comparing two different retrieval policies.

Pre-reranking needs a real list-to-score rule. The implementation promotes a component's reranked top-ten prefix, appends the unselected pool items in that component's original score order, and assigns its original descending standardized score values to those positions. For example, scores `{a:3,b:2,c:1}` and prefix `[c]` become `{c:3,a:2,b:1}`. The multiset of values is preserved; those values are not replaced with arbitrary reciprocal ranks. Frozen Ridge weights then combine the reassigned scores, without standardizing again.

Post-reranking combines the original standardized scores first, then applies one reranker to the hybrid. Each method/order chooses one strength on validation under the inherited 5% relative NDCG-loss limit. Pre uses that same strength across all six components, rather than tuning six independent strengths. Zero strength must reproduce the unchanged hybrid in both paths.

The original frozen results are:

| Hybrid | Method and strength | Order | Test NDCG | Relative NDCG loss | Target reduction |
| --- | --- | --- | ---: | ---: | ---: |
| Weighted, full | Genre, 0.5 | Pre | 0.386796 | 5.45% | 27.41% |
| Weighted, full | Genre, 0.5 | Post | 0.379720 | 7.18% | 37.15% |
| Weighted, full | Popularity, 0.75 | Pre | 0.395378 | 3.35% | 31.68% |
| Weighted, full | Popularity, 0.75 | Post | 0.391661 | 4.26% | 37.85% |
| Weighted, full | Exposure, 0.5 | Pre | 0.399308 | 2.39% | 3.95% |
| Weighted, full | Exposure, 0.5 | Post | 0.388479 | 5.04% | 5.86% |
| Switch, warm | Genre, 0.5 | Pre = post | 0.352646 | 5.19% | 36.10% |
| Switch, warm | Popularity, 0.75 | Pre = post | 0.357064 | 4.00% | 41.15% |
| Switch, warm | Exposure, 0.5 | Pre | 0.364221 | 2.08% | 4.64% |
| Switch, warm | Exposure, 0.5 | Post | 0.356549 | 4.14% | 5.27% |

Changes use each hybrid's own baseline: weighted **0.409095** with 1,682 catalogue items; switching **0.371960** within its 1,615-item warm scope. Targets are genre KL, popularity JS, and exposure Gini. Their reduction percentages are not a shared unit of benefit.

For the weighted system, post-reranking improves each designated target more, at a larger accuracy cost. The result is conditional on the stated score-preserving conversion; it does not prove that every form of pre-reranking is weaker. Equal original scores can also make some promoted rank changes invisible to the combiner.

For switching, local genre/popularity reranking commutes with fixed routing: the selected component supplies the same user-specific pool, scores, profile, and strength in either path. The experiment checks exact list equality at every validation strength and at the frozen test choices. That is stronger evidence than merely finding equal mean NDCG. Exposure does not commute under the declared policy. Pre-reranking allocates exposure over all 943 potential users separately for each component, including lists later discarded; post uses the final mixed cohort. Test exposure lists differ for **747 users**. Routing remains frozen at 733 UserKNN and 210 BPR users.

**New reproduction evidence.** The original Windows caches were unavailable locally, so a separate reconstruction rebuilt thirty OOF fits and six full-training fits, recovered raw targets, and independently reselected Ridge alpha 30,000. Its split and fold memberships match the recorded design. Baseline test NDCG is **0.409041**, compared with original 0.409095. The largest selected-test discrepancy is post-exposure, **0.386825 instead of 0.388479**. All six selected strengths and the qualitative pre/post conclusions agree.

The reconstruction's saved 38 evaluations and 35,834 lists were audited; this report refresh also recomputed their saved metric arithmetic, masks, full relevance, and validation choices. It is close reproduction, not byte-identical replay. Different runtime versions and KNN neighbor tie handling are possible contributors, not a demonstrated causal explanation. The original and rebuilt result folders remain separate. Neither result removes the reused-test limitation. [Sources: order implementation][orders], [switching order code][switch-orders], [experiment protocol][order-protocol], [original combined results][order-results], [independent reconstruction][weighted-reproduction].

### B12. Gabriel's newly merged diversification: three separate objectives

Merged PR #28 implements three variants behind `diversification.rerank`, selected by `attribute="genre"`, `"popularity"`, or `"coverage"`. All share the existing candidate checks, deterministic tie-breaking, min-max relevance normalization, and exact strength-zero baseline. They are separate alternatives, not one method simultaneously optimizing all beyond-accuracy properties.

| Variant | Bonus added to `(1-strength)*relevance` | Selection target |
| --- | --- | --- |
| Genre variety | Mean Jaccard distance to items already selected for that user | Maximize ILD@10. |
| Long-tail discovery | Smoothed training self-information, min-max normalized within that user's pool | Maximize MeanSelfInformation@10. |
| Catalogue reach | One if the item has not yet been recommended in this cohort call; otherwise zero | Maximize CatalogueCoverage@10. |

Genre diversity rewards being different from the growing prefix. At its first position, that bonus is zero, so the original score and external ID resolve the choice. It does not compare against the user's historical genre proportions. That is the contrast with your genre calibration: a varied list and a historically proportionate list are not necessarily the same list.

Long-tail discovery uses `-log2((count+1)/(training_interactions+catalogue_size))`. The context now includes `item_counts`, validated as nonnegative integer counts for external IDs. The additional normalization gives novelty a bounded within-pool scale before combining it with relevance. Your popularity calibration instead minimizes disagreement with a user's head/middle/tail profile. A user with a popular-history profile can receive a well-calibrated list that is not very novel.

Coverage is cohort-dependent. It traverses rank rounds and sorted users with a fresh global covered-item set. Once an item appears, additional appearances receive no coverage bonus. Exposure fairness instead accumulates discounted appearances and continues distinguishing differently exposed movies. Neither objective guarantees the other, and neither can retrieve outside the supplied pool.

The registry adds maximizing target directions and method-specific manifest fields. Compatible frozen selections that omit a newly registered method mark it unavailable rather than tuning it on test. Input and source identity checks remain in force; this rule does not make an old selection reusable after arbitrary code changes.

<figure class="report-figure" id="figure-b5"><a href="figures/diversification-validation-tradeoffs.png"><img src="figures/diversification-validation-tradeoffs.png" alt="PR twenty eight validation curves for genre variety, novelty and catalogue coverage across six models; stars show the frozen choices and the five percent loss boundary is positive on the horizontal axis."></a></figure>

**Figure B5 — Read the sign convention carefully.** Unlike A2's accuracy-change axis, the horizontal axis here is **NDCG loss**, so the feasible side of the +5% line is to the left. The three panels are different objectives; colors distinguish base models. Stars mark the highest available target within the validation constraint. Greater strengths can improve the target while imposing much larger accuracy costs, especially for novelty. The connecting segments are the declared discrete strength sweep, not a continuously estimated frontier.

<figure class="report-figure" id="figure-b6"><a href="figures/diversification-test-effects.png"><img src="figures/diversification-test-effects.png" alt="Frozen test effects of the three diversification methods across six models, with grey accuracy changes, colored target gains and red outlines for losses greater than five percent."></a></figure>

**Figure B6 — What the new six-model evidence supports.** The experiment covers ItemKNN, UserKNN, BPR, SLIMElastic, EASE, and FISM using checked-in configurations without another model-tuning search. Test ILD gains range from **4.66–12.11%**, novelty gains from **1.00–2.47%**, and catalogue-coverage gains from **44.55–143.20%**, relative to each model's own original value. Those three scales cannot be compared as interchangeable improvements in user benefit.

Four of eighteen diversification cases exceed 5% test NDCG loss: FISM genre/novelty and SLIM/EASE coverage. The largest loss is EASE coverage, **7.15%**. Some novelty points improve NDCG too; no monotonic accuracy penalty is assumed. The frozen choices are retained without reselection.

UserKNN coverage illustrates the relative/absolute distinction. Its catalogue fraction rises from about **10.05% to 24.44%**: roughly **143.20% relative gain**, but about **14.39 percentage points** and still far below covering the whole catalogue. A large coverage bar does not establish equal exposure or subgroup fairness. Panel ranges differ, and red outlines flag the test accuracy-budget overruns.

Both figures are copied unchanged from PR #28's committed evidence, now on master. That evidence reports an audit of 228 evaluations and 215,004 lists. The present refresh independently checks the compact validation choices and final tables across all six models; it does not claim a new full replay of every original raw list. The evidence retains its generating runtime and source hashes, including generation before the final packaging commit. Merged source and generating experiment identity are therefore kept distinct. These remain single-seed results on the reused split. [Sources: diversification implementation][diversification], [six-model evidence][diversification-results], [plotting source][diversification-plots].

To defend a diversification result, name the input pool, the bonus, its state, the target, and the frozen strength. Then trace one chosen item. This connects an attractive aggregate chart to a concrete algorithmic choice and makes it easier to detect accidental substitutions between novelty, calibration, coverage, and exposure.

## Evidence, navigation, and reproduction

The two substantive halves have equal word counts under the same rule, including figure explanations; the measured counts are in `verification.json`. Each half has six annotated figures. Snapshot metadata, references, and this navigation section sit outside that allocation.

| Figure | Existing work shown | How to use it |
| --- | --- | --- |
| [A1: Training popularity](#figure-a1) | Saved training counts and item-band memberships | Separate catalogue share from interaction share. |
| [A2: Validation trade-offs](#figure-a2) | Original six-model validation plot | Read feasible strengths and selected operating points. |
| [A3: Frozen test effects](#figure-a3) | Original six-model test plot | Read gains, losses, and failures of budget transfer. |
| [A4: Information flow](#figure-a4) | Illustrated training/selection/test responsibilities | Trace which labels can influence each decision. |
| [A5: One calibration step](#figure-a5) | Numerical example of the implemented objective | Recalculate smoothing, KL and the greedy choice. |
| [A6: Shared exposure](#figure-a6) | Two-user illustration of cohort state | Explain why one recommendation changes another user's result. |
| [B1: Individual tuning](#figure-b1) | All 34 EASE/SLIM trial records | Assess regularization sensitivity and small selection gains. |
| [B2: Hybrid comparisons](#figure-b2) | Final OOF and corrected alternative-hybrid results | Compare within experiment families using the proper controls. |
| [B3: Coefficients and ablation](#figure-b3) | Frozen coefficients and recorded removal experiments | Distinguish regression weight from measured ranking utility. |
| [B4: Hybrid/reranker orders](#figure-b4) | Illustrated pre/post paths on merged master | Identify the fixed controls and list-to-score conversion. |
| [B5: Diversification validation](#figure-b5) | Original latest-PR six-model plot | Read maximizing targets and the positive loss axis. |
| [B6: Diversification test](#figure-b6) | Original latest-PR frozen test plot | Separate relative target gains from percentage-point changes. |

`figures/manifest.json` records source hashes, asset hashes, and plotted values. Four original plots are copied byte-for-byte; four views visualize saved measurements; four diagrams explain implemented behavior or toy arithmetic. The generator does not train models, tune parameters, or reevaluate recommendation lists. PR #28's figure-data JSON has LF line endings in Git; restoring CRLF reproduces its recorded Windows hash. Its PNG/PDF hashes match directly. The packaged summary and figure data preserve their own checked-out hashes.

| What to inspect | Main entrypoints | Report sections |
| --- | --- | --- |
| Metric arithmetic and edge cases | `metrics/accuracy.py`, `metrics/beyond_accuracy.py`, `metrics/_validation.py` | A2–A4 |
| Training-only context and metadata | `data.py:prepare_metadata`, `activity_terciles`, `item_popularity_groups` | A5, B9 |
| Reranker algorithms in PR #24 | `rerankers/calibration.py`, `item_fairness.py`, `_common.py` | A6–A7 |
| Strength choice and provenance in PR #24 | `rerank.py:run_reranking`, `select_strength`, `source_identity` | A8–A10 |
| Data/model boundary | `recbole_adapter.py:train`, `export_scores`, `partition_predictions` | B2–B4 |
| Weighted hybrid | `hybrids.py`, `oof_hybrid.py`, `raw_rating_target.py`, `ridge_search.py` | B5 |
| Alternative hybrids | `switching.py`, `cascade.py`, `feature_combination.py`, `residual_combination.py`, `feature_augmentation.py`, `meta_level.py` | B6–B7 |
| Portable replay | `hybrid_bundle.py`, `scripts/evaluate_alternate_bundle.py` | B2, B7 |
| Contribution analysis | `analysis/coefficients.py`, `scripts/analyze_hybrid_coefficients.py` | B8 |
| Final comparison and subgroup reporting | `comparisons.py`, PR #26 `analysis/groups.py` | B9 |
| Merged hybrid/reranker orders | `experiments/task3.py:score_preserving_permutation`, `switching_orders.py` | B11 |
| Newly merged diversification | PR #28 `rerankers/diversification.py`, `rerank.py:METHODS` | A13, B12 |

Fresh verification: final merged master and PR #28 each ran **196 tests**, with one plotting test skipped because Matplotlib was unavailable in the project runtime. Warnings were treated as errors. The weighted reconstruction's 38 evaluations/35,834 lists were rechecked for accuracy, independent target arithmetic, masks, relevance, pool membership, baseline controls, and frozen choices. PR #28's compact evidence passed 36 selection checks, 42 final-row checks, 18 diversification summaries, and source-figure hashes. A real saved-cohort smoke run of updated PR #26 passed; its full suite and the combined PR integration were not run.

The earlier report's original six-model reranking audit, 1,237 bundle checksums, and alternative-hybrid portable replay remain dated historical checks in `verification.json`. They are not relabeled as fresh runs. This refresh did not retrain models. Figure generation, image inspection, assessment checks, browser rendering, and portable packaging are verified separately.

For the normal project suite, run from a checkout containing the desired source:

```sh
/home/senteran/programming/RecBole_DSAIT4335/.venv/bin/python \
  -W error -m unittest discover -s project/tests -v
```

For the saved reranking audit, use PR #24's checkout and the local bundle:

```sh
cd /home/senteran/programming/RecBole_DSAIT4335/project/artifacts/worktrees/jacek-rerankers
/home/senteran/programming/RecBole_DSAIT4335/.venv/bin/python \
  -m project.scripts.verify_reranking \
  --run /home/senteran/programming/RecBole_DSAIT4335/project/artifacts/reranking-20261022-001/evaluation \
  --output /tmp/reranking-study-verification.json
```

The saved bundle is local and ignored by Git. A fresh clone can inspect the committed compact hybrid evidence and portable alternative weights; it does not automatically contain the large reranking checkpoints or historical OOF caches. Historical notes are useful for provenance, but use the final corrected reports linked above for current numbers. No generalization or causal claim is added merely because a reconstruction matches stored values.

[assignments]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/work_distribution.md
[accuracy]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/metrics/accuracy.py
[metric-validation]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/metrics/_validation.py
[contracts]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/contracts.py
[evaluation]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/evaluation.py
[beyond]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/metrics/beyond_accuracy.py
[data]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/data.py
[pr24]: https://github.com/jacek-kulik/RecBole-6/pull/24
[pr26]: https://github.com/jacek-kulik/RecBole-6/pull/26
[rerank]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/rerank.py
[calibration]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/rerankers/calibration.py
[item-fairness]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/rerankers/item_fairness.py
[rerank-common]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/rerankers/_common.py
[rerank-audit]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/scripts/verify_reranking.py
[protocol]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/configs/protocol.yaml
[adapter]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/recbole_adapter.py
[artifacts]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/artifacts.py
[bundle]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/hybrid_bundle.py
[model-configs]: https://github.com/jacek-kulik/RecBole-6/tree/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/configs/models
[framework]: https://github.com/jacek-kulik/RecBole-6/tree/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/recbole/model/general_recommender
[fism]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/recbole/model/general_recommender/fism.py
[retuning]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/reports/task-1.1-retuning-ease-slim.md
[oof-doc]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/OOF_HYBRID.md
[hybrids]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/hybrids.py
[oof]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/oof_hybrid.py
[oof-results]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/reports/task-1.3-final-results.md
[alternatives]: https://github.com/jacek-kulik/RecBole-6/tree/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys
[alternate-results]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/reports/task-1.4-corrected-hybrid-comparison.md
[coefficients]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/analysis/coefficients.py
[coefficient-results]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/reports/task-2.3-coefficient-analysis.md
[comparisons]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/comparisons.py
[groups-pr]: https://github.com/jacek-kulik/RecBole-6/blob/a3a55153f426727afd20091a33f39050c02cf815/project/recsys/analysis/groups.py
[content]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/content_based.py
[orders]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/experiments/task3.py
[explanations]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/analysis/explanations.py
[rerank-plots]: https://github.com/jacek-kulik/RecBole-6/blob/580bfaa63de4b176b2018e6e2a5108e8f9e390c7/project/scripts/plot_reranking.py

[pr27]: https://github.com/jacek-kulik/RecBole-6/pull/27
[pr28]: https://github.com/jacek-kulik/RecBole-6/pull/28
[switch-orders]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/recsys/experiments/switching_orders.py
[order-protocol]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/RERANKING_ORDERS.md
[order-results]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/project/reports/task-3.3-reranking-orders.md
[weighted-reproduction]: https://github.com/jacek-kulik/RecBole-6/blob/1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf/hyperparameters/results/task-3.3-weighted-reproduction-2026-10-25/README.md
[diversification]: https://github.com/jacek-kulik/RecBole-6/blob/87321da05b01eaf825420371d44899a4583f0fa2/project/recsys/rerankers/diversification.py
[diversification-results]: https://github.com/jacek-kulik/RecBole-6/tree/87321da05b01eaf825420371d44899a4583f0fa2/hyperparameters/results/task-3.1-diversification-six-models-60-20-20
[diversification-plots]: https://github.com/jacek-kulik/RecBole-6/blob/87321da05b01eaf825420371d44899a4583f0fa2/project/scripts/plot_diversification.py
