# Task 3.3: the two hybrid/reranker orders

The experiment compares genre calibration, popularity calibration and item-exposure
fairness separately, using the implementations from PR #24. Diversification and
user-side fairness remain outside this run because their methods are placeholders.

## Matched systems

The source is the corrected six-component OOF Ridge hybrid from Task 1.3. Retain
all component fits, the fitted coefficients/intercept and per-user candidate
z-score normalization. No base recommender or combiner is retrained. The five
historical component settings and corrected FISM are retained together; newer
standalone fits are not substituted.

For each partition, mask training history (validation) or training plus validation
history (test) before standardization. Build a common 50-item candidate pool per
user from the unchanged hybrid's highest eligible scores. Both orders and all
strengths use exactly this pool and return K=10. Relevance remains all held-out
interactions, including positives outside the pool. Training profiles and
popularity groups are prepared once from training interactions only. Full-catalogue
metrics include zero-exposure items.

The unchanged hybrid is evaluated as the control. The restricted pool preserves
its exact top 10. Existing source cache hashes, all thirty OOF folds, split
membership, raw rating targets and reconstructed Ridge coefficients are verified
using the Task 2.3 source checker before running either order. Saved original
validation and test accuracy must replay exactly.

## Explicit rank-to-score rule

Rerankers return lists; Ridge consumes numeric scores. For **pre**:

1. Rerank each component on the common pool, returning its top 10.
2. Append its remaining pool items in that component's original score order.
3. Assign its original descending standardized scores to these positions.
4. Combine these reassigned scores with the unchanged Ridge coefficients and
   return the hybrid's top 10.

This preserves each component's score multiset, units, mean and variance within
the pool; no second z-score transformation is applied. It transfers the reranked
order without fitting new weights or pretending that a reranker returns calibrated
prediction scores. Equal original scores can make some rank changes invisible to
the combiner. This is one chosen operational definition of rerank-then-combine,
and should not be generalized to all rank-fusion conventions.

For **post**, combine the original standardized component scores using the same
Ridge coefficients and then rerank that hybrid score table to K=10. The rerankers
apply their own existing min-max relevance normalization within the fixed pool.

All methods must reproduce the unchanged hybrid exactly at strength zero in both
orders. Official exposure reranker calls reset state for each component, strength,
order and partition and use the same entire cohort in sorted user-ID order. No
exposure state is shared between component calls or partitions.

## Validation and frozen test evaluation

For each of three methods and two orders, sweep strengths 0, 0.25, 0.5, 0.75, 1.
Use one shared strength across all six components in the pre order, rather than
searching six component-specific strengths. Select the lowest designated target
(genre KL, popularity JS, or exposure Gini) subject to validation NDCG@10 being at
least 95% of the unchanged hybrid's NDCG. Ties prefer higher NDCG and then lower
strength. The same operating-point policy applies to both orders.

Complete all 30 validation points plus the baseline before writing the frozen
six-pipeline selection. Only then evaluate the six selected pipelines and baseline
on test. Report test loss-budget overruns without reselection. This gives 38
evaluations in total. Test was already examined in earlier coursework experiments;
results describe this single reused split and this chosen mapping/pool/grid.

The 50-item pool and 5% validation budget are explicit inherited design choices,
not requirements of the brief or established optima. The common hybrid-selected
pool controls retrieval differences between orders; results are conditional on
that retrieval policy. End-to-end performance includes missed positives outside it.

## Reproduction and output

Run on a branch containing merged Task 1.3/2.3 code and PR #24's rerankers:

```powershell
.\.venv\Scripts\python.exe -m unittest project.tests.test_reranking_orders project.tests.test_rerankers project.tests.test_rerank -q
.\.venv\Scripts\python.exe -u -m project.scripts.run_reranking_orders `
  --output project/artifacts/task-3.3-reranking-orders-reproduction `
  --results hyperparameters/results/task-3.3-reranking-orders-reproduction
.\.venv\Scripts\python.exe -m project.scripts.verify_reranking_orders `
  --run project/artifacts/task-3.3-reranking-orders-reproduction `
  --results hyperparameters/results/task-3.3-reranking-orders-reproduction
```

Use fresh directories. Exact source verification currently requires the original
local Task 1.3 OOF caches identified in `project/OOF_HYBRID.md`. Both partition pools,
all 38 evaluation folders, full per-user metrics, rankings and beyond-accuracy
outputs remain in the local run. Compact settings, frozen choices, coefficients,
aggregate metrics and provenance go in the requested results directory. The final
report is `project/reports/task-3.3-reranking-orders.md`.

The weighted-only verifier writes `project/reports/task-3.3-weighted-reranking-orders.md`.
The combined report is generated after both audits using the command below.

The verifier recomputes aggregate and per-user accuracy, beyond-accuracy metrics,
fractional-genre KL, popularity JS, Lorenz-curve exposure Gini and coverage; checks
every saved list against its pool/history mask and the full held-out relevance;
reproduces each selection using validation only; verifies the freeze precedes
test; and creates the seven-row Task 2.2 comparison. It renders a compact results
table and discussion below 200 words after verification.

## Frozen switching extension

The additional hybrid is the existing UserKNN/BPR switch from Task 1.4. Retain its
training-derived routing rule and assignments (733 UserKNN / 210 BPR), component
fits and warm scope: 943 users, 1,615 training-supported items, 19,590 validation
positives and 19,588 test positives. This differs from the weighted experiment's
full catalogue, so compare orders to the matching hybrid baseline.

Build each user's common top-50 warm candidate pool from the unreranked switched
scores. In the pre order, rerank both components over all 943 users independently,
then select each user's list using the frozen routing. No rank-to-score conversion
is needed. In the post order, select the original score rows first and rerank the
resulting mixed cohort. The same strength grid, final K, validation selection rule
and freeze-before-test sequence apply. Routes are never retuned.

Genre and popularity calibration depend only on the individual user's pool,
scores and training profile, so the two operations commute: each user's selected
model supplies exactly the same reranked list. This equality is asserted at all
five validation strengths and at the frozen test selections, not inferred from
similar aggregate metrics.

Exposure allocation depends on other users. Pre-reranking maintains separate
exposure counters for the entire potential cohort of each component; these count
some lists discarded by subsequent routing. Post-reranking maintains one counter
for the final mixed output. This explicit cohort policy explains the difference
and is not equivalent to reranking only the users assigned to each component.
All counters reset between calls. The conclusion is conditional on this policy.

```powershell
.\.venv\Scripts\python.exe -m project.scripts.run_switching_orders `
  --output project/artifacts/task-3.3-switching-reproduction `
  --results hyperparameters/results/task-3.3-switching-reproduction
.\.venv\Scripts\python.exe -m project.scripts.verify_switching_orders `
  --run project/artifacts/task-3.3-switching-reproduction `
  --results hyperparameters/results/task-3.3-switching-reproduction
.\.venv\Scripts\python.exe -m project.scripts.report_hybrid_orders `
  --weighted hyperparameters/results/task-3.3-reranking-orders-reproduction `
  --switching hyperparameters/results/task-3.3-switching-reproduction
```

The switching source is the existing committed portable model bundle. Its run
does not require the weighted hybrid's local OOF arrays. Both baseline accuracy
partitions must reproduce the saved switching evaluation exactly. The switching
audit repeats all list/mask/metric/selection checks within the warm scope. Its
canonical test folders are compared separately from the full-catalogue weighted
folders. Both suites together contain 76 evaluations and 71,668 user lists.
