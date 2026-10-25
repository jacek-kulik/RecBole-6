# Task 3.3 switching extension

The frozen UserKNN/BPR switch was evaluated under both reranking orders with genre
calibration, popularity calibration and item-exposure fairness. There are 30
validation points and six selected test pipelines, plus two unchanged baselines:
38 evaluations, covering 35,834 user recommendation lists. The original switch's
validation/test accuracy replays exactly. No models or routes were retrained.

The warm scope remains 943 users and 1,615 supported items, with 19,590 validation
and 19,588 test positives. Routing remains 733 UserKNN / 210 BPR. Each user's pool
is the original switch's top 50 eligible items. All methods and both orders use
the same pool and final K=10, with full warm relevance denominators.

Pre-reranking applies each component to the entire cohort independently before
selecting its users' lists. Post-reranking selects the score rows before applying
the reranker once across the mixed cohort. Genre and popularity calibration
produce exactly equal lists in both orders at every validation strength and the
frozen test choices. Exposure allocation differs because its cohort counters
include different hypothetical/final lists. The cohort policy is explicit and
the results should not be generalized to all possible pre-routing allocations.

`protocol.json`, `routes.json`, `selection.json` and `manifest.json` record the
source identities, frozen rule, pool/strength policy and completion. `metrics.csv`
and `metrics.json` retain all trials and selected test metrics. `invariance.json`
records the exact list equalities; `verification.json` records independent audits.
Full canonical evaluation folders, pools, user metrics and lists remain at
`project/artifacts/task-3.3-switching-orders-2026-10-25`. Its `comparison/` contains
the seven warm test rows and should remain separate from full-catalogue comparisons.

The source weights, catalogue, split labels and training matrix are already in
the committed alternate-hybrid portable bundle. Switching reproduction does not
need the weighted experiment's local OOF caches. Commands and implementation
choices are documented in `project/RERANKING_ORDERS.md`; the combined report is
`project/reports/task-3.3-reranking-orders.md`.

All six pipeline selections were frozen after validation and before test. Test
was previously inspected in earlier project work; these remain descriptive
single-split results. Neither a strength nor a routing rule was selected on test.
