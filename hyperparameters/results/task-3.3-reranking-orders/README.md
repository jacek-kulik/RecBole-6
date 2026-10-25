# Completed Task 3.3 experiment

See `project/RERANKING_ORDERS.md` for the exact protocol, implementation convention,
limitations and reproduction commands. The numerical runner completed 38 evaluations:
one original hybrid plus thirty validation method/order/strength points, followed
by one original hybrid plus six frozen selected test pipelines. Each method uses
the same selected strength in both orders (genre 0.5, popularity 0.75, exposure 0.5).

The unchanged corrected OOF Ridge hybrid has test NDCG@10 0.409095. Post-reranking
improves each designated target more but incurs larger NDCG losses than pre-reranking.
Both genre orders and post-exposure exceed 5% NDCG loss on test; the 5% constraint
was imposed on validation only and no test-based reselection was performed.

All six component fits, coefficients, split membership, evaluation users and
training-derived profiles remain fixed. Both orders use the original hybrid's
top-50 eligible candidates and return ten items. The pre order uses the explicit
score-preserving rank conversion documented in the protocol. The conclusions are
conditional on that conversion and retrieval policy, not a comparison of every
possible hybrid/reranker composition. The existing single test split was already
inspected in previous project work.

`metrics.csv` / `metrics.json` contain every validation point and selected test row,
with all six accuracy and all aggregate beyond-accuracy metrics. `selection.json`
records the six choices frozen before test. `hybrid.json`, `weighted-inputs.json`,
`frozen-protocol.json` and `manifest.json` retain the source identities, settings,
fixed coefficients and completion evidence. `verification.json` records the audit.

Detailed canonical evaluation folders, per-user metrics, rankings, relevance,
the two partition pools, numerical source snapshots, and the seven-row Task 2.2
comparison remain in the local ignored run:
`project/artifacts/task-3.3-reranking-orders-2026-10-24`.

Verification checks all 35,834 recommendation lists, full relevance denominators,
candidate/history exclusion, all zero-strength controls, all six validation-only
choices and the unchanged freeze through test. It independently calculates the
target metrics; maximum error is at floating-point rounding precision. Original
validation/test accuracy replays the corrected Task 1.3 results exactly. The
recorded test run is in `test-results.json`; the plotting test is skipped because
this environment does not have Matplotlib.

The weighted-only report is `project/reports/task-3.3-weighted-reranking-orders.md`,
with a 167-word discussion. The combined report, including the separately verified
warm-scope switching extension, is `project/reports/task-3.3-reranking-orders.md`,
with a 175-word discussion. Numerical reproduction requires the original local Task 1.3
caches. This compact result directory alone is not a portable training bundle.
