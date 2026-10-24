# Rerankers: implementation and six-model evaluation

Jacek owns the calibration and item-fairness contributions, their integration
and the Task 3.2 trade-off evaluation. This record covers those implementations.
Diversification and user-side fairness remain pending with their owners.

## Status after the 22 October master merge

The numerical tables below record the 21 October experiment at implementation
`7fc5e6a82`. The branch subsequently merged master `5730b718a`, retaining
Bogdan's alternate hybrids and shared evaluation settings, and fixed the
metadata-isolation and strength-directory issues found in review. The combined
code passes 148 tests with warnings treated as errors. Ruff passes for the
branch's changed Python files; the full-project scan reports 42 existing issues
in files identical to master.

The six-model experiment has not been rerun after that merge. Its ItemKNN and
UserKNN configurations used 100 and 50 neighbours; current master uses 200 and
160. The original ignored checkpoints, score exports, evaluation folders and
figures lived in a temporary worktree that is no longer available. The tables
remain as the recorded results, but a fresh run is needed to regenerate the
supporting artifacts and establish results for the current configurations.

## Experiment identity

- Base: `origin/master` at `887b6c38a`; implementation: `7fc5e6a82`.
- Run: `project/artifacts/reranking-001`.
- Training runs: `project/artifacts/<model>-rerankers-001` for all six models.
- MovieLens-100K; seed 42; random per-user 60/20/20; 943 users, 1,682 movies.
- CPU runs with OMP/MKL/OpenBLAS thread counts set to 2.
- Saved counts: 60,734 training / 19,633 validation / 19,633 test interactions.
- Saved CSV split ID: `2502b22df7c34768e49e395a2b07640796ed20fbe081dec4a55d29eb7f61acbb`.
- Top 50 eligible candidates, final K=10. Test masks training and validation;
  profiles and item groups use training only.
- Six checked-in configurations, without further tuning. EASE and SLIMElastic
  use their merged NDCG-selected configurations; this experiment does not claim
  to retune or independently establish optimal settings for the other models.

All validation sweeps completed before test evaluation. Runtime manifests
record the in-progress task checkout, including its dirty flag; source-file,
checkpoint, score-export and data hashes bind the frozen choices to the
implementation committed in `7fc5e6a82`. The run IDs have not been relabeled.

## Original test rankings

| Model | NDCG@10 | Exposure Gini | Catalogue coverage |
| --- | ---: | ---: | ---: |
| itemknn | 0.365694 | 0.957623 | 14.09% |
| userknn | 0.361511 | 0.951332 | 16.83% |
| bpr | 0.354497 | 0.934923 | 26.99% |
| slimelastic | 0.403266 | 0.946017 | 18.49% |
| ease | 0.404872 | 0.950662 | 17.42% |
| fism | 0.348765 | 0.909388 | 37.63% |

## Selected strengths and test effects

Losses are relative to the same model's original NDCG. Target reduction
compares the reranked target metric with its original value on test:
genre KL, popularity UPD, or exposure Gini. Positive reduction is better.

| Model | Method | Strength | Validation NDCG loss | Test NDCG loss | Test target reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| itemknn | Genre | 0.5 | 2.99% | 5.58% | 36.09% |
| itemknn | Popularity | 0.5 | 2.17% | 2.60% | 21.53% |
| itemknn | Exposure | 0.5 | 1.98% | 2.11% | 3.91% |
| userknn | Genre | 0.5 | 3.56% | 4.66% | 37.45% |
| userknn | Popularity | 0.75 | 3.94% | 5.91% | 36.66% |
| userknn | Exposure | 0.25 | 0.89% | 1.21% | 1.69% |
| bpr | Genre | 0.5 | 4.74% | 3.77% | 38.70% |
| bpr | Popularity | 0.5 | 3.24% | 2.81% | 20.75% |
| bpr | Exposure | 0.25 | 1.37% | 2.44% | 2.48% |
| slimelastic | Genre | 0.5 | 4.85% | 5.54% | 38.12% |
| slimelastic | Popularity | 0.75 | 3.94% | 3.79% | 37.41% |
| slimelastic | Exposure | 0.5 | 4.64% | 4.49% | 6.86% |
| ease | Genre | 0.5 | 3.50% | 6.73% | 37.09% |
| ease | Popularity | 0.75 | 3.09% | 4.31% | 37.51% |
| ease | Exposure | 0.5 | 3.23% | 5.60% | 6.39% |
| fism | Genre | 0.5 | 4.02% | 4.80% | 37.60% |
| fism | Popularity | 0.5 | 2.96% | 3.47% | 20.10% |
| fism | Exposure | 0.25 | 2.56% | 2.74% | 3.84% |

All 18 choices satisfy the 5% validation loss rule. Five test cases exceed it:
ItemKNN genre, UserKNN popularity, SLIMElastic genre, EASE genre and EASE
exposure fairness. The largest test loss is 6.73% for EASE genre calibration.
Those outcomes remain in the comparison; test data did not trigger reselection.

At the frozen strengths, genre miscalibration falls by 36.09–38.70%,
popularity deviation by 20.10–37.51%, and Gini by 1.69–6.86%. These are
changes in different target metrics, not a common scale of user benefit.
The exposure heuristic gives smaller Gini reductions with the constrained
candidate pool. Exposure remains unequal, and a top-50 reranker cannot
recommend movies outside that pool. The fixed user order can affect its
cohort allocation. A single seed does not establish statistical significance.

## Verification

At the original experiment snapshot, the project suite passed 99 tests with
warnings treated as errors. Ruff passed for the project modules, tests and new
scripts. Three pre-existing test lint issues were corrected without changing
their assertions. The current integrated checks are recorded above.

The saved-run audit passed for 120 evaluated points and 113,160 ranking
lists. It independently recalculated all six accuracy metrics and exposure
Gini/coverage, checked candidate/history exclusion and all 18 validation
choices, and matched the 24 final comparison rows. The combined per-user
table contains 22,632 rows. Machine-readable evidence is in
`project/artifacts/reranking-001/verification.json`.

## Reproduction and figures

Use the commands in [RERANKING.md](../RERANKING.md) with a new suffix/output.
On this machine the RecBole Python is the original checkout's
`/home/senteran/programming/RecBole_DSAIT4335/.venv/bin/python`.
The plots were rendered with the system `python3`, which has Matplotlib.

Selected outputs:

- [Full test comparison](../artifacts/reranking-001/comparison/comparison.csv).
- [Verification](../artifacts/reranking-001/verification.json).
- [Validation curves](../artifacts/reranking-001/figures/validation-tradeoffs.png).
- [Test effects](../artifacts/reranking-001/figures/test-effects.png).
- The figure directory also contains standalone PDFs.

These links identify the original artifact locations, which are not available
in the restored worktree or committed to Git. The tables above preserve the
compact result record; use the reproduction commands to regenerate the outputs.

## Review handoffs and remaining dependencies

Availability was checked against the fetched branch heads below. These
are pending reviews where no runnable contribution exists, not completed
peer reviews or permission to implement another owner's work.

| Contribution | Inspected snapshot | Handoff |
| --- | --- | --- |
| Diversification | master `887b6c38a`; Gabriel `8ed2ff3b0` | Still raises `NotImplementedError`; await Gabriel's method, then review relevance/ILD balance and deterministic ties. |
| User-side fairness | master `887b6c38a`; Victor `8e139b098` | Still raises `NotImplementedError`; await Victor's method, then check development-only tuning and group coverage. |
| Both hybrid/reranker orders | Bogdan alternate-hybrid branch `c366705f9` | Both Task 3.3 entrypoints remain placeholders. Supply the working reranker interface; review fusion, candidate truncation and matched inputs when the order implementation arrives. |
| Task 2.6 insights | master and Victor snapshot above | The explanatory entrypoint remains a placeholder; no completed insight/improvement contribution was available to review. |
| Group interpretation | master | Victor retains Tasks 2.5/3.4; existing group-report placeholders remain explicit in every evaluation. |

Caio can review Task 3.2 using the comparison and audit. Gabriel reviews
item fairness/integration; Victor reviews calibration. No branches were
merged during the original reranking experiment. The later master merge
preserves its hybrid implementations; no teammate rerankers were replaced.

## Task 3.1 report draft

We implemented genre and popularity calibration and an item-side exposure
reranker. Each method selects ten movies from the original top fifty eligible
candidates, balancing normalized relevance against its secondary objective.
Genre calibration uses the course's smoothed KL divergence with fractional
genre weights. Popularity calibration uses Jensen–Shannon divergence between
training and recommendation popularity distributions. Item fairness greedily
rewards movies with less accumulated discounted exposure across the cohort.
Its exposure state resets for each experiment; it does not guarantee parity.
Profiles and popularity groups use training data only. Strength zero reproduces
the original ranking, and ties are deterministic. Diversification and user-side
fairness remain pending teammate contributions, so this experiment covers
the implemented calibration and item-fairness methods.

## Task 3.2 report draft

We evaluated six individual recommenders on MovieLens-100K using a shared
seed-42, per-user 60/20/20 split. For each method, validation selected the best
target metric from five strengths while limiting relative NDCG loss to 5%.
The choices were frozen before test evaluation. Genre calibration reduced
test miscalibration by 36.09–38.70%, at 3.77–6.73% NDCG loss. Popularity
calibration reduced UPD by 20.10–37.51%, at 2.60–5.91% loss. Exposure fairness
reduced Gini by 1.69–6.86%, at 1.21–5.60% loss. Five of eighteen choices
exceeded the validation loss budget on test, illustrating that validation
constraints do not guarantee the same test trade-off. We retained those
results without reselection. The results show measurable target improvements
at an accuracy cost, with limited exposure redistribution inside the top-fifty
pool. Single-seed results do not establish significance or uniform benefits
across user groups.

Course references: W3S1 Calibration, slides 12–18; W3S2 Fairness, slides
39 and 41–48; W2S2 Diversity, slides 7–12. The popularity variant adapts
the calibration framework to the existing UPD metric.
