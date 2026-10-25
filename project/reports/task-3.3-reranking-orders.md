# Task 3.3: reranking before or after hybrid combination

| Hybrid | Target (strength) | Order | Test NDCG | NDCG loss | Target reduction |
|---|---|---|---:|---:|---:|
| Weighted | genre (0.5) | pre | 0.386796 | 5.45% | 27.41% |
| Weighted | genre (0.5) | post | 0.379720 | 7.18% | 37.15% |
| Weighted | popularity (0.75) | pre | 0.395378 | 3.35% | 31.68% |
| Weighted | popularity (0.75) | post | 0.391661 | 4.26% | 37.85% |
| Weighted | exposure (0.5) | pre | 0.399308 | 2.39% | 3.95% |
| Weighted | exposure (0.5) | post | 0.388479 | 5.04% | 5.86% |
| Switch | genre (0.5) | pre = post | 0.352646 | 5.19% | 36.10% |
| Switch | popularity (0.75) | pre = post | 0.357064 | 4.00% | 41.15% |
| Switch | exposure (0.5) | pre | 0.364221 | 2.08% | 4.64% |
| Switch | exposure (0.5) | post | 0.356549 | 4.14% | 5.27% |

Baseline NDCG: Weighted: 0.409095 (full, 1,682 items); Switch: 0.371960 (warm, 1,615 items).

Pre = rerank components then combine/route; post = combine/route then rerank. Changes use each hybrid's own baseline. Targets: genre KL, popularity JS and exposure Gini.

**Discussion**

Both hybrids retain their fitted components, coefficients/routing and original evaluation scopes. Each uses identical top-50 eligible candidates in both orders and returns ten items. Weighted pre-reranking transfers the reranked prefix through the documented score-preserving permutation before fixed Ridge combination. Each method/order selects strength on validation under a 5% NDCG-loss budget; all choices are frozen before test. For the weighted hybrid, post-reranking improves each target more but incurs greater accuracy loss. For switching, genre and popularity calibration produce identical lists in both orders at every validation strength and their frozen test choices: per-user reranking commutes with fixed routing. Exposure fairness is cohort-dependent. Pre-reranking maintains separate counters for each component's entire cohort, including lists later discarded by routing; post-reranking allocates exposure across the final switched output. Its stronger Gini reduction therefore comes with a larger NDCG loss. These findings depend on the stated conversion, pool and exposure policy. The validation budget does not guarantee test accuracy. Results are descriptive on one previously inspected split, and warm/full absolute scores should not be treated as a controlled cross-hybrid comparison.
