# Task 3.3: rerank before or after hybrid combination

Baseline test NDCG@10: **0.409095**. Pre: rerank components then combine. Post: combine then rerank. Target reductions are relative to the corresponding baseline metric.

| Target | Order | Strength | Test NDCG@10 | NDCG loss | Target reduction |
|---|---|---:|---:|---:|---:|
| genre | pre | 0.5 | 0.386796 | 5.45% | 27.41% |
| genre | post | 0.5 | 0.379720 | 7.18% | 37.15% |
| popularity | pre | 0.75 | 0.395378 | 3.35% | 31.68% |
| popularity | post | 0.75 | 0.391661 | 4.26% | 37.85% |
| exposure | pre | 0.5 | 0.399308 | 2.39% | 3.95% |
| exposure | post | 0.5 | 0.388479 | 5.04% | 5.86% |

Baseline targets: genre KL 0.7109; popularity JS 0.2270; exposure Gini 0.9498.

**Discussion**

We compare both orders using the corrected six-component Ridge hybrid, the same 943 users, training-derived profiles and the same hybrid-selected top-50 eligible candidates. The pre order promotes each component's reranked top ten, appends its original-order remainder and assigns original descending standardized scores to those positions. Frozen Ridge coefficients then combine the reassigned scores. This explicit conversion preserves score distributions and reproduces the original hybrid at zero strength. Each method/order selects strength on validation under a 5% NDCG-loss budget; all choices are frozen before test. Paired orders select identical strengths. Post-reranking produces larger reductions in every designated target but incurs greater NDCG loss. Both genre pipelines and post-exposure reranking exceed the validation loss budget on test, illustrating that the constraint does not guarantee test accuracy. Pre-reranking retains more accuracy while subsequent combination partly dilutes the component rerankers' objectives. These descriptive findings are conditional on the conversion rule, candidate pool and single previously inspected split. Target reduction percentages refer to different metrics and are not directly comparable across targets.
