> **Provisional:** A subsequent lecture/spec audit confirmed that FISM training includes the target item in its own history. FISM-dependent results require correction and rerunning; see `task-1.3-spec-audit.md`. The five other component caches remain verified.

> Historical result, superseded by [the final corrected FISM and Ridge results](task-1.3-final-results.md).

# Six components and OOF Ridge regressor

MovieLens-100K; per-user 60/20/20 split, seed 42; 943 users; binary relevance for every held-out interaction. Five-fold OOF regression targets raw ratings. Ridge alpha 100.0 selected by validation NDCG@10 before test evaluation.

All component scores come from fixed-budget full-training refits. Test scores preserve those exact training histories and weights, excluding training and validation interactions. The other five component caches were reused; FISM uses the corrected logits and its newly selected parameters. The regressor uses all six candidate-normalized score features.

The cached loader ratings were normalized to 0-1. Original 1-5 ratings were recovered by external user/item pair and Ridge refitted at the frozen alpha. Coefficients equal four times the normalized-target coefficients, and the intercept equals four times its old value plus one; validation rankings are verified unchanged. Original normalized artifacts are preserved. No component retraining or new alpha selection was needed.

Metrics use the existing independent accuracy implementation. MAP divides by the full number of relevant items; F1 is averaged per user. Deterministic ties use external item IDs. These single-seed comparisons do not establish statistical significance.

## Validation (selection evidence)

| Model | Precision@10 | Recall@10 | F1@10 | MRR@10 | NDCG@10 | MAP@10 |
|---|---:|---:|---:|---:|---:|---:|
| itemknn | 0.2285 | 0.1696 | 0.1620 | 0.4862 | 0.2729 | 0.0859 |
| userknn | 0.2324 | 0.1732 | 0.1656 | 0.4819 | 0.2748 | 0.0873 |
| bpr | 0.2261 | 0.1677 | 0.1607 | 0.4860 | 0.2693 | 0.0831 |
| slimelastic | 0.2512 | 0.1857 | 0.1788 | 0.5252 | 0.3002 | 0.0954 |
| ease | 0.2509 | 0.1863 | 0.1788 | 0.5217 | 0.2979 | 0.0946 |
| fism | 0.2192 | 0.1697 | 0.1606 | 0.4798 | 0.2626 | 0.0840 |
| regressor | 0.2427 | 0.1800 | 0.1729 | 0.4968 | 0.2868 | 0.0910 |

## Test (frozen models)

| Model | Precision@10 | Recall@10 | F1@10 | MRR@10 | NDCG@10 | MAP@10 |
|---|---:|---:|---:|---:|---:|---:|
| itemknn | 0.3039 | 0.2042 | 0.2021 | 0.5968 | 0.3657 | 0.1186 |
| userknn | 0.3029 | 0.2072 | 0.2038 | 0.5842 | 0.3613 | 0.1204 |
| bpr | 0.2987 | 0.1939 | 0.1951 | 0.5768 | 0.3545 | 0.1119 |
| slimelastic | 0.3330 | 0.2239 | 0.2220 | 0.6450 | 0.4019 | 0.1362 |
| ease | 0.3308 | 0.2232 | 0.2205 | 0.6364 | 0.3992 | 0.1355 |
| fism | 0.2817 | 0.2037 | 0.1990 | 0.5794 | 0.3399 | 0.1130 |
| regressor | 0.3314 | 0.2210 | 0.2202 | 0.6420 | 0.4002 | 0.1337 |

## Regressor parameters

Intercept: 3.18341343. Training rows: 60734.

| Component | Coefficient |
|---|---:|
| itemknn | 0.09729670 |
| userknn | -0.02091330 |
| bpr | 0.02795407 |
| slimelastic | 0.02016452 |
| ease | 0.01354667 |
| fism | 0.06313308 |

Coefficients are unconstrained and apply to correlated, normalized scores; their magnitudes are not causal importance.

Reproduce after the OOF run: `.venv/Scripts/python.exe -m project.recsys.hybrid_report --output project/artifacts/hybrid-60-20-20`.
