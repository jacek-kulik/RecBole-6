# Ridge 100k: updated seven-model comparison

> Historical result, superseded by [the final corrected FISM and Ridge results](task-1.3-final-results.md).

> Provisional: FISM training still includes the candidate in its own history; FISM and the hybrid require correction and rerunning.

MovieLens-100K; 60/20/20 per-user split, seed 42; 943 users. Ridge alpha 100000 selected using validation NDCG@10 and refitted on 60734 OOF raw-rating targets. Component models and histories are unchanged. Test candidates exclude train and validation interactions; per-user score normalization is recomputed for that candidate pool. Original alpha-100 results are preserved.

## Valid

| Model | Precision@10 | Recall@10 | F1@10 | MRR@10 | NDCG@10 | MAP@10 |
|---|---:|---:|---:|---:|---:|---:|
| itemknn | 0.2285 | 0.1696 | 0.1620 | 0.4862 | 0.2729 | 0.0859 |
| userknn | 0.2324 | 0.1732 | 0.1656 | 0.4819 | 0.2748 | 0.0873 |
| bpr | 0.2261 | 0.1677 | 0.1607 | 0.4860 | 0.2693 | 0.0831 |
| slimelastic | 0.2512 | 0.1857 | 0.1788 | 0.5252 | 0.3002 | 0.0954 |
| ease | 0.2509 | 0.1863 | 0.1788 | 0.5217 | 0.2979 | 0.0946 |
| fism | 0.2192 | 0.1697 | 0.1606 | 0.4798 | 0.2626 | 0.0840 |
| regressor | 0.2513 | 0.1871 | 0.1795 | 0.5152 | 0.2986 | 0.0963 |

## Test

| Model | Precision@10 | Recall@10 | F1@10 | MRR@10 | NDCG@10 | MAP@10 |
|---|---:|---:|---:|---:|---:|---:|
| itemknn | 0.3039 | 0.2042 | 0.2021 | 0.5968 | 0.3657 | 0.1186 |
| userknn | 0.3029 | 0.2072 | 0.2038 | 0.5842 | 0.3613 | 0.1204 |
| bpr | 0.2987 | 0.1939 | 0.1951 | 0.5768 | 0.3545 | 0.1119 |
| slimelastic | 0.3330 | 0.2239 | 0.2220 | 0.6450 | 0.4019 | 0.1362 |
| ease | 0.3308 | 0.2232 | 0.2205 | 0.6364 | 0.3992 | 0.1355 |
| fism | 0.2817 | 0.2037 | 0.1990 | 0.5794 | 0.3399 | 0.1130 |
| regressor | 0.3388 | 0.2282 | 0.2261 | 0.6490 | 0.4086 | 0.1397 |

## Coefficients

| Component | Alpha 100 | Alpha 100000 |
|---|---:|---:|
| itemknn | 0.09729670 | 0.03179964 |
| userknn | -0.02091330 | 0.01834298 |
| bpr | 0.02795407 | 0.01510170 |
| slimelastic | 0.02016452 | 0.02793943 |
| ease | 0.01354667 | 0.02527947 |
| fism | 0.06313308 | 0.01928777 |

Intercept: 3.18341343 → 3.27423626.

Reproduce: `.venv/Scripts/python.exe -m project.recsys.ridge_selected_report --run project/artifacts/hybrid-60-20-20 --selected project/artifacts/hybrid-60-20-20/ridge-extended`.
