# Task 2.3: hybrid coefficients and component contributions

MovieLens-100K, 60/20/20, seed 42; 943 users, 1,682 items. NDCG@10 throughout. Removal deltas are relative to the original weighted hybrid; positive means improvement.

**Weighted hybrid: test contribution analysis**

| Removed component | Original coefficient | Frozen removal delta | Refit delta |
|---|---:|---:|---:|
| itemknn | 0.046954 | +0.004194 | +0.002524 |
| userknn | 0.007655 | -0.000890 | -0.001142 |
| bpr | 0.021182 | -0.001167 | -0.000698 |
| slimelastic | 0.028186 | -0.004289 | -0.004673 |
| ease | 0.026056 | -0.002643 | -0.002549 |
| fism | 0.041267 | -0.004608 | -0.003950 |

Original hybrid NDCG@10: valid 0.299155, test 0.409095.

**UserKNN feature augmentation: receiver retrained at each strength**

| Receiver | Strength | Validation NDCG | Test NDCG |
|---|---:|---:|---:|
| EASE | 0 | 0.298099 | 0.404872 |
| EASE | 0.25 | 0.298156 | 0.403747 |
| EASE | 1 | 0.298565 | 0.402431 |
| SLIM | 0 | 0.301112 | 0.403254 |
| SLIM | 0.25 | 0.304670 | 0.411904 |
| SLIM | 1 | 0.303055 | 0.407374 |

**Discussion**

Coefficients multiply per-user standardized component scores; their sizes are not contribution percentages. Frozen removals set one coefficient to zero without changing the others. Refits remove the same component and relearn the remaining coefficients from identical out-of-fold rating targets, holding the Ridge penalty at 30,000. Component fits and candidate masks remain fixed. Despite receiving the largest coefficient, ItemKNN lowers ranking accuracy in this fitted hybrid: both removal procedures improve validation and test NDCG. FISM and SLIM cause the largest frozen test losses; refitting does not fully recover either loss. Thus coefficient magnitude does not establish ranking utility, and the rating-regression objective need not align perfectly with NDCG. Augmentation scales the same top-50 normalized UserKNN predictions, keeps observed targets unchanged, and retrains each receiver with fixed regularization. Strength 0.25 helps SLIM more than strength 1; EASE gains slightly on validation but loses on test at both strengths. More transferred prediction mass therefore need not help. These are descriptive results on one previously inspected test split, with no new configuration selection; they do not establish generalizable improvements.
