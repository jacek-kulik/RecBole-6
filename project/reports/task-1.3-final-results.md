# Task 1.3: final OOF Ridge hybrid results

MovieLens-100K; per-user 60/20/20 split; seed 42; 943 users; 60,734 OOF raw-rating targets. Candidate scores are z-scored per user.

FISM excludes target self-contributions and was retuned over the existing 48-setting grid using independent validation NDCG@10. Training uses a fixed 20 epochs, matching the OOF budget. The other five components and all outer/inner splits are unchanged.

Ridge alpha 30000 was selected from 24 fixed penalties by validation NDCG@10 before test evaluation. Test candidates exclude training and validation histories; score normalization is recomputed on eligible test candidates.

The test partition was already inspected in earlier experiments. These are corrected results on that same single split, not a new untouched generalization estimate. Five component hyperparameters still retain the previously approved transfer from the older tuning protocol.

## Valid

| Model | Precision@10 | Recall@10 | F1@10 | MRR@10 | NDCG@10 | MAP@10 |
|---|---:|---:|---:|---:|---:|---:|
| itemknn | 0.228526 | 0.169613 | 0.161972 | 0.486180 | 0.272891 | 0.085924 |
| userknn | 0.232450 | 0.173156 | 0.165562 | 0.481911 | 0.274841 | 0.087261 |
| bpr | 0.226087 | 0.167718 | 0.160687 | 0.485968 | 0.269292 | 0.083106 |
| slimelastic | 0.251220 | 0.185749 | 0.178772 | 0.525217 | 0.300154 | 0.095448 |
| ease | 0.250901 | 0.186311 | 0.178830 | 0.521718 | 0.297866 | 0.094645 |
| fism | 0.234995 | 0.173882 | 0.166812 | 0.494305 | 0.279497 | 0.087088 |
| regressor | 0.251962 | 0.187427 | 0.179739 | 0.514283 | 0.299155 | 0.096483 |

## Test

| Model | Precision@10 | Recall@10 | F1@10 | MRR@10 | NDCG@10 | MAP@10 |
|---|---:|---:|---:|---:|---:|---:|
| itemknn | 0.303924 | 0.204216 | 0.202074 | 0.596847 | 0.365716 | 0.118641 |
| userknn | 0.302863 | 0.207183 | 0.203795 | 0.584204 | 0.361251 | 0.120419 |
| bpr | 0.298727 | 0.193918 | 0.195053 | 0.576779 | 0.354497 | 0.111934 |
| slimelastic | 0.332980 | 0.223861 | 0.222020 | 0.645019 | 0.401882 | 0.136235 |
| ease | 0.330753 | 0.223166 | 0.220527 | 0.636427 | 0.399226 | 0.135472 |
| fism | 0.316225 | 0.209986 | 0.210234 | 0.596126 | 0.372549 | 0.118284 |
| regressor | 0.338494 | 0.225766 | 0.224400 | 0.651654 | 0.409095 | 0.139554 |

## Coefficients

| Component | Coefficient |
|---|---:|
| itemknn | 0.046953733 |
| userknn | 0.007655277 |
| bpr | 0.021182347 |
| slimelastic | 0.028186345 |
| ease | 0.026056366 |
| fism | 0.041267189 |

Intercept: 3.226650160. Coefficients apply to correlated normalized scores and are not causal importance.

Previous provisional hybrid test NDCG@10: 0.408580. Corrected hybrid: 0.409095.

## Before and after

| Model | Previous validation NDCG@10 | Corrected validation NDCG@10 | Previous test NDCG@10 | Corrected test NDCG@10 |
|---|---:|---:|---:|---:|
| fism | 0.262635 | 0.279497 | 0.339889 | 0.372549 |
| regressor | 0.298631 | 0.299155 | 0.408580 | 0.409095 |

The corrected hybrid changes test NDCG@10 by +0.000515 (+0.126%). Its test difference versus the preserved SLIMElastic component is +0.007212 (+1.795%). These are small single-split differences; no statistical-significance claim is made.

The corrected model with the old FISM hyperparameters obtains validation NDCG@10 0.271198. The selected configuration obtains 0.279497. This control isolates hyperparameter retuning under the corrected model and fixed training budget.

The earlier FISM tuning selected by MRR using a two-observation checkpoint schedule; this rerun selects fixed-20-epoch models by NDCG. The aggregate before/after comparison therefore includes the implementation fix, hyperparameter retuning, and the selection-policy change.

All 25 reused fold hashes and OOF values were checked. The final five unaffected columns, targets, fold assignments, and per-user validation/test metrics match the original artifacts exactly. FISM full-training validation metrics exactly reproduce its selected tuning trial. Frozen-selection hashes were verified after test scoring.

Reproduction: `project/OOF_HYBRID.md`. Committed metrics, all 48 FISM tuning trials, the 24-penalty Ridge search, coefficients, verification and source provenance: `hyperparameters/results/oof-ridge-final-60-20-20/`. Large OOF arrays, checkpoints and candidate score tables remain in local experiment artifacts.
