# Extended Ridge penalty search

> Provisional: current FISM features retain the target self-contribution issue recorded in the spec audit.

Validation-only search on the existing 60/20/20 OOF experiment. No component retraining and no test evaluation. The original fitted model and reported test results are preserved.

| Ridge alpha | Validation NDCG@10 | MRR@10 | Recall@10 | Precision@10 |
|---:|---:|---:|---:|---:|
| 100 | 0.286757 | 0.496803 | 0.179988 | 0.242736 |
| 300 | 0.287279 | 0.496758 | 0.180322 | 0.243054 |
| 1000 | 0.288131 | 0.498305 | 0.180972 | 0.243690 |
| 3000 | 0.291273 | 0.504022 | 0.182833 | 0.245705 |
| 10000 | 0.295182 | 0.510433 | 0.185110 | 0.248887 |
| 30000 | 0.298137 | 0.518211 | 0.186323 | 0.249629 |
| 100000 | 0.298631 | 0.515175 | 0.187132 | 0.251326 |
| 300000 | 0.298023 | 0.510934 | 0.186619 | 0.251750 |
| 1e+06 | 0.298005 | 0.510307 | 0.186903 | 0.251962 |
| 3e+06 | 0.298196 | 0.511024 | 0.186992 | 0.251962 |
| 1e+07 | 0.298001 | 0.510910 | 0.186963 | 0.251750 |
| 3e+07 | 0.297951 | 0.510963 | 0.186938 | 0.251644 |
| 1e+08 | 0.297940 | 0.510963 | 0.186938 | 0.251644 |
| 3e+08 | 0.297934 | 0.510982 | 0.186938 | 0.251644 |
| 1e+09 | 0.297934 | 0.510982 | 0.186938 | 0.251644 |
| 3e+09 | 0.297934 | 0.510982 | 0.186938 | 0.251644 |
| 1e+10 | 0.297934 | 0.510982 | 0.186938 | 0.251644 |
| 3e+10 | 0.297934 | 0.510982 | 0.186938 | 0.251644 |
| 1e+11 | 0.297934 | 0.510982 | 0.186938 | 0.251644 |
| 3e+11 | 0.297934 | 0.510982 | 0.186938 | 0.251644 |
| 1e+12 | 0.297934 | 0.510982 | 0.186938 | 0.251644 |

Selected alpha: 100000. Validation NDCG@10: 0.298631; previous alpha 100: 0.286757.

Highest searched penalty selected: False. SLIMElastic validation NDCG@10: 0.300154.

This is a validation-selection result. Earlier test results belong to the original alpha-100 model; no test metric is claimed for the newly selected model.
