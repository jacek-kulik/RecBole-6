# FISM logits fix and retuning

The historical results accompany an 80/10/10 configuration (default seed 2020); their runtime provenance was not recorded. The corrected runs use the shared per-user random 60/20/20 protocol, seed 42, full-sort evaluation, 20 epochs, validation every 10 epochs, and validation MRR@10 selection.

All 48 existing grid combinations were evaluated without search-level early stopping. Test was evaluated only after selection, for the historical parameter configuration and the selected configuration. The original historical files are preserved.

Historical versus corrected results also change the protocol; they do not isolate the bug fix. The two corrected rows isolate parameter retuning on the same split, seed, training budget, and scoring code. History vectors are summed before candidate dot products to avoid large repeated tensors; regression tests check equivalent scores and gradients.

| Run | Embedding size | Regularization (source, destination) | Alpha | Validation MRR@10 | Test MRR@10 | Test NDCG@10 | Test Recall@10 |
|---|---:|---|---:|---:|---:|---:|---:|
| Historical (buggy, 80/10/10) | 64 | [0.0001, 0.0001] | 0.0 | 0.3002 | 0.3838 | 0.2262 | 0.2033 |
| Corrected, old parameters (60/20/20) | 64 | [0.0001, 0.0001] | 0.0 | 0.4889 | 0.5804 | 0.3470 | 0.1926 |
| Corrected, retuned (60/20/20) | 128 | [0.001, 0.001] | 0.0 | 0.4992 | 0.5908 | 0.3501 | 0.1972 |

On the matched 60/20/20 protocol, retuning doubles the embedding size (64 to 128), increases each regularization weight tenfold (0.0001 to 0.001), and leaves alpha at zero. Validation MRR@10 increases by 0.0103 (2.11% relative); test MRR@10 increases by 0.0104 (1.79%), test NDCG@10 by 0.0031 (0.89%), and test Recall@10 by 0.0046 (2.39%). These are single-seed results under the existing 20-epoch budget, without an uncertainty estimate.

These are RecBole training diagnostics; the shared protocol calls for independent metrics for final reporting.

Reproduce: `.venv/Scripts/python.exe -m project.scripts.retune_fism --threads 2`.

`summary.json` contains all metrics and selected checkpoint paths; `trial-*.json` contains every validation result and runtime. `identity.json` records input hashes and runtime versions. Checkpoints remain local in ignored `saved/`. Load `best-params.yaml` after the model and shared protocol YAMLs to reuse the selected hyperparameters.
