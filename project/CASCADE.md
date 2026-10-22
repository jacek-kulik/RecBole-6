> Historical experiment notes. Corrected fits and current comparisons are documented in [task-1.4-corrected-hybrid-comparison.md](reports/task-1.4-corrected-hybrid-comparison.md).

# ItemKNN retrieval followed by BPR reranking

For each user, ItemKNN selects the top M eligible items. BPR orders that
shortlist and returns 10 recommendations. BPR scores are used directly; the
two models' scores are never blended. M is the retrieval budget, distinct
from ItemKNN's neighbourhood size k.

Changing this inference pipeline does not require changing either training
objective. It does change which hyperparameters are useful. This experiment
therefore fits new component configurations and selects their combination by
the final cascade's validation NDCG@10, rather than independently selecting
each model by standalone accuracy. BPR retains uniform negative sampling;
candidate-conditioned hard-negative training would be a separate experiment.

## Search

- ItemKNN: k in {10, 20, 50, 100, 200}, shrink in {0, 10, 100, 200}: 20 fits.
- BPR: embedding size in {32, 64, 128}, learning rate in {0.0005, 0.0015},
  epochs in {20, 50}: 12 fits. Adam, batch size 512, one uniformly sampled
  negative per positive. No validation early stopping.
- M in {10, 20, 50, 100, 200, 500, 1000, full catalogue}: 1,920 combinations.
- Maximize final validation NDCG@10; break ties by smaller M, then grid order.
  Full catalogue is a BPR-only control and is permitted to win.

The same fitted BPR configuration can score different shortlists, so each fit
is cached once. This evaluates all combinations without redundant training.
These are bounded grids, not an exhaustive search over all training options.

## Evaluation

Reuse the exact Task 1.3 per-user 60/20/20 split, seed 42. Every held-out
interaction is a binary positive. The complete catalogue remains eligible,
including items without training evidence. Validation masks training history;
test masks training and validation history. Test shortlists are constructed
again after masking, rather than masking an already truncated shortlist.

The runner verifies each reconstructed split byte for byte before fitting.
All new component models use only training interactions. Training epochs are
fixed per configuration. Model parameters, M, and standalone control choices
are frozen before test scoring; there is no train+validation refit.

Final metrics use the independent project evaluator. Missed retrieval positives
remain in the relevance denominator. Candidate recall and an oracle NDCG@10
ceiling diagnose retrieval loss. Item ID strings break score ties everywhere.

Controls include the original Task 1.3 ItemKNN/BPR models and independently
retuned standalone models from the same component grids. Their selection uses
validation NDCG@10 as well. MRR@10 is reported; the shared YAML's provisional
MRR selection setting does not drive this search.

This experiment caches complete score tables to reuse fits. Candidate fraction
describes potential BPR scoring volume and is not a measured latency gain.
Results use one seed. Validation gains include selection bias, and the test
set is the existing benchmark holdout already evaluated in earlier experiments.

## Run

From the repository root:

```powershell
.venv/Scripts/python.exe -m unittest discover -s project/tests -p test_cascade.py -v
.venv/Scripts/python.exe -m project.recsys.cascade --output project/artifacts/cascade-itemknn-bpr
```

Add `--resume` for an interrupted run. A run lock prevents concurrent writers.
Completed component score files have SHA-256 markers; source, configuration,
dataset and split changes require a new output directory. Existing experiments
are preserved. The output includes component checkpoints, all trial metrics,
the frozen selection, validation/test metrics, shortlisted score tables,
retrieval diagnostics and `comparison.md`.
