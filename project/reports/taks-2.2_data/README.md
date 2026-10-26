# Task 2.2: corrected baseline comparison

Compare Random and training-popularity baselines with the six checked-in
individual configurations. All eight models were fitted afresh on MovieLens-100K,
seed 42, with the shared per-user random 60/20/20 split and validation NDCG@10
checkpoint selection. Final test evaluation uses the independent project metrics
at K=10, excluding training and validation history from candidates.

The initial PR results used two incorrect RecBole baselines. Random repeated one
random score vector across all users in a batch. Pop counted batch appearances
and sampled negatives rather than actual training frequencies. Random now draws
independent scores for each user/item pair; Pop fits its frequencies directly
from the training dataset and keeps them fixed during loader iteration. The Pop
YAML also disables unnecessary negative sampling.

## Fresh test results

| Model | Original NDCG@10 | Reproduced NDCG@10 |
| --- | ---: | ---: |
| Random | 0.007124 | 0.014494 |
| Popularity | 0.162372 | 0.211725 |
| BPR | 0.354497 | 0.354497 |
| EASE | 0.404872 | 0.404872 |
| SLIM | 0.403266 | 0.403266 |
| ItemKNN | 0.363707 | 0.363396 |
| UserKNN | 0.368319 | 0.368656 |
| FISM | 0.348765 | 0.348765 |

Random now recommends 1,675 distinct movies across 943 users, compared with 15
originally: catalogue coverage increases from 0.8918% to 99.5838%. Its 943 users
have distinct ranked lists. Pop's checkpoint counts exactly match all 60,734
training interactions. Its NDCG differs slightly from the review's standalone
count-based reference (0.211650), because this run retains the shared checkpoint
evaluator's `torch.topk` tie handling.

BPR, EASE, SLIM and FISM reproduce all aggregate metrics within 1e-12. Fresh KNN
fits differ slightly from the original Windows results: ItemKNN NDCG changes by
-0.000311420 and UserKNN by +0.000337618. No KNN code or configuration was changed
for this fix. The original checkpoints and dirty source state were not supplied,
so the cause of those differences cannot be isolated. The table and per-user
CSV consistently use the fresh fits; original results remain in commit
`2c1280551f323507185a1e3a416a86831834c3d3`.

These are results on the same previously inspected single split, with no new
hyperparameter search. The six existing model YAMLs are reproduced as supplied;
this run does not establish that they are optimal.

## Reproduction

From the repository root, with the project's Python environment active:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -m project.scripts.reproduce_baseline_comparison \
  --output project/artifacts/task-2.2-reproduction
```

Use a new output directory. The runner fits all eight models, exports exact
splits, checkpoints and full candidate scores, evaluates validation and test,
and writes the comparison. It reloads both baseline checkpoints and verifies
byte-identical rankings, relevance, per-user metrics and beyond-accuracy results
at CPU batch sizes 1024 and 128. It also checks Pop's saved counts against the
reconstructed training interactions. Progress and an approximate ETA are printed
after each model.

The completed run took 211 seconds on CPU with two threads at clean source
commit `92a2c9461ddbc438eab0fc8da1959bb9b82893ba`. The original split hash is
preserved: `2502b22df7c34768e49e395a2b07640796ed20fbe081dec4a55d29eb7f61acbb`.

`compare-final/` contains the regenerated comparison CSV/JSON, per-user CSV and
manifest. `verification.json` records source/config/data/checkpoint hashes and
the baseline replay checks. `reproduction-audit.json` records runtime versions,
original-versus-fresh metric differences, and independent artifact checks.
Published CSVs use LF line endings; the audit records their hashes separately
from the runner's raw CRLF CSV outputs, with identical parsed values.
Large checkpoints and score exports remain under the ignored artifact directory
and can be regenerated with the command above.

The five focused regressions failed before the fixes and pass afterward. All
152 project tests pass with warnings treated as errors; Ruff lint and formatting
pass for the four changed Python files.
