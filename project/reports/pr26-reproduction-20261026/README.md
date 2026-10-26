# PR 26: merge, lint, and reproduction

Merged `origin/master` at `193a52a9d1d004475bd1afa64c6c34f57ec1bd0a`
into PR head `4140ad835927cb68c292fe21fde2f5d993f1d6d4`. The resulting
merge commit is `ed47e2b2b40256807b8b4952774400a07c253636`.

The merge preserves master's shared evaluation helper and passes rankings,
relevance, and K to the completed group analysis. Original and reranked
evaluations retain metadata-availability reasons and report pending group work
only when inputs are missing. Regression coverage checks both complete and
missing-gender cases. Ruff findings in `project/recsys` and `project/tests`,
including those inherited from master, are fixed.

All 202 project tests pass with warnings treated as errors, including plotting.
Ruff passes for those directories and the new verification script. Plotting
dependencies were installed in the ignored `project/artifacts/pr26-plot-deps`
directory; the shared Python environment was left intact.

Eight models were fitted afresh, with validation-selected checkpoints evaluated
on both validation and test. MovieLens-100K, seed 42, per-user random 60/20/20,
full-sort K=10, all held-out interactions relevant. Saved model YAMLs and the
shared protocol were used without a new hyperparameter search. The run took
215.59 seconds. Split fingerprint:

`2502b22df7c34768e49e395a2b07640796ed20fbe081dec4a55d29eb7f61acbb`

All models use 943 evaluation users, 1,682 items and 60,734 training interactions.
Activity groups use training counts: low <=26 (321 users), medium >26 and <=69
(310), high >69 (312). Recorded gender groups contain 273 F and 670 M users.

| Model | Test NDCG@10 | F | M | Low activity | Medium | High |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.014494 | 0.011874 | 0.015561 | 0.003500 | 0.011124 | 0.029152 |
| Popularity | 0.211725 | 0.174494 | 0.226895 | 0.118554 | 0.174186 | 0.344882 |
| BPR | 0.354497 | 0.298247 | 0.377417 | 0.238290 | 0.287089 | 0.541033 |
| EASE | 0.404872 | 0.360967 | 0.422761 | 0.292260 | 0.328824 | 0.596292 |
| SLIM | 0.403266 | 0.357950 | 0.421730 | 0.286488 | 0.333907 | 0.592326 |
| ItemKNN | 0.363396 | 0.319213 | 0.381399 | 0.258846 | 0.294745 | 0.539172 |
| UserKNN | 0.368656 | 0.317320 | 0.389574 | 0.269148 | 0.295200 | 0.544021 |
| FISM | 0.348765 | 0.306493 | 0.365989 | 0.253546 | 0.305356 | 0.489861 |

The aggregate results match the committed Task 2.2 comparison, with maximum
absolute difference `2.220446049250313e-16` across its numeric metrics. All
accuracy metrics agree exactly with that saved comparison. This verifies that
the merge and lint fixes preserve the existing model results.

The independent audit checks all 16 evaluations: six binary accuracy formulas
for every user; cohort sizes and mean metrics; catalogue/history
masks; item-group counts, hits, relevance shares, audience metrics, coverage,
and rank-discounted exposure. Random and Popularity group outputs replay
byte-for-byte at batch sizes 1024 and 128. Deliberately corrupted accuracy,
cohort counts, and history masks are rejected. Task 2.4 explanations were also
regenerated: sparsity curves, all 28 model-pair disagreements, gender/activity
tables, and genre profiles, for each partition.

EASE and SLIM lead this fixed configuration comparison. Every model has higher
NDCG among users with longer training histories. Even Random shows that
pattern, so it cannot be attributed solely to a learned model: held-out
positive counts and the evaluation task also vary with user activity. EASE's
aggregate M-versus-F relative gap is 17.12%; within the three activity groups
it is 6.82%, 12.17%, and 14.77%. These descriptive differences persist after
this coarse activity stratification, but do not establish a cause.

The PR contains no saved group outputs or reference numbers, so these group
results are a fresh execution rather than an exact historical replay of PR 26.
Genre memberships overlap: users join their top three training genres and
multi-genre items contribute to each applicable item group. Genre shares use
genre memberships as their denominator. Thirteen of the 19 genre cohorts have
fewer than the configured minimum of 30 users. This single-seed run does not
support significance or causal claims, and does not implement the separate
pending user-fairness reranker.

Local evidence is saved under
`project/artifacts/pr26-reproduction-20261026/group-evidence/`, including
`group-summary.png`. `verification.json` records audit checks, output hashes, and historical metric
deltas; `training-verification.json` records fresh-fit and baseline checksums.
`summary.json` and `explanations-{valid,test}.json` contain the numeric results.
The UserKNN and FISM training manifests mark the checkout dirty because the
new audit script was being added during training. The computational source
and model settings remained identical to `ed47e2b2b`; `source-checks.json`
records that verification. Checkpoints, score exports, full rankings, and
per-user outputs remain locally in the ignored artifact directory below.

Run from the task worktree, using new output directories on each replay:

```bash
export PYTHONPATH=.:project/artifacts/pr26-plot-deps
export MPLCONFIGDIR=/tmp/recbole-pr26-mpl
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/home/senteran/programming/RecBole_DSAIT4335/.venv/bin/python

$PY -W error -m unittest discover -s project/tests -v
$PY -m ruff check project/recsys project/tests project/scripts/verify_group_reproduction.py
$PY -m project.scripts.reproduce_baseline_comparison \
  --output project/artifacts/pr26-reproduction-replay
$PY -m project.scripts.verify_group_reproduction \
  --runs project/artifacts/pr26-reproduction-replay \
  --output project/artifacts/pr26-reproduction-replay/group-audit --plot
```

The actual run's full artifacts are in
`project/artifacts/pr26-reproduction-20261026/` in this worktree. The optional
`--plot` flag requires Matplotlib; JSON verification works without it.
