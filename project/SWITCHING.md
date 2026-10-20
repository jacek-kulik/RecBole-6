> Historical experiment notes. Corrected fits and current comparisons are documented in [task-1.4-corrected-hybrid-comparison.md](reports/task-1.4-corrected-hybrid-comparison.md).

# UserKNN/BPR switching experiment

Question: can training-derived user evidence identify which of UserKNN and BPR
will rank better for that user? This is a hard switch of the entire user ranking,
not score blending. Content-based cold-start handling was dropped. No new users
or items without training evidence are claimed to be supported.

## Architecture and search

Reuse the verified cached full-training UserKNN and BPR validation ScoreTables.
Check cache completion hashes, split IDs, users and candidate alignment. Neither
component is retrained, and the original artifacts are preserved.

Compute user history size and a neighborhood-support proxy from training only:
rank other users by binary interaction cosine similarity, exclude self, retain
the top 50, and count how many share at least `m` training items. Similarity ties
use external user-ID ordering. This proxy is reconstructed independently of
the cached UserKNN's tie-dependent neighborhood selection.

Route to UserKNN if that reliable-neighbor count is at least `q`, OR if training
history length is less than `h`. Route to BPR otherwise. Thus reliable neighbors
take priority; UserKNN is also the short-history fallback. There is no third
model, and no cross-model score calibration is needed.

Fixed search grid:

- `m`: 1, 2, 5, 10, 20 shared training items.
- `q`: 1, 5, 10, 20, 40, 51 reliable neighbors. The value 51 disables the
  neighborhood gate because there are at most 50 neighbors.
- `h`: 0, 10, 20, 40, 80, 160, 320, 100000 training interactions. The endpoints
  allow disabling the history gate or always using UserKNN.

Select by macro validation NDCG@10, consistent with previous hybrid experiments.
Also report MRR@10 and the other independent project metrics; the shared YAML's
provisional selection field is MRR@10, so the NDCG choice is explicit here.
There are 240 grid entries plus always-UserKNN and always-BPR controls. Prefer
constant controls on equal scores, then the first deterministic grid entry.

## Warm evaluation scope

Keep the existing 60/20/20 seed-42 split. Every evaluated user must have training
history and at least one supported validation positive. Every candidate item
must have at least one training interaction. All components and hybrids use
identical resulting candidate masks and relevance sets.

This retains 943 users and 1,615/1,682 items. It retains 19,590/19,633 validation
interactions, excluding 43 positives on zero-training-history items. No users
are excluded. Unsupported items are removed from candidates AND relevance;
they are not counted as silently solved. These warm-scope scores differ from
the earlier full-catalogue evaluation and must not be directly equated with it.
All retained validation interactions are relevant regardless of rating.

## Selected parameters and results (18 October 2026)

The best grid entry is `m=1, q=51, h=320`. Since `q=51` disables the overlap
gate, the effective fitted rule is simply:

**Use UserKNN below 320 training interactions; use BPR at 320 or above.**

| Model | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| Always UserKNN | 0.173338 | 0.481911 | 0.274916 |
| Always BPR | 0.167873 | 0.485968 | 0.269347 |
| Selected switch | 0.173357 | 0.483554 | 0.275416 |

The switch routes 939 users to UserKNN and only four to BPR. In those four users,
mean component NDCG is 0.358102 for UserKNN versus 0.475983 for BPR. Across all
users, BPR beats UserKNN on 417, UserKNN wins on 449, and 77 tie. This indicates
component differences exist, but the tested evidence-based routing rules exploit
very little of them.

The gain over UserKNN is about 0.000500 absolute NDCG, or 0.18% relative, on the
same validation data used to search 242 entries. This is weak development evidence,
not independent proof of benefit; the four-user BPR branch may be unstable.
Within this grid, the strong-overlap routing hypothesis was not supported beyond
the simpler history-only rule. Do not claim switching or cold-start success from
these results. No test split was opened or evaluated for this experiment.

## Reproduce and inspect

```powershell
.venv/Scripts/python.exe -m project.recsys.switching --output project/artifacts/switch-userknn-bpr
```

Use a fresh output directory. The completed run saves `selected-rule.json`,
`trials.json`, training-only `user-features.json`, `routes.json`, aligned full
warm-candidate `scores-valid.json.gz`, per-user `metrics.json`, coverage and route
`diagnostics.json`, input hashes in `manifest.json`, and `validation.md`.

Five switching tests cover self-exclusion, overlap calculations, inclusive gate
boundaries, identical warm filtering of candidates/relevance, known optimal
routing and constant-control selection. Ten independent accuracy tests also
pass. The exported hybrid's metrics reproduce the grid-selection objective.
