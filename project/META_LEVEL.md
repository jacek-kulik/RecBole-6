> Historical experiment notes. Corrected fits and current comparisons are documented in [task-1.4-corrected-hybrid-comparison.md](reports/task-1.4-corrected-hybrid-comparison.md).

# BPR user embeddings transferred to UserKNN

The sender is the existing 64-dimensional BPR model, fitted on the seed-42
60/20/20 protocol's training partition for 20 fixed epochs. The transferred
object is its frozen user embedding matrix. The receiver computes user cosine
similarities from these vectors, retains up to k positive neighbours excluding
self, then sums neighbours' binary training interactions weighted by cosine.
BPR item vectors are used only to prove that the checkpoint reproduces the
cached BPR scores, never to produce the hybrid scores. No metadata is used.

Neighbour ties follow sorted external user IDs; ranking ties follow external
item IDs. Users without positive neighbours get zero scores with that same
deterministic tie rule. No popularity or score blending is added.

Run from the repository root using its environment:

```powershell
.\.venv\Scripts\python.exe -m project.recsys.meta_level --output project/artifacts/meta-bpr-userknn
.\.venv\Scripts\python.exe -m unittest project.tests.test_meta_level
```

The output directory must be new. The completed first run already occupies
the example path. Artifacts include frozen user representations with external
ID mapping, selected neighbour lists, complete eligible validation scores,
per-user metrics, every k trial, and a manifest with hashes. The checkpoint's
original dataset is loaded to reconstruct token mapping, without building
splits or fitting on its interactions. Test CSV is not read or evaluated.

## First validation experiment

All 943 validation users and 1682 catalogue items use the original candidate
mask (catalogue minus each user's training interactions). The grid is
k = 10, 20, 40, 50, 80, 160, 320, selecting validation NDCG@10 and smaller k
on exact ties. Cached BPR and UserKNN scores are left unchanged. A matched
interaction-cosine control changes only the representation and searches the
same k grid using the identical scoring and tie logic.

| Model | k | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Cached BPR | - | 0.167718 | 0.485968 | 0.269292 |
| Cached UserKNN | 50 | 0.173156 | 0.481911 | 0.274841 |
| BPR embedding UserKNN | 80 | 0.173724 | 0.495663 | 0.282314 |
| Matched interaction-cosine control | 160 | 0.176867 | 0.497641 | 0.283823 |

At k=50 the hybrid reaches 0.279257 NDCG@10 versus 0.274887 for the matched
control. After selecting k separately, the control is better by 0.001509.
The hybrid improves on the cached component baselines, but this experiment
does not establish a benefit over an equally tuned neighbourhood control.
These are validation-selected development results; independent test performance
has not been measured. Embedding dimension, training length and random seed
were inherited from the existing BPR checkpoint, without further search.

Detailed results: [validation report](artifacts/meta-bpr-userknn/validation.md).
