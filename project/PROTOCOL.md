# Experiment protocol to agree

Owner: Caio, with Jacek. Use `work_distribution.md` for the other owners
and reviewers. This is a decision sheet, not an approved experimental protocol.
The starter YAML contains provisional values so training can run immediately.

| Choice | Starter value / decision still needed |
| --- | --- |
| Dataset | MovieLens 100K from the repository's `dataset/` directory. Record filtering and metadata sources. |
| Split | Provisional per-user random 80/10/10, seed 42. Approve it, or change `configs/protocol.yaml` before comparisons. Save the actual split files and their hash. |
| Relevance | The example uses all observed interactions as implicit positives. Decide whether/how ratings determine relevance and update preparation and independent evaluation together. |
| Candidates | Provisional full-sort evaluation. Agree catalog scope, cold items, and train/validation history masking for each evaluation partition. |
| Cutoffs and selection | Provisional K=10 and validation MRR@10. Agree metrics, direction, seed count, and uncertainty reporting. |
| Individual models | TODO Gabriel: list every required lecture model and add its model YAML. BPR is only a starter example. |
| Hybrids | TODO Bogdan: list lecture alternatives and regression target; choose coefficient-fitting data and separate selection data, normalization, and constraints. |
| Metrics | TODO Jacek: write exact accuracy and beyond-accuracy definitions, averaging, and missing-data behavior. |
| Rerankers | TODO Gabriel, Jacek and Victor: record diversity, calibration, user-fairness, and item-fairness objectives and strength grids. |
| Groups | TODO Victor: training-derived user/item groups, metadata, counts, and minimum useful group size. |
| Compute/search budget | TODO Caio, Gabriel and Bogdan: trials, seeds, failure handling, and maximum run times. |

Record each agreement with its date, rationale, and reviewer. Re-run affected
experiments if the split, candidate policy, metadata processing, or metric
definition changes. Use validation results to select settings, then freeze them
before reporting test results. Ground-truth test labels do not enter hybrid
fitting, rerankers, or group construction.

The `train` command writes RecBole's validation metrics as training diagnostics.
The group's final comparisons must call the independent metrics under
`recsys/metrics/`. The demo's synthetic results are only a wiring example.
