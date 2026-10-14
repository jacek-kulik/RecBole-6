# Experiment protocol

Caio and Jacek own the shared protocol. The choices below record Jacek's
14 October implementation request; Caio's scheduled review still applies.
Use `work_distribution.md` for ownership and review dates.

## Current evaluation choices

| Choice | Definition |
| --- | --- |
| Dataset | MovieLens 100K from `dataset/ml-100k/`, with original user/movie IDs. |
| Split | Random per-user **60/20/20**, seed 42. Export the actual train/valid/test CSVs and their shared hash. Older checkpoints retain their saved configuration and splits. |
| Relevance | Every held-out observed interaction is an implicit positive. No rating threshold; this keeps the assignment's relevance convention. |
| Candidates | Full catalogue from the checkpoint dataset, including cold movies and excluding padding. Validation masks training history; test masks training plus validation history. |
| Selection | Validation MRR@10, larger is better. K=10 is the primary reporting cutoff. Select settings using validation and freeze them before final test evaluation. |
| Profiles | Training interactions only, with equal interaction weights. Movie genre mass is divided equally across its genres; an unavailable genre is retained as `unknown`. |
| Activity reports | Linear terciles of training counts: low <= q1, q1 < medium <= q2, high > q2. Equal counts stay together. Zero-history users join low without changing the training-derived boundaries. |
| Gender fairness | Recorded F/M values from `ml-100k.user`, joined by external user ID. Gender membership is separate from activity membership. Unavailable gender affects only demographic comparisons. |
| Movie popularity groups | Complete descending count buckets cross cumulative training interaction shares 20% and 80%. Record actual head/middle/tail shares; cold movies belong to tail. |

The notebook compares the previous 80/10/10 split with 60/20/20. With the
current data and seed, 60/20/20 yields 60,734 training interactions and 19,633
in each held-out partition. The median held-out count rises from six to
thirteen. This gives more evaluation labels at the cost of training data;
it does not establish better model performance. Training activity boundaries
are 26 and 69, with 321 low, 310 medium, and 312 high users. These are results
of the current split, not hard-coded thresholds.

## Independent metrics

The six accuracy metrics are Precision, Recall, F1, MRR, NDCG and MAP at K.
Precision divides by K; Recall and MAP divide by all relevant items. Binary
NDCG uses `min(K, number_of_relevant_items)` ideal hits. F1 is computed per
user before averaging. Each evaluation user has equal weight.

| Beyond-accuracy measurement | Definition |
| --- | --- |
| ILD | Mean Jaccard distance between unordered pairs of movie genre sets. Undefined at K=1. |
| Mean self-information | Mean `-log2((training_count + 1) / (training_interactions + catalogue_size))`. The pseudocount gives cold movies finite novelty. |
| Genre miscalibration | `KL(p || 0.99*q + 0.01*p)` using natural logarithms. p is the training genre distribution and q is the recommendation distribution. |
| Group Recommendation Unfairness | Absolute difference between F and M mean accuracy for each of the six accuracy metrics. Also save both means and the signed F-minus-M difference. |
| Catalogue coverage | Distinct recommended movies divided by checkpoint catalogue size. |
| Exposure Gini | Gini over all catalogue movies, including those with no exposure. Rank r contributes `1/log2(r+1)` exposure. |
| User Popularity Deviation | Base-2 Jensen-Shannon divergence between each user's training and recommendation head/middle/tail proportions. |
| Item-group exposure | Discounted exposure total, fraction of cohort exposure, and mean exposure per movie in each popularity group. |

UPD uses the distribution comparison in [Abdollahpouri et al.](https://arxiv.org/pdf/2103.06364).
Our implicit-feedback version weights training interactions equally and
averages users equally, rather than weighting ratings or equally averaging
popularity-taste groups. The local diversity, calibration and fairness
lectures supply the remaining metric definitions. Position-discounted exposure
and Jaccard genre distance are explicit choices for this implementation.

Compute every measurement overall and separately for low/medium/high activity
users. Within each cohort, gender unfairness compares that cohort's F/M users.
Activity levels select reports; recorded gender supplies the fairness groups.
Coverage, Gini and item-group exposure are recomputed from each cohort's lists.
Report cohort sizes, demographic counts and the same catalogue denominator.

Unavailable measurements are JSON `null`, accompanied by reasons and usable
sample counts. This includes empty cohorts, missing training profiles and a
gender comparison missing F or M users. Other users and measurements remain
in the reports. A small group gap alone does not establish fairness: quality
could be poor for both groups, and held-out observations are incomplete labels.

## Running and recording evaluations

Use `python -m project.recsys validate --run ... --output ...` for development
and `python -m project.recsys evaluate --run ... --output ...` for final test
reporting. Both accept `--k` and `--batch-size`. The training command saves
RecBole validation diagnostics; final comparisons use the independent code.

Accuracy artifacts retain their existing format. `beyond_accuracy.json` stores
per-user and aggregate values, availability, gender gaps and movie-group
exposure. `groups.json` records memberships, boundaries and counts;
`group_metrics.json` contains overall and activity-cohort reports. The manifest
records the partition, checkpoint/split identity, source metadata hashes,
formulas, smoothing and grouping rules. Only validation is compared against
saved RecBole validation diagnostics. Synthetic examples are wiring evidence,
not MovieLens results.

## Decisions still owned by the team

| Choice | Owner / remaining work |
| --- | --- |
| Individual models | Gabriel: agree the required lecture set and add configurations; BPR remains a small starter. |
| Hybrids | Bogdan: agree fitting/selection subsets, regression targets, normalization and constraints. |
| Rerankers | Gabriel, Jacek and Victor: implement objectives and agree validation strength grids. |
| Further group analysis | Victor: interpret activity/gender and movie-group results; other demographic attributes are outside this implementation. |
| Compute/search budget | Caio, Gabriel and Bogdan: agree trials, seeds and uncertainty reporting. |

Record later changes with their date, rationale and reviewer. A changed split,
candidate policy, profile rule or metric definition requires new comparable
runs. Test labels do not enter fitting, profiles or group construction.
