> Historical experiment notes. Corrected fits and current comparisons are documented in [task-1.4-corrected-hybrid-comparison.md](reports/task-1.4-corrected-hybrid-comparison.md).

# Genre-based content recommender experiments

This component is intended for the switching hybrid's new/sparse-item branch.
It represents movies from metadata, so an item needs no interaction-trained
embedding. These experiments test useful content signal; improvements are not
assumed. No transformer or external metadata is used.

## Frozen V1

For each movie, construct a multi-hot vector `x_i` over genres in the repository's
MovieLens item metadata. Repeated genre tokens count once. `unknown` is not a genre.
Titles and release years are excluded from V1.

For user `u`, form `p_u = sum(x_i for i in training_history[u])`. Every distinct
training movie contributes equally, regardless of rating. Do not normalize the
movie vectors before this sum. Score candidate `j` with `cosine(p_u, x_j)`.
Sum and mean profiles give identical cosine scores. Overall history magnitude
does not change cosine scores; retain history length separately for future routing.

Zero-profile users receive training-popularity rankings. Movies with no usable
genres rank below content-scorable movies, ordered by training popularity within
that fallback tier. Missing metadata records cause an error rather than silent
omission. All score ties use the shared external item-ID ordering.

## Validation protocol

Reuse `project/artifacts/hybrid-60-20-20/splits/train.csv` and `valid.csv` and the
exact eligible candidate sets from `scores-valid-itemknn.json.gz`. The reference's
score values do not affect V1. User profiles and popularity use training only;
validation labels are used only for evaluation. Do not read or evaluate test.
The saved split is per-user random 60/20/20, seed 42.

Report the existing independent Precision, Recall, F1, MRR, NDCG and MAP at 10,
with every validation interaction relevant, consistent with the existing run.
Compare with training-popularity and the cached ItemKNN on identical candidates.
MRR follows the provisional shared protocol; also report NDCG used in the prior
hybrid work. This V1 has no hyperparameter selection.

Report fixed item-training-count bins: 0, 1-5, 6-20, 21+. Give support counts and
conditional recall using the original full-catalogue top-10, without restricting
candidates to the bin. These descriptive bins do not select switching thresholds.
A random interaction split may provide little genuine new-item evidence; a later
item-held-out validation design is needed for a strong cold-start claim.

```powershell
.venv/Scripts/python.exe -m project.recsys.content_based --output project/artifacts/content-genres-v1
```

The output directory must be new. It contains a full candidate ScoreTable,
per-user and aggregate metrics, item-history diagnostics, a readable report, and
input/source hashes. Existing component artifacts are read only.

## Controlled progression

| Variant | Single change relative to V1 | Hypothesis |
|---|---|---|
| V2 | L2-normalize each item vector before aggregation and scoring | Reducing the extra profile influence of multi-genre movies helps |
| V3 | Weight genre dimensions by smoothed IDF before aggregation and cosine | Ubiquitous genres are less discriminative |
| V4 | Weight each training movie's contribution by its raw rating, optionally divided by 5 | Favourites provide stronger evidence than low-rated movies |

For V3, fit IDF on the training-item catalogue, freeze it, then apply to candidate
metadata; record the exact formula and catalogue. For V4, recover raw 1-5 ratings
by external user/item IDs: the current saved CSV ratings were normalized by
RecBole. Dividing every rating by 5 is a uniform profile scaling and does not
alter cosine rankings. No centered-rating weights in this progression.

Test V2-V4 separately against V1 on identical splits, candidates, metrics and
tie rules. Only then combine promising changes and revalidate. Keep test outside
all design and threshold decisions. Report failures as well as gains.

The switching architecture remains a hypothesis: content for new items,
UserKNN when neighborhoods have strong support, BPR for users with sufficient
history otherwise, plus a fallback. Routing priority, thresholds and cross-model
score calibration require separate validation; this module does not implement
the switch.

## V1 validation result (18 October 2026)

Completed on 943 users, 1,682 items, and 18 genre features. No users needed the
zero-profile fallback; two catalogue items had no usable genre features.

| Model | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| Genre V1 | 0.0156 | 0.0774 | 0.0312 |
| Training popularity | 0.1081 | 0.3662 | 0.1807 |
| Cached ItemKNN | 0.1696 | 0.4862 | 0.2729 |

V1 is substantially worse overall on this split. The zero-training-history
validation cohort has only 43 interactions, involving 36 items and 28 users.
V1 retrieved one of those interactions in top-10 (micro recall 1/43); popularity
and ItemKNN retrieved none. This is insufficient support for an effectiveness
claim or a routing threshold. No test evaluation or switching fitting occurred.

Full per-user metrics, candidate scores, hashes, and fixed-bin diagnostics are
in `project/artifacts/content-genres-v1/`. Reproduce into a fresh directory with
the command above. Fifteen targeted content and accuracy checks passed, including
known-answer cosine scores, scoring unseen items, masking, fallback behavior,
and conditional metrics on unchanged rankings.

## V2 validation result (18 October 2026)

V2 changes only the profile aggregation: sum L2-normalized movie vectors rather
than raw multi-hot movie vectors. Candidate vectors are cosine-normalized in
both versions. Profiles still use equal movie weights and no rating information.

```powershell
.venv/Scripts/python.exe -m project.recsys.content_based --variant v2 --baseline project/artifacts/content-genres-v1 --output project/artifacts/content-genres-v2
```

The runner verifies that metadata, training/validation CSVs, reference candidates,
and accuracy code hashes match V1, and reloads its saved scores for comparison.
V1 artifacts are preserved. Both exports align across 943 users and 1,525,392
eligible user-item scores.

| Model | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| Genre V1 | 0.0156 | 0.0774 | 0.0312 |
| Genre V2 | 0.0128 | 0.0644 | 0.0253 |

V2 is worse overall, so retain V1 as the baseline for the next independent
experiment. Zero-history validation retrieval remains 1/43 for both variants.
Hits in training-count bins 1-5, 6-20, and 21+ change from 3/17/249 to 2/21/200.
These results do not support adopting normalization before aggregation on this
validation split. Seventeen targeted content/accuracy checks passed, including
hand-calculated V2 cosine scores and equivalence to V1 when training items all
have the same number of genres. Test data was not read or evaluated.

Detailed evidence is in `project/artifacts/content-genres-v2/validation.md`,
`metrics.json`, `manifest.json`, and `scores-valid.json.gz`.

## V3 validation result (18 October 2026)

V3 changes only genre weights relative to V1. Fit smoothed IDF on the 1,615
distinct items with at least one training interaction:

`idf(g) = 1 + ln((1 + 1615) / (1 + number_of_training_items_with_genre_g))`.

Multiply each genre dimension of every movie vector by its frozen IDF, sum
training-history movie vectors without per-movie normalization, and score with
cosine in that same weighted space. No ratings are used. Repeated interactions
across users do not count an item multiple times in IDF fitting. Unseen-item
metadata is transformed with the frozen weights, without fitting to validation
or test interactions. Genre counts, exact weights and the formula are saved in
`diagnostics.json`. IDF applies to both profile and candidate coordinates.

```powershell
.venv/Scripts/python.exe -m project.recsys.content_based --variant v3 --baseline project/artifacts/content-genres-v1 --output project/artifacts/content-genres-v3
```

| Model | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| Genre V1 | 0.0156 | 0.0774 | 0.0312 |
| Genre V2 | 0.0128 | 0.0644 | 0.0253 |
| Genre V3 | 0.0244 | 0.1230 | 0.0509 |
| Training popularity | 0.1081 | 0.3662 | 0.1807 |

V3 improves over V1 overall on this development split, although it remains far
below popularity. Cold-item retrieval is 2/43, versus V1's 1/43; the small cohort
does not establish a reliable cold-start benefit. Hits in the other bins
(1-5, 6-20, 21+) are 5/15/412, versus V1's 3/17/249. The 6-20 bin worsened.

Nineteen targeted content/accuracy checks passed, including hand-calculated
IDF/cosine scores, distinct-training-item counting, a genre absent from training,
and equivalence to V1 when genre IDFs are equal. Full score exports align with
V1 across 943 users and 1,525,392 candidates, with identical input hashes.
Test data was not read or evaluated. V4 remains a rating-only change relative
to V1; any combination with IDF should be a subsequent separate experiment.

Detailed evidence is in `project/artifacts/content-genres-v3/validation.md`,
`metrics.json`, `diagnostics.json`, `manifest.json`, and `scores-valid.json.gz`.

## V4 validation result (18 October 2026)

V4 changes only the user profile relative to V1:
`p_u = sum(raw_rating_ui * multi_hot_genres_i)` over training-history movies.
Weights are positive raw 1-5 ratings, without centering. No IDF or per-movie
normalization is applied. Candidate scoring remains cosine similarity.

Raw ratings are recovered by external user/item pair from the original `.inter`
file, retaining only saved training pairs. This avoids using RecBole's normalized
0-1 CSV ratings as weights. Validation/test ratings do not enter the profiles;
validation relevance still treats all recorded held-out interactions as positive.
The raw source hash and number of training rating pairs are saved in the manifest.

```powershell
.venv/Scripts/python.exe -m project.recsys.content_based --variant v4 --baseline project/artifacts/content-genres-v1 --output project/artifacts/content-genres-v4
```

| Model | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| Genre V1 | 0.0156 | 0.0774 | 0.0312 |
| Genre V2 | 0.0128 | 0.0644 | 0.0253 |
| Genre V3 (IDF only) | 0.0244 | 0.1230 | 0.0509 |
| Genre V4 (ratings only) | 0.0164 | 0.0844 | 0.0334 |
| Training popularity | 0.1081 | 0.3662 | 0.1807 |

Rating weights give a small overall improvement over V1 on this validation
split; V3 remains the best content variant. V4 retrieves 1/43 zero-history
validation interactions, unchanged from V1. Other-bin hits (1-5, 6-20, 21+)
are 3/19/264, versus V1's 3/17/249. These results do not establish statistical
significance or a reliable cold-start effect. No combined rating+IDF experiment
has been run.

Twenty-two targeted checks passed. V4 checks cover hand-calculated weighted
cosines, equal ratings matching V1 without cancellation, rejecting incomplete
rating coverage, and retaining only training-pair ratings from the source.
V1 inputs and candidate alignment were verified; test split was not read or
evaluated. Full evidence is in `project/artifacts/content-genres-v4/`.

## Regularized genre rating predictor (18 October 2026)

Fit a separate linear model per user using raw multi-hot genres and raw 1-5
training ratings: `predicted_rating(u,i) = intercept_u + genres_i @ weights_u`.
Minimize training squared errors plus `alpha * sum(weights_u ** 2)`; do not
penalize the intercept. Genre coefficients may be positive or negative. Do not
use cosine, clipping, IDF, release year or external metadata. A genre-less item
receives the user's fitted intercept; a user without training data falls back
to training popularity. Exact score ties retain external-ID ordering.

Select one shared alpha from the declared grid `0.01, 0.1, 1, 10, 100, 1000,
10000` using validation NDCG@10; ties select the larger alpha. Coefficients and
intercepts are always fitted using training ratings only. The selected alpha
is 1000 (an interior grid value). The test split is not opened or evaluated.
Note that rating squared error and all-interaction ranking relevance are
different objectives; this is the requested rating-prediction experiment.

```powershell
.venv/Scripts/python.exe -m project.recsys.genre_ridge --output project/artifacts/content-genre-ridge
```

| Model | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| V1 | 0.0156 | 0.0774 | 0.0312 |
| V3 (IDF) | 0.0244 | 0.1230 | 0.0509 |
| V4 (positive rating weights) | 0.0164 | 0.0844 | 0.0334 |
| Per-user Ridge | 0.0187 | 0.1032 | 0.0409 |
| Popularity | 0.1081 | 0.3662 | 0.1807 |

Ridge improves over the raw and positively rating-weighted profiles, but remains
below IDF and far below popularity. Cold-item retrieval is 0/43 interactions.
There are only 216 distinct genre patterns across the eligible catalogue, and
655/943 users have exact ties crossing the top-10 boundary for the selected
Ridge model. Identical genre vectors necessarily receive identical predictions.
The fitted models contain 6,182 negative genre coefficients, so the implementation
does learn signed preferences, rather than merely reweighting positive counts.

These results show limited usefulness of the tested genre-only designs for this
ranking protocol; they do not establish that content-based recommendation is
impossible on MovieLens. Release year, another training objective, or a dedicated
item-held-out validation would be separate experiments. Do not infer that all
hybrid categories require abandoning from this single branch's results.

Twenty-five targeted content, Ridge and accuracy checks passed, including a
known-answer signed regression, coefficient shrinkage, an unpenalized intercept,
constant ratings, unseen items, and ignoring held-out ratings in fitting.
The saved output includes all alpha trials, selected per-user coefficients,
full candidate scores, per-user metrics, cohort diagnostics and input hashes in
`project/artifacts/content-genre-ridge/`.

## Genre + release year Ridge (18 October 2026)

Add one continuous release-year feature to the per-user genre Ridge predictor.
Fit its z-score using the 1,613 distinct training items with valid years:
`year_z = (year - 1989.1413515189088) / 14.145549574156592`.
The two malformed years (`267`: `unkonwn`, `1412`: `V`) are imputed with the
training-item mean, giving standardized value zero. Candidate metadata does not
affect scaler fitting. Genre indicators remain raw 0/1 features. No titles,
IDF, year bins, nonlinear features or extra metadata are used.

Each user gets one signed linear year coefficient alongside genre coefficients.
The intercept remains unpenalized; the same alpha penalizes all 19 coefficients.
Retune using the same seven-value validation grid and NDCG@10 criterion. Selected
alpha is 10. Prior genre-only Ridge scores and all V1-V4 artifacts are preserved.

```powershell
.venv/Scripts/python.exe -m project.recsys.genre_ridge --include-year --output project/artifacts/content-genre-year-ridge
```

| Model | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| Genre Ridge | 0.0187 | 0.1032 | 0.0409 |
| Genre + year Ridge | 0.0100 | 0.0603 | 0.0244 |
| V3 genre IDF | 0.0244 | 0.1230 | 0.0509 |

Cold-item retrieval remains 0/43. Feature patterns increase from 216 genre-only
patterns to 711 genre/year patterns. Exact ties crossing top-10 decrease from
655/943 users to 132/943 users, but validation ranking quality deteriorates.
Reducing ties alone did not help. This rejects this particular linear-year
extension on the development split; it does not establish that year is useless
under other encodings or objectives. V3 remains the best content variant tested.

Twenty-seven targeted content, Ridge and accuracy checks passed, including
training-only scaling, missing-year imputation, and a hand-calculated example
where year distinguishes otherwise identical genre vectors. Sampled fitted
coefficients and intercepts were independently compared with scikit-learn Ridge.
Candidate alignment and input/source hashes were verified. Test split was not
read or evaluated. Full evidence is in `project/artifacts/content-genre-year-ridge/`.
