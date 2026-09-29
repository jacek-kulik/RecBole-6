# Project work plan

This plan is for five students working from **30 September to 26 October
2026**. Replace PERSON-1 to PERSON-5 with names once the group agrees on roles.
The project uses RecBole and MovieLens 100K for offline ranking experiments.

The course requires all numbered tasks in Tasks 1-3: the class-covered
individual models (except content-based), hybrid models, tuning, separate
metric implementations, model and group analysis, and four rerankers
(diversification, calibration, and user-side and item-side fairness). Keep the
experiment set small enough to finish: limit repeated tuning and redundant
comparisons, while covering each required task. The course brief emphasizes
depth of analysis over the number of models or metrics.

The group submits one PDF report and runnable code in a ZIP named after the
group number. Each student submits a peer-evaluation Excel file separately.
The report has a one-page limit per numbered task and a 200-word limit for
experimental-results discussion per task. Keep the main body self-contained
and use Times New Roman, 12 pt, with 1.15 line spacing.
The official deadline is **26 October, 23:59**. Aim to approve the package by
**25 October, 18:00**, with results and corrections settled by **24 October,
18:00**.

## Who does what, and when

Each owner writes a short report section for their work and asks the listed
collaborators for help or a quick check. Dates below are proposed handoffs;
unless another time is shown, aim for **18:00 Europe/Amsterdam**. Raise a
blocker as soon as it could delay someone else's work.

### PERSON-1 — shared experiments, baselines, and final package

**Owns:** experiment setup and shared runner; random and popularity baselines;
Task 2.2 (final model comparison); runnable instructions and clean-checkout
reproduction.

- **30 Sep-5 Oct:** coordinate the kickoff, agree the shared split and result
  format with PERSON-2/3/4, and prepare the runner and baselines. Coordinate
  with PERSON-2/3 so initial model predictions are ready by **5 Oct**.
- **6 Oct:** combine PERSON-2's predictions, PERSON-3's first weighted-hybrid
  output (due **12:00**), the baselines, and PERSON-4's metric pilot into one
  working example. Assemble the feedback-session questions by **16:00**; all
  five check them at **17:00** and finalize by **18:00**.
- **7-9 Oct:** attend the course feedback session on **7 or 8 Oct**; record any
  agreed changes to the protocol by **9 Oct**. Agree the final experiment
  matrix and compute budget with the group by **11 Oct**.
- **18-21 Oct:** record selected model configurations on **18 Oct**; hand off
  the final comparison to PERSON-2 for review by **20 Oct**, and close any
  corrections by **21 Oct**. Send results to PERSON-5 by **19 Oct**.
- **23-25 Oct:** prepare the runnable-code instructions and support a clean
  checkout check by **24 Oct**. Help PERSON-5 inspect the ZIP by **25 Oct,
  12:00**; coordinate upload readiness by **18:00**.

**Collaborates with / may be blocked by:** PERSON-2's model outputs and
PERSON-3's hybrid output are needed for the 6 Oct example. Final comparisons
wait on accepted model and hybrid configurations from PERSON-2/3 and reranker
results from PERSON-4.

### PERSON-2 — individual recommenders and diversification

**Owns:** Tasks 1.1 and 1.2 (class-covered individual models and tuning);
diversification reranker in Task 3.1.

- **1-6 Oct:** agree the required model set with PERSON-3 by **2 Oct**. Supply
  initial candidate-aligned predictions to PERSON-3 by **5 Oct** and complete
  Task 1.1 by **6 Oct**.
- **7-14 Oct:** address the first model review, then hand off tuned individual
  models on **12 Oct**; incorporate review and corrections by **14 Oct**.
  Deliver the diversification method to PERSON-4 by **10 Oct** and close its
  review by **12 Oct**.
- **15-24 Oct:** help PERSON-4 check the integrated rerankers by **15 Oct**;
  review the final model comparison on **20 Oct** and help resolve any issues
  by **21 Oct**. Update the report section with final results by **24 Oct**.

**Collaborates with / may be blocked by:** PERSON-3 needs the initial
predictions to build the first hybrid. Tuning and final comparison depend on
PERSON-1's agreed evaluation setup and runner. PERSON-4 needs the
diversification method to integrate all rerankers.

### PERSON-3 — hybrid models and coefficient analysis

**Owns:** Tasks 1.3-1.5 (weighted hybrid, other class-covered hybrid methods,
and tuning); Task 2.3 (coefficient analysis); Task 3.3 (the two required
hybrid/reranker orders).

- **1-6 Oct:** agree hybrid scope and the coefficient-fitting approach; bring
  any unresolved regression question to the feedback session. Use PERSON-2's
  initial predictions to provide PERSON-1 a first weighted-hybrid output by
  **6 Oct, 12:00**.
- **7-18 Oct:** incorporate feedback and hand off Task 1.3 by **10 Oct**;
  finish other hybrid methods by **13 Oct** and hybrid tuning by **16 Oct**.
  Complete their reviews and corrections by **18 Oct**.
- **19-21 Oct:** send coefficient/contribution results to PERSON-5 by
  **19 Oct**. Run both Task 3.3 pipeline orders with PERSON-4; hand off by
  **19 Oct**, complete review by **20 Oct**, and corrections by **21 Oct**.
- **22-24 Oct:** update the report section from final results and help review
  the combined Task 1 and Task 3 write-ups.

**Collaborates with / may be blocked by:** the first hybrid depends on PERSON-2
providing predictions by **5 Oct**. Both Task 3.3 comparisons depend on
PERSON-4's integrated rerankers and the agreed common experiment conditions.

### PERSON-4 — metrics and rerankers

**Owns:** Task 2.1 (separate accuracy and beyond-accuracy metric
implementations); Task 3.1 (integration of all four rerankers); Task 3.2
(evaluation of reranked models).

- **1-6 Oct:** agree metric and fairness definitions with PERSON-5 by **5
  Oct**. Prepare an initial independent accuracy calculation for the 6 Oct
  example. Define the simple reranker interface with PERSON-2/3/5 by **7 Oct**.
- **7-16 Oct:** hand off metric implementations by **10 Oct** and accept
  corrections by **12 Oct**. Integrate the four rerankers by **14 Oct**;
  PERSON-2 reviews the combined implementation by **15 Oct**, and integration
  fixes are complete by **16 Oct**.
- **17-21 Oct:** evaluate reranked models and trade-offs with PERSON-1; send
  results to PERSON-5 by **19 Oct**, finish review by **20 Oct**, and close
  corrections by **21 Oct**. Coordinate with PERSON-3 on Task 3.3 during this
  period.
- **22-24 Oct:** update the report section and support clean-checkout
  reproduction of the metrics and rerankers.

**Collaborates with / may be blocked by:** the reranker integration needs
PERSON-2's diversification method and PERSON-5's user-side fairness method by
**11 Oct**. Evaluation depends on PERSON-1's shared predictions and agreed
metric inputs.

### PERSON-5 — analysis and report coordination

**Owns:** Tasks 2.4-2.6 (explanatory analysis, user/item groups, and
conclusions); Task 3.4 (group effects of reranking); user-side fairness
reranker contribution; report coordination.

- **1-7 Oct:** prepare the report outline by **2 Oct** and agree user/item
  groups with PERSON-1/3 by **7 Oct**. Work with PERSON-4 on the user-side
  fairness definition for the feedback session.
- **8-13 Oct:** hand off the user-side fairness method to PERSON-4 by **11
  Oct**; finish review and corrections by **13 Oct**. Prepare the explanatory
  and group analyses so they can use results as soon as they arrive.
- **19-24 Oct:** receive coefficient and experiment results from PERSON-1/3/4
  by **19 Oct**. Complete explanatory analysis by **19 Oct**, group analysis
  by **20 Oct**, and reranker group analysis by **21 Oct**. Draft the report by
  **22 Oct**, check course formatting and content by **23 Oct**, and resolve
  remaining comments by **24 Oct**.
- **25 Oct:** inspect the actual ZIP with PERSON-1 by **12:00**, collect
  sign-off, and confirm upload readiness by **18:00**.

**Collaborates with / may be blocked by:** the analyses depend on final,
comparable results from PERSON-1, PERSON-3, and PERSON-4. Start with available
results and mark provisional findings clearly rather than waiting for every
experiment to finish.

## Shared checkpoints

- **30 Sep, 10:00:** kickoff; assign names and agree how the group will
  communicate and record tasks.
- **By 2 Oct:** agree the model list and shared evaluation setup (split,
  candidates, relevance rule, cutoff, and primary metric).
- **By 5 Oct:** agree the hybrid fitting approach and metric/fairness
  definitions. Take unresolved questions to the course feedback session on
  **7 or 8 Oct**; record any resulting changes by **9 Oct**.
- **By 7 Oct:** agree user/item groups. **By 11 Oct:** agree the final,
  manageable experiment matrix.
- **17:00 check-ins:** 2, 6, 9, 13, 16, 20, and 23 Oct. Each person gives a
  short update: completed work, next handoff, and any blocker.

Use one shared evaluation setup so model results can be compared. Keep enough
run information to identify the configuration and data split behind each
reported result. Fit and tune on development data; use the test data for final
evaluation. Write report sections as results become available. A teammate
should check the key result and matching report claim before it is treated as
final.

## Final dates

- **22 Oct:** full report draft assembled.
- **23 Oct:** runnable-code instructions and report compliance check complete.
- **24 Oct, 18:00:** results and corrections accepted; representative runs
  reproduced from a clean checkout.
- **25 Oct, 12:00:** inspect the actual group ZIP. By **18:00**, approve the
  package, confirm the submitter, and have individual peer evaluations ready.
- **26 Oct, 18:00:** target upload time for the group ZIP and individual peer
  evaluations. The course deadline is **23:59**.
