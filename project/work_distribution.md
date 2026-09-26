# Work distribution and proposed timeline

This is a proposed working agreement for five students. Replace `PERSON-1` through
`PERSON-5` with names once the group agrees on roles. The course brief is
[Project-RecSys.pdf](Project-RecSys.pdf). All numbered subtasks are in scope.

The project uses RecBole, MovieLens 100K, and an offline **ranking** evaluation.
The group will submit one PDF report and a directory of runnable code in a ZIP
named after its group number. Each student submits a peer evaluation separately.
The course deadline is **26 October 2026, 23:59**. Our proposed internal deadline
for a complete, reviewable submission is **25 October, 18:00**.

## How to use this plan

- The **owner** gets the subtask working, records decisions and results, drafts
  its report section, and addresses review comments.
- The **reviewer** checks the implementation, experiment protocol, evidence,
  and report claims. Review means reproducing a representative result or
  checking its inputs and outputs; it is more than reading the prose.
- Supporting contributors can help with code or experiments, but the owner
  remains responsible for integration. Tell the group early if a milestone is
  slipping so another person can help.
- Every subtask gets a concise main-body report section. The confirmed limit is
  **one page per numbered subtask** and **at most 200 words of discussion of
  experimental results per subtask**. Draft sections during the project, not
  after all experiments finish.

## Ownership at a glance

| Person | Primary area | Owns numbered subtasks | Designated reviews |
| --- | --- | --- | --- |
| PERSON-1 | Experiment protocol, shared runner, baselines, reproducibility | 2.2 | 1.5, 2.5, 3.2 |
| PERSON-2 | Individual recommenders and tuning | 1.1, 1.2 | 1.3, 2.2, 3.1 |
| PERSON-3 | Hybrid models and coefficient analysis | 1.3, 1.4, 1.5, 2.3, 3.3 | 1.1, 2.4, 3.4 |
| PERSON-4 | Independent metrics and reranker framework | 2.1, 3.1, 3.2 | 1.2, 2.6, 3.3 |
| PERSON-5 | Group analysis, interpretation, report coordination | 2.4, 2.5, 2.6, 3.4 | 1.4, 2.1, 2.3 |

The number of assigned subtasks does not by itself measure workload. PERSON-1
owns the shared run and result pipeline, which every other area depends on.
PERSON-4 leads Task 3.1, with PERSON-2 contributing a diversification method
and PERSON-5 contributing a user-side fairness or calibration method. The team
should rebalance work after its first complete experiment.

### PERSON-1: experiment and integration lead

- Agree with the team on the train/validation/test protocol, candidate set,
  ranking cutoff(s), relevance definition, seeds, and primary selection metric.
  Record these before comparing models.
- Maintain the shared command or script that runs experiments and exports the
  same user IDs, item IDs, candidate items, scores, and ranked lists for all
  models. Ensure the split and candidate set can be identified for each run.
- Own the random and popularity baselines, the final comparison table, and
  Task 2.2. Keep a record of the code commit, config, seed, split, and output
  location for every reported experiment.
- Review hybrid tuning (1.5), group comparisons (2.5), and reranker trade-offs
  (3.2). In particular, check that comparisons use the same evaluation data
  and that test results were not used to choose settings.
- Assemble a clean run instruction with the other owners; check that it works
  from a fresh checkout before submission.

### PERSON-2: individual-model lead

- Own the class-covered individual recommenders in Task 1.1 and their tuning
  in Task 1.2. Inventory the models discussed in class, run the models the
  brief requires, and record any scope question for the feedback session.
  The brief says content-based models are an exception to the models already
  implemented in RecBole.
- Coordinate score export with PERSON-1 so every model supplies predictions
  on the same candidates for hybrids and separate evaluation.
- Select configurations on development data using the agreed metric. Keep the
  search space, budget, failures, chosen parameters, and run identifiers.
- Implement one diversification reranker under PERSON-4's common interface.
- Review the weighted hybrid (1.3), final model comparison (2.2), and all
  Task 3.1 rerankers. Check score alignment and whether reported gains reflect
  consistent inputs.

### PERSON-3: hybrid lead

- Own the weighted hybrid (1.3), other class-covered or justified hybrid
  approaches (1.4), and hybrid tuning (1.5). Specify how base scores are
  normalized and how coefficients are fitted without test-set information.
- Own coefficient/contribution analysis (2.3). Report more than coefficient
  magnitudes: show what changes when a component is removed or its weight
  changes, and connect that to ranking outcomes.
- Own the two required orders in Task 3.3: combine individually reranked
  outputs, and rerank after combining individual outputs. Coordinate the
  interfaces and the experiment grid with PERSON-4.
- Review individual recommenders (1.1), explanatory analysis (2.4), and
  group effects of reranking (3.4). Check that hybrid inputs come from the
  intended model versions and split.

### PERSON-4: metrics and reranking lead

- Own separately implemented accuracy and beyond-accuracy metrics (2.1).
  Translate the lecture definitions into an agreed specification: formula,
  cutoff, aggregation, missing-data behavior, required metadata, and an
  example with a known answer for each metric. RecBole's metrics can be used
  for comparison, but the project evaluation must use the group's code.
- Own the reranker interface and Task 3.1 as a whole: diversification,
  calibration, item-side fairness, and user-side fairness. Share concrete
  implementation work with PERSON-2 and PERSON-5 as described above. Keep
  each method's objective and control parameter explicit.
- Own Task 3.2: evaluate reranked individual models, including accuracy and
  beyond-accuracy effects across several reranking strengths.
- Review individual tuning (1.2), conclusions and proposed improvement
  (2.6), and both Task 3.3 pipeline orders. Check that metric definitions
  match the reported claims and that rerankers operate on comparable inputs.

### PERSON-5: analysis and report lead

- Own the explanatory analysis design (2.4), user and item group comparisons
  (2.5), interpretation or improvement proposal (2.6), and group effects of
  reranking (3.4). Use groups defined from information available before test
  evaluation, such as training-set activity or popularity. Document group
  sizes and avoid conclusions from tiny groups.
- Contribute one user-side fairness or calibration reranker to Task 3.1 under
  PERSON-4's interface. Clarify the required attributes and metric from the
  lectures before implementing it.
- Coordinate the report outline, figure style, terminology, and page/word
  limits. Each owner still writes and revises their own numbered section;
  PERSON-5 checks that the full report tells a coherent story.
- Review other hybrid choices (1.4), metric implementations (2.1), and
  coefficient analysis (2.3). Check whether evidence supports the stated
  explanations rather than only restating which score was higher.

## Subtask deliverables and review questions

Each row names the person accountable for completion and the person who gives
the first substantive review. A report section is part of every row's output.

| Subtask | Owner | Reviewer | Minimum handoff and review focus |
| --- | --- | --- | --- |
| 1.1 Individual recommenders | PERSON-2 | PERSON-3 | Runnable model configs, candidate-aligned outputs, and rationale for the model set; reviewer checks model identity and output alignment. |
| 1.2 Tune individual models | PERSON-2 | PERSON-4 | Search spaces, validation selection, chosen configurations, and run records; reviewer checks comparable budgets and split use. |
| 1.3 Weighted hybrid | PERSON-3 | PERSON-2 | Regression target, fitting split, score transformation, learned coefficients, and ranked outputs; reviewer checks leakage and score alignment. |
| 1.4 Other hybrids | PERSON-3 | PERSON-5 | Implemented class-covered alternatives and reasons for choosing them; reviewer checks that they differ meaningfully from 1.3. |
| 1.5 Tune hybrids | PERSON-3 | PERSON-1 | Tuning ranges, validation-based selection, and selected settings; reviewer checks that test data did not guide choices. |
| 2.1 Implement metrics | PERSON-4 | PERSON-5 | Standalone accuracy and beyond-accuracy code plus definitions and hand-calculated examples; reviewer checks formulas and aggregation. |
| 2.2 Compare models and baselines | PERSON-1 | PERSON-2 | Final comparison of selected individual/hybrid models with random and popularity baselines; reviewer checks same split, candidates, and metrics. |
| 2.3 Analyze hybrid coefficients | PERSON-3 | PERSON-5 | Coefficients, normalization, component ablations or sensitivity, and interpretation; reviewer checks that the contribution claim has evidence. |
| 2.4 Design explanatory analysis | PERSON-5 | PERSON-3 | At least one analysis that investigates a mechanism, such as sparsity or component agreement; reviewer checks the hypothesis and comparison. |
| 2.5 Analyze user/item groups | PERSON-5 | PERSON-1 | Group definitions, counts, accuracy and beyond-accuracy results; reviewer checks group construction and small-sample caveats. |
| 2.6 Draw insights and propose improvement | PERSON-5 | PERSON-4 | Evidence-backed synthesis and a feasible hybrid change or follow-up experiment; reviewer checks whether the claim follows from 2.4-2.5. |
| 3.1 Implement rerankers | PERSON-4 | PERSON-2 | Runnable diversity, calibration, user-side fairness and item-side fairness methods, including contributions from PERSON-2 and PERSON-5; reviewer checks objectives and controls. |
| 3.2 Evaluate reranked models | PERSON-4 | PERSON-1 | Results at multiple reranking strengths and accuracy/beyond-accuracy trade-offs; reviewer checks matched inputs and baselines. |
| 3.3 Compare hybrid/reranker orders | PERSON-3 | PERSON-4 | Both required orders on the same underlying models and conditions; reviewer checks what changed between the two pipelines. |
| 3.4 Analyze reranker group effects | PERSON-5 | PERSON-3 | Before/after results for user and item groups, including harms as well as gains; reviewer checks group sizes and interpretation. |

## Shared decisions to settle early

Record each decision, its reason, and who agreed to it in a short project
protocol document or issue. The group owns these choices; the table above
assigns someone to drive each discussion.

| Decision | Driver | Latest proposed decision date | Why it matters |
| --- | --- | --- | --- |
| Models and hybrid methods from the lectures | PERSON-2 and PERSON-3 | 30 Sep | Sets implementation and compute scope while covering the required methods. |
| Ranking cutoff(s), candidate policy, relevance definition, splits, seed, primary metric | PERSON-1 and PERSON-4 | 30 Sep | Makes every model and reranker comparison comparable. |
| Regression target, coefficient fitting data, score normalization | PERSON-3 | 2 Oct | Prevents leakage and makes coefficients interpretable. Bring unresolved points to the 7-8 Oct feedback session. |
| Exact beyond-accuracy and fairness definitions from lectures | PERSON-4 and PERSON-5 | 2 Oct | Determines metadata, reranker objectives, and group analyses. |
| User and item groups, minimum useful group size | PERSON-5 | 4 Oct | Avoids choosing groups after seeing test results. |
| Final experiment grid and compute budget | PERSON-1 | 9 Oct | Keeps every required comparison feasible before writing starts. |

The regression requirement deserves a specific instructor question: should
Task 1.3 fit coefficients to raw ratings, a binary relevance label, or another
target, and are constraints on coefficient signs/weights expected? Until the
group gets an answer, its protocol should state a defensible choice and fit
coefficients using development data only.

## Interfaces and evidence everyone must share

1. **One experiment protocol.** Use the same MovieLens 100K data version,
   split, user/item IDs, candidate policy, seen-item filtering, ranking cutoff,
   relevance rule, and evaluation code for compared methods. Fit models,
   hybrid coefficients, and reranker settings only on their assigned
   development data. Keep test data for the final evaluation.
2. **One prediction contract.** A model output identifies the model, run,
   split, user, item, score, and rank. The contract also identifies the
   candidate set and any preprocessing. PERSON-1 and PERSON-3 agree on it
   before the first hybrid is built; PERSON-4 confirms rerankers can consume it.
3. **One result record.** For every plotted or reported number, preserve the
   source run ID, Git commit, configuration, seed, data split, metric version,
   and output file. Report means and variation if multiple seeds are used;
   do not mix numbers from different protocols in one comparison.
4. **Separate evaluation.** Use the group's metric implementations for all
   final tables, including metrics RecBole also computes. Compare against
   RecBole only as a diagnostic when the definitions match.
5. **A small, purposeful experiment matrix.** Cover every required task, but
   limit redundant model/parameter combinations. For Task 3.3, use the same
   base models and reranker strengths in both pipeline orders.
6. **A runnable code package.** Keep project code and instructions in a clear
   project directory; identify any patched RecBole files. Generated logs,
   environments, checkpoints, and bulky intermediate outputs should be
   reproducible from code and configs rather than required to run the ZIP.

## Proposed timeline

Dates are targets for a first complete pass, not permission to leave later
tasks until the final week. Start report sections when their first results
appear. Hold a 20-30 minute team check-in twice a week; each owner reports a
completed artifact, the next handoff, and any blocker.

| Period | Milestone | Owner and handoff |
| --- | --- | --- |
| **26-30 Sep** | Agree on scope from lectures and evaluation protocol. Create the task board, assign names, establish shared result formats and a clean runnable baseline. | PERSON-1 drives protocol and runner; PERSON-2 lists individual models; PERSON-3 lists hybrid methods; PERSON-4/5 list metric, reranker, and group definitions. Everyone signs off. |
| **1-6 Oct** | Complete first end-to-end run: individual model outputs, random/popularity baselines, independent accuracy metrics, initial hybrid, and report skeleton. | PERSON-2 hands predictions to PERSON-3; PERSON-1 hands run/split details to PERSON-4; PERSON-5 starts report sections and group definitions. |
| **7-8 Oct** | Attend the scheduled course feedback session with a working run and a short list of unresolved methodological questions, especially the regression target and fairness definitions. | PERSON-3 prepares the coefficient question; PERSON-4/5 prepare metric and fairness questions; PERSON-1 records answers and updates the protocol. |
| **9-13 Oct** | Finish individual tuning and hybrid variants; implement all required metrics and the first version of every reranker. Review interfaces before large experiment runs. | PERSON-2 and PERSON-3 hand selected configurations to PERSON-1; PERSON-4 integrates rerankers, with contributions from PERSON-2/5. |
| **14-18 Oct** | Run hybrid tuning, final model comparisons, reranking strength sweeps, and both hybrid/reranker orders. Begin coefficient, explanatory, and group analyses. | PERSON-1 coordinates run records; PERSON-3 and PERSON-4 hand results to PERSON-5; designated reviewers inspect representative runs. |
| **19-22 Oct** | Complete all 15 subtask analyses and draft sections. Rerun only experiments needed to resolve gaps or contradictory evidence. | Each owner writes their sections; each designated reviewer checks code/results and prose; PERSON-5 assembles the report. |
| **23-25 Oct** | Freeze results, reproduce representative outputs from a clean checkout, check report limits and formatting, prepare the runnable code directory and submission ZIP. | PERSON-1 checks run instructions; PERSON-5 checks report and ZIP contents; all five sign off on their owned and reviewed sections. **Internal complete target: 25 Oct, 18:00.** |
| **26 Oct** | Buffer for final corrections. Submit the group ZIP before **23:59**; each student submits their own peer evaluation. | One nominated submitter uploads the group ZIP and shares the submission receipt; every person handles their individual form. |

## Review and merge agreement

- Use short branches for changes and request review from the designated
  reviewer. A reviewer checks the changed code/config, one representative
  output or metric calculation, and the matching report claim before merging.
- Changes to shared split logic, evaluation definitions, user/item ID mapping,
  or prediction format need a second look from PERSON-1 or PERSON-4 because
  they affect several tasks.
- An experiment is report-ready when another team member can identify its
  source configuration and reproduce its evaluation from recorded outputs.
  A screenshot or an unlabeled CSV is not sufficient evidence.
- If a task slips, preserve the minimum required comparison and analysis for
  every subtask first. Reduce redundant model variants or tuning trials by
  group agreement; record the reduced scope in the report.
- Before submission, each owner confirms their code and section; each
  reviewer confirms their assigned checks; PERSON-1 and PERSON-5 verify the
  combined runnable package and report.
