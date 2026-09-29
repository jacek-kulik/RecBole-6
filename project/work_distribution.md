# Work distribution and proposed timeline

This is a proposed working agreement for five students. Replace `PERSON-1` through
`PERSON-5` with names once the group agrees on roles. The course brief is
[Project-RecSys.pdf](Project-RecSys.pdf). All numbered subtasks are in scope.

The project uses RecBole, MovieLens 100K, and an offline **ranking** evaluation.
The group will submit one PDF report and a directory of runnable code in a ZIP
named after its group number. Each student submits a peer evaluation separately.
The proposed project start is **30 September 2026**. The course deadline is
**26 October 2026, 23:59**. Our proposed internal deadline
for a complete, approved submission package is **25 October, 18:00**. All
subtask corrections should be accepted by **24 October, 18:00**, so packaging
does not depend on unfinished experiments or reviews.

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
and PERSON-5 contributing a user-side fairness method. PERSON-4 implements
calibration and item-side fairness. The team
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
- Implement the diversification reranker under PERSON-4's common interface.
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
  implementation work with PERSON-2 and PERSON-5 as described above; implement
  calibration and item-side fairness yourself. Keep
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
- Contribute the user-side fairness reranker to Task 3.1 under
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

| Decision | Driver | Proposal recorded by | Reviewer(s) | Decision agreed by | Why it matters |
| --- | --- | --- | --- | --- | --- |
| Models and hybrid methods from the lectures | PERSON-2 and PERSON-3 | 1 Oct | All five | 2 Oct | Sets implementation and compute scope while covering the required methods. |
| Ranking cutoff(s), candidate policy, relevance definition, splits, seed, primary metric | PERSON-1 and PERSON-4 | 1 Oct | All five | 2 Oct | Makes every model and reranker comparison comparable. |
| Regression target, coefficient fitting data, score normalization | PERSON-3 | 3 Oct | PERSON-1 and PERSON-2 | 5 Oct | Prevents leakage and makes coefficients interpretable. Bring unresolved points to the 7-8 Oct feedback session; record any resulting revision by 9 Oct. |
| Exact beyond-accuracy and fairness definitions from lectures | PERSON-4 and PERSON-5 | 3 Oct | PERSON-1 and PERSON-3 | 5 Oct | Determines metadata, reranker objectives, and group analyses. Record feedback-session clarifications by 9 Oct. |
| User and item groups, minimum useful group size | PERSON-5 | 5 Oct | PERSON-1 and PERSON-3 | 7 Oct | Avoids choosing groups after seeing test results. |
| Final experiment grid and compute budget | PERSON-1 | 10 Oct | All five | 11 Oct | Keeps every required comparison feasible before the final runs. Writing begins with the first results. |

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

The periods below give the overall sequence; the tables in **Detailed internal
deadlines** give the actual handoff, review, and acceptance dates. Start report
sections when their first results appear. Hold a 20-30 minute kickoff at
**10:00 on 30 Sep**, then team check-ins at **17:00 on Tuesdays and Fridays**:
2, 6, 9, 13, 16, 20, and 23 Oct.
Each owner reports a completed artifact, the next handoff, and any blocker.

| Period | Milestone | Owner and handoff |
| --- | --- | --- |
| **30 Sep-5 Oct** | Kick off, assign names, set up the task board and report outline, agree scope and protocol, and produce working runner, baseline, and metric pilots. | PERSON-1 coordinates setup; PERSON-2/3 list models and hybrids; PERSON-4/5 settle metric, reranker, and group definitions. |
| **6 Oct** | Complete the first end-to-end example: individual outputs, random/popularity baselines, an independent accuracy calculation, and an initial weighted hybrid. Prepare the feedback-session questions. | PERSON-2 hands predictions to PERSON-3; PERSON-1 supplies run/split details to PERSON-4; all five inspect the example. |
| **7-8 Oct** | Attend the scheduled course feedback session with a working run and a short list of unresolved methodological questions, especially the regression target and fairness definitions. | PERSON-3 prepares the coefficient question; PERSON-4/5 prepare metric and fairness questions; PERSON-1 records answers and updates the protocol. |
| **9-16 Oct** | Accept the metrics, tuned individual models, hybrid implementations, and all four rerankers. Agree the final experiment grid before committing the compute budget. | PERSON-2 hands selected individual configs to PERSON-1/3; PERSON-4 integrates PERSON-2/5's rerankers; designated reviewers check code and representative outputs. |
| **17-19 Oct** | Accept hybrid tuning, hand off the final model comparison and reranking sweeps, and run both hybrid/reranker orders. Start analyses as provisional results become available. | PERSON-1 coordinates run records; PERSON-3/4 supply results to PERSON-5. |
| **20-22 Oct** | Complete all 15 numbered subtask drafts, coefficient and explanatory analyses, and user/item-group results. Review the sections as they arrive. | Each owner writes their sections; each designated reviewer checks code/results and prose; PERSON-5 assembles the report. |
| **23-24 Oct** | Close the remaining reviews and corrections, reproduce representative results from a clean checkout, and check the report against the brief. | All subtask fixes accepted by **24 Oct, 18:00**; PERSON-1/5 finish the code and report checks. |
| **25 Oct** | Inspect the actual submission ZIP, complete individual peer evaluations, and obtain all five students' sign-off. | **Complete approved package by 18:00**; PERSON-1 coordinates upload readiness and PERSON-5 records sign-off. |
| **26 Oct** | Submit by the team's target of **18:00**, share the receipt, and use the remaining time to resolve upload problems. The official deadline is **23:59**. | One nominated submitter uploads the group ZIP; every person submits their individual peer-evaluation file. |

## Detailed internal deadlines

These are **proposed team deadlines**, not additional course requirements. All
dates below are in **2026**, and deadlines are **18:00 in Europe/Amsterdam**
unless a different time is stated. Agree the schedule at the 30 Sep kickoff.
No work is scheduled before the 30 Sep kickoff; the final package and official
submission dates are unchanged.

- **Owner handoff:** the deliverable in the subtask table above is ready for
  review, including runnable code/configs where applicable, result provenance,
  figures/tables, and a draft of its numbered report section. Starting a run
  on this date does not count as handing it off.
- **Review due:** the designated reviewer has checked the evidence and section
  and returned specific comments or approval. They should flag major issues
  as soon as they find them, rather than waiting for this date.
- **Fixes accepted:** the owner has addressed the comments, the reviewer has
  checked the corrections, and the code/results/section can be integrated.
  The owner remains responsible for getting this acceptance recorded.

Early implementations and provisional runs can overlap with reviews. Final
tables must use the accepted protocol, metric implementations, and selected
model configurations. If a review changes one of those inputs, rerun the
affected comparisons and update their report sections.

### Shared setup and early handoffs

The decision deadlines above cover methodological choices. This table covers
the shared artifacts that the numbered subtasks need.

| Shared deliverable | Owner | Reviewer | Handoff | Review due | Fixes accepted |
| --- | --- | --- | --- | --- | --- |
| Names assigned to all roles; task board with owners, reviewers, and dates; feedback-session slot booked | PERSON-1 | PERSON-5 | 30 Sep | 1 Oct | 2 Oct |
| Report outline with all 15 numbered sections, shared terminology, and figure/table conventions | PERSON-5 | PERSON-2 | 2 Oct | 3 Oct | 4 Oct |
| Runnable environment and dependency instructions; record of any RecBole patches | PERSON-1 | PERSON-2 | 3 Oct | 4 Oct | 5 Oct |
| Shared runner, fixed split/candidates, prediction contract, and result record containing run ID/config/seed/code version | PERSON-1, with PERSON-3 | PERSON-4 | 3 Oct | 4 Oct | 5 Oct |
| Reranker interface and a small example that consumes the shared predictions | PERSON-4 | PERSON-3 | 5 Oct | 6 Oct | 7 Oct |
| Random and popularity baselines with candidate-aligned outputs | PERSON-1 | PERSON-2 | 5 Oct | 6 Oct | 7 Oct |
| First independent accuracy calculation on recorded outputs; known-answer examples | PERSON-4 | PERSON-5 | 5 Oct | 6 Oct | 7 Oct |
| Task 2.4 analysis proposal: hypothesis, comparison, required outputs, and planned figure | PERSON-5 | PERSON-3 | 5 Oct | 6 Oct | 7 Oct |

Additional milestones for the feedback session:

- **5 Oct:** PERSON-2 supplies initial candidate-aligned predictions to
  PERSON-3, ahead of the full Task 1.1 handoff on 6 Oct.
- **6 Oct, 12:00:** PERSON-3 supplies a first weighted-hybrid output to
  PERSON-1, using those initial predictions. PERSON-2 inspects its alignment
  before the feedback pack is finalized; the final Task 1.3 handoff below
  incorporates course feedback.
- **6 Oct, 16:00:** PERSON-1 assembles the working end-to-end example and the
  question list. PERSON-3 supplies the regression question; PERSON-4/5 supply
  metric and fairness questions. All five check the pack at the 17:00 meeting
  and finalize it by **18:00**.
- **7 or 8 Oct:** attend the booked course feedback session.
- **9 Oct:** PERSON-1 records the answers and updates the protocol with the
  affected owners; PERSON-3/4/5 check that their questions and decisions are
  represented. Share any change that requires rerunning earlier work.

### Numbered subtasks: implementation, evidence, and section review

The dates include the draft section and its first substantive review, not
just implementation. Early sections are updated again during the final report
review if later results change their claims.

| Subtask | Owner | Reviewer | Owner handoff | Review due | Fixes accepted |
| --- | --- | --- | --- | --- | --- |
| 1.1 Individual recommenders | PERSON-2 | PERSON-3 | 6 Oct | 7 Oct | 8 Oct |
| 1.2 Tune individual models | PERSON-2 | PERSON-4 | 12 Oct | 13 Oct | 14 Oct |
| 1.3 Weighted hybrid | PERSON-3 | PERSON-2 | 10 Oct | 11 Oct | 12 Oct |
| 1.4 Other hybrids | PERSON-3 | PERSON-5 | 13 Oct | 14 Oct | 15 Oct |
| 1.5 Tune hybrids | PERSON-3 | PERSON-1 | 16 Oct | 17 Oct | 18 Oct |
| 2.1 Implement all agreed metrics | PERSON-4 | PERSON-5 | 10 Oct | 11 Oct | 12 Oct |
| 2.2 Compare selected models and baselines | PERSON-1 | PERSON-2 | 19 Oct | 20 Oct | 21 Oct |
| 2.3 Analyze hybrid coefficients/contributions | PERSON-3 | PERSON-5 | 19 Oct | 20 Oct | 21 Oct |
| 2.4 Complete explanatory analysis | PERSON-5 | PERSON-3 | 19 Oct | 20 Oct | 21 Oct |
| 2.5 Analyze user/item groups | PERSON-5 | PERSON-1 | 20 Oct | 21 Oct | 22 Oct |
| 2.6 Synthesize insights and propose improvement | PERSON-5 | PERSON-4 | 22 Oct | 23 Oct | 24 Oct |
| 3.1 Integrate all four rerankers | PERSON-4 | PERSON-2 | 14 Oct | 15 Oct | 16 Oct |
| 3.2 Evaluate reranked models and trade-offs | PERSON-4 | PERSON-1 | 19 Oct | 20 Oct | 21 Oct |
| 3.3 Compare both hybrid/reranker orders | PERSON-3 | PERSON-4 | 19 Oct | 20 Oct | 21 Oct |
| 3.4 Analyze reranker effects on user/item groups | PERSON-5 | PERSON-3 | 21 Oct | 22 Oct | 23 Oct |

Dependencies that determine the order:

- **1.2 and 2.1 are accepted by 14 Oct** before the final individual-model
  evaluation; **1.3-1.4 are accepted by 15 Oct** before final hybrid selection.
- **1.5 is accepted by 18 Oct** before the final 2.2/3.2 comparisons. PERSON-1
  records the selected base/hybrid configurations and their validation results
  on **18 Oct**; PERSON-2/3/4 check the selection before final evaluation runs.
- **3.1 is accepted by 16 Oct**, so 3.2 and 3.3 can consume the same reviewed
  reranker implementations. Both 3.3 orders use the same underlying models
  and conditions.
- PERSON-3 sends coefficient/ablation outputs to PERSON-5 by **19 Oct**.
  PERSON-1/4 send the final comparison and reranking outputs by **19 Oct**.
  PERSON-5 uses these for 2.4-2.6 and 3.4, rather than waiting for prose reviews
  to finish before starting the analyses.
- **2.5 is accepted by 22 Oct** before 2.6's review. Any proposed improvement
  that needs a new experiment is agreed with PERSON-1/3 by **20 Oct**, with
  results delivered by **22 Oct**. Otherwise, describe it as an evidence-based
  proposal and do not claim an improvement that has not been measured.

### Task 3.1 contributions

The separate method deadlines leave time for PERSON-4 to integrate the
rerankers before the overall 3.1 review. Each contribution includes its control
parameter, required metadata, example output, and explanation for the report.

| Method | Implementer | Method reviewer | Handoff | Review due | Fixes accepted |
| --- | --- | --- | --- | --- | --- |
| Diversification | PERSON-2 | PERSON-4 | 10 Oct | 11 Oct | 12 Oct |
| Calibration | PERSON-4 | PERSON-5 | 11 Oct | 12 Oct | 13 Oct |
| Item-side fairness | PERSON-4 | PERSON-2 | 11 Oct | 12 Oct | 13 Oct |
| User-side fairness | PERSON-5 | PERSON-4 | 11 Oct | 12 Oct | 13 Oct |

PERSON-2's overall 3.1 review on **15 Oct** checks all four methods together,
including the common interface and candidate handling. PERSON-4 closes any
integration issues by **16 Oct**.

### Full-task reviews and submission deadlines

These reviews check consistency across sections after the individual subtask
reviews. They do not replace the code/evidence reviews above.

| Deliverable or check | Responsible | Reviewer / sign-off | Deadline and completion condition |
| --- | --- | --- | --- |
| Task 1 combined section | PERSON-2 and PERSON-3 | PERSON-1 and PERSON-4 | **19 Oct:** review model identities, tuning protocol, score transformations, and coefficient claims across 1.1-1.5. Owners close comments by **20 Oct**. |
| Task 2 combined section | PERSON-1, PERSON-3, and PERSON-5 | PERSON-2 and PERSON-4 | **23 Oct:** review consistency across 2.1-2.6 and the link from evidence to conclusions. Owners close comments by **24 Oct**. |
| Task 3 combined section | PERSON-3, PERSON-4, and PERSON-5 | PERSON-1 and PERSON-2 | **23 Oct:** review consistency across 3.1-3.4, especially the two pipeline orders and group effects. Owners close comments by **24 Oct**. |
| Full report draft | PERSON-5, with all section owners | All five | **22 Oct:** assemble every section, figure, citation, and any appendix; outstanding review comments may remain explicitly tracked until 24 Oct. |
| Results ready for final report | Every owner | Every designated reviewer | **24 Oct:** accept all fixes, resolve disagreements, and record the final run IDs. Later changes need a stated reason and another check of affected tables/claims. |
| Runnable code directory and instructions | PERSON-1, with all code owners | PERSON-4 | **23 Oct:** hand off dependencies, configs, patched files, and commands to recreate reported evaluations. |
| Clean-checkout reproduction | PERSON-4 | PERSON-1 and affected owners | **24 Oct:** reproduce representative individual, hybrid, metric, and reranker outputs; record commands/results. Owners resolve blockers before packaging. |
| Report compliance check | PERSON-5 | PERSON-2 | **23 Oct:** check group details, all 15 sections, one-page/200-word limits, Times New Roman 12 pt, 1.15 spacing, captions, references, and a self-contained main body. Owners fix issues by **24 Oct**. |
| Submission ZIP assembled | PERSON-1 and PERSON-5 | PERSON-2 and PERSON-4 | **25 Oct, 12:00:** inspect the actual ZIP, named after the group number, for the final PDF and complete runnable code directory; verify instructions against its extracted contents. |
| Group sign-off and upload readiness | All five; PERSON-5 records confirmation | Each owner and reviewer confirms their sections | **25 Oct, 18:00:** approve the final package and nominate the submitter, with PERSON-1 as backup. |
| Individual peer-evaluation Excel files ready | Each person individually | Each person checks their own file | **25 Oct, 18:00:** complete the required file and have it ready to upload separately. |
| Group ZIP uploaded and receipt shared | Nominated submitter; PERSON-1 as backup | PERSON-5 checks receipt | **26 Oct, 18:00:** submit to Brightspace and share confirmation. |
| Individual peer evaluations submitted | Each person individually | Each person checks their own receipt | **26 Oct, 18:00:** upload the individual Excel file and confirm submission. |

The official deadline for the submissions remains **26 Oct, 23:59**. The gap
after the internal upload target is a buffer for submission problems.

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
- Raise a likely missed deadline at the next check-in, or immediately if a
  downstream handoff is at risk. The owner and reviewer agree a new date and
  tell everyone whose work depends on it. Record the revised handoff, review,
  and acceptance dates on the task board; do not silently move only the owner
  deadline or consume the final submission buffer.
- Before submission, each owner confirms their code and section; each
  reviewer confirms their assigned checks; PERSON-1 and PERSON-5 verify the
  combined runnable package and report.
