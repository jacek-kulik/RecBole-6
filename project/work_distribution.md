# Project work distribution and timeline

This is a proposed plan for five students from **30 September to 26 October
2026**. Roles are assigned to the team members listed below. The
[course brief](Project-RecSys.pdf) sets the required tasks; the previous,
more detailed plan is kept in [work_distribution_OLD.md](work_distribution_OLD.md).

The project uses RecBole and MovieLens 100K for offline ranking experiments.
Cover all 15 numbered subtasks, but keep the experiment set manageable: limit
repeated tuning and redundant comparisons. The brief emphasizes depth of
analysis over the number of models or metrics.

The group submits a PDF report and runnable code in a ZIP named after its
group number. Each student separately submits a peer-evaluation Excel file.
The report allows at most one page per numbered task and 200 words of
experimental-results discussion per task. Keep its main body self-contained
and use Times New Roman, 12 pt, with 1.15 line spacing.

## Team assignments

Assignments use each person's first choice in `preferences.csv`, with Caio's
updated first choice of role 1 overriding his earlier entry. The five first
choices are then distinct, so everyone receives their preferred role without
a tie-break. The CSV remains the recorded ranking; this plan records Caio's
subsequent change without inferring his remaining preferences.

| Original role | Team member | Responsibility |
| --- | --- | --- |
| 1 | Caio | Shared experiments, baselines, and cross-task review |
| 2 | Gabriel | Individual models and diversification |
| 3 | Bogdan | Hybrid models and coefficient analysis |
| 4 | Jacek | Metrics and rerankers |
| 5 | Victor | Analysis and report coordination |

## Tasks and reviewers

The **worker** completes the implementation or analysis and writes its report
section. The **reviewer** checks the result and the matching claim. The two
dates show when work is ready to share and when feedback is due; they are
proposed team dates, not course deadlines. All times are Europe/Amsterdam,
with **18:00** assumed unless shown otherwise.

| Task | Work | Worker | Reviewer | Handoff | Review |
| --- | --- | --- | --- | --- | --- |
| 1.1 | Individual recommenders | Gabriel | Bogdan | 6 Oct | 7 Oct |
| 1.2 | Tune individual models | Gabriel | Caio | 12 Oct | 13 Oct |
| 1.3 | Weighted hybrid | Bogdan | Gabriel | 10 Oct | 11 Oct |
| 1.4 | Other hybrid models | Bogdan | Caio | 13 Oct | 14 Oct |
| 1.5 | Tune hybrid models | Bogdan | Caio | 16 Oct | 17 Oct |
| 2.1 | Implement accuracy and beyond-accuracy metrics separately | Jacek | Caio | 10 Oct | 11 Oct |
| 2.2 | Compare selected models with random and popularity baselines | Caio | Gabriel | 19 Oct | 20 Oct |
| 2.3 | Analyze hybrid coefficients and component contributions | Bogdan | Victor | 19 Oct | 20 Oct |
| 2.4 | Explain model behavior with a focused analysis | Victor | Bogdan | 19 Oct | 20 Oct |
| 2.5 | Compare results across user and item groups | Victor | Caio | 20 Oct | 21 Oct |
| 2.6 | Draw insights and propose a feasible improvement | Victor | Jacek | 22 Oct | 23 Oct |
| 3.1 | Integrate the four rerankers | Jacek | Gabriel | 14 Oct | 15 Oct |
| 3.2 | Evaluate reranked models and accuracy trade-offs | Jacek | Caio | 19 Oct | 20 Oct |
| 3.3 | Compare both hybrid/reranker orders | Bogdan | Jacek | 19 Oct | 20 Oct |
| 3.4 | Analyze reranker effects on user and item groups | Victor | Caio | 21 Oct | 22 Oct |

Task 3.1 is shared work. Jacek owns the combined result and implements
calibration and item-side fairness. Each contribution has a worker and reviewer:

| Reranker | Worker | Reviewer | Handoff | Review |
| --- | --- | --- | --- | --- |
| Diversification | Gabriel | Jacek | 10 Oct | 11 Oct |
| Calibration | Jacek | Victor | 11 Oct | 12 Oct |
| Item-side fairness | Jacek | Gabriel | 11 Oct | 12 Oct |
| User-side fairness | Victor | Jacek | 11 Oct | 12 Oct |

## Review workload

Implementation, analysis ownership, and report-section authorship stay with the
original workers. Caio takes a larger share of independent review: Tasks
2.1 (11 Oct), 1.2 (13 Oct), 1.4 (14 Oct), 1.5 (17 Oct), 3.2 (20 Oct),
2.5 (21 Oct), and 3.4 (22 Oct). This is seven numbered-task reviews, up from
three, distributed across the project rather than concentrated at submission.

These reviews are substantive: inspect the relevant code and experimental
choices, reproduce a small representative calculation or result, check the
matching report claims, and record actionable feedback plus its resolution.
Workers supply runnable evidence with each handoff and implement corrections;
Caio checks closure. Reranker contribution reviews remain distributed among
the other participants, and Gabriel independently reviews Caio's Task 2.2.

## Timeline for each person

The last column names the collaboration or input that can delay the work.
Share a usable result as soon as it exists; the report section can be polished
after the dependent person has started.

### Caio — shared experiments, baselines, and cross-task review

| When | Work and handoff | Collaboration or possible blocker |
| --- | --- | --- |
| 30 Sep-5 Oct | Agree the split, candidate set, metric, and result format; prepare the runner and random/popularity baselines. | Work with Gabriel, Bogdan, and Jacek on the shared format. Their later results need this setup. |
| 6 Oct | Assemble the first end-to-end example and feedback questions by 16:00; group checks them at 17:00. | Needs Gabriel's predictions by 5 Oct, Bogdan's first hybrid by 6 Oct at 12:00, and Jacek's metric pilot. |
| 7-11 Oct | Record feedback-session answers by 9 Oct; agree the final experiment set and compute budget by 11 Oct; review the independent metrics (Task 2.1) on 11 Oct. | Decide the experiment scope with all five. Needs Jacek's metric implementation and worked examples by 10 Oct; unresolved course answers may change earlier choices. |
| 12-16 Oct | Review individual-model tuning (Task 1.2) on 13 Oct and alternative hybrids (Task 1.4) on 14 Oct; verify development/test separation, comparison fairness, and supporting evidence. | Needs Gabriel's tuning results by 12 Oct and Bogdan's alternative hybrids by 13 Oct. Workers make corrections and return them for review. |
| 17-21 Oct | Review Task 1.5 on 17 Oct; record selected configurations on 18 Oct; deliver Task 2.2 on 19 Oct and resolve its review by 21 Oct. Review Task 3.2 on 20 Oct and Task 2.5 on 21 Oct; send results to Victor. | Needs selected models from Gabriel and Bogdan and comparable outputs from Jacek. |
| 22 Oct | Review reranker effects across user/item groups (Task 3.4), checking group definitions, sample counts, and whether the evidence supports the conclusions. | Needs Victor's analysis by 21 Oct; send feedback in time for report corrections. |
| 23-26 Oct | Close outstanding review comments; prepare run instructions, help check a clean checkout, inspect the ZIP with Victor, and support the group upload. | Needs runnable code and final sections from all workers; Jacek helps reproduce results. |

### Gabriel — individual models and diversification

| When | Work and handoff | Collaboration or possible blocker |
| --- | --- | --- |
| 30 Sep-6 Oct | Agree the class-covered model set with Bogdan by 2 Oct; send initial predictions by 5 Oct; deliver Task 1.1 on 6 Oct. | Needs Caio's split and output format. Bogdan needs the predictions for the first hybrid. |
| 7-14 Oct | Deliver diversification on 10 Oct and tuned models on 12 Oct; address reviews by 14 Oct. Review Task 1.3 on 11 Oct and item-side fairness on 12 Oct. | Jacek needs diversification for Task 3.1. Tuning uses Caio's shared evaluation setup; Caio reviews Task 1.2 on 13 Oct. |
| 15-24 Oct | Review the combined rerankers on 15 Oct and Task 2.2 on 20 Oct; update model sections from final results. | Depends on Jacek's integration and Caio's comparison table. |

### Bogdan — hybrid models and coefficient analysis

| When | Work and handoff | Collaboration or possible blocker |
| --- | --- | --- |
| 30 Sep-6 Oct | Agree hybrid scope and fitting approach; supply a first weighted-hybrid output by 6 Oct at 12:00. Bring unresolved regression questions to the feedback session. | Needs Gabriel's predictions by 5 Oct and Caio's shared output format. |
| 7-18 Oct | Review Task 1.1 on 7 Oct; incorporate course feedback; deliver Tasks 1.3, 1.4, and 1.5 on 10, 13, and 16 Oct; close corrections by 18 Oct. | Gabriel reviews Task 1.3; Caio reviews Tasks 1.4 and 1.5. Hybrid tuning depends on the agreed evaluation setup. |
| 19-24 Oct | Deliver coefficient analysis and both Task 3.3 orders on 19 Oct; review Task 2.4 on 20 Oct; update report sections. | Needs Jacek's rerankers by 16 Oct. Victor needs the analysis results for the report. |

### Jacek — metrics and rerankers

| When | Work and handoff | Collaboration or possible blocker |
| --- | --- | --- |
| 30 Sep-6 Oct | Agree metric and fairness definitions with Victor by 5 Oct; provide an independent accuracy calculation for the 6 Oct example. | Needs Caio's split, candidate, and result format. |
| 7-16 Oct | Agree the reranker interface by 7 Oct; deliver Task 2.1 on 10 Oct, calibration and item-side fairness on 11 Oct, and the combined Task 3.1 on 14 Oct. Review diversification on 11 Oct and user-side fairness on 12 Oct; address Caio's Task 2.1 review from 11 Oct; close integration fixes by 16 Oct. | Needs Gabriel's diversification by 10 Oct and Victor's user-side fairness by 11 Oct. |
| 17-24 Oct | Deliver Task 3.2 on 19 Oct; review Task 3.3 on 20 Oct and Task 2.6 on 23 Oct; send results to Victor and help reproduce key outputs. | Needs Caio's shared predictions and the selected model configurations. |

### Victor — analysis and report coordination

| When | Work and handoff | Collaboration or possible blocker |
| --- | --- | --- |
| 30 Sep-7 Oct | Outline the report by 2 Oct; agree user/item groups with Caio and Bogdan by 7 Oct; clarify user-side fairness with Jacek. | Needs the shared evaluation setup and fairness definition before fixing groups or metadata. |
| 8-14 Oct | Deliver user-side fairness by 11 Oct; review calibration on 12 Oct; prepare the analyses and address Jacek's review by 13 Oct. | Needs Jacek's reranker interface. Jacek needs this contribution for Task 3.1. |
| 19-24 Oct | Deliver Tasks 2.4, 2.5, 3.4, and 2.6 on 19, 20, 21, and 22 Oct; review Task 2.3 on 20 Oct. Assemble the report on 22 Oct, check compliance on 23 Oct, and close comments by 24 Oct. | Needs comparable final results from Caio, Bogdan, and Jacek by 19 Oct; start with provisional outputs if needed. Caio reviews Task 3.4 on 22 Oct. |
| 25-26 Oct | Inspect the ZIP with Caio, collect sign-off, and check the upload receipt. | Needs final code and sections from all workers. |

## Shared checkpoints and submission

| Date | Group milestone |
| --- | --- |
| 30 Sep, 10:00 | Kickoff: confirm role assignments and agree how to share work and blockers. |
| 2 Oct | Agree the model list and common evaluation setup. |
| 5 Oct | Agree hybrid fitting and metric/fairness definitions. |
| 6 Oct | Check the first end-to-end example and questions for the course feedback session. |
| 7 or 8 Oct | Attend the course feedback session; record any changed decisions by 9 Oct. |
| 11 Oct | Agree a manageable final experiment set. |
| 22 Oct | Assemble the full report draft. |
| 24 Oct, 18:00 | Finish corrections and reproduce representative results from a clean checkout. |
| 25 Oct, 12:00 | Inspect the actual ZIP; approve it and have individual peer evaluations ready by 18:00. |
| 26 Oct, 18:00 | Target group ZIP and individual peer-evaluation uploads. The official deadline is **23:59**. |

Hold short group check-ins at **17:00** on 2, 6, 9, 13, 16, 20, and 23 Oct.
At each one, report the next handoff and any blocker. Use the same split,
candidates, and separately implemented metrics for final comparisons. Choose
settings on development data and reserve test data for final evaluation.
