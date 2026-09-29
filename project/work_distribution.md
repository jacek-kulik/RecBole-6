# Project work distribution and timeline

This is a proposed plan for five students from **30 September to 26 October
2026**. Replace PERSON-1 to PERSON-5 with names at the kickoff. The
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

## Tasks and reviewers

The **worker** completes the implementation or analysis and writes its report
section. The **reviewer** checks the result and the matching claim. The two
dates show when work is ready to share and when feedback is due; they are
proposed team dates, not course deadlines. All times are Europe/Amsterdam,
with **18:00** assumed unless shown otherwise.

| Task | Work | Worker | Reviewer | Handoff | Review |
| --- | --- | --- | --- | --- | --- |
| 1.1 | Individual recommenders | PERSON-2 | PERSON-3 | 6 Oct | 7 Oct |
| 1.2 | Tune individual models | PERSON-2 | PERSON-4 | 12 Oct | 13 Oct |
| 1.3 | Weighted hybrid | PERSON-3 | PERSON-2 | 10 Oct | 11 Oct |
| 1.4 | Other hybrid models | PERSON-3 | PERSON-5 | 13 Oct | 14 Oct |
| 1.5 | Tune hybrid models | PERSON-3 | PERSON-1 | 16 Oct | 17 Oct |
| 2.1 | Implement accuracy and beyond-accuracy metrics separately | PERSON-4 | PERSON-5 | 10 Oct | 11 Oct |
| 2.2 | Compare selected models with random and popularity baselines | PERSON-1 | PERSON-2 | 19 Oct | 20 Oct |
| 2.3 | Analyze hybrid coefficients and component contributions | PERSON-3 | PERSON-5 | 19 Oct | 20 Oct |
| 2.4 | Explain model behavior with a focused analysis | PERSON-5 | PERSON-3 | 19 Oct | 20 Oct |
| 2.5 | Compare results across user and item groups | PERSON-5 | PERSON-1 | 20 Oct | 21 Oct |
| 2.6 | Draw insights and propose a feasible improvement | PERSON-5 | PERSON-4 | 22 Oct | 23 Oct |
| 3.1 | Integrate the four rerankers | PERSON-4 | PERSON-2 | 14 Oct | 15 Oct |
| 3.2 | Evaluate reranked models and accuracy trade-offs | PERSON-4 | PERSON-1 | 19 Oct | 20 Oct |
| 3.3 | Compare both hybrid/reranker orders | PERSON-3 | PERSON-4 | 19 Oct | 20 Oct |
| 3.4 | Analyze reranker effects on user and item groups | PERSON-5 | PERSON-3 | 21 Oct | 22 Oct |

Task 3.1 is shared work. PERSON-4 owns the combined result and implements
calibration and item-side fairness. Each contribution has a worker and reviewer:

| Reranker | Worker | Reviewer | Handoff | Review |
| --- | --- | --- | --- | --- |
| Diversification | PERSON-2 | PERSON-4 | 10 Oct | 11 Oct |
| Calibration | PERSON-4 | PERSON-5 | 11 Oct | 12 Oct |
| Item-side fairness | PERSON-4 | PERSON-2 | 11 Oct | 12 Oct |
| User-side fairness | PERSON-5 | PERSON-4 | 11 Oct | 12 Oct |

## Timeline for each person

The last column names the collaboration or input that can delay the work.
Share a usable result as soon as it exists; the report section can be polished
after the dependent person has started.

### PERSON-1 — shared experiments and baselines

| When | Work and handoff | Collaboration or possible blocker |
| --- | --- | --- |
| 30 Sep-5 Oct | Agree the split, candidate set, metric, and result format; prepare the runner and random/popularity baselines. | Work with PERSON-2/3/4 on the shared format. Their later results need this setup. |
| 6 Oct | Assemble the first end-to-end example and feedback questions by 16:00; group checks them at 17:00. | Needs PERSON-2's predictions by 5 Oct, PERSON-3's first hybrid by 6 Oct at 12:00, and PERSON-4's metric pilot. |
| 7-11 Oct | Record feedback-session answers by 9 Oct; agree the final experiment set and compute budget by 11 Oct. | Decide with all five; unresolved course answers may change earlier choices. |
| 17-21 Oct | Review Task 1.5 on 17 Oct; record selected configurations on 18 Oct; deliver Task 2.2 on 19 Oct and resolve its review by 21 Oct. Review Task 3.2 on 20 Oct and Task 2.5 on 21 Oct; send results to PERSON-5. | Needs selected models from PERSON-2/3 and comparable outputs from PERSON-4. |
| 23-26 Oct | Prepare run instructions, help check a clean checkout, inspect the ZIP with PERSON-5, and support the group upload. | Needs runnable code and final sections from all workers; PERSON-4 helps reproduce results. |

### PERSON-2 — individual models and diversification

| When | Work and handoff | Collaboration or possible blocker |
| --- | --- | --- |
| 30 Sep-6 Oct | Agree the class-covered model set with PERSON-3 by 2 Oct; send initial predictions by 5 Oct; deliver Task 1.1 on 6 Oct. | Needs PERSON-1's split and output format. PERSON-3 needs the predictions for the first hybrid. |
| 7-14 Oct | Deliver diversification on 10 Oct and tuned models on 12 Oct; address reviews by 14 Oct. Review Task 1.3 on 11 Oct and item-side fairness on 12 Oct. | PERSON-4 needs diversification for Task 3.1. Tuning uses PERSON-1's shared evaluation setup. |
| 15-24 Oct | Review the combined rerankers on 15 Oct and Task 2.2 on 20 Oct; update model sections from final results. | Depends on PERSON-4's integration and PERSON-1's comparison table. |

### PERSON-3 — hybrid models and coefficient analysis

| When | Work and handoff | Collaboration or possible blocker |
| --- | --- | --- |
| 30 Sep-6 Oct | Agree hybrid scope and fitting approach; supply a first weighted-hybrid output by 6 Oct at 12:00. Bring unresolved regression questions to the feedback session. | Needs PERSON-2's predictions by 5 Oct and PERSON-1's shared output format. |
| 7-18 Oct | Review Task 1.1 on 7 Oct; incorporate course feedback; deliver Tasks 1.3, 1.4, and 1.5 on 10, 13, and 16 Oct; close corrections by 18 Oct. | PERSON-2, PERSON-5, and PERSON-1 review the hybrid tasks, respectively. Hybrid tuning depends on the agreed evaluation setup. |
| 19-24 Oct | Deliver coefficient analysis and both Task 3.3 orders on 19 Oct; review Task 2.4 on 20 Oct and Task 3.4 on 22 Oct; update report sections. | Needs PERSON-4's rerankers by 16 Oct. PERSON-5 needs the analysis results for the report. |

### PERSON-4 — metrics and rerankers

| When | Work and handoff | Collaboration or possible blocker |
| --- | --- | --- |
| 30 Sep-6 Oct | Agree metric and fairness definitions with PERSON-5 by 5 Oct; provide an independent accuracy calculation for the 6 Oct example. | Needs PERSON-1's split, candidate, and result format. |
| 7-16 Oct | Agree the reranker interface by 7 Oct; deliver Task 2.1 on 10 Oct, calibration and item-side fairness on 11 Oct, and the combined Task 3.1 on 14 Oct. Review diversification on 11 Oct, user-side fairness on 12 Oct, and Task 1.2 on 13 Oct; close integration fixes by 16 Oct. | Needs PERSON-2's diversification by 10 Oct and PERSON-5's user-side fairness by 11 Oct. |
| 17-24 Oct | Deliver Task 3.2 on 19 Oct; review Task 3.3 on 20 Oct and Task 2.6 on 23 Oct; send results to PERSON-5 and help reproduce key outputs. | Needs PERSON-1's shared predictions and the selected model configurations. |

### PERSON-5 — analysis and report coordination

| When | Work and handoff | Collaboration or possible blocker |
| --- | --- | --- |
| 30 Sep-7 Oct | Outline the report by 2 Oct; agree user/item groups with PERSON-1/3 by 7 Oct; clarify user-side fairness with PERSON-4. | Needs the shared evaluation setup and fairness definition before fixing groups or metadata. |
| 8-14 Oct | Deliver user-side fairness by 11 Oct; review Task 2.1 on 11 Oct, calibration on 12 Oct, and Task 1.4 on 14 Oct; prepare the analyses and address PERSON-4's review by 13 Oct. | Needs PERSON-4's reranker interface. PERSON-4 needs this contribution for Task 3.1. |
| 19-24 Oct | Deliver Tasks 2.4, 2.5, 3.4, and 2.6 on 19, 20, 21, and 22 Oct; review Task 2.3 on 20 Oct. Assemble the report on 22 Oct, check compliance on 23 Oct, and close comments by 24 Oct. | Needs comparable final results from PERSON-1/3/4 by 19 Oct; start with provisional outputs if needed. |
| 25-26 Oct | Inspect the ZIP with PERSON-1, collect sign-off, and check the upload receipt. | Needs final code and sections from all workers. |

## Shared checkpoints and submission

| Date | Group milestone |
| --- | --- |
| 30 Sep, 10:00 | Kickoff: assign names and agree how to share work and blockers. |
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
