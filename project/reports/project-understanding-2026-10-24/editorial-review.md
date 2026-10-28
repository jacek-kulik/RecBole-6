# Writing review

This review records the reasoning behind the report's structure and checks its claims. The learning report presents the ideas and results directly; paragraph-role annotations remain in its Markdown source.

## Section outline

- Task, data, information boundaries and the individual models.
- Accuracy and beyond-accuracy definitions, followed by baseline interpretation.
- Training profiles and your calibration/exposure algorithms.
- Validation selection, measured trade-offs and absolute group quality.
- Hybrid learning, matched controls, ablation and pipeline order.
- Diversification and the interpretation of group-specific measurements.

## Reverse outline and paragraph roles

| Chapters | Main message | Prerequisites introduced first | Paragraph roles |
| --- | --- | --- | --- |
| 1–2 | Scores arise from different model assumptions under one defined ranking task. | Interaction, catalogue, candidate, score, ranking, split, training, validation, checkpoint and NDCG's purpose. | Opening → definitions → method → motivation → trade-off. |
| 3 | A ranking has several measurable properties with different denominators. | Hits and relevant sets precede formulas; distributions precede divergences; cohorts precede coverage and Gini. | Definition → worked example → aggregation → interpretation → limitation. |
| 4 | Baselines expose the difference between recovery, reach and concentration. | Chapter 3 defines every metric in the table. | Mechanism → evidence → interpretation → scope. |
| 5–6 | Training-only profiles and explicit objectives determine reranking behavior. | Fractional genre mass and popularity bands precede calibration; exposure precedes the shared counter. | Motivation → representation → algorithm → numerical example → trade-off. |
| 7 | Validation chooses an operating point; test reveals its costs. | Strength, objective direction and relative change precede selected-results tables. | Selection rule → example → evidence → group interpretation → limitation. |
| 8–9 | Hybrid design changes the information transferred, and state determines whether order matters. | Standardization and out-of-fold features precede Ridge results; score conversion precedes order results; shared state precedes switching interpretation. | Mechanism → control → evidence → explanation → limitation. |
| 10–11 | Variety, novelty, reach and group measurements remain distinct. | Metric definitions precede bonuses; group memberships precede conditional and audience results. | Definition → mechanism → evidence → denominator example → interpretation. |

Each prose paragraph has a role annotation in the source. Topic sentences state the paragraph's message. Repeated closing commentary was removed; technical terms were defined where their mechanism first needs them. The opening gives the project question without claiming a new state-of-the-art algorithm.

## Five-dimension self-review

| Dimension | Adversarial question | Answer and resulting revision |
| --- | --- | --- |
| Contribution | What understanding does the report add? | It connects mechanisms, controls and observed outcomes. It claims a comparative course-project study rather than methodological novelty. |
| Writing clarity | Can the reader follow each formula and decision without unexplained dependencies? | The report proceeds from task → models → metrics → baselines → profiles → rerankers → results → hybrids → group interpretation. Acronyms and mathematical objects receive local explanations. |
| Experimental strength | Are benefits material and correctly compared? | Each gain is tied to its own model or receiver control. The report includes zero-strength selection, accuracy losses, failed budget transfer and non-improving hybrid variants. |
| Evaluation completeness | Does the evidence support broad generalization or fairness claims? | No. Single-seed/reused-split limits and conditional bootstrap intervals remain explicit. Cross-seed robustness, stronger causal explanations and deployment benefit need new experiments. Claims were narrowed accordingly. |
| Method design | Does the description conceal a different task or optimization guarantee? | Full relevance is retained outside shortlists; warm scope is distinct; inverse exposure is a heuristic; greedy construction has no global-optimum guarantee. Group diagnostics remain separate from an unimplemented user-fairness intervention. |

## Claim-evidence map

Claim: The six personalized configurations recover held-out interactions more accurately than Popularity. | Evidence: Table 2, NDCG 0.348765–0.404872 versus 0.211725. | Status: supported.

Claim: Broad catalogue reach alone does not establish useful recommendation. | Evidence: Figure 5, Random coverage 99.58% with NDCG 0.014494. | Status: supported.

Claim: Calibration and exposure improve their designated targets at model-dependent accuracy costs. | Evidence: Table 3 and Figures 9–10; target reductions and paired original/reranked NDCG. | Status: supported.

Claim: Validation's five-percent budget does not uniformly transfer to test. | Evidence: Seven calibration/exposure and two coverage choices exceed it among 36 fixed choices. | Status: supported.

Claim: A smaller F/M gap need not improve either group's absolute quality. | Evidence: Figure 11; EASE exposure F NDCG 0.360967→0.346933 and M 0.422761→0.396597. | Status: supported.

Claim: The strongest Ridge coefficient is not a contribution percentage. | Evidence: Figure 13; ItemKNN has the largest coefficient, yet its frozen removal increases NDCG by 0.004194. | Status: supported.

Claim: Weak SLIM augmentation improves its matched receiver, while more inferred input is not uniformly better. | Evidence: Figure 12; 0.411904 versus 0.403254 and the weaker improvement at strength one. | Status: supported.

Claim: Post-reranking gives stronger target reductions and greater NDCG losses for the specified weighted conversion. | Evidence: Table 5, comparing each pre/post pair under fixed controls. | Status: supported.

Claim: Local calibration commutes with fixed switching, while the specified exposure policies differ. | Evidence: Exact equality across calibration strengths, and 747 differing exposure test lists. | Status: supported.

Claim: Tuned FISM retains zero novelty strength under the tested grid. | Evidence: Its first positive setting loses 5.4476% validation NDCG; all positive tested strengths are infeasible. | Status: supported.

Claim: High tail precision can coexist with negligible tail recovery. | Evidence: Chapter 11, two hits/five recommendations and two hits/3,891 relevant tail interactions. | Status: supported.

Claim: The results establish statistical significance or improved user satisfaction. | Evidence: No independent evidence for either. | Status: needs evidence. This claim is excluded from the report.

## Final clarity checklist

- Each paragraph develops one message.
- Mechanisms precede results and interpretations.
- Terms keep one meaning across models, metrics and plots.
- Metric direction, units, denominators and scope remain explicit.
- Tables use consistent precision and separate experimental controls.
- Every figure explains its axes or state, result and interpretive limits.
- Development chronology, commit logs and provenance discussion are absent from the learning report.
