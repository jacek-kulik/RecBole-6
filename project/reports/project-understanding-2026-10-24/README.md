# Project understanding study pack

Updated 25 October 2026. Start with the illustrated delivery `report.html`, or attempt `self-test.html` closed-book first. Keep the answer key closed until after the attempt.

The report describes synced master `1dc9587d273b9084c70ab274e2d3d90cbc8a7eaf`. PR #24 (your calibration/exposure delivery) and PR #27 (hybrid/reranker orders and independent weighted reconstruction) are merged. PR #28, inspected at `87321da05b01eaf825420371d44899a4583f0fa2`, merged during this refresh and its diversification code and six-model evidence are on master. The remaining open PR is #26 at `a3a55153f426727afd20091a33f39050c02cf815`, inspected without merging it. User fairness, group integration, and the explanatory entrypoint retain their actual status.

- `report.md`: 11,888 substantive words, exactly 5,944 in each ownership half. Twelve figures are divided six/six. The update adds information-flow and greedy-objective examples, shared exposure state, merged order experiments, reconstruction limits, and all three diversification objectives and plots.
- `self-test.html`: 30 questions and 100 points, split 50/50. Twenty objective items are graded automatically; ten explanations use five one-point self-assessment criteria each. The updated questions cover score-preserving conversion, local versus cohort-dependent order behavior, diversification objectives, and the current group PR.
- `answer-key.md`: the complete question bank, answers, rubrics, scoring guidance and retesting advice. It is generated from the assessment bank to keep answers synchronized.
- `verification.json`: exact source heads, measured allocations, fresh and historical checks, and what those checks establish.
- `figures/`: four unchanged original plots, four new views of recorded results, and four illustrative diagrams. `manifest.json` records source hashes, asset hashes and plotted values. The numerical examples are illustrations rather than new experimental observations.
- `build_visualizations.py`: recreates figures from saved evidence, without training or evaluation. New charts/diagrams have SVG and PNG versions. Original experiment plots are copied unchanged.
- `build_study_pack.py`: generates the answer key, readable HTML, landing page, checksums and offline ZIP using Python-Markdown.

Allow 75–90 minutes for the assessment. Answers lock when grading starts. Award explanation points only when your locked answer demonstrates the criterion; blank explanations score zero. The combined score remains provisional until every rubric criterion is assessed. These study bands are diagnostic guidance, not validated proficiency thresholds. Export before closing or reloading; answers are not saved across reloads. The export also exposes JSON text if file downloads are restricted. No account or network submission is used.

The delivery includes printable `report.html` and `answer-key.html`, plus `index.html`. Generated HTML and the ZIP live in the ignored local artifact directory instead of duplicating the committed report. Extract the entire ZIP before opening `index.html` so the figures remain alongside it.

## Rebuild

Use Python with NumPy and Matplotlib for figures and Python-Markdown for the study pack. Run from the report branch's repository root. The saved local reranking bundle supplies the original calibration/exposure plots and training counts; other older numerical evidence is committed under `hyperparameters/results/`.

```sh
MPLCONFIGDIR=/tmp/recbole-study-matplotlib python -W error \
  project/reports/project-understanding-2026-10-24/build_visualizations.py \
  --reranking-bundle /home/senteran/programming/RecBole_DSAIT4335/project/artifacts/reranking-20261022-001

python project/reports/project-understanding-2026-10-24/build_study_pack.py \
  --output /home/senteran/programming/RecBole_DSAIT4335/project/artifacts/project-understanding-2026-10-24
```

The figure generator retains and verifies the packaged PR #28 plots by default. To recopy them from the inspected source, add `--diversification-source /path/to/checkout-at-87321da05` to the figure command. The checked-in summary, figure-data JSON and source manifest support portable inspection. The JSON source hash recorded on Windows uses CRLF; the committed LF content matches after restoring CRLF. Original PNG/PDF hashes match byte-for-byte.

A fresh clone does not include the ignored large experiment runs or checkpoints. Rendering the already committed figures and assessment into HTML requires no model runtime and no original training bundle. This pack does not change recommendation runtime code.
