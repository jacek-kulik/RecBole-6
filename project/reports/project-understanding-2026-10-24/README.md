# Project understanding study pack

Start with `report.md`, or attempt `self-test.html` closed-book to establish a baseline first. The assessment opens directly in a browser and works offline. Keep `answer-key.md` closed until you have attempted it.

- `report.md`: approximately 9,500 substantive words, split exactly equally between Jacek's assigned work and the other/shared contributions. Each half contains 4,773 words and three annotated figures. It includes important concepts, code structure, decisions, worked examples, results, open PRs, and remaining limitations.
- `self-test.html`: 30 questions worth 100 points, split 50/50. Twenty objective items are graded automatically; ten explanations use explicit self-assessment rubrics. Answers lock when grading starts. Export the attempt before closing or reloading the page.
- `answer-key.md`: every question, objective solutions, explanation models, point-by-point rubrics, and retesting guidance. This also serves as a printable question bank.
- `verification.json`: inspected source heads, checks actually performed, measured word allocation, saved-result audits, and the limits of those checks.
- `figures/`: the two original reranking plots plus four new visualizations of recorded popularity, tuning, hybrid and ablation evidence, with source hashes and plotted values in `manifest.json`. Each report caption explains the axes, result, and limits of its interpretation. Click an image to view it at full size.
- `build_visualizations.py`: reproduces the figure assets from existing experiment records, without training or rerunning evaluations. New charts have both SVG and PNG copies; existing reranking PNGs are copied without alteration.

The report describes master `54e60bffb`, open reranker PR #24 at `580bfaa63`, and open group-analysis PR #26 at `67be0b246`. The PRs were inspected without merging them. Pending algorithms and cross-PR integration work remain explicit. This pack changes documentation and the standalone assessment only.

For the assessment, allow 75–90 minutes. Answer with the report closed, grade the objective portion, then award explanation points only when the locked answer demonstrates the corresponding criterion. A blank explanation scores zero. The page distinguishes automatic and self-assessed scores and withholds a final combined score until the rubric is assessed. Study bands are diagnostic guidance, not validated proficiency thresholds.

No installation is needed to open the HTML file. If a browser restricts file downloads, the export button also exposes the full attempt JSON as text for saving manually. Response data stays in the page; no account or network submission is used.

The delivery copy additionally contains readable, printable `report.html` and `answer-key.html`, plus an `index.html` entrypoint. These are generated from the Markdown for convenience and kept with the ignored local delivery artifacts rather than committed alongside duplicate report content.

To rebuild the figures, use Python with NumPy and Matplotlib installed. The exact local reranking bundle is required for the two original graphs and training-only popularity statistics; other inputs are committed in `hyperparameters/results/`:

```sh
MPLCONFIGDIR=/tmp/recbole-study-matplotlib python \
  project/reports/project-understanding-2026-10-24/build_visualizations.py \
  --reranking-bundle /home/senteran/programming/RecBole_DSAIT4335/project/artifacts/reranking-20261022-001
```
