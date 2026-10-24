# Project understanding study pack

Start with `report.md`, or attempt `self-test.html` closed-book to establish a baseline first. The assessment opens directly in a browser and works offline. Keep `answer-key.md` closed until you have attempted it.

- `report.md`: approximately 7,900 substantive words, split exactly equally between Jacek's assigned work and the other/shared contributions. It includes important concepts, code structure, decisions, worked examples, results, open PRs, and remaining limitations.
- `self-test.html`: 30 questions worth 100 points, split 50/50. Twenty objective items are graded automatically; ten explanations use explicit self-assessment rubrics. Answers lock when grading starts. Export the attempt before closing or reloading the page.
- `answer-key.md`: every question, objective solutions, explanation models, point-by-point rubrics, and retesting guidance. This also serves as a printable question bank.
- `verification.json`: inspected source heads, checks actually performed, measured word allocation, saved-result audits, and the limits of those checks.

The report describes master `54e60bffb`, open reranker PR #24 at `580bfaa63`, and open group-analysis PR #26 at `67be0b246`. The PRs were inspected without merging them. Pending algorithms and cross-PR integration work remain explicit. This pack changes documentation and the standalone assessment only.

For the assessment, allow 75–90 minutes. Answer with the report closed, grade the objective portion, then award explanation points only when the locked answer demonstrates the corresponding criterion. A blank explanation scores zero. The page distinguishes automatic and self-assessed scores and withholds a final combined score until the rubric is assessed. Study bands are diagnostic guidance, not validated proficiency thresholds.

No installation is needed to open the HTML file. If a browser restricts file downloads, the export button also exposes the full attempt JSON as text for saving manually. Response data stays in the page; no account or network submission is used.

The delivery copy additionally contains readable, printable `report.html` and `answer-key.html`, plus an `index.html` entrypoint. These are generated from the Markdown for convenience and kept with the ignored local delivery artifacts rather than committed alongside duplicate report content.
