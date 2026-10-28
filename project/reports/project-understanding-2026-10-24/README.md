# Project understanding study pack

Open the illustrated delivery report, or attempt the self-test closed-book first. The report explains the project through definitions, implementation decisions, worked examples, trade-offs and results. It follows the order in which the ideas depend on one another.

The substantive text contains 10,080 words: 5,040 for your metrics, calibration, exposure and reranker evaluation, and 5,040 for the shared foundations and other contributions. Sixteen figures have the same eight/eight allocation. The reader-facing report excludes development history, commit logs and provenance discussion.

- report.md is the learning text.
- self-test.html contains 30 questions worth 100 points, divided 50/50 by ownership. Twenty objective questions are graded automatically; ten explanations use explicit five-point self-assessment rubrics.
- answer-key.md contains every prompt, solution, rubric and scoring guide.
- editorial-review.md provides the outline, paragraph-role review, five-dimension self-review and claim-evidence map requested by the writing workflow.
- figures/ contains explanatory diagrams and charts; data/ contains the compact numerical inputs.
- build_visualizations.py renders figures from existing measurements and examples.
- build_study_pack.py creates the HTML report, answer key, editorial review, landing page, checksums and offline ZIP.
- verification.json records technical checks separately from the learning text.

The self-test focuses on understanding rather than remembering PR numbers or file history. Its feedback links to the relevant report chapters. Allow 75–90 minutes, answer before grading, and award explanation points only when your locked response meets the stated criterion. Blank explanations score zero. The final score remains provisional until every rubric criterion is assessed.

Export the attempt before reloading or closing the page. Responses are held in the page's memory and are not submitted. Export also displays JSON text for saving when browser file downloads are restricted. The study bands are guidance, not validated proficiency thresholds.

## Rebuild the HTML pack

Use Python with Python-Markdown installed, from the report branch's repository root:

    python project/reports/project-understanding-2026-10-24/build_study_pack.py \
      --output /home/senteran/programming/RecBole_DSAIT4335/project/artifacts/project-understanding-2026-10-24

The generated HTML and ZIP live in the ignored delivery directory. Extract the entire ZIP before opening index.html so the figures remain alongside the pages.

## Rebuild the figures

Use Python with NumPy, Matplotlib and the project source available:

    MPLCONFIGDIR=/tmp/recbole-study-matplotlib python -W error \
      project/reports/project-understanding-2026-10-24/build_visualizations.py \
      --reranking-bundle /home/senteran/programming/RecBole_DSAIT4335/project/artifacts/reranking-20261022-001

The training-popularity view needs that local training bundle. The model-search and hybrid views read committed results. Calibration/diversification plots use the packaged tuned-model measurements through the project's plotting functions; diagrams use explicit toy examples. The existing EASE validation/group illustration is retained unchanged. Rendering the HTML from already packaged figures needs neither training artifacts nor the model runtime.
