"""Render the report and assessment key, then package a portable offline copy.

Requires Python-Markdown. Figure generation is a separate command because it
needs the original local experiment bundle. This command never trains models.
"""

import argparse
import hashlib
import html
import json
import re
import shutil
import zipfile
from pathlib import Path

import markdown

SOURCE = Path(__file__).resolve().parent

CSS = """
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:#f6f4ed;color:#172e2b;font:17px/1.75 system-ui,-apple-system,sans-serif}
main{max-width:1050px;margin:0 auto;padding:40px 32px 90px}
h1{font-size:clamp(2rem,4vw,3rem);line-height:1.12;letter-spacing:-.03em;margin:16px 0 26px}
h2{font-size:1.85rem;line-height:1.25;margin:65px 0 24px;padding-top:24px;border-top:2px solid #b9c9bc;scroll-margin-top:20px}
h3{font-size:1.25rem;line-height:1.35;margin:36px 0 18px;scroll-margin-top:20px}
h4{font-size:1.08rem;margin:30px 0 16px;line-height:1.4}p{max-width:850px}
a{color:#176550;text-underline-offset:3px}
code{font-size:.88em;background:#e7eee6;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}
pre{padding:20px;background:#e7eee6;border-radius:9px;overflow-x:auto;line-height:1.55;font-size:.9rem}
pre code{padding:0;background:transparent;overflow-wrap:normal}
table{border-collapse:collapse;margin:22px 0;width:100%;display:block;overflow-x:auto;font-size:.9rem;line-height:1.5}
th,td{padding:11px 13px;border:1px solid #c6d1c4;text-align:left;vertical-align:top}
th{background:#e1e9df}tr:nth-child(even){background:#eeefe7}
blockquote{border-left:4px solid #176550;margin:20px 0;padding:1px 20px}
.nav{display:flex;gap:20px;flex-wrap:wrap;font-size:.92rem;border-bottom:1px solid #c6d1c4;padding-bottom:16px}
.eyebrow{font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:#176550}
.toc-box{padding:18px 24px;background:#e7eee6;border:1px solid #c6d1c4;border-radius:10px;margin:25px 0}
.toc-box summary{cursor:pointer;font-weight:650}.toc-box ul{padding-left:20px}.toc-box a{font-size:.92rem}
.card{padding:22px 26px;background:#fffefa;border:1px solid #c6d1c4;border-radius:12px;margin:20px 0}
.card h2{margin:0 0 15px;border:0;padding:0;font-size:1.5rem}
.muted,footer{color:#4c615c;font-size:.9rem}footer{margin-top:50px}
figure.report-figure{margin:24px 0;padding:12px;background:white;border:1px solid #c6d1c4;border-radius:9px}
figure.report-figure img{display:block;width:100%;height:auto;cursor:zoom-in}
a:focus-visible,summary:focus-visible{outline:3px solid #b7641b;outline-offset:4px}
@media(max-width:650px){main{padding:24px 16px 65px}table{font-size:.8rem}th,td{padding:8px}}
@media print{body{background:white;font:11pt/1.5 Georgia,serif}main{padding:0;max-width:none}
.nav,.toc-box{display:none}h2{break-before:page}h3{break-after:avoid}table{display:table;font-size:9pt}
tr,figure{break-inside:avoid}pre{white-space:pre-wrap}a{color:inherit;text-decoration:none}}
"""


def load_questions():
    source = (SOURCE / "self-test.html").read_text()
    start = source.index("const questions=") + len("const questions=")
    return json.JSONDecoder().raw_decode(source[start:])[0]


def word_counts(report):
    counts = {}
    for part, end in (("A", "## Part B"), ("B", "## Evidence")):
        content = report.split("## Part " + part, 1)[1].split(end, 1)[0]
        counts[part] = len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", re.sub(r"<[^>]+>", "", content)))
    return counts


def write_answer_key(questions):
    text = ["# Self-assessment: answer key and scoring rubric", "",
            "Refreshed 25 October 2026 for master `1dc9587d2`, merged PRs #24/#27/#28, and open PR #26 `a3a55153f`.", "",
            "Attempt [self-test.html](self-test.html) before reading this key. Every prompt is included for printing. Allow 75–90 minutes.", "",
            "Each half is worth 50 points: ten objective questions worth 2.5 each and five explanations worth 5 each. Multiple choice uses exact-set scoring. Numeric tolerance is 0.0001; simple fractions are accepted. A blank explanation scores zero.", "",
            "For explanations, award one point per criterion only when the locked answer demonstrates it without a material contradiction. These points are self-assessed. Finish all rubrics before interpreting the combined score.", "",
            "Study bands: 85–100 strong on this diagnostic; 70–84 working understanding with gaps; 50–69 partial; below 50 revisit the foundations. These are study guidance, not validated proficiency thresholds. Resolve every marked readiness check even if your total is high.", ""]
    for section, title in (("A", "Your metrics and rerankers"), ("B", "The shared system and other contributions")):
        text += [f"## {section} — {title}", ""]
        for question in [q for q in questions if q["section"] == section]:
            text += [f"### {question['id']} · {question['topic']} · {question['points']:g} points", "", question["prompt"], ""]
            if question["kind"] == "written":
                text += ["Award one point for each demonstrated criterion:", ""]
                for criterion in question["criteria"]:
                    text += ["- " + criterion["text"] + (" **Readiness check.**" if criterion.get("critical") else "")]
                text += ["", "**Model explanation:** " + question["model"], ""]
            else:
                if question.get("options"):
                    text += [f"{i + 1}. {option}" for i, option in enumerate(question["options"])] + [""]
                if question["kind"] == "number":
                    answer = f"{question['answer']:.6f}".rstrip("0").rstrip(".")
                else:
                    indices = question["answer"] if question["kind"] == "multi" else [question["answer"]]
                    answer = "; ".join(f"{i + 1}: {question['options'][i]}" for i in indices)
                text += ["**Answer:** " + answer, "", question["explanation"], ""]
                if question.get("critical"):
                    text += ["**Readiness check.**", ""]
    text += ["## Use the result", "",
             "Keep the first attempt and its export. Revisit the linked report sections, trace the relevant code, and explain one example aloud. Retake after a delay using changed numbers or a different user; recalling an answer is weaker evidence than explaining why it follows from the implementation.", ""]
    (SOURCE / "answer-key.md").write_text("\n".join(text))


def page(title, body):
    navigation = '<nav class="nav" aria-label="Study pack"><a href="index.html">Study pack</a><a href="report.html">Report</a><a href="self-test.html">Self-test</a><a href="answer-key.html">Answer key</a></nav>'
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>'
            f'<style>{CSS}</style></head><body><main>{navigation}{body}'
            '<footer>Updated 25 October 2026 · Source identities and verification scope are in verification.json. '
            'Figures and the assessment work offline.</footer></main></body></html>\n')


def render_markdown(name, title):
    renderer = markdown.Markdown(extensions=["tables", "fenced_code", "toc"], extension_configs={"toc": {"toc_depth": "2-4"}})
    body = renderer.convert((SOURCE / name).read_text())
    body = body.replace('href="report.md"', 'href="report.html"').replace('href="answer-key.md"', 'href="answer-key.html"')
    contents = '<details class="toc-box"><summary>Contents and section navigation</summary>' + renderer.toc + "</details>"
    return page(title, contents + body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == SOURCE or SOURCE in output.parents:
        parser.error("Use a delivery directory outside the report source directory")
    questions = load_questions()
    counts = word_counts((SOURCE / "report.md").read_text())
    if counts["A"] != counts["B"] or len(questions) != 30:
        raise ValueError("The report and question allocations must remain balanced")
    write_answer_key(questions)
    output.mkdir(parents=True, exist_ok=True)
    packaged = []
    for source in sorted(SOURCE.rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts or source.suffix not in {".md", ".html", ".py", ".json", ".csv", ".svg", ".png"}:
            continue
        destination = output / source.relative_to(SOURCE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        packaged.append(destination)
    quiz = (output / "self-test.html").read_text().replace('href="report.md"', 'href="report.html"').replace('href="answer-key.md"', 'href="answer-key.html"')
    (output / "self-test.html").write_text(quiz)
    for source, title in (("report.md", "Our project: implementation and understanding"), ("answer-key.md", "Project understanding: answer key")):
        destination = output / (Path(source).stem + ".html")
        destination.write_text(render_markdown(source, title))
        packaged.append(destination)
    intro = (f'<p class="eyebrow">RecBole course project · refreshed 25 October 2026</p>'
             '<h1>Understand the project,<br>then test your understanding.</h1>'
             f'<p>{counts["A"] + counts["B"]:,} substantive words, split equally between your assigned work and the other/shared contributions. '
             'Twelve explained figures: six in each half. Updated for master 1dc9587d2, merged PRs #24/#27/#28, and open PR #26.</p>'
             '<section class="card"><h2><a href="report.html">Read the illustrated report</a></h2><p>Concepts, code structure, decisions, worked examples, saved results, latest PRs and evidence limits.</p></section>'
             '<section class="card"><h2><a href="self-test.html">Attempt the 100-point self-test</a></h2><p>Thirty questions. Automatic calculations and choices, then explicit rubrics for your explanations. Start closed-book.</p></section>'
             '<section class="card"><h2><a href="answer-key.html">Review the answer key</a></h2><p>Every prompt, solution and reasoning criterion, plus scoring and retesting guidance.</p></section>'
             '<p><a href="study-pack.zip">Download the complete offline pack</a> · <a href="verification.json">Verification record</a> · <a href="figures/manifest.json">Figure provenance</a> · <a href="README.md">Source and rebuild instructions</a></p>')
    (output / "index.html").write_text(page("Project understanding study pack", intro))
    packaged.append(output / "index.html")
    checksums = "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}\n" for path in sorted(packaged))
    (output / "SHA256SUMS").write_text(checksums)
    with zipfile.ZipFile(output / "study-pack.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(packaged) + [output / "SHA256SUMS"]:
            archive.write(path, path.relative_to(output))
    print(json.dumps({"output": str(output), "word_counts": counts, "figures": 12, "questions": len(questions), "packaged_files": len(packaged) + 1}))


if __name__ == "__main__":
    main()
