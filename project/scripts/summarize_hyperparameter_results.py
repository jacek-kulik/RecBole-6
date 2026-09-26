#!/usr/bin/env python3
"""Summarize the last or best trial from each hyperparameter result file."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


EXPERIMENTS = (
    "ItemKNN",
    "UserKNN",
    "BPR",
    "SLIMElastic",
    "EASE",
    "FISM",
    "Pop",
    "Random",
)

METRIC_NAMES = ("precision", "recall", "ndcg", "mrr")
BLOCK_PATTERN = re.compile(
    r"^(?P<parameters>[^\n]+)\s*\n"
    r"Valid result:\s*\n(?P<valid>[^\n]+)\s*\n"
    r"Test result:\s*\n(?P<test>[^\n]+)",
    re.MULTILINE,
)
METRIC_PATTERN = re.compile(
    r"(?P<name>[A-Za-z_]+)@(?P<cutoff>\d+)\s*:\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


@dataclass(frozen=True)
class Trial:
    parameters: str
    valid: dict[str, float]
    test: dict[str, float]


def parse_metrics(line: str) -> dict[str, float]:
    return {
        f"{match.group('name').lower()}@{match.group('cutoff')}": float(
            match.group("value")
        )
        for match in METRIC_PATTERN.finditer(line)
    }


def parse_trials(path: Path) -> list[Trial]:
    text = path.read_text(encoding="utf-8")
    return [
        Trial(
            parameters=match.group("parameters").strip(),
            valid=parse_metrics(match.group("valid")),
            test=parse_metrics(match.group("test")),
        )
        for match in BLOCK_PATTERN.finditer(text)
    ]


def choose_trial(trials: list[Trial], selection: str, cutoff: int) -> Trial:
    if selection == "last":
        return trials[-1]

    selection_metric = f"mrr@{cutoff}"
    eligible = [trial for trial in trials if selection_metric in trial.valid]
    if not eligible:
        raise ValueError(f"no validation {selection_metric} value was found")
    return max(eligible, key=lambda trial: trial.valid[selection_metric])


def format_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def print_table(rows: list[dict[str, str]]) -> None:
    columns = (
        "model",
        "hyperparameters",
        "precision",
        "recall",
        "ndcg",
        "mrr",
    )
    headings = {
        "model": "Model",
        "hyperparameters": "Hyperparameters",
        "precision": "Precision",
        "recall": "Recall",
        "ndcg": "NDCG",
        "mrr": "MRR",
    }
    widths = {
        column: max(len(headings[column]), *(len(row[column]) for row in rows))
        for column in columns
    }

    print(" | ".join(headings[column].ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(row[column].ljust(widths[column]) for column in columns))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read hyper-<model>.result files and report one complete trial per model."
        )
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("hyperparameters/results"),
        help="directory containing hyper-<model>.result files",
    )
    parser.add_argument(
        "--selection",
        choices=("last", "best"),
        default="last",
        help="use the last trial, or the trial with the highest validation MRR",
    )
    parser.add_argument(
        "--metrics-split",
        choices=("test", "valid"),
        default="test",
        help="show test or validation metrics for the selected trial",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=10,
        help="metric cutoff to report, for example 10 for precision@10",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV output path (default: <result-dir>/<selection>-run-summary.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.result_dir / f"{args.selection}-run-summary.csv"
    rows: list[dict[str, str]] = []
    errors: list[str] = []

    for model in EXPERIMENTS:
        path = args.result_dir / f"hyper-{model}.result"
        if not path.is_file():
            errors.append(f"{model}: missing {path}")
            continue

        trials = parse_trials(path)
        if not trials:
            errors.append(f"{model}: no complete trials found in {path}")
            continue

        try:
            trial = choose_trial(trials, args.selection, args.cutoff)
        except ValueError as error:
            errors.append(f"{model}: {error}")
            continue

        metrics = trial.test if args.metrics_split == "test" else trial.valid
        row = {
            "model": model,
            "hyperparameters": trial.parameters,
        }
        for metric_name in METRIC_NAMES:
            key = f"{metric_name}@{args.cutoff}"
            row[metric_name] = format_value(metrics.get(key))
        rows.append(row)

    if not rows:
        print("No results could be summarized.", file=sys.stderr)
        for error in errors:
            print(f"warning: {error}", file=sys.stderr)
        return 1

    print(
        f"Selection: {args.selection}; metrics: {args.metrics_split}; "
        f"cutoff: @{args.cutoff}"
    )
    print_table(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "model",
        "hyperparameters",
        f"precision@{args.cutoff}",
        f"recall@{args.cutoff}",
        f"ndcg@{args.cutoff}",
        f"mrr@{args.cutoff}",
    )
    with output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "hyperparameters": row["hyperparameters"],
                    f"precision@{args.cutoff}": row["precision"],
                    f"recall@{args.cutoff}": row["recall"],
                    f"ndcg@{args.cutoff}": row["ndcg"],
                    f"mrr@{args.cutoff}": row["mrr"],
                }
            )

    print(f"\nCSV written to {output}")
    for error in errors:
        print(f"warning: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
