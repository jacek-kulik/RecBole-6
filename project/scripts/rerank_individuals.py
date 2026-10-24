"""Run all six validation sweeps, freeze choices, then evaluate test effects."""

import argparse
import json
import time
from pathlib import Path

from project.recsys.artifacts import REPO_ROOT, new_run, provenance, write_json
from project.recsys.comparisons import compare_runs
from project.recsys.rerank import run_reranking

MODELS = ("itemknn", "userknn", "bpr", "slimelastic", "ease", "fism")


def run(run_suffix: str, output: Path, source_root: Path) -> Path:
    """Training is separate: use train_individuals.sh with the same suffix."""
    output = new_run(output)
    manifest = {**provenance(), "kind": "six-model-reranking", "status": "running", "models": list(MODELS),
                "source_root": str(source_root.resolve()), "run_suffix": run_suffix,
                "k": 10, "candidate_count": 50, "strengths": [0, .25, .5, .75, 1],
                "selection": "Minimum target metric with at most 5% relative validation NDCG loss"}
    started = time.monotonic()
    try:
        for partition in ("valid", "test"):
            for index, model in enumerate(MODELS, 1):
                print(f"\n{partition}: model {index}/{len(MODELS)} ({model}); elapsed {time.monotonic()-started:.0f}s", flush=True)
                selection = output / model / "valid/selection.json" if partition == "test" else None
                run_reranking(source_root / f"{model}-{run_suffix}", output / model / partition,
                               partition=partition, selection=selection)
        arguments = []
        for model in MODELS:
            folder = output / model / "test"
            for row in json.loads((folder / "summary.json").read_text()):
                arguments.append(f"{model}-{row['method']}={folder / row['evaluation']}")
        compare_runs(arguments, output / "comparison", k=10)
        manifest.update(status="complete", seconds=time.monotonic()-started,
                        comparison=str(output / "comparison/comparison.csv"))
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-suffix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "project/artifacts")
    args = parser.parse_args()
    print(run(args.run_suffix, args.output, args.source_root))


if __name__ == "__main__":
    main()
