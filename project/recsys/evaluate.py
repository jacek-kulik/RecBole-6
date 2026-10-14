"""Final test evaluation. Select model settings using validate beforehand."""

from os import PathLike
from pathlib import Path

from .evaluation import run_evaluation


def evaluate_training_run(
    run_dir: str | PathLike[str], output: str | PathLike[str], k: int = 10, batch_size: int = 1024
) -> Path:
    return run_evaluation(run_dir, output, partition="test", k=k, batch_size=batch_size)
