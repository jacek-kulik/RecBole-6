"""Run from the repository root: python -m project.recsys --help."""

import argparse
import importlib

from .artifacts import REPO_ROOT


def main() -> None: # :)
    parser = argparse.ArgumentParser(description="RecSys project starter commands")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run the synthetic baseline/evaluation example")
    demo.add_argument("--output", required=True, help="New artifact directory (must not exist)")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--k", type=int, default=2)
    train = commands.add_parser("train", help="Train a RecBole model; save validation and splits")
    train.add_argument("--model-config", default=str(REPO_ROOT / "project/configs/models/bpr.yaml"))
    train.add_argument("--output", required=True, help="New artifact directory (must not exist)")
    for name, partition in (("validate", "validation"), ("evaluate", "final test")):
        evaluation = commands.add_parser(name, help=f"Run independent {partition} metrics on a saved training run")
        evaluation.add_argument("--run", required=True, help="Completed project training artifact directory")
        evaluation.add_argument("--output", required=True, help="New artifact directory (must not exist)")
        evaluation.add_argument("--k", type=int, default=10)
        evaluation.add_argument("--batch-size", type=int, default=1024, help="Users scored per batch; reduce to save memory")
    for name in ("task1", "task2", "task3"):
        commands.add_parser(name, help="TODO: project experiment orchestration")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            from .demo import run
            output = run(args.output, seed=args.seed, k=args.k)
            print(f"Synthetic demo saved to {output}; not report evidence.")
        elif args.command == "train":
            from .recbole_adapter import train
            output = train(args.model_config, args.output)
            print(f"Training artifacts saved to {output}; use the validate command for independent validation.")
        elif args.command == "validate":
            from .validate import validate_training_run
            output = validate_training_run(args.run, args.output, k=args.k, batch_size=args.batch_size)
            print(f"Independent validation saved to {output}; see metrics.json, beyond_accuracy.json, and group_metrics.json")
        elif args.command == "evaluate":
            from .evaluate import evaluate_training_run
            output = evaluate_training_run(args.run, args.output, k=args.k, batch_size=args.batch_size)
            print(f"Final test evaluation saved to {output}; see metrics.json, beyond_accuracy.json, and group_metrics.json")
        else:
            importlib.import_module(f"project.recsys.experiments.{args.command}").run()
    except (NotImplementedError, ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
