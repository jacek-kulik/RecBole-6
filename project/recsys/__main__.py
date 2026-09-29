"""Run from the repository root: python -m project.recsys --help."""

import argparse
import importlib

from .artifacts import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description="RecSys project starter commands")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run the synthetic baseline/evaluation example")
    demo.add_argument("--output", required=True, help="New artifact directory (must not exist)")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--k", type=int, default=2)
    train = commands.add_parser("train", help="Train a RecBole model; save validation and splits")
    train.add_argument("--model-config", default=str(REPO_ROOT / "project/configs/models/bpr.yaml"))
    train.add_argument("--output", required=True, help="New artifact directory (must not exist)")
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
            print(f"Training artifacts saved to {output}; independent project evaluation is still TODO.")
        else:
            importlib.import_module(f"project.recsys.experiments.{args.command}").run()
    except (NotImplementedError, ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
