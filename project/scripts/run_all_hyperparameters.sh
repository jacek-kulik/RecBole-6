#!/bin/sh
set -eu

DATASET="ml-100k"
EXPERIMENTS="ItemKNN UserKNN BPR SLIMElastic EASE FISM Pop Random"
RESULT_DIR="hyperparameters/results"

echo "Starting hyperparameter tests for $DATASET"
. .venv/bin/activate

mkdir -p "$RESULT_DIR"

for experiment in $EXPERIMENTS; do
    config_file="recbole/config/$experiment/$DATASET.yaml"
    params_file="hyperparameters/hyper-$experiment.test"
    result_file="$RESULT_DIR/hyper-$experiment.result"
    display_file="$RESULT_DIR/hyper-$experiment.html"

    [ -f "$config_file" ] || {
        echo "Missing configuration: $config_file" >&2
        exit 1
    }

    [ -f "$params_file" ] || {
        echo "Missing parameter space: $params_file" >&2
        exit 1
    }

    echo "Running hyperparameter tests for $experiment"

    python run_hyper.py \
        --config_files="$config_file" \
        --params_file="$params_file" \
        --output_file="$result_file" \
        --display_file="$display_file" \
        --tool=Hyperopt

    echo "Finished $experiment -> $result_file"
done

echo "All hyperparameter tests completed"
