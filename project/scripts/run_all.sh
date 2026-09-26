#!/bin/sh
set -e

DATASET="ml-100k"
EXPERIMENTS="ItemKNN UserKNN BPR SLIMElastic EASE FISM Pop Random"
TOPK=10

echo "Starting all models for $DATASET"
. .venv/bin/activate

for experiment in $EXPERIMENTS; do
    recbole_model="$experiment"

    case "$experiment" in
        UserKNN)
            recbole_model="ItemKNN"
            ;;
    esac

    config_file="recbole/config/$experiment/$DATASET.yaml"

    echo "Running $experiment using RecBole model $recbole_model"

    python run_recbole.py \
        --model="$recbole_model" \
        --dataset="$DATASET" \
        --config_files="$config_file"

    python save_split.py \
        --model="$recbole_model" \
        --dataset="$DATASET" \
        --config_files="$config_file" \
        --output_dir="saved/splits/$experiment"

    model_file=$(
        ls -t "saved/$DATASET-$recbole_model-"*.pth 2>/dev/null |
        head -n 1
    )

    [ -n "$model_file" ] || {
        echo "No checkpoint found for $experiment" >&2
        exit 1
    }

    # Use a temporary experiment-specific directory because both ItemKNN and
    # UserKNN are internally named ItemKNN by RecBole.
    temporary_output="saved/recommendations/.${experiment}.tmp"
    final_output="saved/recommendations/${DATASET}_${experiment}_top${TOPK}.json"

    mkdir -p "$temporary_output" saved/recommendations

    python save_recommendations.py \
        --model_file="$model_file" \
        --k="$TOPK" \
        --output_dir="$temporary_output"

    generated_output="$temporary_output/${DATASET}_${recbole_model}_top${TOPK}.json"

    mv "$generated_output" "$final_output"
    rmdir "$temporary_output"

    echo "Finished $experiment -> $final_output"
done
