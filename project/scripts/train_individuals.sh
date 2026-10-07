#!/bin/sh
# Train every individual model on the shared protocol and
# export its valid/test ScoreTables. Run from the repository root.
# Usage: sh project/scripts/train_individuals.sh [run-suffix]
set -e

MODELS="itemknn userknn bpr slimelastic ease fism"
SUFFIX="${1:-$(date +%Y%m%d-%H%M%S)}"
PYTHON="${PYTHON:-python}"

for model in $MODELS; do
    echo "Training $model"
    "$PYTHON" -m project.recsys train \
        --model-config "project/configs/models/$model.yaml" \
        --output "project/artifacts/$model-$SUFFIX"
done
