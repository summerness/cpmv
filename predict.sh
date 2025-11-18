#!/bin/bash
set -euo pipefail
IMAGE_DIR=$1
CHECKPOINT=$2
OUTPUT=${3:-submission.csv}
python src/infer.py --image-dir "$IMAGE_DIR" --checkpoint "$CHECKPOINT" --output "$OUTPUT" --save-prob
