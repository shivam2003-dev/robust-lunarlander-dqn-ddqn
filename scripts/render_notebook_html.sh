#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK="output/jupyter-notebook/Group_148_Q_learning_DQN_DDQN.ipynb"
OUTPUT_DIR="output/html"
OUTPUT_NAME="Group_148_Q_learning_DQN_DDQN"

mkdir -p "$OUTPUT_DIR"

PYTHONPATH=src uv run jupyter nbconvert \
  --to=html \
  --embed-images \
  --output="$OUTPUT_NAME" \
  --output-dir="$OUTPUT_DIR" \
  "$NOTEBOOK"

PYTHONPATH=src python scripts/inject_notebook_print_css.py \
  "$OUTPUT_DIR/$OUTPUT_NAME.html" \
  submission/notebook-print.css

echo "Wrote $OUTPUT_DIR/$OUTPUT_NAME.html"
