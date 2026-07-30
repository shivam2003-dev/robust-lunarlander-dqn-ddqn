#!/usr/bin/env bash
set -euo pipefail

REPORT_SOURCE="output/report/Group_148_Q_learning_DQN_DDQN.md"
REPORT_OUTPUT="output/html/Group148_Q_learning_DQN_DDQN.html"

mkdir -p output/html

pandoc "$REPORT_SOURCE" \
  --from="markdown+raw_tex+tex_math_dollars+pipe_tables" \
  --to=html5 \
  --standalone \
  --embed-resources \
  --css=../../submission/report-print.css \
  --resource-path=".:output/report:artifacts:submission" \
  --output="$REPORT_OUTPUT"

PYTHONPATH=src python scripts/inject_iteration_html.py "$REPORT_OUTPUT"

echo "Wrote $REPORT_OUTPUT"
