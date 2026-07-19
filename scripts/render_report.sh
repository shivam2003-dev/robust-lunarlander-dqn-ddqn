#!/usr/bin/env bash
set -euo pipefail

REPORT_SOURCE="output/report/Group_148_Q_learning_DQN_DDQN.md"
REPORT_OUTPUT="output/pdf/Group_148_Q_learning_DQN_DDQN.pdf"

mkdir -p output/pdf

pandoc "$REPORT_SOURCE" \
  --from="markdown+raw_tex+tex_math_dollars+pipe_tables" \
  --to=pdf \
  --pdf-engine=xelatex \
  --include-in-header=submission/report-header.tex \
  --syntax-highlighting=tango \
  --resource-path=".:output/report:artifacts:submission" \
  --output="$REPORT_OUTPUT"

echo "Wrote $REPORT_OUTPUT"
