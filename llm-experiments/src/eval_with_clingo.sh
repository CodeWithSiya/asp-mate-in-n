#!/usr/bin/env bash
# Usage: ./scripts/eval_with_clingo.sh results/<date>/<model>/mate_in_one.lp gold/mate_in_one.lp data/boards/sample.lp
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <generated_program.lp> <gold_reference.lp> [additional_facts.lp]" >&2
  exit 1
fi
clingo "$@"
