#!/usr/bin/env bash
# Full run: synthetic matrices, multiply timing, progressive precision, GPT-2 perplexity, report.
set -euo pipefail
cd "$(dirname "$0")"
python -W ignore run_synthetic.py
python -W ignore run_gpt2.py
python make_report.py
