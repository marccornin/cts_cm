#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/experiment/main.yaml}"
python -m cts_cm.observer run-all --config "${CONFIG}"
