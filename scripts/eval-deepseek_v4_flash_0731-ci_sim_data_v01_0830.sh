#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run --env-file .env python -m ci_sim.cli run \
  --scenario_path=ci_sim_pipeline/outputs/ci_sim_data_v01_0830 \
  --model=deepseek-ai/DeepSeek-V4-Flash-0731 \
  --thinking \
  --repetitions=3 \
  --concurrency=30
