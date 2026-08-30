#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run --env-file .env python -m ci_sim.cli run \
  --scenario_path=ci_sim_pipeline/outputs/ci_sim_data_v01_0830 \
  --model=Qwen/Qwen3.8-27B-FP8 \
  --reasoning_effort=low \
  --concurrency=10


uv run --env-file .env python -m ci_sim.cli run \
  --scenario_path=ci_sim_pipeline/outputs/ci_sim_data_v01_0830 \
  --model=Qwen/Qwen3.8-27B-FP8 \
  --reasoning_effort=medium \
  --concurrency=10


uv run --env-file .env python -m ci_sim.cli run \
  --scenario_path=ci_sim_pipeline/outputs/ci_sim_data_v01_0830 \
  --model=Qwen/Qwen3.8-27B-FP8 \
  --reasoning_effort=xhigh \
  --concurrency=10
