#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

qwen_runs_dir="${QWEN_RUNS_DIR:-runs}"

uv run --env-file .env python -m ci_sim.cli run \
  --scenario_path=ci_sim_pipeline/outputs/ci_sim_data_v01_0830 \
  --model=Qwen/Qwen3.8-27B-FP8 \
  --reasoning_effort=low \
  --repetitions=3 \
  --runs_dir="$qwen_runs_dir" \
  --concurrency=30


uv run --env-file .env python -m ci_sim.cli run \
  --scenario_path=ci_sim_pipeline/outputs/ci_sim_data_v01_0830 \
  --model=Qwen/Qwen3.8-27B-FP8 \
  --reasoning_effort=medium \
  --repetitions=3 \
  --runs_dir="$qwen_runs_dir" \
  --concurrency=30


uv run --env-file .env python -m ci_sim.cli run \
  --scenario_path=ci_sim_pipeline/outputs/ci_sim_data_v01_0830 \
  --model=Qwen/Qwen3.8-27B-FP8 \
  --reasoning_effort=xhigh \
  --repetitions=3 \
  --runs_dir="$qwen_runs_dir" \
  --concurrency=30
