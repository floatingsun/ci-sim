#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run --env-file .env python -m ci_sim.cli run \
  --scenario_path=ci_sim_pipeline/outputs/ci_sim_data_v01_0830 \
  --model=nvidia/GLM-5.2-NVFP4 \
  --reasoning_effort=high \
  --repetitions=3 \
  --concurrency=30


uv run --env-file .env python -m ci_sim.cli run \
  --scenario_path=ci_sim_pipeline/outputs/ci_sim_data_v01_0830 \
  --model=nvidia/GLM-5.2-NVFP4 \
  --reasoning_effort=max \
  --repetitions=3 \
  --concurrency=30
