# CI Sim

Contextual integrity means an LLM should use only the information needed for a
task and avoid leaking it into the wrong context. CI Sim turns that idea into a
workplace simulation that measures whether an AI model can complete write tasks
without disclosing information outside the intended context.

## 1. Install

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

Pipeline generation also requires an authenticated `codex` CLI on `PATH`.
Evaluation requires a `.env` file containing:

```text
RIVER_API_KEY=your_river_api_key
```

## 2. Run the pipeline and evaluations

The pipeline is configured in `ci_sim_pipeline/pipeline.yaml`. Its main sections
are:

- `run_stage`: selects one stage or the full pipeline.
- `codex`: sets the model, reasoning effort, timeout, retries, and concurrency.
- `generate_ideas`: creates scenario ideas from the reference examples.
- `construct_scenarios`: turns the ideas into candidate scenario JSON.
- `validate_scenarios`: validates, repairs, and writes the final dataset.

The YAML file also controls input/output paths, checkpoints, coverage targets,
dataset size, and validation thresholds.

Generate ideas, construct scenarios, and validate the final dataset:

```bash
uv run python -m ci_sim_pipeline.runner --stage all
```

By default, the final generated scenarios are written to
`ci_sim_pipeline/outputs/final_dataset/`. This repo already contains a generated dataset `ci_sim_pipeline/outputs/ci_sim_data_v01_0830/` .

And the batch evaluation scripts

```bash
bash scripts/eval-deepseek_v4_flash_0731-ci_sim_data_v01_0830.sh
bash scripts/eval-glm_5_2_nvfp4-ci_sim_data_v01_0830.sh
bash scripts/eval-qwen3_8_27b_fp8-ci_sim_data_v01_0830.sh
```

Evaluation results are saved under `runs/`.

## 3. Project tree

```text
ci_sim/
├── __init__.py
├── cli.py
├── contracts.py
├── runner.py
├── agent/
│   ├── __init__.py
│   ├── contracts.py
│   └── river.py
└── environment/
    ├── __init__.py
    ├── contracts.py
    ├── scenario.py
    ├── grading/
    │   ├── __init__.py
    │   ├── evaluator.py
    │   ├── scenario_validation.py
    │   └── text_matching.py
    └── workplace/
        ├── __init__.py
        ├── environment.py
        └── tools.py

ci_sim_pipeline/
├── contracts.py
├── pipeline.yaml
├── portfolio.py
├── quality.py
├── runner.py
├── scenario_alignment.py
├── codex/
│   └── __init__.py
├── data_examples/
│   ├── gina_ellis_...json
│   └── gregory_parker_...json
├── prompts/
│   ├── generate_ideas.md
│   ├── construct_scenarios.md
│   └── validate_scenarios.md
├── stages/
│   ├── __init__.py
│   ├── generate_ideas.py
│   ├── construct_scenarios.py
│   └── validate_scenarios.py
└── outputs/                 # generated pipeline artifacts (ignored)
```
