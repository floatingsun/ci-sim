"""Config-driven runner for the three-stage dataset generation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from ci_sim.contracts import StrictModel

from .codex import run as default_codex_runner
from .contracts import load_yaml
from .stages import (
    CodexRunner,
    CodexSettings,
    ConstructionConfig,
    IdeaGenerationConfig,
    ValidationConfig,
    construct_scenarios,
    generate_ideas,
    validate_scenarios,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("pipeline.yaml")

StageName = Literal[
    "generate_ideas",
    "construct_scenarios",
    "validate_scenarios",
]
StageSelection = Literal[
    "generate_ideas",
    "construct_scenarios",
    "validate_scenarios",
    "all",
]


class PipelineConfig(StrictModel):
    run_stage: StageSelection = "all"
    codex: CodexSettings
    generate_ideas: IdeaGenerationConfig
    construct_scenarios: ConstructionConfig
    validate_scenarios: ValidationConfig


def load_pipeline_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[Path, PipelineConfig]:
    """Load and validate the single pipeline YAML file."""

    path = Path(config_path).expanduser().resolve()
    return path, PipelineConfig.model_validate(load_yaml(path))


def run_pipeline(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    stage: StageSelection | None = None,
    *,
    codex_runner: CodexRunner = default_codex_runner,
) -> dict[StageName, tuple[Path, ...]]:
    """Run one selected stage, or all stages in dependency order."""

    path, config = load_pipeline_config(config_path)
    selection = stage or config.run_stage
    return {
        name: _run_stage(
            name,
            config,
            config_path=path,
            codex_runner=codex_runner,
        )
        for name in _stage_names(selection)
    }


def _stage_names(selection: StageSelection) -> tuple[StageName, ...]:
    if selection == "all":
        return (
            "generate_ideas",
            "construct_scenarios",
            "validate_scenarios",
        )
    return (selection,)


def _run_stage(
    name: StageName,
    config: PipelineConfig,
    *,
    config_path: Path,
    codex_runner: CodexRunner,
) -> tuple[Path, ...]:
    common = {
        "config_path": config_path,
        "codex_settings": config.codex,
        "codex_runner": codex_runner,
    }
    if name == "generate_ideas":
        return generate_ideas(config.generate_ideas, **common)
    if name == "construct_scenarios":
        return construct_scenarios(config.construct_scenarios, **common)
    return validate_scenarios(config.validate_scenarios, **common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--stage",
        choices=(
            "generate_ideas",
            "construct_scenarios",
            "validate_scenarios",
            "all",
        ),
        default=None,
        help="Override run_stage from the YAML file.",
    )
    arguments = parser.parse_args()

    outputs = run_pipeline(arguments.config, arguments.stage)
    print(
        json.dumps(
            {
                stage_name: [str(path) for path in paths]
                for stage_name, paths in outputs.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
