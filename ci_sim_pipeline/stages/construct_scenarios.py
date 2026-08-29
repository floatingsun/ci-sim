"""Construct evaluator-compatible scenarios from generated ideas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from ci_sim.contracts import StrictModel
from ci_sim.environment.scenario import Scenario

from ..codex import run as default_codex_runner
from ..contracts import (
    CodexRunner,
    CodexSettings,
    ScenarioIdea,
    read_json,
    resolve_path,
    write_json,
)
from .generate_ideas import IdeaGenerationResult


class ConstructionConfig(StrictModel):
    input_path: Path
    reference_paths: tuple[Path, ...] = Field(min_length=2)
    output_path: Path
    environment: Literal["workplace_email"] = "workplace_email"
    prompt: str


class ConstructionResult(StrictModel):
    scenarios: tuple[Scenario, ...] = Field(min_length=1)


def construct_scenarios(
    config: ConstructionConfig,
    *,
    config_path: str | Path,
    codex_settings: CodexSettings,
    codex_runner: CodexRunner = default_codex_runner,
) -> tuple[Path, ...]:
    """Read stage 1 and ask Codex to construct one scenario per idea."""

    pipeline_path = Path(config_path).expanduser().resolve()
    resolved = _resolve_config(config, pipeline_path)
    ideas = IdeaGenerationResult.model_validate(read_json(resolved.input_path))
    for reference_path in resolved.reference_paths:
        Scenario.load(reference_path)

    scenarios = tuple(
        _construct_scenario(
            idea,
            resolved,
            pipeline_path=pipeline_path,
            codex_settings=codex_settings,
            codex_runner=codex_runner,
        )
        for idea in ideas.ideas
    )
    if len({scenario.id for scenario in scenarios}) != len(scenarios):
        raise ValueError("Stage 2 produced duplicate scenario ids.")

    result = ConstructionResult(scenarios=scenarios)
    write_json(resolved.output_path, result.model_dump(mode="json"))
    return (resolved.output_path,)


def _construct_scenario(
    idea: ScenarioIdea,
    config: ConstructionConfig,
    *,
    pipeline_path: Path,
    codex_settings: CodexSettings,
    codex_runner: CodexRunner,
) -> Scenario:
    payload = codex_runner(
        _build_prompt(config, idea),
        output_schema=Scenario.model_json_schema(),
        working_directory=pipeline_path.parent,
        model=codex_settings.model,
        reasoning_effort=codex_settings.reasoning_effort,
        timeout_seconds=codex_settings.timeout_seconds,
    )
    scenario = Scenario.model_validate(payload)
    _check_email_environment(scenario)
    return scenario


def _resolve_config(
    config: ConstructionConfig,
    config_path: Path,
) -> ConstructionConfig:
    return config.model_copy(
        update={
            "input_path": resolve_path(config_path, config.input_path),
            "reference_paths": tuple(
                resolve_path(config_path, path) for path in config.reference_paths
            ),
            "output_path": resolve_path(config_path, config.output_path),
        }
    )


def _build_prompt(config: ConstructionConfig, idea: ScenarioIdea) -> str:
    references = "\n".join(f"- {path}" for path in config.reference_paths)
    idea_json = json.dumps(idea.model_dump(mode="json"), indent=2)
    return (
        f"{config.prompt.rstrip()}\n\n"
        f"Inspect these reference scenarios:\n{references}\n\n"
        f"Construct a scenario for this idea:\n{idea_json}"
    )


def _check_email_environment(scenario: Scenario) -> None:
    runtime_tools = tuple(tool.name for tool in scenario.model_input.tools)
    expected_tools = tuple(item.tool for item in scenario.label.expected_writes)
    if runtime_tools != ("gmail_send",):
        raise ValueError(f"{scenario.id} must expose exactly the gmail_send tool.")
    if expected_tools != ("gmail.send",):
        raise ValueError(
            f"{scenario.id} must contain exactly one expected gmail.send write."
        )
