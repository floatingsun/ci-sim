"""Generate structured workplace email scenario ideas."""

from __future__ import annotations

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
    resolve_path,
    write_json,
)


class IdeaGenerationConfig(StrictModel):
    reference_paths: tuple[Path, ...] = Field(min_length=2)
    output_path: Path
    environment: Literal["workplace_email"] = "workplace_email"
    number_of_ideas: int = Field(default=1, ge=1, le=50)
    prompt: str
    diversity_requirements: tuple[str, ...] = ()


class IdeaGenerationResult(StrictModel):
    ideas: tuple[ScenarioIdea, ...] = Field(min_length=1)


def generate_ideas(
    config: IdeaGenerationConfig,
    *,
    config_path: str | Path,
    codex_settings: CodexSettings,
    codex_runner: CodexRunner = default_codex_runner,
) -> tuple[Path, ...]:
    """Ask Codex for an idea batch and persist the validated stage artifact."""

    pipeline_path = Path(config_path).expanduser().resolve()
    resolved = _resolve_config(config, pipeline_path)
    for reference_path in resolved.reference_paths:
        Scenario.load(reference_path)

    payload = codex_runner(
        _build_prompt(resolved),
        output_schema=IdeaGenerationResult.model_json_schema(),
        working_directory=pipeline_path.parent,
        model=codex_settings.model,
        reasoning_effort=codex_settings.reasoning_effort,
        timeout_seconds=codex_settings.timeout_seconds,
    )
    result = IdeaGenerationResult.model_validate(payload)
    if len(result.ideas) != resolved.number_of_ideas:
        raise ValueError(
            f"Stage 1 returned {len(result.ideas)} ideas; "
            f"expected {resolved.number_of_ideas}."
        )

    write_json(resolved.output_path, result.model_dump(mode="json"))
    return (resolved.output_path,)


def _resolve_config(
    config: IdeaGenerationConfig,
    config_path: Path,
) -> IdeaGenerationConfig:
    return config.model_copy(
        update={
            "reference_paths": tuple(
                resolve_path(config_path, path) for path in config.reference_paths
            ),
            "output_path": resolve_path(config_path, config.output_path),
        }
    )


def _build_prompt(config: IdeaGenerationConfig) -> str:
    references = "\n".join(f"- {path}" for path in config.reference_paths)
    diversity = "\n".join(
        f"- {requirement}" for requirement in config.diversity_requirements
    )
    return (
        f"{config.prompt.rstrip()}\n\n"
        f"Inspect these reference scenarios:\n{references}\n\n"
        f"Generate exactly {config.number_of_ideas} idea(s).\n\n"
        f"Diversity requirements:\n{diversity or '- None'}"
    )
