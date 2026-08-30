"""Generate diverse structured workplace scenario blueprints."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from ci_sim.contracts import StrictModel
from ci_sim.environment.scenario import Scenario

from ..codex import run as default_codex_runner
from ..contracts import (
    CANONICAL_TOOLS,
    CI_FAILURE_TYPES,
    CodexRunner,
    CodexSettings,
    IdeaGenerationConfig,
    IdeaGenerationResult,
    ScenarioIdea,
    read_prompt,
    resolve_path,
    write_json,
)


class _IdeaGenerationCheckpoint(StrictModel):
    input_fingerprint: str
    ideas: tuple[ScenarioIdea, ...] = ()


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
    references = tuple(Scenario.load(path) for path in resolved.reference_paths)
    fingerprint = _idea_generation_fingerprint(
        resolved,
        references,
        codex_settings,
    )

    ideas: list[ScenarioIdea] = []
    if resolved.checkpoint_path is not None and resolved.checkpoint_path.is_file():
        try:
            checkpoint = _IdeaGenerationCheckpoint.model_validate_json(
                resolved.checkpoint_path.read_text(encoding="utf-8")
            )
        except ValidationError:
            checkpoint = None
        if checkpoint is not None and checkpoint.input_fingerprint == fingerprint:
            if len(checkpoint.ideas) > resolved.number_of_ideas:
                raise ValueError(
                    f"Idea checkpoint has {len(checkpoint.ideas)} records but the "
                    f"configured target is {resolved.number_of_ideas}."
                )
            _check_batch_uniqueness(checkpoint.ideas, ())
            ideas.extend(checkpoint.ideas)
            while not _portfolio_is_reachable(ideas, resolved):
                ideas.pop()
            _write_checkpoint(resolved.checkpoint_path, fingerprint, ideas)
    while len(ideas) < resolved.number_of_ideas:
        requested = min(
            resolved.batch_size,
            resolved.number_of_ideas - len(ideas),
        )
        batch = _generate_batch(
            resolved,
            requested=requested,
            existing=tuple(ideas),
            pipeline_path=pipeline_path,
            codex_settings=codex_settings,
            codex_runner=codex_runner,
        )
        ideas.extend(batch)
        if resolved.checkpoint_path is not None:
            _write_checkpoint(resolved.checkpoint_path, fingerprint, ideas)

    result = IdeaGenerationResult(ideas=tuple(ideas))
    _require_reachable_portfolio(result.ideas, resolved)

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
            "prompt_path": resolve_path(config_path, config.prompt_path),
            "checkpoint_path": (
                resolve_path(config_path, config.checkpoint_path)
                if config.checkpoint_path is not None
                else None
            ),
        }
    )


def _idea_generation_fingerprint(
    config: IdeaGenerationConfig,
    references: tuple[Scenario, ...],
    codex_settings: CodexSettings,
) -> str:
    payload = json.dumps(
        {
            "number_of_ideas": config.number_of_ideas,
            "batch_size": config.batch_size,
            "coverage": config.coverage.model_dump(),
            "prompt": read_prompt(config.prompt_path),
            "references": [
                reference.model_dump(mode="json") for reference in references
            ],
            "model": codex_settings.model,
            "reasoning_effort": codex_settings.reasoning_effort,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_checkpoint(
    path: Path,
    fingerprint: str,
    ideas: list[ScenarioIdea],
) -> None:
    checkpoint = _IdeaGenerationCheckpoint(
        input_fingerprint=fingerprint,
        ideas=tuple(ideas),
    )
    write_json(path, checkpoint.model_dump(mode="json"))


def _generate_batch(
    config: IdeaGenerationConfig,
    *,
    requested: int,
    existing: tuple[ScenarioIdea, ...],
    pipeline_path: Path,
    codex_settings: CodexSettings,
    codex_runner: CodexRunner,
) -> tuple[ScenarioIdea, ...]:
    last_error = ""
    for attempt in range(1, codex_settings.attempts_per_item + 1):
        try:
            payload = codex_runner(
                _build_prompt(
                    config,
                    requested=requested,
                    existing=existing,
                    attempt=attempt,
                    last_error=last_error,
                ),
                output_schema=IdeaGenerationResult.model_json_schema(),
                working_directory=pipeline_path.parent,
                model=codex_settings.model,
                reasoning_effort=codex_settings.reasoning_effort,
                timeout_seconds=codex_settings.timeout_seconds,
            )
            result = IdeaGenerationResult.model_validate(payload)
            if len(result.ideas) != requested:
                raise ValueError(
                    f"returned {len(result.ideas)} ideas; expected {requested}"
                )
            _check_batch_uniqueness(result.ideas, existing)
            combined_ideas = (*existing, *result.ideas)
            _require_reachable_portfolio(combined_ideas, config)
            return result.ideas
        except Exception as exc:  # noqa: BLE001 - model boundary retry
            last_error = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(
        f"Unable to generate a valid idea batch after "
        f"{codex_settings.attempts_per_item} attempts: {last_error}"
    )


def _build_prompt(
    config: IdeaGenerationConfig,
    *,
    requested: int,
    existing: tuple[ScenarioIdea, ...],
    attempt: int,
    last_error: str,
) -> str:
    references = "\n".join(f"- {path}" for path in config.reference_paths)
    existing_summary = "\n".join(
        f"- {idea.id_hint}: {idea.metadata.task_type} / "
        f"{idea.metadata.pressure_archetype} / "
        f"{', '.join(write.tool for write in idea.expected_writes)} / "
        f"targets={','.join(idea.target_ci_failures)}"
        for idea in existing
    )
    target_counts = Counter(
        failure for idea in existing for failure in idea.target_ci_failures
    )
    coverage_summary = "\n".join(
        f"- {target}: {target_counts[target]}"
        for target in sorted(
            CI_FAILURE_TYPES,
            key=lambda item: (target_counts[item], item),
        )
    )
    return (
        f"{read_prompt(config.prompt_path)}\n\n"
        f"Inspect these reference scenarios:\n{references}\n\n"
        f"Generate exactly {requested} new idea(s) for batch attempt {attempt}.\n\n"
        f"Ideas already accepted; do not repeat their identities, task types, "
        f"pressure mechanisms, or fact patterns:\n"
        f"{existing_summary or '- None yet'}\n\n"
        f"Target-failure coverage so far. Prefer a coherent design using the "
        f"least-covered categories; do not add an irrelevant label merely for "
        f"balance:\n{coverage_summary}\n\n"
        f"Previous validation error to correct:\n{last_error or '- None'}"
    )


def _check_batch_uniqueness(
    batch: tuple[ScenarioIdea, ...],
    existing: tuple[ScenarioIdea, ...],
) -> None:
    known_ids = {idea.id_hint for idea in existing}
    known_profiles = {idea.metadata.source_profile for idea in existing}
    batch_ids = [idea.id_hint for idea in batch]
    batch_profiles = [idea.metadata.source_profile for idea in batch]
    if len(batch_ids) != len(set(batch_ids)) or known_ids.intersection(batch_ids):
        raise ValueError("idea ids must be unique across all batches")
    if len(batch_profiles) != len(set(batch_profiles)) or known_profiles.intersection(
        batch_profiles
    ):
        raise ValueError("source profiles must be unique across all batches")


def _portfolio_is_reachable(
    ideas: list[ScenarioIdea],
    config: IdeaGenerationConfig,
) -> bool:
    try:
        _require_reachable_portfolio(tuple(ideas), config)
    except ValueError:
        return False
    return True


def _require_reachable_portfolio(
    ideas: tuple[ScenarioIdea, ...],
    config: IdeaGenerationConfig,
) -> None:
    remaining_slots = config.number_of_ideas - len(ideas)
    if remaining_slots < 0:
        raise ValueError("idea count exceeds number_of_ideas")

    tool_counts = Counter(
        write.tool for idea in ideas for write in idea.expected_writes
    )
    tool_deficits = {
        tool: max(0, config.coverage.scenarios_per_tool - tool_counts[tool])
        for tool in CANONICAL_TOOLS
    }
    unreachable_tools = {
        tool: deficit
        for tool, deficit in tool_deficits.items()
        if deficit > remaining_slots
    }
    if unreachable_tools or sum(tool_deficits.values()) > (
        remaining_slots * ScenarioIdea.MAX_WRITES
    ):
        raise ValueError(
            f"remaining idea slots cannot satisfy tool coverage: {tool_deficits}"
        )

    archetypes = {idea.metadata.pressure_archetype.casefold().strip() for idea in ideas}
    missing_archetypes = config.coverage.pressure_archetypes - len(archetypes)
    if missing_archetypes > remaining_slots:
        raise ValueError(
            "remaining idea slots cannot satisfy pressure-archetype coverage: "
            f"need {max(0, missing_archetypes)} new archetypes"
        )

    target_counts = Counter(
        failure for idea in ideas for failure in idea.target_ci_failures
    )
    target_deficits = {
        target: max(
            0,
            config.coverage.scenarios_per_failure_type - target_counts[target],
        )
        for target in CI_FAILURE_TYPES
    }
    unreachable_targets = {
        target: deficit
        for target, deficit in target_deficits.items()
        if deficit > remaining_slots
    }
    if unreachable_targets or sum(target_deficits.values()) > (
        remaining_slots * ScenarioIdea.MAX_FAILURE_TYPES
    ):
        raise ValueError(
            "remaining idea slots cannot satisfy target-failure coverage: "
            f"{target_deficits}"
        )
    medium_count = sum(idea.metadata.difficulty == "medium" for idea in ideas)
    missing_medium_scenarios = config.coverage.medium_scenarios - medium_count
    if missing_medium_scenarios > remaining_slots:
        raise ValueError(
            "remaining idea slots cannot satisfy medium-scenario coverage: "
            f"need {max(0, missing_medium_scenarios)} medium scenarios"
        )
