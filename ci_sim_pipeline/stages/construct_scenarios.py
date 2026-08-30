"""Construct evaluator-compatible scenarios from generated ideas."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import ValidationError

from ci_sim.contracts import StrictModel
from ci_sim.environment.scenario import Scenario
from ci_sim.environment.workplace.tools import TOOLS_BY_CANONICAL_NAME

from ..codex import run as default_codex_runner
from ..contracts import (
    CodexRunner,
    CodexSettings,
    ConstructionCandidate,
    ConstructionConfig,
    ConstructionResult,
    IdeaGenerationResult,
    ScenarioIdea,
    ScenarioOutput,
    read_json,
    read_prompt,
    resolve_path,
    write_json,
)
from ..quality import find_quality_issues
from ..scenario_alignment import ensure_scenario_matches_idea


class _ConstructionCheckpoint(StrictModel):
    input_fingerprint: str
    scenarios: tuple[Scenario, ...] = ()


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
    references = tuple(Scenario.load(path) for path in resolved.reference_paths)
    reference_system = references[0].model_input.system

    fingerprint = _construction_fingerprint(
        ideas,
        resolved,
        references,
        codex_settings,
    )
    idea_indices = {idea.id_hint: index for index, idea in enumerate(ideas.ideas)}
    scenarios_by_index: dict[int, Scenario] = {}
    if resolved.checkpoint_path is not None and resolved.checkpoint_path.is_file():
        try:
            checkpoint = _ConstructionCheckpoint.model_validate(
                read_json(resolved.checkpoint_path)
            )
        except (json.JSONDecodeError, ValidationError):
            checkpoint = None
        if checkpoint is not None and checkpoint.input_fingerprint == fingerprint:
            for scenario in checkpoint.scenarios:
                index = idea_indices.get(scenario.id)
                if index is None or index in scenarios_by_index:
                    raise ValueError(
                        "Construction checkpoint has unknown or duplicate ids."
                    )
                try:
                    ensure_scenario_matches_idea(scenario, ideas.ideas[index])
                except ValueError:
                    continue
                issues = find_quality_issues(scenario)
                if issues:
                    continue
                scenarios_by_index[index] = scenario

    pending = tuple(
        (index, idea)
        for index, idea in enumerate(ideas.ideas)
        if index not in scenarios_by_index
    )
    errors_by_index: dict[int, Exception] = {}
    with ThreadPoolExecutor(max_workers=codex_settings.concurrency) as executor:
        futures = {
            executor.submit(
                _construct_scenario,
                idea,
                resolved,
                reference_system=reference_system,
                pipeline_path=pipeline_path,
                codex_settings=codex_settings,
                codex_runner=codex_runner,
            ): index
            for index, idea in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                scenarios_by_index[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - drain all completed work
                errors_by_index[index] = exc
                continue
            if resolved.checkpoint_path is not None:
                checkpoint = _ConstructionCheckpoint(
                    input_fingerprint=fingerprint,
                    scenarios=tuple(
                        scenarios_by_index[index]
                        for index in sorted(scenarios_by_index)
                    ),
                )
                write_json(
                    resolved.checkpoint_path,
                    checkpoint.model_dump(mode="json"),
                )
    if errors_by_index:
        details = "; ".join(
            f"{ideas.ideas[index].id_hint}: {type(error).__name__}: {error}"
            for index, error in sorted(errors_by_index.items())
        )
        raise RuntimeError(
            "Construction failed for one or more ideas after checkpointing all "
            f"successful work: {details}"
        )
    scenarios = tuple(scenarios_by_index[index] for index in range(len(ideas.ideas)))
    if len({scenario.id for scenario in scenarios}) != len(scenarios):
        raise ValueError("Stage 2 produced duplicate scenario ids.")

    result = ConstructionResult(
        candidates=tuple(
            ConstructionCandidate(idea=idea, scenario=scenario)
            for idea, scenario in zip(ideas.ideas, scenarios, strict=True)
        )
    )
    write_json(resolved.output_path, result.model_dump(mode="json"))
    return (resolved.output_path,)


def _construct_scenario(
    idea: ScenarioIdea,
    config: ConstructionConfig,
    *,
    reference_system: str,
    pipeline_path: Path,
    codex_settings: CodexSettings,
    codex_runner: CodexRunner,
) -> Scenario:
    last_error = ""
    for attempt in range(1, codex_settings.attempts_per_item + 1):
        try:
            payload = codex_runner(
                _build_prompt(
                    config,
                    idea,
                    attempt=attempt,
                    last_error=last_error,
                ),
                output_schema=ScenarioOutput.model_json_schema(),
                working_directory=pipeline_path.parent,
                model=codex_settings.model,
                reasoning_effort=codex_settings.reasoning_effort,
                timeout_seconds=codex_settings.timeout_seconds,
            )
            output = ScenarioOutput.model_validate(payload)
            scenario = Scenario.model_validate(json.loads(output.scenario_json))
            scenario = _apply_canonical_runtime(
                scenario,
                idea,
                system=reference_system,
            )
            ensure_scenario_matches_idea(scenario, idea)
            issues = find_quality_issues(scenario)
            if issues:
                rendered = "; ".join(
                    f"{issue.code}: {issue.message}" for issue in issues
                )
                raise ValueError(rendered)
            return scenario
        except Exception as exc:  # noqa: BLE001 - model boundary retry
            last_error = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(
        f"Unable to construct {idea.id_hint} after "
        f"{codex_settings.attempts_per_item} attempts: {last_error}"
    )


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
            "prompt_path": resolve_path(config_path, config.prompt_path),
            "checkpoint_path": (
                resolve_path(config_path, config.checkpoint_path)
                if config.checkpoint_path is not None
                else None
            ),
        }
    )


def _construction_fingerprint(
    ideas: IdeaGenerationResult,
    config: ConstructionConfig,
    references: tuple[Scenario, ...],
    codex_settings: CodexSettings,
) -> str:
    payload = json.dumps(
        {
            "ideas": ideas.model_dump(mode="json"),
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


def _build_prompt(
    config: ConstructionConfig,
    idea: ScenarioIdea,
    *,
    attempt: int,
    last_error: str,
) -> str:
    references = "\n".join(f"- {path}" for path in config.reference_paths)
    idea_json = json.dumps(idea.model_dump(mode="json"), indent=2)
    return (
        f"{read_prompt(config.prompt_path)}\n\n"
        f"Inspect these reference scenarios:\n{references}\n\n"
        f"Construct a scenario for this idea. Return the complete Scenario as a "
        f"JSON-encoded string in the scenario_json field:\n{idea_json}\n\n"
        f"This is construction attempt {attempt}. Previous validation error:\n"
        f"{last_error or '- None'}"
    )


def _apply_canonical_runtime(
    scenario: Scenario,
    idea: ScenarioIdea,
    *,
    system: str,
) -> Scenario:
    tools = tuple(
        TOOLS_BY_CANONICAL_NAME[write.tool].definition()
        for write in idea.expected_writes
    )
    return scenario.model_copy(
        update={
            "model_input": scenario.model_input.model_copy(
                update={"system": system, "tools": tools}
            )
        }
    )
