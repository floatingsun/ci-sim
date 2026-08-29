"""Validate, filter, and emit evaluator-ready scenario files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from ci_sim.contracts import StrictModel
from ci_sim.environment.grading import validate_scenario
from ci_sim.environment.scenario import Scenario

from ..codex import run as default_codex_runner
from ..contracts import (
    CodexRunner,
    CodexSettings,
    read_json,
    resolve_path,
    write_json,
)
from .construct_scenarios import ConstructionResult

QualityDimension = Literal[
    "schema",
    "task_completion",
    "contextual_integrity",
    "grader_consistency",
    "realism",
    "temptation",
    "answer_leakage",
    "novelty",
    "metadata",
]


class ValidationConfig(StrictModel):
    input_path: Path
    reference_paths: tuple[Path, ...] = Field(min_length=2)
    output_directory: Path
    report_path: Path
    environment: Literal["workplace_email"] = "workplace_email"
    prompt: str


class QualityIssue(StrictModel):
    dimension: QualityDimension
    severity: Literal["info", "warning", "error"]
    path: str | None = None
    message: str
    suggested_fix: str | None = None


class QualityScores(StrictModel):
    task_completion: float = Field(ge=0.0, le=1.0)
    contextual_integrity: float = Field(ge=0.0, le=1.0)
    grader_consistency: float = Field(ge=0.0, le=1.0)
    realism: float = Field(ge=0.0, le=1.0)
    temptation: float = Field(ge=0.0, le=1.0)
    answer_leakage_resistance: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)


class ScenarioReview(StrictModel):
    scenario_id: str
    decision: Literal["pass", "revise", "reject"]
    summary: str
    scores: QualityScores
    issues: tuple[QualityIssue, ...] = ()


class ValidationReport(StrictModel):
    reviews: tuple[ScenarioReview, ...]


def validate_scenarios(
    config: ValidationConfig,
    *,
    config_path: str | Path,
    codex_settings: CodexSettings,
    codex_runner: CodexRunner = default_codex_runner,
) -> tuple[Path, ...]:
    """Review candidates with Codex and emit approved standalone scenarios."""

    pipeline_path = Path(config_path).expanduser().resolve()
    resolved = _resolve_config(config, pipeline_path)
    candidates = ConstructionResult.model_validate(read_json(resolved.input_path))
    for reference_path in resolved.reference_paths:
        Scenario.load(reference_path)

    reviews = tuple(
        _review_scenario(
            scenario,
            resolved,
            pipeline_path=pipeline_path,
            codex_settings=codex_settings,
            codex_runner=codex_runner,
        )
        for scenario in candidates.scenarios
    )

    output_paths: list[Path] = []
    for scenario, review in zip(candidates.scenarios, reviews, strict=True):
        if review.decision != "pass":
            continue
        output_path = resolved.output_directory / (
            f"{_safe_file_stem(scenario.id)}.json"
        )
        write_json(output_path, scenario.model_dump(mode="json"))
        output_paths.append(output_path)

    report = ValidationReport(reviews=reviews)
    write_json(resolved.report_path, report.model_dump(mode="json"))
    return tuple(output_paths)


def _review_scenario(
    scenario: Scenario,
    config: ValidationConfig,
    *,
    pipeline_path: Path,
    codex_settings: CodexSettings,
    codex_runner: CodexRunner,
) -> ScenarioReview:
    deterministic_issues = tuple(
        QualityIssue(
            dimension="grader_consistency",
            severity="error",
            message=issue.message,
            suggested_fix=f"Resolve deterministic check: {issue.code}",
        )
        for issue in validate_scenario(scenario)
    )
    payload = codex_runner(
        _build_prompt(config, scenario, deterministic_issues),
        output_schema=ScenarioReview.model_json_schema(),
        working_directory=pipeline_path.parent,
        model=codex_settings.model,
        reasoning_effort=codex_settings.reasoning_effort,
        timeout_seconds=codex_settings.timeout_seconds,
    )
    review = ScenarioReview.model_validate(payload)
    if review.scenario_id != scenario.id:
        raise ValueError(
            f"Stage 3 reviewed {review.scenario_id}; expected {scenario.id}."
        )
    if not deterministic_issues:
        return review

    decision = "revise" if review.decision == "pass" else review.decision
    return review.model_copy(
        update={
            "decision": decision,
            "issues": (*review.issues, *deterministic_issues),
        }
    )


def _resolve_config(
    config: ValidationConfig,
    config_path: Path,
) -> ValidationConfig:
    return config.model_copy(
        update={
            "input_path": resolve_path(config_path, config.input_path),
            "reference_paths": tuple(
                resolve_path(config_path, path) for path in config.reference_paths
            ),
            "output_directory": resolve_path(
                config_path,
                config.output_directory,
            ),
            "report_path": resolve_path(config_path, config.report_path),
        }
    )


def _build_prompt(
    config: ValidationConfig,
    scenario: Scenario,
    deterministic_issues: tuple[QualityIssue, ...],
) -> str:
    references = "\n".join(f"- {path}" for path in config.reference_paths)
    scenario_json = json.dumps(scenario.model_dump(mode="json"), indent=2)
    issues_json = json.dumps(
        [issue.model_dump(mode="json") for issue in deterministic_issues],
        indent=2,
    )
    return (
        f"{config.prompt.rstrip()}\n\n"
        f"Compare against these reference scenarios:\n{references}\n\n"
        f"Candidate scenario:\n{scenario_json}\n\n"
        f"Deterministic validation issues:\n{issues_json}"
    )


def _safe_file_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")
    return stem or "scenario"
