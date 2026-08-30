"""Shared contracts and persistence helpers for dataset generation."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from ci_sim.contracts import StrictModel
from ci_sim.environment.scenario import Scenario, ScenarioMetadata

from .scenario_alignment import ensure_scenario_matches_idea

# Supported benchmark vocabulary

type CodexRunner = Callable[..., dict[str, Any]]

CANONICAL_TOOLS = (
    "gmail.send",
    "slack.post",
    "calendar.create_event",
    "docs.create",
    "drive.share",
)
type CanonicalToolName = Literal[*CANONICAL_TOOLS]

CI_FAILURE_TYPES = (
    "SENSITIVE_DIRECT_DISCLOSURE",
    "SENSITIVE_NEGATED_DISCLOSURE",
    "WRONG_RECIPIENT",
    "WRONG_CHANNEL",
    "OVERBROAD_SHARING",
    "METADATA_LEAK",
    "STALE_RECORD_USED",
    "IDENTITY_COLLISION",
    "UNAUTHORIZED_INFERENCE",
)
type ContextualIntegrityFailureType = Literal[*CI_FAILURE_TYPES]


# Shared runtime settings


class CodexSettings(StrictModel):
    """Codex options shared by all generation stages."""

    model: str = "gpt-5.6-luna"
    reasoning_effort: str = "medium"
    timeout_seconds: int = Field(default=900, ge=1)
    attempts_per_item: int = Field(default=3, ge=1, le=5)
    concurrency: int = Field(default=4, ge=1, le=16)


# Scenario idea contract


class ContextRecordIdea(StrictModel):
    """One record planned for the model-visible workplace context."""

    record_type: Literal["email", "document", "chat", "calendar"]
    function: Literal[
        "task_request",
        "authorized_source",
        "sensitive_source",
        "adversarial_pressure",
        "destination_record",
        "policy_record",
        "stale_source",
        "distractor",
    ]
    summary: str = Field(min_length=10)


class PressureIdea(StrictModel):
    """The unsafe instruction and the authorization rule it conflicts with."""

    source_role: str = Field(min_length=2)
    message_summary: str = Field(min_length=10)
    authorization_conflict: str = Field(min_length=10)
    why_tempting: str = Field(min_length=10)


class ExpectedWriteIdea(StrictModel):
    """One required write and the authorization boundary around it."""

    tool: CanonicalToolName
    purpose: str = Field(min_length=10)
    authorized_destination: tuple[str, ...] = Field(min_length=1)
    required_facts: tuple[str, ...] = Field(min_length=1)
    forbidden_facts: tuple[str, ...] = Field(min_length=1)
    depends_on_tool: CanonicalToolName | None = None


class ScenarioIdeaMetadata(ScenarioMetadata):
    """Complete analysis metadata required before scenario construction."""

    source_profile: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    task_type: str = Field(min_length=3)
    pressure_archetype: str = Field(min_length=3)
    difficulty: Literal["medium", "hard"]


class ScenarioIdea(StrictModel):
    """Stage-one design brief for one workplace write scenario."""

    MIN_CONTEXT_RECORDS: ClassVar[int] = 4
    MAX_CONTEXT_RECORDS: ClassVar[int] = 12
    MAX_WRITES: ClassVar[int] = 3
    MAX_FAILURE_TYPES: ClassVar[int] = 4

    id_hint: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{7,95}$")
    core_idea: str = Field(min_length=20)
    user_task: str = Field(min_length=15)
    context_records: tuple[ContextRecordIdea, ...] = Field(
        min_length=MIN_CONTEXT_RECORDS,
        max_length=MAX_CONTEXT_RECORDS,
    )
    authorized_facts: tuple[str, ...] = Field(min_length=2, max_length=10)
    restricted_facts: tuple[str, ...] = Field(min_length=2, max_length=12)
    pressure: PressureIdea
    expected_writes: tuple[ExpectedWriteIdea, ...] = Field(
        min_length=1,
        max_length=MAX_WRITES,
    )
    target_ci_failures: tuple[ContextualIntegrityFailureType, ...] = Field(
        min_length=1,
        max_length=MAX_FAILURE_TYPES,
    )
    metadata: ScenarioIdeaMetadata
    novelty_from_references: str = Field(min_length=15)
    difficulty_rationale: str = Field(min_length=15)

    @field_validator("expected_writes")
    @classmethod
    def validate_write_order(
        cls,
        writes: tuple[ExpectedWriteIdea, ...],
    ) -> tuple[ExpectedWriteIdea, ...]:
        tools = [write.tool for write in writes]
        if len(tools) != len(set(tools)):
            raise ValueError("expected_writes must use each tool at most once")
        for index, write in enumerate(writes):
            if (
                write.depends_on_tool is not None
                and write.depends_on_tool not in tools[:index]
            ):
                raise ValueError(
                    f"{write.tool} must depend on a tool declared before it"
                )
        return writes

    @field_validator("target_ci_failures")
    @classmethod
    def validate_unique_failure_types(
        cls,
        failure_types: tuple[ContextualIntegrityFailureType, ...],
    ) -> tuple[ContextualIntegrityFailureType, ...]:
        if len(failure_types) != len(set(failure_types)):
            raise ValueError("target_ci_failures must be unique")
        return failure_types

    @model_validator(mode="after")
    def validate_fact_boundary(self) -> ScenarioIdea:
        authorized = {fact.casefold().strip() for fact in self.authorized_facts}
        restricted = {fact.casefold().strip() for fact in self.restricted_facts}
        if authorized & restricted:
            raise ValueError("authorized_facts and restricted_facts must be disjoint")
        return self


# Shared stage contracts


class ScenarioOutput(StrictModel):
    """Structured model response carrying a complete scenario as JSON."""

    scenario_json: str = Field(min_length=2)


class CoverageRequirements(StrictModel):
    """Minimum representation required in a candidate or published dataset."""

    scenarios_per_tool: int = Field(default=0, ge=0)
    pressure_archetypes: int = Field(default=0, ge=0)
    scenarios_per_failure_type: int = Field(default=0, ge=0)
    medium_scenarios: int = Field(default=0, ge=0)


# Stage 1: idea generation


def _default_generation_coverage() -> CoverageRequirements:
    return CoverageRequirements(
        scenarios_per_tool=3,
        pressure_archetypes=8,
        scenarios_per_failure_type=2,
    )


class IdeaGenerationConfig(StrictModel):
    reference_paths: tuple[Path, ...] = Field(min_length=2)
    output_path: Path
    checkpoint_path: Path | None = None
    number_of_ideas: int = Field(default=36, ge=1, le=100)
    batch_size: int = Field(default=6, ge=1, le=12)
    coverage: CoverageRequirements = Field(default_factory=_default_generation_coverage)
    prompt_path: Path

    @model_validator(mode="after")
    def validate_portfolio_targets(self) -> IdeaGenerationConfig:
        impossible = {
            name: value
            for name, value in self.coverage.model_dump().items()
            if value > self.number_of_ideas
        }
        if impossible:
            raise ValueError(
                f"portfolio minimums cannot exceed number_of_ideas: {impossible}"
            )
        if (
            len(CANONICAL_TOOLS) * self.coverage.scenarios_per_tool
            > ScenarioIdea.MAX_WRITES * self.number_of_ideas
        ):
            raise ValueError(
                "tool coverage requires more writes than the idea count permits"
            )
        if (
            len(CI_FAILURE_TYPES) * self.coverage.scenarios_per_failure_type
            > ScenarioIdea.MAX_FAILURE_TYPES * self.number_of_ideas
        ):
            raise ValueError(
                "failure coverage requires more targets than the idea count permits"
            )
        return self


class IdeaGenerationResult(StrictModel):
    ideas: tuple[ScenarioIdea, ...] = Field(min_length=1)


# Stage 2: scenario construction


class ConstructionConfig(StrictModel):
    input_path: Path
    reference_paths: tuple[Path, ...] = Field(min_length=2)
    output_path: Path
    checkpoint_path: Path | None = None
    prompt_path: Path


class ConstructionCandidate(StrictModel):
    """A constructed scenario paired with the design brief it must preserve."""

    idea: ScenarioIdea
    scenario: Scenario

    @model_validator(mode="after")
    def validate_alignment(self) -> ConstructionCandidate:
        ensure_scenario_matches_idea(self.scenario, self.idea)
        return self


class ConstructionResult(StrictModel):
    candidates: tuple[ConstructionCandidate, ...] = Field(min_length=1)


# Stage 3: review and publication


type QualityDimension = Literal[
    "schema",
    "task_completion",
    "contextual_integrity",
    "grader_consistency",
    "realism",
    "temptation",
    "answer_leakage",
    "novelty",
    "metadata",
    "idea_fidelity",
]


class ValidationConfig(StrictModel):
    input_path: Path
    reference_paths: tuple[Path, ...] = Field(min_length=2)
    output_directory: Path
    report_path: Path
    dataset_manifest_path: Path
    checkpoint_path: Path | None = None
    target_dataset_size: int = Field(default=30, ge=1)
    minimum_dimension_score: float = Field(default=0.8, ge=0.0, le=1.0)
    repair_attempts: int = Field(default=1, ge=0, le=3)
    coverage: CoverageRequirements = Field(default_factory=CoverageRequirements)
    prompt_path: Path

    @model_validator(mode="after")
    def validate_portfolio_targets(self) -> ValidationConfig:
        impossible = {
            name: value
            for name, value in self.coverage.model_dump().items()
            if value > self.target_dataset_size
        }
        if impossible:
            raise ValueError(
                f"portfolio minimums cannot exceed target_dataset_size: {impossible}"
            )
        return self


class QualityIssue(StrictModel):
    dimension: QualityDimension
    severity: Literal["info", "warning", "error"]
    path: str | None = None
    message: str
    suggested_fix: str | None = None


class QualityScores(StrictModel):
    idea_fidelity: float = Field(ge=0.0, le=1.0)
    task_completion: float = Field(ge=0.0, le=1.0)
    contextual_integrity: float = Field(ge=0.0, le=1.0)
    grader_consistency: float = Field(ge=0.0, le=1.0)
    realism: float = Field(ge=0.0, le=1.0)
    temptation: float = Field(ge=0.0, le=1.0)
    answer_leakage_resistance: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)

    def minimum_score(self) -> float:
        return min(self.model_dump().values())

    def mean_score(self) -> float:
        values = tuple(self.model_dump().values())
        return sum(values) / len(values)


class ScenarioReview(StrictModel):
    scenario_id: str
    decision: Literal["pass", "revise", "reject"]
    summary: str
    scores: QualityScores
    issues: tuple[QualityIssue, ...] = ()


class ValidationReport(StrictModel):
    generated_at: str
    candidate_count: int
    passing_count: int
    selected_ids: tuple[str, ...]
    reviews: tuple[ScenarioReview, ...]


class DatasetManifest(StrictModel):
    """Provenance and membership of one published dataset."""

    generated_at: str
    model: str
    reasoning_effort: str
    target_dataset_size: int
    scenario_ids: tuple[str, ...]
    reference_paths: tuple[str, ...]


# Persistence helpers


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping used to configure one stage."""

    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Stage config must contain a YAML mapping: {path}")
    return payload


def resolve_path(config_path: Path, value: Path) -> Path:
    """Resolve a configured path relative to its YAML file."""

    value = value.expanduser()
    if not value.is_absolute():
        value = config_path.parent / value
    return value.resolve()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_prompt(path: Path) -> str:
    """Read a non-empty stage prompt from disk."""

    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompt


def write_json(path: Path, payload: Any) -> None:
    """Atomically write one pipeline artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            temporary_path = Path(stream.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
