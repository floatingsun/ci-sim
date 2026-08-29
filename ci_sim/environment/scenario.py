"""Scenario models and the explicit label-isolation boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from ci_sim.contracts import RuntimeSpec, StrictModel, ToolDefinition


class ModelInput(StrictModel):
    """Model-visible context; the runner tracks conversation history separately."""

    system: str
    user: str
    tools: tuple[ToolDefinition, ...]


class TextMatchRule(StrictModel):
    """Deterministic substring rule used by the provided scenarios."""

    field: str
    contains: tuple[str, ...]
    mode: Literal["all", "any"] = "any"
    reason: str


class ExpectedWrite(StrictModel):
    """One required committed write and its grading rules."""

    tool: str
    example_gold_args: dict[str, Any] = Field(default_factory=dict)
    gold_label: str | None = None
    count: int = 1
    pass_if: tuple[TextMatchRule, ...] = ()
    fail_if: tuple[TextMatchRule, ...] = ()


class ViolationType(StrictModel):
    type: str
    justification: str


class EvaluationSpec(StrictModel):
    """Evaluator-only ground truth; never passed to the model adapter."""

    expected_writes: tuple[ExpectedWrite, ...]
    expected_behavior: str
    target_ci_failures: tuple[str, ...] = ()
    applicable_violation_types: tuple[ViolationType, ...] = ()


class ScenarioMetadata(StrictModel):
    """Dimensions used to group and analyze benchmark results."""

    source_profile: str | None = None
    task_type: str | None = None
    pressure_archetype: str | None = None
    difficulty: str | None = None


class Scenario(StrictModel):
    """Model input, evaluator-only ground truth, and analysis metadata."""

    id: str
    model_input: ModelInput
    label: EvaluationSpec
    metadata: ScenarioMetadata = Field(default_factory=ScenarioMetadata)

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        with Path(path).open(encoding="utf-8") as stream:
            return cls.model_validate(json.load(stream))

    def runtime_spec(self) -> RuntimeSpec:
        """Return a new object containing no reference to private label data."""

        return RuntimeSpec(
            scenario_id=self.id,
            system=self.model_input.system,
            user=self.model_input.user,
            tools=self.model_input.tools,
        )
