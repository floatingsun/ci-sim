"""Scenario models and the explicit label-isolation boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from ci_sim.contracts import RuntimeSpec, StrictModel, ToolDefinition

type TextMatchOperator = Literal[
    "contains",
    "equals",
    "not_equals",
    "exact_set",
    "not_exact_set",
    "empty",
    "nonempty",
    "references_prior_result",
]


class ModelInput(StrictModel):
    """Model-visible context; the runner tracks conversation history separately."""

    system: str
    user: str
    tools: tuple[ToolDefinition, ...] = Field(min_length=1)


class TextMatchRule(StrictModel):
    """Deterministic rule using one of the evaluator's supported comparisons."""

    field: str
    operator: TextMatchOperator = "contains"
    contains: tuple[str, ...] = ()
    mode: Literal["all", "any"] = "any"
    reason: str
    violation_types: tuple[str, ...] = ()
    reference_tool: str | None = None

    @model_validator(mode="after")
    def validate_operator_inputs(self) -> TextMatchRule:
        if self.operator == "contains" and not self.contains:
            raise ValueError("contains rules require at least one search term")
        if self.operator in {"equals", "not_equals"} and len(self.contains) != 1:
            raise ValueError(
                f"{self.operator} rules require exactly one expected value"
            )
        if self.operator in {"exact_set", "not_exact_set"} and not self.contains:
            raise ValueError(
                f"{self.operator} rules require at least one expected value"
            )
        if (
            self.operator in {"empty", "nonempty", "references_prior_result"}
            and self.contains
        ):
            raise ValueError(f"{self.operator} rules do not accept search terms")
        if self.operator == "references_prior_result" and not self.reference_tool:
            raise ValueError("references_prior_result rules require reference_tool")
        if self.operator != "references_prior_result" and self.reference_tool:
            raise ValueError(
                "reference_tool is only valid for references_prior_result rules"
            )
        return self


class ExpectedWrite(StrictModel):
    """One required committed write and its grading rules."""

    tool: str
    example_gold_args: dict[str, Any] = Field(default_factory=dict)
    gold_label: str | None = None
    count: int = Field(default=1, ge=1)
    pass_if: tuple[TextMatchRule, ...] = ()
    fail_if: tuple[TextMatchRule, ...] = ()


class ViolationType(StrictModel):
    type: str
    justification: str
    kind: Literal["contextual_integrity", "task"] = "contextual_integrity"


class EvaluationSpec(StrictModel):
    """Evaluator-only ground truth; never passed to the model adapter."""

    expected_writes: tuple[ExpectedWrite, ...] = Field(min_length=1)
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

    id: str = Field(min_length=1)
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
