"""Checks that reject mechanically inconsistent scenarios before model runs."""

from __future__ import annotations

from pydantic import ValidationError

from ci_sim.contracts import StrictModel

from ..scenario import Scenario
from ..workplace.tools import TOOLS_BY_CANONICAL_NAME, TOOLS_BY_MODEL_NAME
from .text_matching import evaluate_text_rule


class ValidationIssue(StrictModel):
    code: str
    message: str


def validate_scenario(scenario: Scenario) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    runtime_names = [tool.name for tool in scenario.model_input.tools]

    if len(runtime_names) != len(set(runtime_names)):
        issues.append(
            ValidationIssue(
                code="duplicate-runtime-tool",
                message="model_input.tools contains duplicate names.",
            )
        )

    for name in runtime_names:
        if name not in TOOLS_BY_MODEL_NAME:
            issues.append(
                ValidationIssue(
                    code="unknown-runtime-tool",
                    message=f"No implementation exists for {name}.",
                )
            )

    expected_names: set[str] = set()
    for expected in scenario.label.expected_writes:
        if expected.tool in expected_names:
            issues.append(
                ValidationIssue(
                    code="duplicate-expected-tool",
                    message=f"Duplicate expected write for {expected.tool}.",
                )
            )
        expected_names.add(expected.tool)

        tool = TOOLS_BY_CANONICAL_NAME.get(expected.tool)
        if tool is None:
            issues.append(
                ValidationIssue(
                    code="unknown-expected-tool",
                    message=f"No implementation exists for {expected.tool}.",
                )
            )
            continue

        if tool.model_name not in runtime_names:
            issues.append(
                ValidationIssue(
                    code="expected-tool-not-enabled",
                    message=f"{expected.tool} is expected but not model-visible.",
                )
            )

        try:
            tool.validate(expected.example_gold_args)
        except ValidationError as exc:
            issues.append(
                ValidationIssue(
                    code="invalid-gold-args",
                    message=f"Invalid gold arguments for {expected.tool}: {exc}",
                )
            )
            continue

        for rule in expected.pass_if:
            if not evaluate_text_rule(rule, expected.example_gold_args):
                issues.append(
                    ValidationIssue(
                        code="gold-fails-required-rule",
                        message=f"Gold arguments fail: {rule.reason}",
                    )
                )
        for rule in expected.fail_if:
            if evaluate_text_rule(rule, expected.example_gold_args):
                issues.append(
                    ValidationIssue(
                        code="gold-triggers-safety-rule",
                        message=f"Gold arguments trigger: {rule.reason}",
                    )
                )

    return issues
