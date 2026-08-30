"""Checks that reject mechanically inconsistent scenarios before model runs."""

from __future__ import annotations

from collections.abc import Collection

from pydantic import ValidationError

from ci_sim.contracts import RunArtifact, StrictModel, WriteRecord

from ..scenario import ExpectedWrite, Scenario, TextMatchRule, ViolationType
from ..workplace.tools import TOOLS_BY_CANONICAL_NAME, TOOLS_BY_MODEL_NAME
from .text_matching import evaluate_text_rule


class ValidationIssue(StrictModel):
    code: str
    message: str


def validate_scenario(scenario: Scenario) -> list[ValidationIssue]:
    """Return issues that would make deterministic grading incomplete or ambiguous."""

    issues: list[ValidationIssue] = []
    runtime_names = [tool.name for tool in scenario.model_input.tools]
    violation_types = {
        item.type: item for item in scenario.label.applicable_violation_types
    }
    gold_artifact = _build_gold_artifact(scenario)
    gold_writes = {write.canonical_tool: write for write in gold_artifact.writes}

    if len(violation_types) != len(scenario.label.applicable_violation_types):
        issues.append(
            ValidationIssue(
                code="duplicate-violation-type",
                message="applicable_violation_types contains duplicate types.",
            )
        )

    if len(set(scenario.label.target_ci_failures)) != len(
        scenario.label.target_ci_failures
    ):
        issues.append(
            ValidationIssue(
                code="duplicate-target-violation-type",
                message="target_ci_failures contains duplicate types.",
            )
        )

    for target in scenario.label.target_ci_failures:
        violation_type = violation_types.get(target)
        if violation_type is None:
            issues.append(
                ValidationIssue(
                    code="unknown-target-violation-type",
                    message=f"Target CI failure {target} is not declared applicable.",
                )
            )
        elif violation_type.kind != "contextual_integrity":
            issues.append(
                ValidationIssue(
                    code="target-is-not-ci-violation",
                    message=f"Target CI failure {target} is classified as task utility.",
                )
            )

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
    mapped_ci_types: set[str] = set()
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
            issues.extend(
                _validate_rule_field(
                    rule.field,
                    tool.args_model.model_fields,
                    expected.tool,
                )
            )
            issues.extend(
                _validate_rule_reference(
                    rule,
                    expected.tool,
                    scenario.label.expected_writes,
                )
            )
            issues.extend(
                _validate_rule_violation_types(
                    rule.violation_types,
                    violation_types,
                    expected_kind="task",
                    tool=expected.tool,
                )
            )
            gold_write = gold_writes.get(expected.tool)
            if gold_write is None or not evaluate_text_rule(
                rule,
                gold_write.arguments,
                artifact=gold_artifact,
                write_sequence=gold_write.sequence,
            ):
                issues.append(
                    ValidationIssue(
                        code="gold-fails-required-rule",
                        message=f"Gold arguments fail: {rule.reason}",
                    )
                )
        for rule in expected.fail_if:
            issues.extend(
                _validate_rule_field(
                    rule.field,
                    tool.args_model.model_fields,
                    expected.tool,
                )
            )
            issues.extend(
                _validate_rule_reference(
                    rule,
                    expected.tool,
                    scenario.label.expected_writes,
                )
            )
            if violation_types and not rule.violation_types:
                issues.append(
                    ValidationIssue(
                        code="unmapped-safety-rule",
                        message=(
                            f"Safety rule for {expected.tool} has no violation_types: "
                            f"{rule.reason}"
                        ),
                    )
                )
            issues.extend(
                _validate_rule_violation_types(
                    rule.violation_types,
                    violation_types,
                    expected_kind="contextual_integrity",
                    tool=expected.tool,
                )
            )
            mapped_ci_types.update(
                name
                for name in rule.violation_types
                if name in violation_types
                and violation_types[name].kind == "contextual_integrity"
                and _rule_field_is_scoreable(rule.field, tool.args_model.model_fields)
            )
            gold_write = gold_writes.get(expected.tool)
            if gold_write is not None and evaluate_text_rule(
                rule,
                gold_write.arguments,
                artifact=gold_artifact,
                write_sequence=gold_write.sequence,
            ):
                issues.append(
                    ValidationIssue(
                        code="gold-triggers-safety-rule",
                        message=f"Gold arguments trigger: {rule.reason}",
                    )
                )

    for name, violation_type in violation_types.items():
        if (
            violation_type.kind == "contextual_integrity"
            and name not in mapped_ci_types
        ):
            issues.append(
                ValidationIssue(
                    code="unscored-violation-type",
                    message=f"No safety rule scores applicable violation type {name}.",
                )
            )

    for name in runtime_names:
        tool = TOOLS_BY_MODEL_NAME.get(name)
        if tool is not None and tool.canonical_name not in expected_names:
            issues.append(
                ValidationIssue(
                    code="unexpected-enabled-write-tool",
                    message=f"Enabled write tool {name} has no expected write or grader.",
                )
            )

    return issues


def _rule_field_is_scoreable(field: str, tool_fields: Collection[str]) -> bool:
    if field == "any_write_arg":
        return True
    return field in tool_fields


def _validate_rule_field(
    field: str,
    tool_fields: Collection[str],
    tool: str,
) -> list[ValidationIssue]:
    if _rule_field_is_scoreable(field, tool_fields):
        return []
    return [
        ValidationIssue(
            code="invalid-rule-field",
            message=f"Rule for {tool} references unknown argument field {field}.",
        )
    ]


def _validate_rule_reference(
    rule: TextMatchRule,
    current_tool: str,
    expected_writes: tuple[ExpectedWrite, ...],
) -> list[ValidationIssue]:
    if rule.operator != "references_prior_result":
        return []
    reference_tool = rule.reference_tool
    ordered_tools = [item.tool for item in expected_writes]
    if reference_tool not in ordered_tools:
        return [
            ValidationIssue(
                code="unknown-reference-tool",
                message=f"Rule for {current_tool} references unknown tool {reference_tool}.",
            )
        ]
    if ordered_tools.index(reference_tool) >= ordered_tools.index(current_tool):
        return [
            ValidationIssue(
                code="reference-tool-not-prior",
                message=(
                    f"Rule for {current_tool} must reference a tool that appears earlier "
                    f"than {current_tool}."
                ),
            )
        ]
    return []


def _build_gold_artifact(scenario: Scenario) -> RunArtifact:
    writes: list[WriteRecord] = []
    for sequence, expected in enumerate(scenario.label.expected_writes, start=1):
        tool = TOOLS_BY_CANONICAL_NAME.get(expected.tool)
        if tool is None:
            continue
        try:
            arguments = tool.validate(expected.example_gold_args)
        except ValidationError:
            arguments = expected.example_gold_args
        writes.append(
            WriteRecord(
                sequence=sequence,
                call_id=f"gold-{sequence}",
                canonical_tool=tool.canonical_name,
                model_tool=tool.model_name,
                arguments=arguments,
                result_id=f"{tool.id_prefix}_{sequence:04d}",
            )
        )
    return RunArtifact(writes=tuple(writes))


def _validate_rule_violation_types(
    rule_types: tuple[str, ...],
    declared_types: dict[str, ViolationType],
    *,
    expected_kind: str,
    tool: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name in rule_types:
        violation_type = declared_types.get(name)
        if violation_type is None:
            issues.append(
                ValidationIssue(
                    code="unknown-rule-violation-type",
                    message=f"Rule for {tool} references undeclared type {name}.",
                )
            )
        elif violation_type.kind != expected_kind:
            issues.append(
                ValidationIssue(
                    code="wrong-rule-violation-kind",
                    message=(
                        f"Rule for {tool} maps {name} to {expected_kind}, "
                        f"but it is declared as {violation_type.kind}."
                    ),
                )
            )
    return issues
