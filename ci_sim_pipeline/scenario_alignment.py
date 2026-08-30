"""Checks that constructed scenarios preserve their approved design brief."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ci_sim.environment.grading.text_matching import normalize_text
from ci_sim.environment.scenario import Scenario
from ci_sim.environment.workplace.tools import TOOLS_BY_CANONICAL_NAME

if TYPE_CHECKING:
    from .contracts import ScenarioIdea


def ensure_scenario_matches_idea(
    scenario: Scenario,
    idea: ScenarioIdea,
) -> None:
    """Raise when construction changes a design brief's fixed boundaries."""

    runtime_tools = tuple(tool.name for tool in scenario.model_input.tools)
    expected_tools = tuple(item.tool for item in scenario.label.expected_writes)
    idea_tools = tuple(write.tool for write in idea.expected_writes)
    model_tools = tuple(TOOLS_BY_CANONICAL_NAME[tool].model_name for tool in idea_tools)
    if scenario.id != idea.id_hint:
        raise ValueError(f"Scenario id {scenario.id} must equal {idea.id_hint}.")
    if runtime_tools != model_tools:
        raise ValueError(
            f"{scenario.id} exposes {runtime_tools}; expected {model_tools}."
        )
    if expected_tools != idea_tools:
        raise ValueError(
            f"{scenario.id} expects {expected_tools}; idea requires {idea_tools}."
        )
    if tuple(scenario.label.target_ci_failures) != tuple(idea.target_ci_failures):
        raise ValueError("Scenario target_ci_failures must exactly match its idea.")
    if scenario.metadata.model_dump() != idea.metadata.model_dump():
        raise ValueError("Scenario metadata must exactly match its idea.")

    expected_by_tool = {item.tool: item for item in scenario.label.expected_writes}
    destination_fields = {
        "gmail.send": "to",
        "slack.post": "channel",
        "calendar.create_event": "attendees",
        "drive.share": "recipients",
    }
    for write_idea in idea.expected_writes:
        expected = expected_by_tool[write_idea.tool]
        destination_field = destination_fields.get(write_idea.tool)
        if destination_field is not None:
            destination_is_collection = destination_field in {
                "attendees",
                "recipients",
            }
            pass_operator = "exact_set" if destination_is_collection else "equals"
            fail_operator = (
                "not_exact_set" if destination_is_collection else "not_equals"
            )
            authorized_destination = _normalized_values(
                write_idea.authorized_destination
            )
            gold_destination = _normalized_values(
                expected.example_gold_args.get(destination_field)
            )
            if gold_destination != authorized_destination:
                raise ValueError(
                    f"{write_idea.tool} gold destination {gold_destination} does "
                    f"not match the idea destination {authorized_destination}."
                )
            if not any(
                rule.field == destination_field
                and rule.operator == pass_operator
                and _normalized_values(rule.contains) == authorized_destination
                for rule in expected.pass_if
            ):
                raise ValueError(
                    f"{write_idea.tool} needs a {pass_operator} pass rule for "
                    "the idea's authorized destination."
                )
            if not any(
                rule.field == destination_field
                and rule.operator == fail_operator
                and _normalized_values(rule.contains) == authorized_destination
                for rule in expected.fail_if
            ):
                raise ValueError(
                    f"{write_idea.tool} needs a {fail_operator} safety rule for "
                    "the idea's authorized destination."
                )
        if write_idea.depends_on_tool is not None and not any(
            rule.operator == "references_prior_result"
            and rule.reference_tool == write_idea.depends_on_tool
            for rule in expected.pass_if
        ):
            raise ValueError(
                f"{write_idea.tool} must grade its dependency on "
                f"{write_idea.depends_on_tool}."
            )


def _normalized_values(value: object) -> frozenset[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        values = value
    elif value is None:
        values = ()
    else:
        values = (value,)
    return frozenset(normalize_text(item) for item in values)
