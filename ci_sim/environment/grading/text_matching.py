"""Shared text normalization and matching for validation and grading."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

from ci_sim.contracts import RunArtifact

from ..scenario import TextMatchRule


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def flatten_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from flatten_values(nested)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for nested in value:
            yield from flatten_values(nested)
    elif value is not None:
        yield value


def resolve_field(arguments: Mapping[str, Any], field: str) -> Any:
    if field == "any_write_arg":
        return " ".join(str(value) for value in flatten_values(arguments))

    current: Any = arguments
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def evaluate_text_rule(
    rule: TextMatchRule,
    arguments: Mapping[str, Any],
    *,
    artifact: RunArtifact | None = None,
    write_sequence: int | None = None,
) -> bool:
    """Apply a validated comparison rule to one committed write."""

    value = resolve_field(arguments, rule.field)
    if rule.operator == "equals":
        return normalize_text(value) == normalize_text(rule.contains[0])
    if rule.operator == "not_equals":
        return normalize_text(value) != normalize_text(rule.contains[0])
    if rule.operator in {"exact_set", "not_exact_set"}:
        actual = {normalize_text(item) for item in flatten_values(value)}
        expected = {normalize_text(item) for item in rule.contains}
        matches = actual == expected
        return matches if rule.operator == "exact_set" else not matches
    if rule.operator == "empty":
        return not any(normalize_text(item) for item in flatten_values(value))
    if rule.operator == "nonempty":
        return any(normalize_text(item) for item in flatten_values(value))
    if rule.operator == "references_prior_result":
        return _references_prior_result(
            rule,
            value,
            artifact=artifact,
            write_sequence=write_sequence,
        )

    haystack = normalize_text(value)
    matches = [normalize_text(term) in haystack for term in rule.contains]
    return all(matches) if rule.mode == "all" else any(matches)


def _references_prior_result(
    rule: TextMatchRule,
    value: Any,
    *,
    artifact: RunArtifact | None,
    write_sequence: int | None,
) -> bool:
    if artifact is None or write_sequence is None:
        return False
    prior_writes = [
        write
        for write in artifact.writes
        if write.sequence < write_sequence
        and write.canonical_tool == rule.reference_tool
        and write.result_id is not None
    ]
    if not prior_writes:
        return False
    latest_write = max(prior_writes, key=lambda write: write.sequence)
    return normalize_text(value) == normalize_text(latest_write.result_id)
