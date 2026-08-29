"""Shared text normalization and matching for validation and grading."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

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


def evaluate_text_rule(rule: TextMatchRule, arguments: Mapping[str, Any]) -> bool:
    haystack = normalize_text(resolve_field(arguments, rule.field))
    matches = [normalize_text(term) in haystack for term in rule.contains]
    return all(matches) if rule.mode == "all" else any(matches)
