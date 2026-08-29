"""Shared contracts and persistence helpers for dataset generation."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from ci_sim.contracts import StrictModel
from ci_sim.environment.scenario import ScenarioMetadata

CodexRunner = Callable[..., dict[str, Any]]


class CodexSettings(StrictModel):
    """Codex options shared by all generation stages."""

    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "medium"
    timeout_seconds: int = Field(default=900, ge=1)


class ContextRecordIdea(StrictModel):
    """One record planned for the model-visible workplace context."""

    record_type: Literal["email", "document", "chat", "calendar"]
    function: Literal[
        "task_request",
        "authorized_source",
        "sensitive_source",
        "adversarial_pressure",
    ]
    summary: str


class PressureIdea(StrictModel):
    """The unsafe instruction and the authorization rule it conflicts with."""

    message_summary: str
    authorization_conflict: str


class ExpectedEmailIdea(StrictModel):
    """The safe email behavior that stage 2 must make deterministically gradable."""

    recipient: str
    subject_intent: str
    must_include: tuple[str, ...] = Field(min_length=1)
    must_exclude: tuple[str, ...] = Field(min_length=1)


class ScenarioIdea(StrictModel):
    """Stage-one design brief for one workplace email-writing scenario."""

    id_hint: str
    core_idea: str
    user_task: str
    context_records: tuple[ContextRecordIdea, ...] = Field(min_length=1)
    authorized_facts: tuple[str, ...] = Field(min_length=1)
    restricted_facts: tuple[str, ...] = Field(min_length=1)
    pressure: PressureIdea
    expected_email: ExpectedEmailIdea
    target_ci_failures: tuple[str, ...] = Field(min_length=1)
    metadata: ScenarioMetadata
    novelty_from_references: str


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
