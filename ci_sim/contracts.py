"""Core data exchanged between runners, agents, environments, and graders."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolDefinition(StrictModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(StrictModel):
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(StrictModel):
    call_id: str
    content: dict[str, str] | None = None
    error: str | None = None


class WriteRecord(StrictModel):
    sequence: int
    call_id: str
    canonical_tool: str
    model_tool: str
    arguments: dict[str, Any]


class RunArtifact(StrictModel):
    writes: tuple[WriteRecord, ...] = ()


class RuntimeSpec(StrictModel):
    """Scenario data allowed to reach model inference."""

    scenario_id: str
    system: str
    user: str
    tools: tuple[ToolDefinition, ...]
