"""Append-only environment for simulated workplace writes."""

from __future__ import annotations

from pydantic import ValidationError

from ..contracts import RunArtifact, RuntimeSpec, ToolCall, ToolResult, WriteRecord
from .tools import TOOLS_BY_MODEL_NAME


class WorkplaceEnvironment:
    def __init__(self, enabled_tools: tuple[str, ...]) -> None:
        self._enabled_tools = frozenset(enabled_tools)
        self._writes: list[WriteRecord] = []

    def execute(self, call: ToolCall) -> ToolResult:
        tool = TOOLS_BY_MODEL_NAME.get(call.name)
        if tool is None or call.name not in self._enabled_tools:
            return ToolResult(
                call_id=call.call_id,
                error=f"Unknown or disabled tool: {call.name}",
            )

        try:
            arguments = tool.validate(call.arguments)
        except ValidationError as exc:
            return ToolResult(call_id=call.call_id, error=str(exc))

        sequence = len(self._writes) + 1
        self._writes.append(
            WriteRecord(
                sequence=sequence,
                call_id=call.call_id,
                canonical_tool=tool.canonical_name,
                model_tool=tool.model_name,
                arguments=arguments,
            )
        )
        return ToolResult(
            call_id=call.call_id,
            content={
                "status": tool.success_status,
                "id": f"{tool.id_prefix}_{sequence:04d}",
            },
        )

    def artifact(self) -> RunArtifact:
        return RunArtifact(writes=tuple(self._writes))


def build_workplace_environment(spec: RuntimeSpec) -> WorkplaceEnvironment:
    return WorkplaceEnvironment(tuple(tool.name for tool in spec.tools))
