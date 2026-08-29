"""Interface implemented by simulation environments."""

from __future__ import annotations

from typing import Protocol

from ci_sim.contracts import RunArtifact, ToolCall, ToolResult


class Environment(Protocol):
    def execute(self, call: ToolCall) -> ToolResult: ...

    def artifact(self) -> RunArtifact: ...
