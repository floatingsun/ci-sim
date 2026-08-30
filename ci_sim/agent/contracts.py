"""Interface between the simulation runner and an evaluated agent."""

from __future__ import annotations

from typing import Protocol

from ci_sim.contracts import (
    RuntimeSpec,
    StrictModel,
    ToolCall,
    ToolResult,
)


class AgentTurn(StrictModel):
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


type TranscriptEvent = AgentTurn | ToolResult


class AgentState(StrictModel):
    """Model-visible context and history for one simulation run."""

    runtime: RuntimeSpec
    messages: list[TranscriptEvent]


class Agent(Protocol):
    """Stateful model boundary driven by the simulation runner."""

    def get_init_state(
        self,
        runtime: RuntimeSpec,
        message_history: tuple[TranscriptEvent, ...] = (),
    ) -> AgentState:
        """Create isolated state for one simulation run."""
        ...

    async def respond(
        self,
        tool_results: tuple[ToolResult, ...],
        state: AgentState,
        *,
        seed: int,
    ) -> tuple[AgentTurn, AgentState]:
        """Consume new results and return the next turn and updated state."""
        ...
