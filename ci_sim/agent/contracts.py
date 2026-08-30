"""Interface between the simulation runner and an evaluated agent."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import Field

from ci_sim.contracts import (
    RuntimeSpec,
    StrictModel,
    ToolCall,
    ToolResult,
)


class TokenUsage(StrictModel):
    """Normalized token counts with the provider's original usage payload."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    provider_usage: dict[str, Any] | None = None


class AgentTurn(StrictModel):
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage | None = None


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
