"""Model adapter and simulation-loop boundaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from sim_env.agent import Agent, TranscriptEvent

from .contracts import RunArtifact, RuntimeSpec, SimEnvironment, StrictModel


class RunResult(StrictModel):
    scenario_id: str
    events: tuple[TranscriptEvent, ...]
    artifact: RunArtifact
    termination_reason: Literal["completed", "max_tool_rounds"]


class SimulationRunner:
    def __init__(
        self,
        environment_builder: Callable[[RuntimeSpec], SimEnvironment],
        agent: Agent,
        *,
        max_tool_rounds: int = 6,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be positive")
        self._environment_builder = environment_builder
        self._agent = agent
        self._max_tool_rounds = max_tool_rounds

    async def run(self, spec: RuntimeSpec, *, seed: int = 0) -> RunResult:
        environment = self._environment_builder(spec)
        events: list[TranscriptEvent] = []

        for _ in range(self._max_tool_rounds):
            turn = await self._agent.respond(spec, tuple(events), seed=seed)
            events.append(turn)
            if not turn.tool_calls:
                return RunResult(
                    scenario_id=spec.scenario_id,
                    events=tuple(events),
                    artifact=environment.artifact(),
                    termination_reason="completed",
                )
            for call in turn.tool_calls:
                events.append(environment.execute(call))

        return RunResult(
            scenario_id=spec.scenario_id,
            events=tuple(events),
            artifact=environment.artifact(),
            termination_reason="max_tool_rounds",
        )
