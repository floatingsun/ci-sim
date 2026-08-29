"""Simulation loop coordinating an agent with an environment."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from typing import Literal

from ci_sim.agent import Agent, TranscriptEvent
from ci_sim.contracts import RunArtifact, RuntimeSpec, StrictModel, ToolResult
from ci_sim.environment import Environment


class RunResult(StrictModel):
    scenario_id: str
    events: tuple[TranscriptEvent, ...]
    artifact: RunArtifact
    termination_reason: Literal["completed", "max_tool_rounds"]


class Runner:
    def __init__(
        self,
        environment_builder: Callable[[RuntimeSpec], Environment],
        agent: Agent,
        *,
        max_tool_rounds: int = 6,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be positive")
        self._environment_builder = environment_builder
        self._agent = agent
        self._max_tool_rounds = max_tool_rounds

    async def run_single(self, spec: RuntimeSpec, *, seed: int = 0) -> RunResult:
        environment = self._environment_builder(spec)
        agent_state = self._agent.get_init_state(spec)
        events: list[TranscriptEvent] = []
        tool_results: tuple[ToolResult, ...] = ()

        for _ in range(self._max_tool_rounds):
            turn, agent_state = await self._agent.respond(
                tool_results,
                agent_state,
                seed=seed,
            )
            events.append(turn)
            if not turn.tool_calls:
                return RunResult(
                    scenario_id=spec.scenario_id,
                    events=tuple(events),
                    artifact=environment.artifact(),
                    termination_reason="completed",
                )
            tool_results = tuple(environment.execute(call) for call in turn.tool_calls)
            events.extend(tool_results)

        return RunResult(
            scenario_id=spec.scenario_id,
            events=tuple(events),
            artifact=environment.artifact(),
            termination_reason="max_tool_rounds",
        )

    async def run(
        self,
        specs: Iterable[RuntimeSpec],
        *,
        seed: int = 0,
        concurrency: int = 1,
    ) -> tuple[RunResult, ...]:
        """Run scenarios concurrently and return results in input order."""
        if concurrency < 1:
            raise ValueError("concurrency must be positive")

        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(spec: RuntimeSpec) -> RunResult:
            async with semaphore:
                return await self.run_single(spec, seed=seed)

        return tuple(await asyncio.gather(*(run_one(spec) for spec in specs)))
