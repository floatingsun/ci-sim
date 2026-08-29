"""River-backed agent using the OpenAI-compatible chat API."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Protocol

from ci_sim.contracts import RuntimeSpec, ToolCall, ToolResult

from .contracts import AgentState, AgentTurn, TranscriptEvent


class _ChatResult(Protocol):
    response_json: str
    status_code: int


class _RiverClient(Protocol):
    def chat_complete(
        self,
        messages: list[dict[str, Any]],
        *,
        base_model: str,
        **kwargs: Any,
    ) -> _ChatResult: ...


class RiverAgent:
    """Translate benchmark turns to and from River chat completions."""

    def __init__(
        self,
        client: _RiverClient,
        *,
        base_model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        timeout: float = 300.0,
    ) -> None:
        self._client = client
        self._base_model = base_model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout

    @classmethod
    def from_env(cls, *, base_model: str, **kwargs: Any) -> RiverAgent:
        """Build an agent using ``RIVER_API_KEY`` from the environment."""

        import river_client as river

        client = river.Client(api_key=os.environ["RIVER_API_KEY"])
        return cls(client, base_model=base_model, **kwargs)

    def get_init_state(
        self,
        runtime: RuntimeSpec,
        message_history: tuple[TranscriptEvent, ...] = (),
    ) -> AgentState:
        return AgentState(runtime=runtime, messages=list(message_history))

    async def respond(
        self,
        tool_results: tuple[ToolResult, ...],
        state: AgentState,
        *,
        seed: int,
    ) -> tuple[AgentTurn, AgentState]:
        state.messages.extend(tool_results)
        result = await asyncio.to_thread(
            self._client.chat_complete,
            _messages(state.runtime, state.messages),
            base_model=self._base_model,
            tools=_tools(state.runtime),
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            seed=seed,
            timeout=self._timeout,
        )
        if result.status_code >= 400:
            raise RuntimeError(f"River chat completion failed ({result.status_code})")
        turn = _agent_turn(result.response_json)
        state.messages.append(turn)
        return turn, state


def _messages(
    runtime: RuntimeSpec,
    history: list[TranscriptEvent],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": runtime.system},
        {"role": "user", "content": runtime.user},
    ]
    for event in history:
        if isinstance(event, AgentTurn):
            message: dict[str, Any] = {
                "role": "assistant",
                "content": event.content,
            }
            if event.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in event.tool_calls
                ]
            messages.append(message)
            continue

        messages.append(
            {
                "role": "tool",
                "tool_call_id": event.call_id,
                "content": json.dumps(_tool_result_content(event)),
            }
        )
    return messages


def _tools(runtime: RuntimeSpec) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in runtime.tools
    ]


def _tool_result_content(result: ToolResult) -> dict[str, Any]:
    if result.error is not None:
        return {"error": result.error}
    return {"result": result.content}


def _agent_turn(response_json: str) -> AgentTurn:
    response = json.loads(response_json)
    message = response["choices"][0]["message"]
    tool_calls = tuple(
        ToolCall(
            call_id=call["id"],
            name=call["function"]["name"],
            arguments=json.loads(call["function"]["arguments"]),
        )
        for call in message.get("tool_calls") or ()
    )
    return AgentTurn(content=message.get("content"), tool_calls=tool_calls)
