"""River-backed agent using the OpenAI-compatible chat API."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Protocol
from uuid import uuid4

from ci_sim.contracts import RuntimeSpec, ToolCall, ToolResult

from .contracts import AgentState, AgentTurn, TokenUsage, TranscriptEvent


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
        reasoning_effort: str | None = None,
        thinking: bool | None = None,
    ) -> None:
        self._client = client
        self._base_model = base_model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._reasoning_effort = reasoning_effort
        self._thinking = thinking

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
        request = {
            "tools": _tools(state.runtime),
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "seed": seed,
        }
        chat_template_kwargs: dict[str, Any] = {}
        if self._reasoning_effort is not None:
            chat_template_kwargs["reasoning_effort"] = self._reasoning_effort
        if self._thinking is not None:
            chat_template_kwargs["thinking"] = self._thinking
        if chat_template_kwargs:
            request["chat_template_kwargs"] = chat_template_kwargs
        result = await asyncio.to_thread(
            self._client.chat_complete,
            _messages(state.runtime, state.messages),
            base_model=self._base_model,
            timeout=self._timeout,
            **request,
        )
        if result.status_code >= 400:
            raise RuntimeError(
                f"River chat completion failed ({result.status_code}): "
                f"{result.response_json}"
            )
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
    content = message.get("content")
    tool_call_payloads = tuple(message.get("tool_calls") or ())
    if not tool_call_payloads and isinstance(content, str):
        content, tool_call_payloads = _parse_xml_tool_calls(content)

    tool_calls = tuple(
        ToolCall(
            call_id=call.get("id") or f"xml_call_{uuid4().hex}",
            name=call["function"]["name"],
            arguments=json.loads(call["function"]["arguments"]),
        )
        for call in tool_call_payloads
    )
    return AgentTurn(
        content=content,
        tool_calls=tool_calls,
        usage=_parse_token_usage(response.get("usage")),
    )


def _parse_token_usage(raw_usage: Any) -> TokenUsage | None:
    if not isinstance(raw_usage, dict):
        return None

    input_tokens = _token_count(raw_usage, "prompt_tokens")
    output_tokens = _token_count(raw_usage, "completion_tokens")
    total_tokens = _token_count(raw_usage, "total_tokens")
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    input_details = raw_usage.get("prompt_tokens_details")
    output_details = raw_usage.get("completion_tokens_details")
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=_token_count(input_details, "cached_tokens"),
        reasoning_tokens=_token_count(output_details, "reasoning_tokens"),
        provider_usage=raw_usage,
    )


def _token_count(values: Any, key: str) -> int:
    if not isinstance(values, dict):
        return 0
    value = values.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _parse_xml_tool_calls(
    content: str,
) -> tuple[str | None, tuple[dict[str, Any], ...]]:
    """Recover XML tool calls when the server returns them as text."""

    from river_client.renderers.qwen3 import parse_qwen_content_blocks

    parsed = parse_qwen_content_blocks(content)
    if parsed is None:
        return content, ()

    parts, parsed_calls = parsed
    tool_calls = tuple(call for call in parsed_calls if "function" in call)
    if not tool_calls:
        return content, ()

    visible_content = "".join(
        part["text"] for part in parts if part["type"] == "text"
    )
    if "</think>" in visible_content:
        visible_content = visible_content.rsplit("</think>", 1)[1]
    return visible_content.strip() or None, tool_calls
