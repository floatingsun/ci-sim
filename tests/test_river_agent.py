from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from ci_sim.agent.river import RiverAgent, _agent_turn
from ci_sim.contracts import RuntimeSpec


class RiverAgentRequestTest(unittest.IsolatedAsyncioTestCase):
    async def test_sends_thinking_chat_template_kwarg(self) -> None:
        client = MagicMock()
        client.chat_complete.return_value = SimpleNamespace(
            response_json=json.dumps(
                {"choices": [{"message": {"content": "Done."}}]}
            ),
            status_code=200,
        )
        agent = RiverAgent(
            client,
            base_model="deepseek-ai/DeepSeek-V4-Flash-0731",
            thinking=True,
        )
        runtime = RuntimeSpec(
            scenario_id="scenario-one",
            system="System",
            user="User",
            tools=(),
        )

        await agent.respond((), agent.get_init_state(runtime), seed=0)

        self.assertEqual(
            client.chat_complete.call_args.kwargs["chat_template_kwargs"],
            {"thinking": True},
        )


class RiverAgentTurnTest(unittest.TestCase):
    def test_recovers_xml_tool_calls_from_message_content(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "private reasoning\n</think>\n\n"
                            "<tool_call>\n"
                            "<function=docs_create>\n"
                            "<parameter=title>\nTest brief\n</parameter>\n"
                            "<parameter=body>\nApproved body\n</parameter>\n"
                            "</function>\n"
                            "</tool_call>"
                        )
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 20},
                "completion_tokens_details": {"reasoning_tokens": 10},
            },
        }

        turn = _agent_turn(json.dumps(response))

        self.assertIsNone(turn.content)
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertTrue(turn.tool_calls[0].call_id.startswith("xml_call_"))
        self.assertEqual(turn.tool_calls[0].name, "docs_create")
        self.assertEqual(
            turn.tool_calls[0].arguments,
            {"title": "Test brief", "body": "Approved body"},
        )
        self.assertIsNotNone(turn.usage)
        assert turn.usage is not None
        self.assertEqual(turn.usage.input_tokens, 120)
        self.assertEqual(turn.usage.output_tokens, 30)
        self.assertEqual(turn.usage.total_tokens, 150)
        self.assertEqual(turn.usage.cached_input_tokens, 20)
        self.assertEqual(turn.usage.reasoning_tokens, 10)
        self.assertEqual(turn.usage.provider_usage, response["usage"])

    def test_prefers_structured_tool_calls_from_chat_completion(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "Keep this content.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "slack_post",
                                    "arguments": json.dumps(
                                        {"channel": "#ops", "text": "Ready"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }

        turn = _agent_turn(json.dumps(response))

        self.assertEqual(turn.content, "Keep this content.")
        self.assertEqual(turn.tool_calls[0].call_id, "call_123")
        self.assertEqual(turn.tool_calls[0].name, "slack_post")
        self.assertIsNone(turn.usage)


if __name__ == "__main__":
    unittest.main()
