from __future__ import annotations

import unittest

from ci_sim.agent import AgentTurn, TokenUsage
from ci_sim.contracts import ToolResult
from ci_sim.runner import _aggregate_token_usage


class RunnerTokenUsageTest(unittest.TestCase):
    def test_aggregates_usage_across_model_turns(self) -> None:
        usage = _aggregate_token_usage(
            [
                AgentTurn(
                    usage=TokenUsage(
                        input_tokens=100,
                        output_tokens=20,
                        total_tokens=120,
                        cached_input_tokens=10,
                        reasoning_tokens=5,
                    )
                ),
                ToolResult(call_id="call_1", content={"status": "ok"}),
                AgentTurn(
                    usage=TokenUsage(
                        input_tokens=150,
                        output_tokens=30,
                        total_tokens=180,
                        cached_input_tokens=40,
                        reasoning_tokens=8,
                    )
                ),
            ]
        )

        self.assertEqual(usage.input_tokens, 250)
        self.assertEqual(usage.output_tokens, 50)
        self.assertEqual(usage.total_tokens, 300)
        self.assertEqual(usage.cached_input_tokens, 50)
        self.assertEqual(usage.reasoning_tokens, 13)


if __name__ == "__main__":
    unittest.main()
