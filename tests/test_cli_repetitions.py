from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ci_sim.cli import run


class CliRepetitionsTest(unittest.TestCase):
    @patch.dict(os.environ, {"RIVER_API_KEY": "test-key"})
    @patch("river_client.Client")
    @patch("ci_sim.cli.RuleBasedWriteEvaluator")
    @patch("ci_sim.cli.asyncio.run")
    @patch("ci_sim.cli.Runner")
    @patch("ci_sim.cli.RiverAgent")
    @patch("ci_sim.cli.Scenario.load")
    def test_run_uses_distinct_seeds_and_paths_for_repetitions(
        self,
        load_scenario: MagicMock,
        river_agent: MagicMock,
        runner: MagicMock,
        asyncio_run: MagicMock,
        evaluator: MagicMock,
        river_client: MagicMock,
    ) -> None:
        del river_agent, river_client
        scenario = SimpleNamespace(
            id="scenario-one",
            label=object(),
            runtime_spec=lambda: object(),
        )
        load_scenario.return_value = scenario
        results = [
            SimpleNamespace(
                artifact=object(),
                model_dump=lambda **_: {"termination_reason": "completed"},
            )
            for _ in range(2)
        ]
        asyncio_run.side_effect = [(results[0],), (results[1],)]
        evaluator.return_value.grade.return_value = SimpleNamespace(
            model_dump=lambda **_: {"overall_success": True}
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "scenario.json").touch()

            stdout = StringIO()
            with redirect_stdout(stdout):
                run(
                    str(dataset),
                    "provider/model",
                    reasoning_effort="medium",
                    repetitions=2,
                    runs_dir=str(root / "runs"),
                )

            saved_paths = [
                root
                / "runs"
                / "dataset"
                / "provider-model"
                / "reasoning-medium"
                / f"repetition-{repetition}"
                / "scenario-one.json"
                for repetition in range(2)
            ]
            saved_repetitions = [
                json.loads(path.read_text())["repetition"] for path in saved_paths
            ]

        self.assertEqual(
            [call.kwargs["seed"] for call in runner.return_value.run.call_args_list],
            [0, 1],
        )
        self.assertEqual(saved_repetitions, [0, 1])
        self.assertEqual(json.loads(stdout.getvalue()), [str(p) for p in saved_paths])

    def test_repetitions_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            scenario_path = Path(temporary_directory) / "scenario.json"
            scenario_path.touch()

            with self.assertRaisesRegex(ValueError, "must be positive"):
                run(str(scenario_path), "provider/model", repetitions=0)


if __name__ == "__main__":
    unittest.main()
