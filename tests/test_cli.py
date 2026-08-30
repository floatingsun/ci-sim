from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ci_sim.cli import (
    _default_output_path,
    _safe_path_component,
    _write_json_atomic,
    run,
)


class ResultArtifactPathTest(unittest.TestCase):
    def test_default_path_includes_dataset_model_and_reasoning_effort(self) -> None:
        path = _default_output_path(
            "gina/status update",
            "Qwen/Qwen3.8 27B",
            scenario_group="generated-dataset",
            reasoning_effort="xhigh",
            runs_dir="artifacts",
        )

        self.assertEqual(
            path,
            Path(
                "artifacts/generated-dataset/Qwen-Qwen3.8-27B/"
                "reasoning-xhigh/gina-status-update.json"
            ),
        )

    def test_path_components_cannot_escape_the_runs_directory(self) -> None:
        self.assertEqual(_safe_path_component("../../"), "unnamed")
        self.assertEqual(_safe_path_component("../scenario/name"), "scenario-name")

    def test_atomic_writer_creates_parent_directories_and_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "nested" / "result.json"
            payload = {"scenario_id": "scenario-1"}

            _write_json_atomic(payload, output_path)

            self.assertEqual(json.loads(output_path.read_text()), payload)
            self.assertEqual(list(output_path.parent.glob("*.tmp")), [])

    @patch.dict(os.environ, {"RIVER_API_KEY": "test-key"})
    @patch("river_client.Client")
    @patch("ci_sim.cli.RuleBasedWriteEvaluator")
    @patch("ci_sim.cli.asyncio.run")
    @patch("ci_sim.cli.Runner")
    @patch("ci_sim.cli.RiverAgent")
    @patch("ci_sim.cli.Scenario.load")
    def test_run_saves_to_the_generated_path_by_default(
        self,
        load_scenario: MagicMock,
        river_agent: MagicMock,
        runner: MagicMock,
        asyncio_run: MagicMock,
        evaluator: MagicMock,
        river_client: MagicMock,
    ) -> None:
        del river_agent, runner, river_client
        scenario = SimpleNamespace(
            id="scenario/one",
            label=object(),
            runtime_spec=lambda: object(),
        )
        load_scenario.return_value = scenario
        asyncio_run.return_value = (
            SimpleNamespace(
                artifact=object(),
                model_dump=lambda **_: {"termination_reason": "completed"},
            ),
        )
        evaluator.return_value.grade.return_value = SimpleNamespace(
            model_dump=lambda **_: {"overall_success": True}
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                run(
                    "scenario.json",
                    "provider/model name",
                    runs_dir=temporary_directory,
                )

            output_path = (
                Path(temporary_directory)
                / "scenario"
                / "provider-model-name"
                / "reasoning-default"
                / "scenario-one.json"
            )
            saved = json.loads(output_path.read_text())

        self.assertEqual(saved["scenario_id"], "scenario/one")
        self.assertEqual(saved["model"], "provider/model name")
        self.assertNotIn("seed", saved)
        self.assertIn(str(output_path), stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue()), saved)

    @patch.dict(os.environ, {"RIVER_API_KEY": "test-key"})
    @patch("river_client.Client")
    @patch("ci_sim.cli.RuleBasedWriteEvaluator")
    @patch("ci_sim.cli.asyncio.run")
    @patch("ci_sim.cli.Runner")
    @patch("ci_sim.cli.RiverAgent")
    @patch("ci_sim.cli.Scenario.load")
    def test_run_accepts_a_directory_and_runs_scenarios_concurrently(
        self,
        load_scenario: MagicMock,
        river_agent: MagicMock,
        runner: MagicMock,
        asyncio_run: MagicMock,
        evaluator: MagicMock,
        river_client: MagicMock,
    ) -> None:
        del river_client
        scenarios = [
            SimpleNamespace(id="scenario-one", label=object(), runtime_spec=object),
            SimpleNamespace(id="scenario-two", label=object(), runtime_spec=object),
        ]
        results = tuple(
            SimpleNamespace(
                artifact=object(),
                model_dump=lambda **_: {"termination_reason": "completed"},
            )
            for _ in scenarios
        )
        load_scenario.side_effect = scenarios
        asyncio_run.return_value = results
        evaluator.return_value.grade.return_value = SimpleNamespace(
            model_dump=lambda **_: {"overall_success": True}
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "one.json").touch()
            (dataset / "two.json").touch()

            stdout = StringIO()
            with redirect_stdout(stdout):
                run(
                    str(dataset),
                    "provider/model",
                    concurrency=2,
                    reasoning_effort="xhigh",
                    runs_dir=str(root / "runs"),
                )

            saved_paths = [
                root
                / "runs"
                / "dataset"
                / "provider-model"
                / "reasoning-xhigh"
                / f"{scenario.id}.json"
                for scenario in scenarios
            ]
            all_results_saved = all(path.is_file() for path in saved_paths)

        self.assertEqual(
            runner.return_value.run.call_args.kwargs["concurrency"],
            2,
        )
        self.assertEqual(
            river_agent.call_args.kwargs["reasoning_effort"],
            "xhigh",
        )
        self.assertTrue(all_results_saved)
        self.assertEqual(
            json.loads(stdout.getvalue()), [str(path) for path in saved_paths]
        )


if __name__ == "__main__":
    unittest.main()
