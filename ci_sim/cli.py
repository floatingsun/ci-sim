"""Command-line interface for validating and running CI simulation scenarios."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import fire

from ci_sim.agent import RiverAgent
from ci_sim.environment.grading import RuleBasedWriteEvaluator
from ci_sim.environment.scenario import Scenario
from ci_sim.environment.workplace import build_workplace_environment
from ci_sim.runner import Runner

DEFAULT_RUNS_DIR = Path("runs")


def _safe_path_component(value: str) -> str:
    """Return a readable, portable directory name for a run identifier."""

    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")
    return component or "unnamed"


def _default_output_path(
    scenario_id: str,
    model: str,
    *,
    scenario_group: str,
    reasoning_effort: str | None,
    thinking: bool | None = None,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
) -> Path:
    """Build the conventional path for a saved run result."""

    path = (
        Path(runs_dir)
        / _safe_path_component(scenario_group)
        / _safe_path_component(model)
        / f"reasoning-{_safe_path_component(reasoning_effort or 'default')}"
    )
    if thinking is not None:
        path /= f"thinking-{'enabled' if thinking else 'disabled'}"
    return path / f"{_safe_path_component(scenario_id)}.json"


def _write_json_atomic(payload: dict[str, Any], output_path: Path) -> None:
    """Write a JSON artifact without leaving a partial result on failure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(payload, temporary_file, indent=2)
            temporary_file.write("\n")
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def models() -> None:
    """List the River models available to the configured API key."""

    import river_client as river

    api_key = os.environ.get("RIVER_API_KEY")
    if not api_key:
        raise RuntimeError("RIVER_API_KEY is not set")

    client = river.Client(api_key=api_key)
    try:
        print(json.dumps({"models": list(client.get_capabilities())}, indent=2))
    finally:
        client.close()


def run(
    scenario_path: str,
    model: str,
    max_tool_rounds: int = 6,
    max_tokens: int = 16_384,
    temperature: float = 0.0,
    timeout: float = 300.0,
    reasoning_effort: str | None = None,
    thinking: bool | None = None,
    concurrency: int = 4,
    repetitions: int = 1,
    output: str | None = None,
    runs_dir: str = str(DEFAULT_RUNS_DIR),
) -> None:
    """Run, grade, and save a scenario file or directory of scenarios.

    Results default to
    ``runs/<dataset>/<model>/reasoning-<effort>/<scenario>.json``. Pass
    ``--output`` to choose an exact file path, or ``--runs-dir`` to change the
    root of the generated path. Directories run concurrently. With multiple
    repetitions, results are grouped under ``repetition-<index>`` directories.
    """

    import river_client as river

    input_path = Path(scenario_path)
    is_directory = input_path.is_dir()
    scenario_group = input_path.name if is_directory else input_path.stem
    scenario_paths = sorted(input_path.glob("*.json")) if is_directory else [input_path]
    if not scenario_paths:
        raise ValueError(f"No JSON scenarios found in {input_path}")
    if repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if is_directory and output is not None:
        raise ValueError("--output can only be used with one scenario file")
    if repetitions > 1 and output is not None:
        raise ValueError("--output can only be used with one repetition")

    scenarios = [Scenario.load(path) for path in scenario_paths]
    api_key = os.environ.get("RIVER_API_KEY")
    if not api_key:
        raise RuntimeError("RIVER_API_KEY is not set")

    client = river.Client(api_key=api_key)
    try:
        agent = RiverAgent(
            client,
            base_model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            thinking=thinking,
        )
        runner = Runner(
            build_workplace_environment,
            agent,
            max_tool_rounds=max_tool_rounds,
        )
        repetition_results = []
        for repetition in range(repetitions):
            results = asyncio.run(
                runner.run(
                    (scenario.runtime_spec() for scenario in scenarios),
                    seed=repetition,
                    concurrency=concurrency,
                )
            )
            repetition_results.append((repetition, results))
    finally:
        client.close()

    evaluator = RuleBasedWriteEvaluator()
    output_paths: list[str] = []
    for repetition, results in repetition_results:
        for scenario, result in zip(scenarios, results, strict=True):
            grade = evaluator.grade(scenario.label, result.artifact)
            payload = {
                "scenario_id": scenario.id,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "thinking": thinking,
                "result": result.model_dump(mode="json"),
                "grade": grade.model_dump(mode="json"),
            }
            if repetitions > 1:
                payload["repetition"] = repetition
            output_path = (
                Path(output)
                if output is not None
                else _default_output_path(
                    scenario.id,
                    model,
                    scenario_group=scenario_group,
                    reasoning_effort=reasoning_effort,
                    thinking=thinking,
                    runs_dir=runs_dir,
                )
            )
            if output is None and repetitions > 1:
                output_path = (
                    output_path.parent
                    / f"repetition-{repetition}"
                    / output_path.name
                )
            _write_json_atomic(payload, output_path)
            print(f"Saved result artifact to {output_path}", file=sys.stderr)
            output_paths.append(str(output_path))

    rendered = output_paths if is_directory or repetitions > 1 else payload
    print(json.dumps(rendered, indent=2))


def main() -> None:
    fire.Fire({"models": models, "run": run})


if __name__ == "__main__":
    main()
