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
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
) -> Path:
    """Build the conventional path for a saved run result."""

    return (
        Path(runs_dir)
        / _safe_path_component(scenario_id)
        / f"{_safe_path_component(model)}.json"
    )


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
    output: str | None = None,
    runs_dir: str = str(DEFAULT_RUNS_DIR),
) -> None:
    """Run, grade, and save one scenario with a River-hosted model.

    Results default to ``runs/<scenario>/<model>.json``. Pass
    ``--output`` to choose an exact file path, or ``--runs-dir`` to change the
    root of the generated path.
    """

    import river_client as river

    scenario = Scenario.load(scenario_path)
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
        )
        runner = Runner(
            build_workplace_environment,
            agent,
            max_tool_rounds=max_tool_rounds,
        )
        result = asyncio.run(runner.run_single(scenario.runtime_spec()))
    finally:
        client.close()

    grade = RuleBasedWriteEvaluator().grade(scenario.label, result.artifact)
    payload = {
        "scenario_id": scenario.id,
        "model": model,
        "result": result.model_dump(mode="json"),
        "grade": grade.model_dump(mode="json"),
    }
    rendered = json.dumps(payload, indent=2)
    output_path = (
        Path(output)
        if output is not None
        else _default_output_path(
            scenario.id,
            model,
            runs_dir=runs_dir,
        )
    )
    _write_json_atomic(payload, output_path)
    print(f"Saved result artifact to {output_path}", file=sys.stderr)
    print(rendered)


def main() -> None:
    fire.Fire({"models": models, "run": run})


if __name__ == "__main__":
    main()
