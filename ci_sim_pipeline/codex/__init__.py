"""Structured-output Codex execution for pipeline stages."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

BINARY = "codex"


def run(
    prompt: str,
    *,
    output_schema: Mapping[str, Any] | str | Path,
    working_directory: str | Path | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Run Codex and return its structured response."""

    binary = shutil.which(BINARY)
    if binary is None:
        raise RuntimeError("Codex CLI is not installed or is not on PATH")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")

    cwd = Path(working_directory or Path.cwd()).expanduser().resolve()
    if not cwd.is_dir():
        raise NotADirectoryError(cwd)

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        output_path = temporary_root / "result.json"
        schema_path = _materialize_schema(output_schema, temporary_root)
        command = _command(
            output_path=output_path,
            schema_path=schema_path,
            working_directory=cwd,
            model=model,
            reasoning_effort=reasoning_effort,
            binary=binary,
        )
        completed = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"Codex failed with exit code {completed.returncode}: {detail}"
            )
        if not output_path.is_file():
            raise RuntimeError("Codex completed without writing structured output")
        payload = json.loads(output_path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError("Codex structured output must be a JSON object")
    return payload


def _command(
    *,
    output_path: Path,
    schema_path: Path,
    working_directory: Path,
    model: str | None,
    reasoning_effort: str | None,
    binary: str = BINARY,
) -> list[str]:
    command = [
        binary,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ask-for-approval",
        "never",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--cd",
        str(working_directory),
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
    ]
    if model:
        command.extend(("--model", model))
    if reasoning_effort:
        command.extend(
            ("--config", f'model_reasoning_effort="{reasoning_effort}"')
        )
    command.append("-")
    return command


def _materialize_schema(
    output_schema: Mapping[str, Any] | str | Path,
    temporary_root: Path,
) -> Path:
    if isinstance(output_schema, Mapping):
        schema_path = temporary_root / "schema.json"
        schema_path.write_text(
            json.dumps(dict(output_schema), indent=2),
            encoding="utf-8",
        )
        return schema_path

    schema_path = Path(output_schema).expanduser().resolve()
    if not schema_path.is_file():
        raise FileNotFoundError(schema_path)
    return schema_path
