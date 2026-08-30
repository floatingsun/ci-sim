"""Aggregate contextual-integrity violation rates by tool argument field."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any

from ci_sim.environment.grading import CheckResult, score_field_violations
from ci_sim.environment.scenario import Scenario


def analyze(
    result_paths: list[Path],
    scenario_dir: Path,
) -> dict[str, Any]:
    """Return field-level rates across saved result artifacts."""

    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    models: set[str] = set()
    reasoning_efforts: set[str] = set()

    @cache
    def load_scenario(scenario_id: str) -> Scenario:
        return Scenario.load(scenario_dir / f"{scenario_id}.json")

    for result_path in result_paths:
        payload = json.loads(result_path.read_text())
        models.add(payload["model"])
        if payload.get("reasoning_effort") is not None:
            reasoning_efforts.add(payload["reasoning_effort"])
        scenario = load_scenario(payload["scenario_id"])
        checks = [
            CheckResult.model_validate(check)
            for check in payload["grade"]["checks"]
        ]
        for score in score_field_violations(scenario.label, checks):
            key = (score.tool, score.field)
            field = aggregate.setdefault(
                key,
                {
                    "tool": score.tool,
                    "field": score.field,
                    "applicable_result_count": 0,
                    "affected_result_count": 0,
                    "applicable_type_opportunities": 0,
                    "violated_type_opportunities": 0,
                    "by_violation_type": defaultdict(
                        lambda: {"opportunities": 0, "violations": 0}
                    ),
                },
            )
            field["applicable_result_count"] += 1
            field["affected_result_count"] += bool(
                score.violated_violation_types
            )
            field["applicable_type_opportunities"] += len(
                score.applicable_violation_types
            )
            field["violated_type_opportunities"] += len(
                score.violated_violation_types
            )
            violated = frozenset(score.violated_violation_types)
            for violation_type in score.applicable_violation_types:
                type_score = field["by_violation_type"][violation_type]
                type_score["opportunities"] += 1
                type_score["violations"] += violation_type in violated

    field_scores = []
    for field in aggregate.values():
        applicable_results = field["applicable_result_count"]
        opportunities = field["applicable_type_opportunities"]
        by_violation_type = [
            {
                "type": violation_type,
                **counts,
                "violation_rate": counts["violations"]
                / counts["opportunities"],
            }
            for violation_type, counts in field.pop("by_violation_type").items()
        ]
        by_violation_type.sort(
            key=lambda item: (-item["violation_rate"], item["type"])
        )
        field_scores.append(
            {
                **field,
                "affected_result_rate": field["affected_result_count"]
                / applicable_results,
                "violation_rate": field["violated_type_opportunities"]
                / opportunities,
                "by_violation_type": by_violation_type,
            }
        )

    field_scores.sort(
        key=lambda item: (-item["violation_rate"], item["tool"], item["field"])
    )
    return {
        "result_count": len(result_paths),
        "models": sorted(models),
        "reasoning_efforts": sorted(reasoning_efforts),
        "field_scores": field_scores,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument(
        "--result-glob",
        default="**/*.json",
        help="Glob relative to run_dir (default: **/*.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_paths = sorted(args.run_dir.glob(args.result_glob))
    if not result_paths:
        raise SystemExit(
            f"No result files matched {args.result_glob!r} in {args.run_dir}"
        )
    print(json.dumps(analyze(result_paths, args.scenario_dir), indent=2))


if __name__ == "__main__":
    main()
