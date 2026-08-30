# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib>=3.10,<4",
#   "pydantic>=2,<3",
# ]
# ///
"""Plot dollar cost against task completion and CI violation rate."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from statistics import mean, stdev

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

from ci_sim.environment.grading import CheckResult, score_field_violations
from ci_sim.environment.scenario import Scenario


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs" / "ci_sim_data_v01_0830"
SCENARIO_ROOT = PROJECT_ROOT / "ci_sim_pipeline" / "outputs" / "ci_sim_data_v01_0830"
QWEN_RUN_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "qwen_token_usage_rerun_0830"
    / "ci_sim_data_v01_0830"
    / "Qwen-Qwen3.8-27B-FP8"
)
OUTPUT_STEM = Path(__file__).with_name("cost_performance_icml")


@dataclass(frozen=True)
class Configuration:
    label: str
    run_dir: Path
    result_pattern: str
    expected_results: int
    input_price: float
    cached_input_price: float
    output_price: float
    color: str
    marker: str


@dataclass(frozen=True)
class Metrics:
    label: str
    mean_cost: float
    total_cost: float
    completion_rate: float
    completion_std: float
    violation_rate: float
    color: str
    marker: str


CONFIGURATIONS = (
    Configuration(
        label="DeepSeek",
        run_dir=(
            RUN_ROOT
            / "deepseek-ai-DeepSeek-V4-Flash-0731"
            / "reasoning-default"
            / "thinking-enabled"
        ),
        result_pattern="repetition-*/*.json",
        expected_results=90,
        input_price=1.50,
        cached_input_price=0.30,
        output_price=3.00,
        color="#009E73",
        marker="D",
    ),
    Configuration(
        label="GLM-high",
        run_dir=RUN_ROOT / "nvidia-GLM-5.2-NVFP4" / "reasoning-high",
        result_pattern="repetition-*/*.json",
        expected_results=90,
        input_price=1.46,
        cached_input_price=0.292,
        output_price=3.67,
        color="#72B7E3",
        marker="o",
    ),
    Configuration(
        label="GLM-max",
        run_dir=RUN_ROOT / "nvidia-GLM-5.2-NVFP4" / "reasoning-max",
        result_pattern="repetition-*/*.json",
        expected_results=90,
        input_price=1.46,
        cached_input_price=0.292,
        output_price=3.67,
        color="#00629B",
        marker="o",
    ),
    Configuration(
        label="Qwen-low",
        run_dir=QWEN_RUN_ROOT / "reasoning-low",
        result_pattern="repetition-*/*.json",
        expected_results=90,
        input_price=1.80,
        cached_input_price=0.36,
        output_price=5.50,
        color="#F3C76F",
        marker="^",
    ),
    Configuration(
        label="Qwen-medium",
        run_dir=QWEN_RUN_ROOT / "reasoning-medium",
        result_pattern="repetition-*/*.json",
        expected_results=90,
        input_price=1.80,
        cached_input_price=0.36,
        output_price=5.50,
        color="#E58B17",
        marker="^",
    ),
    Configuration(
        label="Qwen-xhigh",
        run_dir=QWEN_RUN_ROOT / "reasoning-xhigh",
        result_pattern="repetition-*/*.json",
        expected_results=90,
        input_price=1.80,
        cached_input_price=0.36,
        output_price=5.50,
        color="#984807",
        marker="^",
    ),
)


@cache
def load_scenario(scenario_id: str) -> Scenario:
    return Scenario.load(SCENARIO_ROOT / f"{scenario_id}.json")


def task_field_violation_at_3(payloads: list[dict]) -> float:
    """Return fields violated in any of their three task repetitions."""

    outcomes: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for payload in payloads:
        checks = [
            CheckResult.model_validate(check)
            for check in payload["grade"]["checks"]
        ]
        scores = score_field_violations(
            load_scenario(payload["scenario_id"]).label,
            checks,
        )
        for score in scores:
            key = (payload["scenario_id"], score.tool, score.field)
            outcomes[key].append(bool(score.violated_violation_types))
    if not outcomes:
        raise ValueError("No applicable contextual-integrity task fields found")
    repetition_counts = {len(values) for values in outcomes.values()}
    if repetition_counts != {3}:
        raise ValueError(
            f"violation@3 requires three runs per task field; found {repetition_counts}"
        )
    return sum(any(values) for values in outcomes.values()) / len(outcomes)


def run_cost(payload: dict, configuration: Configuration) -> float:
    usage = payload["result"].get("usage")
    if usage is None:
        raise ValueError(
            f"Missing token usage for {configuration.label}: "
            f"{payload['scenario_id']}"
        )
    cached = usage["cached_input_tokens"]
    uncached = usage["input_tokens"] - cached
    return (
        uncached * configuration.input_price
        + cached * configuration.cached_input_price
        + usage["output_tokens"] * configuration.output_price
    ) / 1_000_000


def load_metrics(configuration: Configuration) -> Metrics:
    result_files = sorted(configuration.run_dir.glob(configuration.result_pattern))
    payloads = [json.loads(path.read_text()) for path in result_files]
    if len(payloads) != configuration.expected_results:
        raise ValueError(
            f"Expected {configuration.expected_results} results for "
            f"{configuration.label}; found {len(payloads)}"
        )

    costs = [run_cost(item, configuration) for item in payloads]
    completion_rates = [
        100 * item["grade"]["task_completeness"] for item in payloads
    ]
    completion_by_repetition: dict[int, list[float]] = defaultdict(list)
    for payload, completion_rate in zip(
        payloads,
        completion_rates,
        strict=True,
    ):
        repetition = payload["repetition"]
        completion_by_repetition[repetition].append(completion_rate)
    if sorted(completion_by_repetition) != [0, 1, 2]:
        raise ValueError(
            f"Expected repetitions 0, 1, and 2 for {configuration.label}; "
            f"found {sorted(completion_by_repetition)}"
        )
    repetition_completion = [
        mean(completion_by_repetition[index]) for index in range(3)
    ]
    return Metrics(
        label=configuration.label,
        mean_cost=mean(costs),
        total_cost=sum(costs),
        completion_rate=mean(completion_rates),
        completion_std=stdev(repetition_completion),
        violation_rate=100 * task_field_violation_at_3(payloads),
        color=configuration.color,
        marker=configuration.marker,
    )


def configure_icml_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw() -> tuple[Path, Path]:
    metrics = [load_metrics(configuration) for configuration in CONFIGURATIONS]
    configure_icml_style()

    fig, axes = plt.subplots(1, 2, figsize=(5.20, 2.35), sharex=True)
    completion_ax, violation_ax = axes

    for item in metrics:
        completion_ax.errorbar(
            item.mean_cost,
            item.completion_rate,
            yerr=item.completion_std,
            fmt="none",
            ecolor=item.color,
            elinewidth=0.75,
            capsize=2,
            capthick=0.75,
            alpha=0.55,
            zorder=2,
        )
        for ax, y_value in (
            (completion_ax, item.completion_rate),
            (violation_ax, item.violation_rate),
        ):
            ax.scatter(
                item.mean_cost,
                y_value,
                s=46,
                marker=item.marker,
                facecolor=item.color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )

    completion_ax.set_title("(a) Task completion")
    completion_ax.set_ylabel("Completion rate (%)")
    completion_ax.set_ylim(78.5, 84.5)
    completion_ax.yaxis.set_major_locator(MultipleLocator(2))
    completion_ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:.0f}")
    )
    violation_ax.set_title("(b) Task-field violation@3")
    violation_ax.set_ylabel("Violation@3 (%)")
    violation_ax.set_ylim(0.3, 4.4)
    violation_ax.yaxis.set_major_locator(MultipleLocator(1))
    violation_ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:.0f}")
    )
    costs = [item.mean_cost for item in metrics]
    cost_padding = 0.00030
    for ax in axes:
        ax.set_xlim(min(costs) - cost_padding, max(costs) + cost_padding)
        ax.xaxis.set_major_locator(MultipleLocator(0.004))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"${value:.3f}")
        )
        ax.grid(color="#D9D9D9", linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Mean cost/run (USD)")

    family_styles = (
        ("DeepSeek", "D", "#009E73"),
        ("GLM", "o", "#00629B"),
        ("Qwen", "^", "#984807"),
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6,
            label=family,
        )
        for family, marker, color in family_styles
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        title="Model  ·  darker shade = greater reasoning effort",
        title_fontsize=6.1,
        fontsize=6.5,
        handletextpad=0.35,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.86, bottom=0.31, wspace=0.22)

    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    png_path = OUTPUT_STEM.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    for item in metrics:
        print(
            f"{item.label}: cost/run=${item.mean_cost:.6f}, "
            f"total=${item.total_cost:.6f}, completion={item.completion_rate:.2f}%, "
            f"completion_std={item.completion_std:.2f}pp, "
            f"field_violation_at_3={item.violation_rate:.2f}%"
        )
    return pdf_path, png_path


if __name__ == "__main__":
    for output_path in draw():
        print(output_path)
