# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib>=3.10,<4",
#   "pydantic>=2,<3",
# ]
# ///
"""Plot Qwen reasoning effort against completion and CI violation rates."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import cache
from pathlib import Path
from statistics import mean, stdev

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

from ci_sim.environment.grading import CheckResult, score_field_violations
from ci_sim.environment.scenario import Scenario

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ROOT = PROJECT_ROOT / "ci_sim_pipeline" / "outputs" / "ci_sim_data_v01_0830"
RUN_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "qwen_token_usage_rerun_0830"
    / "ci_sim_data_v01_0830"
    / "Qwen-Qwen3.8-27B-FP8"
)
OUTPUT_STEM = Path(__file__).with_name("qwen_reasoning_effort_icml")

EFFORTS = ("low", "medium", "xhigh")
EFFORT_LABELS = ("Low", "Medium", "XHigh")
MODEL_LABEL = "Qwen3.8-27B-FP8"

# Colorblind-safe Okabe-Ito colors. Distinct markers and line styles keep the
# figure legible when printed in grayscale.
COMPLETION_COLOR = "#0072B2"
VIOLATION_COLOR = "#D55E00"


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


def load_metrics() -> tuple[list[float], list[float], list[float], list[int]]:
    """Load three repetitions for each Qwen reasoning effort."""

    completion_rates: list[float] = []
    completion_stds: list[float] = []
    violation_rates: list[float] = []
    sample_sizes: list[int] = []

    for effort in EFFORTS:
        run_dir = RUN_ROOT / f"reasoning-{effort}"
        repetition_dirs = sorted(run_dir.glob("repetition-*"))
        if len(repetition_dirs) != 3:
            raise ValueError(
                f"Expected three repetitions for {effort}; "
                f"found {len(repetition_dirs)}"
            )

        payloads: list[dict] = []
        repetition_rates: list[float] = []
        repetition_sizes: list[int] = []
        for repetition_dir in repetition_dirs:
            result_files = sorted(repetition_dir.glob("*.json"))
            repetition_payloads = [
                json.loads(path.read_text()) for path in result_files
            ]
            if not repetition_payloads:
                raise FileNotFoundError(
                    f"No results found in {repetition_dir}"
                )
            payloads.extend(repetition_payloads)
            repetition_rates.append(
                100
                * mean(
                    item["grade"]["task_completeness"]
                    for item in repetition_payloads
                )
            )
            repetition_sizes.append(len(repetition_payloads))

        if len(set(repetition_sizes)) != 1:
            raise ValueError(
                f"Repetitions have unequal sample sizes for {effort}: "
                f"{repetition_sizes}"
            )
        if not payloads:
            raise FileNotFoundError(f"No repetition results found in {run_dir}")

        completion_rates.append(mean(repetition_rates))
        completion_stds.append(stdev(repetition_rates))
        violation_rates.append(100 * task_field_violation_at_3(payloads))
        sample_sizes.append(len(payloads))

    if len(set(sample_sizes)) != 1:
        raise ValueError(f"Efforts have unequal sample sizes: {sample_sizes}")
    return completion_rates, completion_stds, violation_rates, sample_sizes


def configure_icml_style() -> None:
    """Apply a compact, publication-oriented Matplotlib style."""

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
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
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


def add_value_labels(
    ax: mpl.axes.Axes,
    values: list[float],
    offset: float,
    stds: list[float] | None = None,
    horizontal_offsets: list[float] | None = None,
) -> None:
    for index, value in enumerate(values):
        std = stds[index] if stds is not None else None
        label = f"{value:.2f}%" if std is None else f"{value:.2f}% ± {std:.2f}"
        horizontal_offset = (
            horizontal_offsets[index] if horizontal_offsets is not None else 0
        )
        ax.annotate(
            label,
            xy=(index, value + (std or 0)),
            xytext=(horizontal_offset, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )


def draw() -> tuple[Path, Path]:
    completion, completion_stds, violation, _sample_sizes = load_metrics()
    configure_icml_style()

    # Keep the two-panel figure compact enough for flexible paper placement.
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.65), sharex=True)
    fig.suptitle(MODEL_LABEL, y=0.985, fontsize=9, fontweight="bold")
    percent = FuncFormatter(lambda value, _position: f"{value:.0f}")

    completion_ax, violation_ax = axes
    completion_ax.errorbar(
        EFFORT_LABELS,
        completion,
        yerr=completion_stds,
        color=COMPLETION_COLOR,
        marker="o",
        linestyle="-",
        capsize=3,
        capthick=1.0,
        elinewidth=1.0,
    )
    completion_ax.set_title("(a) Task completion")
    completion_ax.set_ylabel("Task completion rate (%)")
    completion_ax.set_ylim(74, 87)
    completion_ax.yaxis.set_major_locator(MultipleLocator(2))
    completion_ax.yaxis.set_major_formatter(percent)
    add_value_labels(
        completion_ax,
        completion,
        offset=6,
        stds=completion_stds,
        horizontal_offsets=[6, 0, -6],
    )

    violation_ax.plot(
        EFFORT_LABELS,
        violation,
        color=VIOLATION_COLOR,
        marker="s",
        linestyle="--",
    )
    violation_ax.set_title("(b) Task-field violation@3")
    violation_ax.set_ylabel("Task-field violation@3 (%)")
    violation_ax.set_ylim(0, 5.0)
    violation_ax.yaxis.set_major_locator(MultipleLocator(1))
    violation_ax.yaxis.set_major_formatter(percent)
    add_value_labels(violation_ax, violation, offset=6)

    for ax in axes:
        ax.set_xlabel("Reasoning effort")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.17)

    fig.subplots_adjust(left=0.095, right=0.985, top=0.86, bottom=0.22, wspace=0.10)

    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    png_path = OUTPUT_STEM.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


if __name__ == "__main__":
    for output_path in draw():
        print(output_path)
