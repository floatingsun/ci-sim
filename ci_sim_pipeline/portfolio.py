"""Coverage-aware selection for the final scenario portfolio."""

from __future__ import annotations

from functools import cache

from .contracts import (
    CANONICAL_TOOLS,
    CI_FAILURE_TYPES,
    CoverageRequirements,
    ScenarioIdea,
)


def select_covered_idea_indices(
    ideas: tuple[ScenarioIdea, ...],
    *,
    size: int,
    coverage: CoverageRequirements,
) -> tuple[int, ...]:
    """Return positions of the highest-ranked ideas satisfying all coverage."""

    tools = tuple(sorted(CANONICAL_TOOLS))
    targets = tuple(sorted(CI_FAILURE_TYPES))
    idea_tools = tuple(
        frozenset(write.tool for write in idea.expected_writes) for idea in ideas
    )
    idea_targets = tuple(frozenset(idea.target_ci_failures) for idea in ideas)
    archetypes = tuple(
        idea.metadata.pressure_archetype.casefold().strip() for idea in ideas
    )
    medium = tuple(idea.metadata.difficulty == "medium" for idea in ideas)
    tool_requirements = (coverage.scenarios_per_tool,) * len(tools)
    target_requirements = (coverage.scenarios_per_failure_type,) * len(targets)
    initial_archetypes = frozenset() if coverage.pressure_archetypes else None

    def reachable(
        index: int,
        slots: int,
        tool_counts: tuple[int, ...],
        target_counts: tuple[int, ...],
        medium_count: int,
        selected_archetypes: frozenset[str] | None,
    ) -> bool:
        remaining_tools = idea_tools[index:]
        remaining_targets = idea_targets[index:]
        if len(remaining_tools) < slots:
            return False
        for tool, count, required in zip(
            tools,
            tool_counts,
            tool_requirements,
            strict=True,
        ):
            available = sum(tool in values for values in remaining_tools)
            if required - count > min(slots, available):
                return False
        for target, count, required in zip(
            targets,
            target_counts,
            target_requirements,
            strict=True,
        ):
            available = sum(target in values for values in remaining_targets)
            if required - count > min(slots, available):
                return False
        if coverage.medium_scenarios - medium_count > min(
            slots,
            sum(medium[index:]),
        ):
            return False
        if selected_archetypes is not None:
            available = frozenset(archetypes[index:]) - selected_archetypes
            if coverage.pressure_archetypes - len(selected_archetypes) > min(
                slots, len(available)
            ):
                return False
        return True

    @cache
    def search(
        index: int,
        slots: int,
        tool_counts: tuple[int, ...],
        target_counts: tuple[int, ...],
        medium_count: int,
        selected_archetypes: frozenset[str] | None,
    ) -> tuple[int, ...] | None:
        if slots == 0:
            return (
                ()
                if reachable(
                    index,
                    0,
                    tool_counts,
                    target_counts,
                    medium_count,
                    selected_archetypes,
                )
                else None
            )
        if not reachable(
            index,
            slots,
            tool_counts,
            target_counts,
            medium_count,
            selected_archetypes,
        ):
            return None

        next_tool_counts = tuple(
            min(required, count + (tool in idea_tools[index]))
            for tool, count, required in zip(
                tools,
                tool_counts,
                tool_requirements,
                strict=True,
            )
        )
        next_target_counts = tuple(
            min(required, count + (target in idea_targets[index]))
            for target, count, required in zip(
                targets,
                target_counts,
                target_requirements,
                strict=True,
            )
        )
        next_medium_count = min(
            coverage.medium_scenarios,
            medium_count + medium[index],
        )
        next_archetypes = selected_archetypes
        if selected_archetypes is not None:
            next_archetypes = selected_archetypes | {archetypes[index]}
            if len(next_archetypes) >= coverage.pressure_archetypes:
                next_archetypes = None

        selected = search(
            index + 1,
            slots - 1,
            next_tool_counts,
            next_target_counts,
            next_medium_count,
            next_archetypes,
        )
        if selected is not None:
            return (index, *selected)
        return search(
            index + 1,
            slots,
            tool_counts,
            target_counts,
            medium_count,
            selected_archetypes,
        )

    selected = search(
        0,
        size,
        (0,) * len(tools),
        (0,) * len(targets),
        0,
        initial_archetypes,
    )
    if selected is None:
        raise RuntimeError(
            f"No set of {size} passing candidates satisfies dataset coverage."
        )
    return selected
