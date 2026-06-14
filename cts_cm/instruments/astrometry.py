from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cts_cm.aperture.frames import MECHANICAL_MEDIATORS, METABOLIC_MEDIATORS

Edge = tuple[str, str]

TREATMENTS: tuple[str, ...] = (
    "bmi",
    "physical_activity",
    "occupational_loading",
    "metabolic_syndrome",
)
CONFOUNDERS: tuple[str, ...] = ("age", "sex", "race", "baseline_kl")
TRAJECTORY: str = "trajectory_group"

VARIABLE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "treatment": TREATMENTS,
    "mechanical": MECHANICAL_MEDIATORS,
    "metabolic": METABOLIC_MEDIATORS,
    "confounder": CONFOUNDERS,
    "trajectory": (TRAJECTORY,),
}


@dataclass
class CausalGraph:
    edges: set[Edge]

    @classmethod
    def expert(cls) -> CausalGraph:
        edges: set[Edge] = set()
        for confounder in CONFOUNDERS:
            for treatment in TREATMENTS:
                edges.add((confounder, treatment))
            edges.add((confounder, TRAJECTORY))
        for treatment in TREATMENTS:
            for mediator in MECHANICAL_MEDIATORS + METABOLIC_MEDIATORS:
                edges.add((treatment, mediator))
        for mediator in MECHANICAL_MEDIATORS + METABOLIC_MEDIATORS:
            edges.add((mediator, TRAJECTORY))
        return cls(edges=edges)

    def parents(self, node: str) -> set[str]:
        return {src for src, dst in self.edges if dst == node}

    def children(self, node: str) -> set[str]:
        return {dst for src, dst in self.edges if src == node}

    @property
    def mechanical_path(self) -> set[Edge]:
        path: set[Edge] = set()
        for mediator in MECHANICAL_MEDIATORS:
            path.update({(src, dst) for src, dst in self.edges if dst == mediator})
            path.add((mediator, TRAJECTORY))
        return {edge for edge in path if edge in self.edges}

    @property
    def metabolic_path(self) -> set[Edge]:
        path: set[Edge] = set()
        for mediator in METABOLIC_MEDIATORS:
            path.update({(src, dst) for src, dst in self.edges if dst == mediator})
            path.add((mediator, TRAJECTORY))
        return {edge for edge in path if edge in self.edges}

    def adjustment_set(self, treatment: str) -> tuple[str, ...]:
        confounders = self.parents(treatment) & self.parents(TRAJECTORY)
        return tuple(sorted(confounders))


def stability_select_edges(edge_sets: Sequence[set[Edge]], threshold: float = 0.8) -> set[Edge]:
    if not edge_sets:
        return set()
    counts: dict[Edge, int] = {}
    for edges in edge_sets:
        for edge in edges:
            counts[edge] = counts.get(edge, 0) + 1
    cutoff = threshold * len(edge_sets)
    return {edge for edge, count in counts.items() if count >= cutoff}
