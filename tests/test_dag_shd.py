from __future__ import annotations

from cts_cm.instruments.astrometry import CausalGraph, stability_select_edges
from cts_cm.magnitudes.testing import structural_hamming_distance


def test_expert_graph_self_distance_zero() -> None:
    graph = CausalGraph.expert()
    assert structural_hamming_distance(graph.edges, graph.edges) == 0


def test_removing_edge_increases_distance() -> None:
    graph = CausalGraph.expert()
    reduced = set(graph.edges)
    reduced.pop()
    assert structural_hamming_distance(graph.edges, reduced) == 1


def test_adjustment_set_returns_confounders() -> None:
    graph = CausalGraph.expert()
    adjust = graph.adjustment_set("bmi")
    assert set(adjust) == {"age", "sex", "race", "baseline_kl"}


def test_stability_selection_threshold() -> None:
    a: set[tuple[str, str]] = {("x", "y"), ("y", "z")}
    b: set[tuple[str, str]] = {("x", "y")}
    selected = stability_select_edges([a, a, b], threshold=0.8)
    assert ("x", "y") in selected
    assert ("y", "z") not in selected


def test_pathways_are_disjoint_and_nonempty() -> None:
    graph = CausalGraph.expert()
    assert graph.mechanical_path
    assert graph.metabolic_path
    assert graph.mechanical_path.isdisjoint(graph.metabolic_path)
