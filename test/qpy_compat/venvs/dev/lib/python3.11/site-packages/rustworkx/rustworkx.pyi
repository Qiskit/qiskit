# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/lib.rs

from .visit import BFSVisitor, DFSVisitor, DijkstraVisitor
from types import GenericAlias
from typing import (
    Callable,
    final,
    Any,
    Generic,
    overload,
)
from collections.abc import (
    Iterable,
    Iterator,
    Sequence,
    ItemsView,
    KeysView,
    ValuesView,
    Mapping,
    Hashable,
)
from abc import ABC
from rustworkx import generators  # noqa

import numpy as np
import numpy.typing as npt
import sys

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

_S = TypeVar("_S", default=Any)
_T = TypeVar("_T", default=Any)

class DAGHasCycle(Exception): ...
class DAGWouldCycle(Exception): ...
class InvalidNode(Exception): ...
class NoEdgeBetweenNodes(Exception): ...
class NoPathFound(Exception): ...
class NoSuitableNeighbors(Exception): ...
class NullGraph(Exception): ...
class NegativeCycle(Exception): ...
class JSONSerializationError(Exception): ...
class JSONDeserializationError(Exception): ...
class FailedToConverge(Exception): ...
class InvalidMapping(Exception): ...
class GraphNotBipartite(Exception): ...

@final
class ColoringStrategy:
    Degree: Any
    Saturation: Any
    IndependentSet: Any

# Cartesian product

def digraph_cartesian_product(
    first: PyDiGraph,
    second: PyDiGraph,
    /,
) -> tuple[PyDiGraph, ProductNodeMap]: ...
def graph_cartesian_product(
    first: PyGraph,
    second: PyGraph,
    /,
) -> tuple[PyGraph, ProductNodeMap]: ...

# Centrality

def digraph_eigenvector_centrality(
    graph: PyDiGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    max_iter: int = ...,
    tol: float = ...,
) -> CentralityMapping: ...
def graph_eigenvector_centrality(
    graph: PyGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    max_iter: int = ...,
    tol: float = ...,
) -> CentralityMapping: ...
def digraph_betweenness_centrality(
    graph: PyDiGraph[_S, _T],
    /,
    normalized: bool = ...,
    endpoints: bool = ...,
    parallel_threshold: int = ...,
) -> CentralityMapping: ...
def graph_betweenness_centrality(
    graph: PyGraph[_S, _T],
    /,
    normalized: bool = ...,
    endpoints: bool = ...,
    parallel_threshold: int = ...,
) -> CentralityMapping: ...
def digraph_edge_betweenness_centrality(
    graph: PyDiGraph[_S, _T],
    /,
    normalized: bool = ...,
    parallel_threshold: int = ...,
) -> EdgeCentralityMapping: ...
def graph_edge_betweenness_centrality(
    graph: PyGraph[_S, _T],
    /,
    normalized: bool = ...,
    parallel_threshold: int = ...,
) -> EdgeCentralityMapping: ...
def digraph_closeness_centrality(
    graph: PyDiGraph[_S, _T],
    wf_improved: bool = ...,
) -> CentralityMapping: ...
def graph_closeness_centrality(
    graph: PyGraph[_S, _T],
    wf_improved: bool = ...,
) -> CentralityMapping: ...
def digraph_degree_centrality(
    graph: PyDiGraph[_S, _T],
    /,
) -> CentralityMapping: ...
def in_degree_centrality(
    graph: PyDiGraph[_S, _T],
    /,
) -> CentralityMapping: ...
def out_degree_centrality(
    graph: PyDiGraph[_S, _T],
    /,
) -> CentralityMapping: ...
def graph_degree_centrality(
    graph: PyGraph[_S, _T],
    /,
) -> CentralityMapping: ...
def digraph_katz_centrality(
    graph: PyDiGraph[_S, _T],
    /,
    alpha: float | None = ...,
    beta: float | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float | None = ...,
    max_iter: int | None = ...,
    tol: float | None = ...,
) -> CentralityMapping: ...
def graph_katz_centrality(
    graph: PyGraph[_S, _T],
    /,
    alpha: float | None = ...,
    beta: float | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float | None = ...,
    max_iter: int | None = ...,
    tol: float | None = ...,
) -> CentralityMapping: ...

# Coloring

def graph_greedy_color(
    graph: PyGraph,
    /,
    preset_color_fn: Callable[[int], int | None] | None = ...,
    strategy: int = ...,
) -> dict[int, int]: ...
def graph_greedy_edge_color(
    graph: PyGraph,
    /,
    preset_color_fn: Callable[[int], int | None] | None = ...,
    strategy: int = ...,
) -> dict[int, int]: ...
def graph_is_bipartite(graph: PyGraph) -> bool: ...
def digraph_is_bipartite(graph: PyDiGraph) -> bool: ...
def graph_two_color(graph: PyGraph) -> dict[int, int]: ...
def digraph_two_color(graph: PyDiGraph) -> dict[int, int]: ...
def graph_misra_gries_edge_color(graph: PyGraph, /) -> dict[int, int]: ...
def graph_bipartite_edge_color(graph: PyGraph, /) -> dict[int, int]: ...

# Connectivity

def connected_components(graph: PyGraph, /) -> list[set[int]]: ...
def is_connected(graph: PyGraph, /) -> bool: ...
def is_weakly_connected(graph: PyDiGraph, /) -> bool: ...
def is_semi_connected(graph: PyDiGraph, /) -> bool: ...
def number_connected_components(graph: PyGraph, /) -> int: ...
def number_weakly_connected_components(graph: PyDiGraph, /) -> bool: ...
def node_connected_component(graph: PyGraph, node: int, /) -> set[int]: ...
def strongly_connected_components(graph: PyDiGraph, /) -> list[list[int]]: ...
def weakly_connected_components(graph: PyDiGraph, /) -> list[set[int]]: ...
def digraph_adjacency_matrix(
    graph: PyDiGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    null_value: float = ...,
    parallel_edge: str = ...,
) -> npt.NDArray[np.float64]: ...
def graph_adjacency_matrix(
    graph: PyGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    null_value: float = ...,
    parallel_edge: str = ...,
) -> npt.NDArray[np.float64]: ...
def cycle_basis(graph: PyGraph, /, root: int | None = ...) -> list[list[int]]: ...
def articulation_points(graph: PyGraph, /) -> set[int]: ...
def bridges(graph: PyGraph, /) -> set[tuple[int]]: ...
def biconnected_components(graph: PyGraph, /) -> BiconnectedComponents: ...
def chain_decomposition(graph: PyGraph, /, source: int | None = ...) -> Chains: ...
def digraph_find_cycle(
    graph: PyDiGraph[_S, _T],
    /,
    source: int | None = ...,
) -> EdgeList: ...
def digraph_complement(graph: PyDiGraph[_S, _T], /) -> PyDiGraph[_S, _T | None]: ...
def graph_complement(
    graph: PyGraph[_S, _T],
    /,
) -> PyGraph[_S, _T | None]: ...
def digraph_all_simple_paths(
    graph: PyDiGraph,
    origin: int,
    to: int,
    /,
    min_depth: int | None = ...,
    cutoff: int | None = ...,
) -> list[list[int]]: ...
def graph_all_simple_paths(
    graph: PyGraph,
    origin: int,
    to: int,
    /,
    min_depth: int | None = ...,
    cutoff: int | None = ...,
) -> list[list[int]]: ...
def digraph_all_pairs_all_simple_paths(
    graph: PyDiGraph,
    /,
    min_depth: int | None = ...,
    cutoff: int | None = ...,
) -> AllPairsMultiplePathMapping: ...
def graph_all_pairs_all_simple_paths(
    graph: PyGraph,
    /,
    min_depth: int | None = ...,
    cutoff: int | None = ...,
) -> AllPairsMultiplePathMapping: ...
def digraph_longest_simple_path(graph: PyDiGraph, /) -> NodeIndices | None: ...
def graph_longest_simple_path(graph: PyGraph, /) -> NodeIndices | None: ...
def digraph_core_number(
    graph: PyDiGraph,
    /,
) -> int: ...
def graph_core_number(
    graph: PyGraph,
    /,
) -> int: ...
def stoer_wagner_min_cut(
    graph: PyGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
) -> tuple[float, NodeIndices] | None: ...
def simple_cycles(graph: PyDiGraph, /) -> Iterator[NodeIndices]: ...
def graph_isolates(graph: PyGraph) -> NodeIndices: ...
def digraph_isolates(graph: PyDiGraph) -> NodeIndices: ...
def connected_subgraphs(graph: PyGraph, k: int, /) -> list[list[int]]: ...

# DAG Algorithms

def collect_runs(
    graph: PyDiGraph[_S, _T],
    filter_fn: Callable[[_S], bool],
) -> list[list[_S]]: ...
def collect_bicolor_runs(
    graph: PyDiGraph[_S, _T],
    filter_fn: Callable[[_S], bool],
    color_fn: Callable[[_T], int],
) -> list[list[_S]]: ...
def dag_longest_path(
    graph: PyDiGraph[_S, _T], /, weight_fn: Callable[[int, int, _T], int] | None = ...
) -> NodeIndices: ...
def dag_longest_path_length(
    graph: PyDiGraph[_S, _T], /, weight_fn: Callable[[int, int, _T], int] | None = ...
) -> int: ...
def dag_weighted_longest_path(
    graph: PyDiGraph[_S, _T],
    weight_fn: Callable[[int, int, _T], float],
    /,
) -> NodeIndices: ...
def dag_weighted_longest_path_length(
    graph: PyDiGraph[_S, _T],
    weight_fn: Callable[[int, int, _T], float],
    /,
) -> float: ...
def is_directed_acyclic_graph(graph: PyDiGraph, /) -> bool: ...
def topological_sort(graph: PyDiGraph, /) -> NodeIndices: ...
def topological_generations(dag: PyDiGraph, /) -> list[NodeIndices]: ...
def lexicographical_topological_sort(
    dag: PyDiGraph[_S, _T],
    /,
    key: Callable[[_S], str],
    *,
    reverse: bool = ...,
    initial: Iterable[int] | None = ...,
) -> list[_S]: ...
def transitive_reduction(graph: PyDiGraph, /) -> tuple[PyDiGraph, dict[int, int]]: ...
def layers(
    dag: PyDiGraph[_S, _T],
    first_layer: list[int],
    /,
    index_output: bool = ...,
) -> list[list[_S]] | list[list[int]]: ...
@final
class TopologicalSorter:
    def __init__(
        self,
        dag: PyDiGraph,
        /,
        check_cycle: bool,
        *,
        reverse: bool = ...,
        initial: Iterable[int] | None = ...,
        check_args: bool = ...,
    ) -> None: ...
    def is_active(self) -> bool: ...
    def get_ready(self) -> list[int]: ...
    def done(self, nodes: int | Sequence[int]) -> None: ...

# isomorpism

def digraph_is_isomorphic(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    /,
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    call_limit: int | None = ...,
) -> bool: ...
def graph_is_isomorphic(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    /,
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    call_limit: int | None = ...,
) -> bool: ...
def digraph_is_subgraph_isomorphic(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    /,
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    induced: bool = ...,
    call_limit: int | None = ...,
) -> bool: ...
def graph_is_subgraph_isomorphic(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    /,
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    induced: bool = ...,
    call_limit: int | None = ...,
) -> bool: ...
def digraph_vf2_mapping(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    /,
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    subgraph: bool = ...,
    induced: bool = ...,
    call_limit: int | None = ...,
) -> Iterator[NodeMap]: ...
def graph_vf2_mapping(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    /,
    node_matcher: Callable[[_S, _S], bool] | None = ...,
    edge_matcher: Callable[[_T, _T], bool] | None = ...,
    id_order: bool = ...,
    subgraph: bool = ...,
    induced: bool = ...,
    call_limit: int | None = ...,
) -> Iterator[NodeMap]: ...

# Layout

def digraph_bipartite_layout(
    graph: PyDiGraph,
    first_nodes: set[int],
    /,
    horizontal: bool | None = ...,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
    aspect_ratio: float | None = ...,
) -> Pos2DMapping: ...
def graph_bipartite_layout(
    graph: PyGraph,
    first_nodes: set[int],
    /,
    horizontal: bool | None = ...,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
    aspect_ratio: float | None = ...,
) -> Pos2DMapping: ...
def digraph_circular_layout(
    graph: PyDiGraph,
    /,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
) -> Pos2DMapping: ...
def graph_circular_layout(
    graph: PyGraph,
    /,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
) -> Pos2DMapping: ...
def digraph_random_layout(
    graph: PyDiGraph,
    /,
    center: tuple[float, float] | None = ...,
    seed: int | None = ...,
) -> Pos2DMapping: ...
def graph_random_layout(
    graph: PyGraph,
    /,
    center: tuple[float, float] | None = ...,
    seed: int | None = ...,
) -> Pos2DMapping: ...
def digraph_shell_layout(
    graph: PyDiGraph,
    /,
    nlist: list[list[int]] | None = ...,
    rotate: float | None = ...,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
) -> Pos2DMapping: ...
def graph_shell_layout(
    graph: PyGraph,
    /,
    nlist: list[list[int]] | None = ...,
    rotate: float | None = ...,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
) -> Pos2DMapping: ...
def digraph_spiral_layout(
    graph: PyDiGraph,
    /,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
    resolution: float | None = ...,
    equidistant: bool | None = ...,
) -> Pos2DMapping: ...
def graph_spiral_layout(
    graph: PyGraph,
    /,
    scale: float | None = ...,
    center: tuple[float, float] | None = ...,
    resolution: float | None = ...,
    equidistant: bool | None = ...,
) -> Pos2DMapping: ...
def digraph_spring_layout(
    graph: PyDiGraph[_S, _T],
    pos: dict[int, tuple[float, float]] | None = ...,
    fixed: set[int] | None = ...,
    k: float | None = ...,
    repulsive_exponent: int = ...,
    adaptive_cooling: bool = ...,
    num_iter: int = ...,
    tol: float | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    scale: float = ...,
    center: tuple[float, float] | None = ...,
    seed: int | None = ...,
    /,
) -> Pos2DMapping: ...
def graph_spring_layout(
    graph: PyGraph[_S, _T],
    pos: dict[int, tuple[float, float]] | None = ...,
    fixed: set[int] | None = ...,
    k: float | None = ...,
    repulsive_exponent: int = ...,
    adaptive_cooling: bool = ...,
    num_iter: int = ...,
    tol: float | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    scale: float = ...,
    center: tuple[float, float] | None = ...,
    seed: int | None = ...,
    /,
) -> Pos2DMapping: ...

# Line graph

def graph_line_graph(graph: PyGraph, /) -> tuple[PyGraph, dict[int, int]]: ...

# Link Analysis

def hits(
    graph: PyDiGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    nstart: dict[int, float] | None = ...,
    tol: float | None = ...,
    max_iter: int | None = ...,
    normalized: bool | None = ...,
) -> tuple[CentralityMapping, CentralityMapping]: ...
def pagerank(
    graph: PyDiGraph[_S, _T],
    /,
    alpha: float | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    nstart: dict[int, float] | None = ...,
    personalization: dict[int, float] | None = ...,
    tol: float | None = ...,
    max_iter: int | None = ...,
    dangling: dict[int, float] | None = ...,
) -> CentralityMapping: ...

# Matching

def max_weight_matching(
    graph: PyGraph[_S, _T],
    /,
    max_cardinality: bool = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: int = ...,
    verify_optimum: bool = ...,
) -> set[tuple[int, int]]: ...
def is_matching(
    graph: PyGraph,
    matching: set[tuple[int, int]],
    /,
) -> bool: ...
def is_maximal_matching(
    graph: PyGraph,
    matching: set[tuple[int, int]],
    /,
) -> bool: ...

# Maximum Bisimulation

def digraph_maximum_bisimulation(graph: PyDiGraph) -> RelationalCoarsestPartition: ...

# Planar

def is_planar(graph: PyGraph, /) -> bool: ...

# Random Graph

def directed_gnm_random_graph(
    num_nodes: int,
    num_edges: int,
    /,
    seed: int | None = ...,
) -> PyDiGraph: ...
def undirected_gnm_random_graph(
    num_nodes: int,
    num_edges: int,
    /,
    seed: int | None = ...,
) -> PyGraph: ...
def directed_gnp_random_graph(
    num_nodes: int,
    probability: float,
    /,
    seed: int | None = ...,
) -> PyDiGraph: ...
def undirected_gnp_random_graph(
    num_nodes: int,
    probability: float,
    /,
    seed: int | None = ...,
) -> PyGraph: ...
def directed_sbm_random_graph(
    sizes: list[int],
    probabilities: npt.NDArray[np.float64],
    loops: bool,
    /,
    seed: int | None = ...,
) -> PyDiGraph: ...
def undirected_sbm_random_graph(
    sizes: list[int],
    probabilities: npt.NDArray[np.float64],
    loops: bool,
    /,
    seed: int | None = ...,
) -> PyGraph: ...
def random_geometric_graph(
    num_nodes: int,
    radius: float,
    /,
    dim: int = ...,
    pos: list[list[float]] | None = ...,
    p: float = ...,
    seed: int | None = ...,
) -> PyGraph: ...
def hyperbolic_random_graph(
    pos: list[list[float]],
    r: float,
    beta: float | None,
    /,
    seed: int | None = ...,
) -> PyGraph: ...
def barabasi_albert_graph(
    n: int,
    m: int,
    seed: int | None = ...,
    initial_graph: PyGraph | None = ...,
) -> PyGraph: ...
def directed_barabasi_albert_graph(
    n: int,
    m: int,
    seed: int | None = ...,
    initial_graph: PyDiGraph | None = ...,
) -> PyDiGraph: ...
def undirected_random_bipartite_graph(
    num_l_nodes: int,
    num_r_nodes: int,
    probability: float,
    /,
    seed: int | None = ...,
) -> PyGraph: ...
def directed_random_bipartite_graph(
    num_l_nodes: int,
    num_r_nodes: int,
    probability: float,
    /,
    seed: int | None = ...,
) -> PyDiGraph: ...

# Read Write

def read_graphml(
    path: str,
    /,
    compression: str | None = ...,
) -> list[PyGraph | PyDiGraph]: ...
def digraph_node_link_json(
    graph: PyDiGraph[_S, _T],
    /,
    path: str | None = ...,
    graph_attrs: Callable[[Any], dict[str, str]] | None = ...,
    node_attrs: Callable[[_S], dict[str, str]] | None = ...,
    edge_attrs: Callable[[_T], dict[str, str]] | None = ...,
) -> str | None: ...
def graph_node_link_json(
    graph: PyGraph[_S, _T],
    /,
    path: str | None = ...,
    graph_attrs: Callable[[Any], dict[str, str]] | None = ...,
    node_attrs: Callable[[_S], dict[str, str]] | None = ...,
    edge_attrs: Callable[[_T], dict[str, str]] | None = ...,
) -> str | None: ...
def parse_node_link_json(
    data: str,
    graph_attrs: Callable[[dict[str, str]], Any] | None = ...,
    node_attrs: Callable[[dict[str, str]], _S] | None = ...,
    edge_attrs: Callable[[dict[str, str]], _T] | None = ...,
) -> PyDiGraph[_S, _T] | PyGraph[_S, _T]: ...
def from_node_link_json_file(
    path: str,
    graph_attrs: Callable[[dict[str, str]], Any] | None = ...,
    node_attrs: Callable[[dict[str, str]], _S] | None = ...,
    edge_attrs: Callable[[dict[str, str]], _T] | None = ...,
) -> PyDiGraph[_S, _T] | PyGraph[_S, _T]: ...

# Shortest Path

def digraph_bellman_ford_shortest_paths(
    graph: PyDiGraph[_S, _T],
    source: int,
    /,
    target: int | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    as_undirected: bool = ...,
) -> PathMapping: ...
def graph_bellman_ford_shortest_paths(
    graph: PyDiGraph[_S, _T],
    source: int,
    /,
    target: int | None = ...,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
) -> PathMapping: ...
def digraph_bellman_ford_shortest_path_lengths(
    graph: PyDiGraph[_S, _T],
    node: int,
    edge_cost_fn: Callable[[_T], float] | None,
    /,
    goal: int | None = ...,
) -> PathLengthMapping: ...
def graph_bellman_ford_shortest_path_lengths(
    graph: PyGraph[_S, _T],
    node: int,
    edge_cost_fn: Callable[[_T], float] | None,
    /,
    goal: int | None = ...,
) -> PathLengthMapping: ...
def digraph_dijkstra_shortest_paths(
    graph: PyDiGraph[_S, _T],
    source: int,
    /,
    target: int | None,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    as_undirected: bool = ...,
) -> PathMapping: ...
def graph_dijkstra_shortest_paths(
    graph: PyDiGraph[_S, _T],
    source: int,
    /,
    target: int | None,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
) -> PathMapping: ...
def digraph_dijkstra_shortest_path_lengths(
    graph: PyDiGraph[_S, _T],
    node: int,
    edge_cost_fn: Callable[[_T], float] | None,
    /,
    goal: int | None = ...,
) -> PathLengthMapping: ...
def graph_dijkstra_shortest_path_lengths(
    graph: PyGraph[_S, _T],
    node: int,
    edge_cost_fn: Callable[[_T], float] | None,
    /,
    goal: int | None = ...,
) -> PathLengthMapping: ...
def digraph_all_pairs_bellman_ford_path_lengths(
    graph: PyDiGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathLengthMapping: ...
def graph_all_pairs_bellman_ford_path_lengths(
    graph: PyGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathLengthMapping: ...
def digraph_all_pairs_bellman_ford_shortest_paths(
    graph: PyDiGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathMapping: ...
def graph_all_pairs_bellman_ford_shortest_paths(
    graph: PyDiGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathMapping: ...
def digraph_all_pairs_dijkstra_path_lengths(
    graph: PyDiGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathLengthMapping: ...
def graph_all_pairs_dijkstra_path_lengths(
    graph: PyGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathLengthMapping: ...
def digraph_all_pairs_dijkstra_shortest_paths(
    graph: PyDiGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathMapping: ...
def graph_all_pairs_dijkstra_shortest_paths(
    graph: PyDiGraph[_S, _T],
    edge_cost: Callable[[_T], float],
    /,
) -> AllPairsPathMapping: ...
def digraph_astar_shortest_path(
    graph: PyDiGraph[_S, _T],
    node: int,
    goal_fn: Callable[[_S], bool],
    edge_cost_fn: Callable[[_T], float],
    estimate_cost_fn: Callable[[_S], float],
    /,
) -> NodeIndices: ...
def graph_astar_shortest_path(
    graph: PyGraph[_S, _T],
    node: int,
    goal_fn: Callable[[_S], bool],
    edge_cost_fn: Callable[[_T], float],
    estimate_cost_fn: Callable[[_S], float],
    /,
) -> NodeIndices: ...
def digraph_k_shortest_path_lengths(
    graph: PyDiGraph[_S, _T],
    start: int,
    k: int,
    edge_cost: Callable[[_T], float],
    /,
    goal: int | None = ...,
) -> PathLengthMapping: ...
def graph_k_shortest_path_lengths(
    graph: PyGraph[_S, _T],
    start: int,
    k: int,
    edge_cost: Callable[[_T], float],
    /,
    goal: int | None = ...,
) -> PathLengthMapping: ...
def digraph_has_path(
    graph: PyDiGraph,
    source: int,
    target: int,
    /,
    as_undirected: bool | None = ...,
) -> bool: ...
def graph_has_path(
    graph: PyGraph,
    source: int,
    target: int,
) -> bool: ...
def digraph_num_shortest_paths_unweighted(
    graph: PyDiGraph,
    source: int,
    /,
) -> NodesCountMapping: ...
def graph_num_shortest_paths_unweighted(
    graph: PyGraph,
    source: int,
    /,
) -> NodesCountMapping: ...
def digraph_unweighted_average_shortest_path_length(
    graph: PyDiGraph,
    /,
    parallel_threshold: int | None = ...,
    as_undirected: bool | None = ...,
    disconnected: bool | None = ...,
) -> float: ...
def graph_unweighted_average_shortest_path_length(
    graph: PyGraph,
    /,
    parallel_threshold: int | None = ...,
    disconnected: bool | None = ...,
) -> float: ...
def digraph_distance_matrix(
    graph: PyDiGraph,
    /,
    parallel_threshold: int | None = ...,
    as_undirected: bool | None = ...,
    null_value: float | None = ...,
) -> npt.NDArray[np.float64]: ...
def graph_distance_matrix(
    graph: PyGraph,
    /,
    parallel_threshold: int | None = ...,
    null_value: float | None = ...,
) -> npt.NDArray[np.float64]: ...
def digraph_floyd_warshall(
    graph: PyDiGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    as_undirected: bool | None = ...,
    default_weight: float | None = ...,
    parallel_threshold: int | None = ...,
) -> AllPairsPathLengthMapping: ...
def graph_floyd_warshall(
    graph: PyGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float | None = ...,
    parallel_threshold: int | None = ...,
) -> AllPairsPathLengthMapping: ...
def digraph_floyd_warshall_numpy(
    graph: PyDiGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    as_undirected: bool | None = ...,
    default_weight: float | None = ...,
    parallel_threshold: int | None = ...,
) -> npt.NDArray[np.float64]: ...
def graph_floyd_warshall_numpy(
    graph: PyGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float | None = ...,
    parallel_threshold: int | None = ...,
) -> npt.NDArray[np.float64]: ...
def digraph_floyd_warshall_successor_and_distance(
    graph: PyDiGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    as_undirected: bool | None = ...,
    default_weight: float | None = ...,
    parallel_threshold: int | None = ...,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def graph_floyd_warshall_successor_and_distance(
    graph: PyGraph[_S, _T],
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float | None = ...,
    parallel_threshold: int | None = ...,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def find_negative_cycle(
    graph: PyDiGraph[_S, _T],
    edge_cost_fn: Callable[[_T], float],
    /,
) -> NodeIndices: ...
def negative_edge_cycle(
    graph: PyDiGraph[_S, _T],
    edge_cost_fn: Callable[[_T], float],
    /,
) -> bool: ...
def digraph_all_shortest_paths(
    graph: PyDiGraph[_S, _T],
    source: int,
    target: int,
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
    as_undirected: bool = ...,
) -> list[list[int]]: ...
def graph_all_shortest_paths(
    graph: PyGraph[_S, _T],
    source: int,
    target: int,
    /,
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
) -> list[list[int]]: ...

# Tensor Product

def digraph_tensor_product(
    first: PyDiGraph,
    second: PyDiGraph,
    /,
) -> tuple[PyDiGraph, ProductNodeMap]: ...
def graph_tensor_product(
    first: PyGraph,
    second: PyGraph,
    /,
) -> tuple[PyGraph, ProductNodeMap]: ...

# Token Swapper

def graph_token_swapper(
    graph: PyGraph,
    mapping: dict[int, int],
    /,
    trials: int | None = ...,
    seed: int | None = ...,
    parallel_threshold: int | None = ...,
) -> EdgeList: ...

# Transitivity

def digraph_transitivity(graph: PyDiGraph, /) -> float: ...
def graph_transitivity(graph: PyGraph, /) -> float: ...

# Traversal

_BFSVisitor = TypeVar("_BFSVisitor", bound=BFSVisitor, default=BFSVisitor)
_DFSVisitor = TypeVar("_DFSVisitor", bound=DFSVisitor, default=DFSVisitor)
_DijkstraVisitor = TypeVar("_DijkstraVisitor", bound=DijkstraVisitor, default=DijkstraVisitor)

def digraph_bfs_search(
    graph: PyDiGraph,
    source: Sequence[int] | None = ...,
    visitor: _BFSVisitor | None = ...,
) -> None: ...
def graph_bfs_search(
    graph: PyGraph,
    source: Sequence[int] | None = ...,
    visitor: _BFSVisitor | None = ...,
) -> None: ...
def digraph_dfs_search(
    graph: PyDiGraph,
    source: Sequence[int] | None = ...,
    visitor: _DFSVisitor | None = ...,
) -> None: ...
def graph_dfs_search(
    graph: PyGraph,
    source: Sequence[int] | None = ...,
    visitor: _DFSVisitor | None = ...,
) -> None: ...
def digraph_dijkstra_search(
    graph: PyDiGraph,
    source: Sequence[int] | None = ...,
    weight_fn: Callable[[Any], float] | None = ...,
    visitor: _DijkstraVisitor | None = ...,
) -> None: ...
def graph_dijkstra_search(
    graph: PyGraph,
    source: Sequence[int] | None = ...,
    weight_fn: Callable[[Any], float] | None = ...,
    visitor: _DijkstraVisitor | None = ...,
) -> None: ...
def digraph_dfs_edges(graph: PyDiGraph[_S, _T], /, source: int | None = ...) -> EdgeList: ...
def graph_dfs_edges(graph: PyGraph[_S, _T], /, source: int | None = ...) -> EdgeList: ...
def ancestors(graph: PyDiGraph, node: int, /) -> set[int]: ...
def bfs_predecessors(graph: PyDiGraph, node: int, /) -> BFSPredecessors: ...
def bfs_successors(graph: PyDiGraph, node: int, /) -> BFSSuccessors: ...
def descendants(graph: PyDiGraph, node: int, /) -> set[int]: ...

# Tree

def minimum_spanning_edges(
    graph: PyGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
) -> WeightedEdgeList: ...
def minimum_spanning_tree(
    graph: PyGraph[_S, _T],
    weight_fn: Callable[[_T], float] | None = ...,
    default_weight: float = ...,
) -> PyGraph[_S, _T]: ...
def steiner_tree(
    graph: PyGraph[_S, _T],
    terminal_nodes: list[int],
    weight_fn: Callable[[_T], float],
    /,
) -> PyGraph[_S, _T]: ...
def metric_closure(
    graph: PyGraph[_S, _T],
    weight_fn: Callable[[_T], float],
    /,
) -> PyGraph: ...

# Union

def digraph_union(
    first: PyDiGraph[_S, _T],
    second: PyDiGraph[_S, _T],
    /,
    merge_nodes: bool = ...,
    merge_edges: bool = ...,
) -> PyDiGraph[_S, _T]: ...
def graph_union(
    first: PyGraph[_S, _T],
    second: PyGraph[_S, _T],
    /,
    merge_nodes: bool = ...,
    merge_edges: bool = ...,
) -> PyGraph[_S, _T]: ...

# Dominance

def immediate_dominators(graph: PyDiGraph[_S, _T], start_node: int, /) -> dict[int, int]: ...
def dominance_frontiers(graph: PyDiGraph[_S, _T], start_node: int, /) -> dict[int, set[int]]: ...

# Iterators

_T_co = TypeVar("_T_co", covariant=True, default=Any)

class _RustworkxCustomVecIter(Generic[_T_co], Sequence[_T_co], ABC):
    def __init__(self) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    @overload
    def __getitem__(self, index: int) -> _T_co: ...
    @overload
    def __getitem__(self: Self, index: slice) -> Self: ...
    def __getstate__(self) -> Any: ...
    def __hash__(self) -> int: ...
    def __len__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: Sequence[_T_co]) -> None: ...
    def __array__(
        self, dtype: np.dtype[Any] | None = ..., copy: bool | None = ...
    ) -> npt.NDArray[Any]: ...
    def __iter__(self) -> Iterator[_T_co]: ...
    def __reversed__(self) -> Iterator[_T_co]: ...

class _RustworkxCustomHashMapIter(Generic[_S, _T_co], Mapping[_S, _T_co], ABC):
    def __init__(self) -> None: ...
    def items(self) -> ItemsView[_S, _T_co]: ...
    def keys(self) -> KeysView[_S]: ...
    def values(self) -> ValuesView[_T_co]: ...
    def __contains__(self, other: object) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __getitem__(self, index: _S) -> _T_co: ...
    def __getstate__(self) -> Any: ...
    def __hash__(self) -> int: ...
    def __iter__(self) -> Iterator[_S]: ...
    def __len__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __setstate__(self, state: Mapping[_S, _T_co]) -> None: ...

@final
class NodeIndices(_RustworkxCustomVecIter[int]): ...

@final
class PathLengthMapping(_RustworkxCustomHashMapIter[int, float]): ...

@final
class PathMapping(_RustworkxCustomHashMapIter[int, NodeIndices]): ...

@final
class AllPairsPathLengthMapping(_RustworkxCustomHashMapIter[int, PathLengthMapping]): ...

@final
class AllPairsPathMapping(_RustworkxCustomHashMapIter[int, PathMapping]): ...

@final
class BFSSuccessors(Generic[_T_co], _RustworkxCustomVecIter[tuple[_T_co, list[_T_co]]]): ...

@final
class BFSPredecessors(Generic[_T_co], _RustworkxCustomVecIter[tuple[_T_co, list[_T_co]]]): ...

@final
class EdgeIndexMap(Generic[_T_co], _RustworkxCustomHashMapIter[int, tuple[int, int, _T_co]]): ...

@final
class EdgeIndices(_RustworkxCustomVecIter[int]): ...

@final
class Chains(_RustworkxCustomVecIter[EdgeIndices]): ...

@final
class IndexPartitionBlock(_RustworkxCustomVecIter[int]): ...

@final
class RelationalCoarsestPartition(_RustworkxCustomVecIter[IndexPartitionBlock]): ...

@final
class EdgeList(_RustworkxCustomVecIter[tuple[int, int]]): ...

@final
class NodeMap(_RustworkxCustomHashMapIter[int, int]): ...

@final
class NodesCountMapping(_RustworkxCustomHashMapIter[int, int]): ...

@final
class Pos2DMapping(_RustworkxCustomHashMapIter[int, tuple[float, float]]): ...

@final
class WeightedEdgeList(Generic[_T_co], _RustworkxCustomVecIter[tuple[int, int, _T_co]]): ...

@final
class CentralityMapping(_RustworkxCustomHashMapIter[int, float]): ...

@final
class EdgeCentralityMapping(_RustworkxCustomHashMapIter[int, float]): ...

@final
class BiconnectedComponents(_RustworkxCustomHashMapIter[tuple[int, int], int]): ...

@final
class ProductNodeMap(_RustworkxCustomHashMapIter[tuple[int, int], int]): ...

@final
class MultiplePathMapping(_RustworkxCustomHashMapIter[int, list[list[int]]]): ...

@final
class AllPairsMultiplePathMapping(_RustworkxCustomHashMapIter[int, MultiplePathMapping]): ...

# Graph

class PyGraph(Generic[_S, _T]):
    attrs: Any
    multigraph: bool = ...
    def __init__(
        self,
        /,
        multigraph: bool = ...,
        attrs: Any = ...,
        *,
        node_count_hint: int | None = ...,
        edge_count_hint: int | None = ...,
    ) -> None: ...
    def add_edge(self, node_a: int, node_b: int, edge: _T, /) -> int: ...
    def add_edges_from(
        self,
        obj_list: Iterable[tuple[int, int, _T]],
        /,
    ) -> list[int]: ...
    def add_edges_from_no_data(
        self: PyGraph[_S, _T | None], obj_list: Iterable[tuple[int, int]], /
    ) -> list[int]: ...
    def add_node(self, obj: _S, /) -> int: ...
    def add_nodes_from(self, obj_list: Iterable[_S], /) -> NodeIndices: ...
    def adj(self, node: int, /) -> dict[int, _T]: ...
    def clear(self) -> None: ...
    def clear_edges(self) -> None: ...
    def compose(
        self,
        other: PyGraph[_S, _T],
        node_map: dict[int, tuple[int, _T]],
        /,
        node_map_func: Callable[[_S], int] | None = ...,
        edge_map_func: Callable[[_T], int] | None = ...,
    ) -> dict[int, int]: ...
    def contract_nodes(
        self,
        nodes: Sequence[int],
        obj: _S,
        /,
        weight_combo_fn: Callable[[_T, _T], _T] | None = ...,
    ) -> int: ...
    def copy(self) -> PyGraph[_S, _T]: ...
    def degree(self, node: int, /) -> int: ...
    def edge_index_map(self) -> EdgeIndexMap[_T]: ...
    def edge_indices(self) -> EdgeIndices: ...
    def edge_indices_from_endpoints(self, node_a: int, node_b: int) -> EdgeIndices: ...
    def edge_list(self) -> EdgeList: ...
    def edges(self) -> list[_T]: ...
    def edge_subgraph(self, edge_list: Sequence[tuple[int, int]], /) -> PyGraph[_S, _T]: ...
    def extend_from_edge_list(
        self: PyGraph[_S | None, _T | None], edge_list: Iterable[tuple[int, int]], /
    ) -> None: ...
    def extend_from_weighted_edge_list(
        self: PyGraph[_S | None, _T],
        edge_list: Iterable[tuple[int, int, _T]],
        /,
    ) -> None: ...
    def filter_edges(self, filter_function: Callable[[_T], bool]) -> EdgeIndices: ...
    def filter_nodes(self, filter_function: Callable[[_S], bool]) -> NodeIndices: ...
    def find_node_by_weight(
        self,
        obj: _S,
        /,
    ) -> int | None: ...
    @staticmethod
    def from_adjacency_matrix(
        matrix: npt.NDArray[np.float64], /, null_value: float = ...
    ) -> PyGraph[int, float]: ...
    @staticmethod
    def from_complex_adjacency_matrix(
        matrix: npt.NDArray[np.complex64], /, null_value: complex = ...
    ) -> PyGraph[int, complex]: ...
    def get_all_edge_data(self, node_a: int, node_b: int, /) -> list[_T]: ...
    def get_edge_data(self, node_a: int, node_b: int, /) -> _T: ...
    def get_edge_data_by_index(self, edge_index: int, /) -> _T: ...
    def get_edge_endpoints_by_index(self, edge_index: int, /) -> tuple[int, int]: ...
    def get_node_data(self, node: int, /) -> _S: ...
    def has_node(self, node: int, /) -> bool: ...
    def has_edge(self, node_a: int, node_b: int, /) -> bool: ...
    def has_parallel_edges(self) -> bool: ...
    def in_edges(self, node: int, /) -> WeightedEdgeList[_T]: ...
    def incident_edge_index_map(self, node: int, /) -> EdgeIndexMap: ...
    def incident_edges(self, node: int, /) -> EdgeIndices: ...
    def neighbors(self, node: int, /) -> NodeIndices: ...
    def node_indexes(self) -> NodeIndices: ...
    def node_indices(self) -> NodeIndices: ...
    def nodes(self) -> list[_S]: ...
    def num_edges(self) -> int: ...
    def num_nodes(self) -> int: ...
    def out_edges(self, node: int, /) -> WeightedEdgeList[_T]: ...
    @staticmethod
    def read_edge_list(
        path: str,
        /,
        comment: str | None = ...,
        deliminator: str | None = ...,
        labels: bool = ...,
    ) -> PyGraph: ...
    def remove_edge(self, node_a: int, node_b: int, /) -> None: ...
    def remove_edge_from_index(self, edge: int, /) -> None: ...
    def remove_edges_from(self, index_list: Iterable[tuple[int, int]], /) -> None: ...
    def remove_node(self, node: int, /) -> None: ...
    def remove_nodes_from(self, index_list: Iterable[int], /) -> None: ...
    def subgraph(self, nodes: Sequence[int], /, preserve_attrs: bool = ...) -> PyGraph[_S, _T]: ...
    def substitute_node_with_subgraph(
        self,
        node: int,
        other: PyGraph[_S, _T],
        edge_map_fn: Callable[[int, int, _T], int | None],
        /,
        node_filter: Callable[[_S], bool] | None = ...,
        edge_weight_map: Callable[[_T], _T] | None = ...,
    ) -> NodeMap: ...
    def to_dot(
        self,
        /,
        node_attr: Callable[[_S], dict[str, str]] | None = ...,
        edge_attr: Callable[[_T], dict[str, str]] | None = ...,
        graph_attr: dict[str, str] | None = ...,
        filename: str | None = None,
    ) -> str | None: ...
    def to_directed(self) -> PyDiGraph[_S, _T]: ...
    def update_edge(
        self,
        source: int,
        target: int,
        edge: _T,
        /,
    ) -> None: ...
    def update_edge_by_index(self, edge_index: int, edge: _T, /) -> None: ...
    def weighted_edge_list(self) -> WeightedEdgeList[_T]: ...
    def write_edge_list(
        self,
        path: str,
        /,
        deliminator: str | None = ...,
        weight_fn: Callable[[_T], str] | None = ...,
    ) -> None: ...
    def __delitem__(self, idx: int, /) -> None: ...
    def __getitem__(self, idx: int, /) -> _S: ...
    @classmethod
    def __class_getitem__(cls, key: Any, /) -> GenericAlias: ...
    def __getnewargs_ex__(self) -> tuple[tuple[Any, ...], dict[str, Any]]: ...
    def __getstate__(self) -> Any: ...
    def __len__(self) -> int: ...
    def __setitem__(self, idx: int, value: _S, /) -> None: ...
    def __setstate__(self, state: Any, /) -> None: ...

# Digraph

class PyDiGraph(Generic[_S, _T]):
    attrs: Any
    check_cycle: bool = ...
    multigraph: bool = ...
    def __init__(
        self,
        /,
        check_cycle: bool = ...,
        multigraph: bool = ...,
        attrs: Any = ...,
        *,
        node_count_hint: int | None = ...,
        edge_count_hint: int | None = ...,
    ) -> None: ...
    def add_child(self, parent: int, obj: _S, edge: _T, /) -> int: ...
    def add_edge(self, parent: int, child: int, edge: _T, /) -> int: ...
    def add_edges_from(
        self,
        obj_list: Iterable[tuple[int, int, _T]],
        /,
    ) -> list[int]: ...
    def add_edges_from_no_data(
        self: PyDiGraph[_S, _T | None], obj_list: Iterable[tuple[int, int]], /
    ) -> list[int]: ...
    def add_node(self, obj: _S, /) -> int: ...
    def add_nodes_from(self, obj_list: Iterable[_S], /) -> NodeIndices: ...
    def add_parent(self, child: int, obj: _S, edge: _T, /) -> int: ...
    def adj(self, node: int, /) -> dict[int, _T]: ...
    def adj_direction(self, node: int, direction: bool, /) -> dict[int, _T]: ...
    def clear(self) -> None: ...
    def clear_edges(self) -> None: ...
    def compose(
        self,
        other: PyDiGraph[_S, _T],
        node_map: dict[int, tuple[int, _T]],
        /,
        node_map_func: Callable[[_S], int] | None = ...,
        edge_map_func: Callable[[_T], int] | None = ...,
    ) -> dict[int, int]: ...
    def contract_nodes(
        self,
        nodes: Sequence[int],
        obj: _S,
        /,
        check_cycle: bool | None = ...,
        weight_combo_fn: Callable[[_T, _T], _T] | None = ...,
    ) -> int: ...
    def copy(self) -> PyDiGraph[_S, _T]: ...
    def edge_index_map(self) -> EdgeIndexMap[_T]: ...
    def edge_indices(self) -> EdgeIndices: ...
    def edge_indices_from_endpoints(self, node_a: int, node_b: int) -> EdgeIndices: ...
    def edge_list(self) -> EdgeList: ...
    def edges(self) -> list[_T]: ...
    def edge_subgraph(self, edge_list: Sequence[tuple[int, int]], /) -> PyDiGraph[_S, _T]: ...
    def extend_from_edge_list(
        self: PyDiGraph[_S | None, _T | None], edge_list: Iterable[tuple[int, int]], /
    ) -> None: ...
    def extend_from_weighted_edge_list(
        self: PyDiGraph[_S | None, _T],
        edge_list: Iterable[tuple[int, int, _T]],
        /,
    ) -> None: ...
    def filter_edges(self, filter_function: Callable[[_T], bool]) -> EdgeIndices: ...
    def filter_nodes(self, filter_function: Callable[[_S], bool]) -> NodeIndices: ...
    def find_adjacent_node_by_edge(self, node: int, predicate: Callable[[_T], bool], /) -> _S: ...
    def find_node_by_weight(
        self,
        obj: _S,
        /,
    ) -> int | None: ...
    def find_predecessors_by_edge(
        self, node: int, filter_fn: Callable[[_T], bool], /
    ) -> list[_S]: ...
    def find_predecessor_node_by_edge(
        self, node: int, predicate: Callable[[_T], bool], /
    ) -> _S: ...
    def find_successors_by_edge(
        self, node: int, filter_fn: Callable[[_T], bool], /
    ) -> list[_S]: ...
    @staticmethod
    def from_adjacency_matrix(
        matrix: npt.NDArray[np.float64], /, null_value: float = ...
    ) -> PyDiGraph[int, float]: ...
    @staticmethod
    def from_complex_adjacency_matrix(
        matrix: npt.NDArray[np.complex64], /, null_value: complex = ...
    ) -> PyDiGraph[int, complex]: ...
    def get_all_edge_data(self, node_a: int, node_b: int, /) -> list[_T]: ...
    def get_edge_data(self, node_a: int, node_b: int, /) -> _T: ...
    def get_node_data(self, node: int, /) -> _S: ...
    def get_edge_data_by_index(self, edge_index: int, /) -> _T: ...
    def get_edge_endpoints_by_index(self, edge_index: int, /) -> tuple[int, int]: ...
    def has_node(self, node: int, /) -> bool: ...
    def has_edge(self, node_a: int, node_b: int, /) -> bool: ...
    def has_parallel_edges(self) -> bool: ...
    def in_degree(self, node: int, /) -> int: ...
    def in_edges(self, node: int, /) -> WeightedEdgeList[_T]: ...
    def incident_edge_index_map(self, node: int, /, all_edges: bool = ...) -> EdgeIndexMap: ...
    def incident_edges(self, node: int, /, all_edges: bool = ...) -> EdgeIndices: ...
    def insert_node_on_in_edges(self, node: int, ref_node: int, /) -> None: ...
    def insert_node_on_in_edges_multiple(self, node: int, ref_nodes: Sequence[int], /) -> None: ...
    def insert_node_on_out_edges(self, node: int, ref_node: int, /) -> None: ...
    def insert_node_on_out_edges_multiple(self, node: int, ref_nodes: Sequence[int], /) -> None: ...
    def is_symmetric(self) -> bool: ...
    def make_symmetric(self, edge_payload_fn: Callable[[_T], _T] | None = ...) -> None: ...
    def merge_nodes(self, u: int, v: int, /) -> None: ...
    def neighbors(self, node: int, /) -> NodeIndices: ...
    def neighbors_undirected(self, node: int, /) -> NodeIndices: ...
    def node_indexes(self) -> NodeIndices: ...
    def node_indices(self) -> NodeIndices: ...
    def nodes(self) -> list[_S]: ...
    def num_edges(self) -> int: ...
    def num_nodes(self) -> int: ...
    def out_degree(self, node: int, /) -> int: ...
    def out_edges(self, node: int, /) -> WeightedEdgeList[_T]: ...
    def predecessor_indices(self, node: int, /) -> NodeIndices: ...
    def predecessors(self, node: int, /) -> list[_S]: ...
    @staticmethod
    def read_edge_list(
        path: str,
        /,
        comment: str | None = ...,
        deliminator: str | None = ...,
        labels: bool = ...,
    ) -> PyDiGraph: ...
    def remove_edge(self, parent: int, child: int, /) -> None: ...
    def remove_edge_from_index(self, edge: int, /) -> None: ...
    def remove_edges_from(self, index_list: Iterable[tuple[int, int]], /) -> None: ...
    def remove_node(self, node: int, /) -> None: ...
    def remove_node_retain_edges(
        self,
        node: int,
        /,
        use_outgoing: bool = ...,
        condition: Callable[[_T, _T], bool] | None = ...,
    ) -> None: ...
    def remove_node_retain_edges_by_id(self, node: int, /) -> None: ...
    def remove_node_retain_edges_by_key(
        self,
        node: int,
        /,
        key: Callable[[_T], Hashable] | None = ...,
        *,
        use_outgoing: bool = ...,
    ) -> None: ...
    def remove_nodes_from(self, index_list: Iterable[int], /) -> None: ...
    def subgraph(
        self, nodes: Sequence[int], /, preserve_attrs: bool = ...
    ) -> PyDiGraph[_S, _T]: ...
    def substitute_node_with_subgraph(
        self,
        node: int,
        other: PyDiGraph[_S, _T],
        edge_map_fn: Callable[[int, int, _T], int | None],
        /,
        node_filter: Callable[[_S], bool] | None = ...,
        edge_weight_map: Callable[[_T], _T] | None = ...,
    ) -> NodeMap: ...
    def successor_indices(self, node: int, /) -> NodeIndices: ...
    def successors(self, node: int, /) -> list[_S]: ...
    def to_dot(
        self,
        /,
        node_attr: Callable[[_S], dict[str, str]] | None = ...,
        edge_attr: Callable[[_T], dict[str, str]] | None = ...,
        graph_attr: dict[str, str] | None = ...,
        filename: str | None = None,
    ) -> str | None: ...
    def to_undirected(
        self,
        /,
        multigraph: bool = ...,
        weight_combo_fn: Callable[[_T, _T], _T] | None = ...,
    ) -> PyGraph[_S, _T]: ...
    def update_edge(
        self,
        source: int,
        target: int,
        edge: _T,
        /,
    ) -> None: ...
    def update_edge_by_index(self, edge_index: int, edge: _T, /) -> None: ...
    def weighted_edge_list(self) -> WeightedEdgeList[_T]: ...
    def write_edge_list(
        self,
        path: str,
        /,
        deliminator: str | None = ...,
        weight_fn: Callable[[_T], str] | None = ...,
    ) -> None: ...
    def reverse(self) -> None: ...
    def __delitem__(self, idx: int, /) -> None: ...
    def __getitem__(self, idx: int, /) -> _S: ...
    @classmethod
    def __class_getitem__(cls, key: Any, /) -> GenericAlias: ...
    def __getnewargs_ex__(self) -> tuple[tuple[Any, ...], dict[str, Any]]: ...
    def __getstate__(self) -> Any: ...
    def __len__(self) -> int: ...
    def __setitem__(self, idx: int, value: _S, /) -> None: ...
    def __setstate__(self, state: Any, /) -> None: ...
