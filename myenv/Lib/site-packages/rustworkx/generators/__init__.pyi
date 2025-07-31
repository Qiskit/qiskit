# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file contains only type annotations for PyO3 functions and classes
# For implementation details, see __init__.py and src/shortest_path/mod.rs

from rustworkx import PyGraph
from rustworkx import PyDiGraph

from typing import Any
from collections.abc import Sequence

def cycle_graph(
    num_nodes: int | None = ..., weights: Sequence[Any] | None = ..., multigraph: bool = ...
) -> PyGraph: ...
def directed_cycle_graph(
    num_nodes: int | None = ...,
    weights: Sequence[Any] | None = ...,
    bidirectional: bool = ...,
    multigraph: bool = ...,
) -> PyDiGraph: ...
def path_graph(
    num_nodes: int | None = ..., weights: Sequence[Any] | None = ..., multigraph: bool = ...
) -> PyGraph: ...
def directed_path_graph(
    num_nodes: int | None = ...,
    weights: Sequence[Any] | None = ...,
    bidirectional: bool = ...,
    multigraph: bool = ...,
) -> PyDiGraph: ...
def star_graph(
    num_nodes: int | None = ..., weights: Sequence[Any] | None = ..., multigraph: bool = ...
) -> PyGraph: ...
def directed_star_graph(
    num_nodes: int | None = ...,
    weights: Sequence[Any] | None = ...,
    inward: bool = ...,
    bidirectional: bool = ...,
    multigraph: bool = ...,
) -> PyDiGraph: ...
def mesh_graph(
    num_nodes: int | None = ..., weights: Sequence[Any] | None = ..., multigraph: bool = ...
) -> PyGraph: ...
def directed_mesh_graph(
    num_nodes: int | None = ...,
    weights: Sequence[Any] | None = ...,
    multigraph: bool = ...,
) -> PyDiGraph: ...
def grid_graph(
    rows: int | None = ...,
    cols: int | None = ...,
    weights: Sequence[Any] | None = ...,
    multigraph: bool = ...,
) -> PyGraph: ...
def directed_grid_graph(
    rows: int | None = ...,
    cols: int | None = ...,
    weights: Sequence[Any] | None = ...,
    bidirectional: bool = ...,
    multigraph: bool = ...,
) -> PyDiGraph: ...
def heavy_square_graph(d: int, multigraph: bool = ...) -> PyGraph: ...
def directed_heavy_square_graph(
    d: int,
    bidirectional: bool = ...,
    multigraph: bool = ...,
) -> PyDiGraph: ...
def heavy_hex_graph(d: int, multigraph: bool = ...) -> PyGraph: ...
def directed_heavy_hex_graph(
    d: int,
    bidirectional: bool = ...,
    multigraph: bool = ...,
) -> PyDiGraph: ...
def binomial_tree_graph(
    order: int, weights: Sequence[Any] | None = ..., multigraph: bool = ...
) -> PyGraph: ...
def directed_binomial_tree_graph(
    order: int,
    weights: Sequence[Any] | None = ...,
    bidirectional: bool = ...,
    multigraph: bool = ...,
) -> PyDiGraph: ...
def full_rary_tree(
    branching_factor: int,
    num_nodes: int,
    weights: Sequence[Any] | None = ...,
    multigraph: bool = ...,
) -> PyGraph: ...
def hexagonal_lattice_graph(
    rows: int, cols: int, multigraph: bool = ..., periodic: bool = ..., with_positions: bool = ...
) -> PyGraph: ...
def directed_hexagonal_lattice_graph(
    rows: int,
    cols: int,
    bidirectional: bool = ...,
    multigraph: bool = ...,
    periodic: bool = ...,
    with_positions: bool = ...,
) -> PyDiGraph: ...
def lollipop_graph(
    num_mesh_nodes: int | None = ...,
    num_path_nodes: int | None = ...,
    mesh_weights: Sequence[Any] | None = ...,
    path_weights: Sequence[Any] | None = ...,
    multigraph: bool = ...,
) -> PyGraph: ...
def barbell_graph(
    num_mesh_nodes: int | None = ...,
    num_path_nodes: int | None = ...,
    multigraph: bool = ...,
    mesh_weights: Sequence[Any] | None = ...,
    path_weights: Sequence[Any] | None = ...,
) -> PyGraph: ...
def generalized_petersen_graph(
    n: int,
    k: int,
    multigraph: bool = ...,
) -> PyGraph: ...
def empty_graph(n: int, multigraph: bool = ...) -> PyGraph: ...
def directed_empty_graph(n: int, multigraph: bool = ...) -> PyDiGraph: ...
def complete_graph(
    num_nodes: int | None = ..., weights: Sequence[Any] | None = ..., multigraph: bool = ...
) -> PyGraph: ...
def directed_complete_graph(
    num_nodes: int | None = ...,
    weights: Sequence[Any] | None = ...,
    multigraph: bool = ...,
) -> PyDiGraph: ...
def dorogovtsev_goltsev_mendes_graph(n: int) -> PyGraph: ...
def karate_club_graph(multigraph: bool = ...) -> PyGraph: ...
