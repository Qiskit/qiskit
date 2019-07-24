"""Implementations for permuting on a cartesian graph."""
#  arct performs circuit transformations of quantum circuit for architectures
#  Copyright (C) 2019  Andrew M. Childs, Eddie Schoute, Cem M. Unsal
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import itertools
import logging
import random
from collections import defaultdict
from typing import List, Dict, Tuple, TypeVar, Callable, Iterator, Iterable, Mapping, NamedTuple, \
    Union, Any
import networkx as nx

from qiskit.transpiler.routing import util, Permutation, Swap, fast_path

_V = TypeVar('_V')
PERMUTER = Callable[[Permutation[int]], Iterable[List[Swap[int]]]]
PARTIAL_PERMUTER = Callable[[Mapping[int, int]], Iterable[List[Swap[int]]]]

logger = logging.getLogger(__name__)


class Point(NamedTuple):
    """A node in the cartesian product of graphs."""
    x: int
    y: int

    @staticmethod
    def from_int(identifier: int, width: int) -> 'Point':
        x = identifier % width
        y = identifier // width
        return Point(x=x, y=y)

    def to_int(self, width: int) -> int:
        return self.x + width * self.y


class UnmappedQubit:
    """A qubit that has not been mapped by the mapping."""

    def __repr__(self) -> str:
        """A string representation of the UnmappedQubit."""
        return "UnmappedQubit()"


def permute_cartesian_partial(mapping: Mapping[int, int],
                              width: int,
                              height: int,
                              permute_x: PARTIAL_PERMUTER,
                              permute_y: PARTIAL_PERMUTER,
                              trials: int = 1) \
        -> List[List[Swap[int]]]:
    """Compute swaps that implement a mapping on a cartesian product of graphs.

    Based on the paper "Routing Permutations on Graphs via Matchings"
    by Noga Alon, F. R. K. Chung, and R. L. Graham,
    DOI: https://doi.org/10.1137/S0895480192236628,
    and "Circuit Transformations for Quantum Architectures"
    by A.M. Childs, E. Schoute, C.M. Unsal,
    DOI: https://doi.org/10.4230/LIPIcs.TQC.2019.3

    :param mapping: A mapping of origin to destination nodes.
    :param width: Length of x graph
    :param height: Length of y graph
    :param permute_x: Function to permute in the x direction
    :param permute_y: function to permute in y direction
    :param trials: Retry doing the permutation this many times, and take the best solution.
    :return: A list describing which matchings to swap at each step.
    """
    if height == 0 or width == 0:
        if mapping:
            # Mapping not empty
            raise ValueError("The mapping is not empty, but the graph has a zero-length axis:"
                             f"width={width}, height={height}")
        else:
            return []

    trial_results = iter(list(_partial_cartesian_trial(mapping, width, height,
                                                       permute_x, permute_y))
                     for _ in range(trials))

    # Once we find a zero solution we stop.
    def take_until_zero(results: Iterable[List[_V]]) -> Iterator[List[_V]]:
        """Take results until one is emitted of length zero (and also emit that)."""
        for result in results:
            if len(result) > 0:
                yield result
            else:
                yield result
                break

    trial_results = take_until_zero(trial_results)
    return min(trial_results, key=util.longest_path)


def _partial_cartesian_trial(mapping: Mapping[int, int],
                             width: int,
                             height: int,
                             permute_x: PARTIAL_PERMUTER,
                             permute_y: PARTIAL_PERMUTER) -> Iterator[List[Swap[int]]]:
    # Reshape mapping to Point(x,y) mapping
    dest = {Point.from_int(orig, width): Point.from_int(dest, width)
            for orig, dest in mapping.items()}

    def right_node(vertex: int) -> int:
        """Compute the node index on the right side of bipartite graph."""
        return width + vertex

    epsilon = 1 / (height ** 2)
    # Each row has a mapping from y coordinates to y coordinates.
    current_mappings: List[Mapping[int, int]] = [dict() for x in range(width)]

    def cost(point: Point,
             destination: Point,
             y: int) -> float:
        """Compute overhead cost induced by routing to this row.

        In the case of an unmapped qubit, the cost is a positive ε value.

        Otherwise, the overhead cost is the cost of implementing the current mapping
        plus the mapping of point.y to y to destination.y,
        minus the cost of just routing directly from point.y to destination.y
        and the cost of implementing the current mapping

        :param point: The origin point.
        :param destination: The destination point
        :param y: The current row that is being matched.
        :return: A float that could be negative if cost was actually saved
            over the starting position (by the current mapping).
        """
        current_mapping = current_mappings[point.x]
        return _it_len(permute_y({**current_mapping, **{point.y: y}})) \
               - _it_len(permute_y(current_mapping)) \
               + _it_len(permute_y({y: destination.y})) \
               - _it_len(permute_y({point.y: destination.y}))

    unmapped_qubit = UnmappedQubit()
    # For each column, keep track of the number of unmapped qubits.
    unmapped_qubits = {
        x: height - sum(1 for origin in dest if origin.x == x)
        for x in range(width)}
    # Iterate over rows in random order.
    remaining_rows = list(range(height))
    random.shuffle(remaining_rows)
    remaining_destinations = dest.copy()
    while remaining_rows:
        row = remaining_rows.pop()

        destination_graph = nx.MultiGraph()
        destination_graph.add_nodes_from(range(width), bipartite=0)
        destination_graph.add_nodes_from((right_node(node) for node in range(width)), bipartite=1)

        # Add edges for the "real" mapped qubits
        for origin, destination in remaining_destinations.items():
            c = cost(origin, destination, row)
            destination_graph.add_edge(origin.x, right_node(destination.x), weight=c, origin=origin)

        # Add connectivity with weight epsilon for each column with an unmapped qubit.
        # Only add edges to columns where the in-degree is less
        # than the remaining "slots" (nr of rows remaining) in the column.
        unmapped_graph = nx.Graph()
        for out_col in range(width):
            if unmapped_qubits[out_col] == 0:
                continue
            for in_col in (right_node(v) for v in range(width)):
                if destination_graph.degree(in_col) < len(remaining_rows) + 1:
                    unmapped_graph.add_edge(out_col, in_col, weight=epsilon, origin=unmapped_qubit)

        # Merge destination graph and unmapped graph to weighted simple graph
        # with the minimum weight on each edge
        simple_graph = nx.Graph()
        for graph in (destination_graph, unmapped_graph):
            for e0, e1, data in graph.edges(data=True):
                weight = data["weight"]
                if not(simple_graph.has_edge(e0, e1)) or weight < simple_graph[e0][e1]["weight"]:
                    simple_graph.add_edge(e0, e1, **data)

        # Convert "costs" that need to be minimized to "weights" that need to be maximized.
        # We shrink the range of weights to (-1, 1) by dividing by the absolute maximum cost size+1
        max_cost = max(abs(v[2]) for v in simple_graph.edges(data='weight'))
        if max_cost == 0:  # Avoid divide by zero
            # All costs are 0; set all weights to 1.
            for e0, e1 in simple_graph.edges:
                simple_graph[e0][e1]["weight"] = 1
        else:
            # High cost means low weight. But it still needs to be positive or it won't be included.
            # Furthermore, we need to guarantee the maximum weighted matching is perfect.
            # Therefore, we divide by height, so that no height-1 weighted edges are larger
            # than height in weight.
            for e0, e1 in simple_graph.edges:
                weight = simple_graph[e0][e1]["weight"]
                # Calculate the new weight and overwrite the old weight
                simple_graph[e0][e1]["weight"] = (-weight / (max_cost + 1) + 1) / (2 * height) + 1

        matching = nx.algorithms.max_weight_matching(simple_graph, maxcardinality=True)
        assert len(matching) == width, "The matching is not perfect."
        for e0, e1 in matching:
            origin = simple_graph[e0][e1]["origin"]
            if isinstance(origin, UnmappedQubit):
                # Sort e0 and e1 since e0 is on the "left" side of the bipartition
                # and is used for indexing unmapped_qubits.
                e0, e1 = sorted((e0, e1))
                # We dont map unmapped qubits, but now one less is available in the column
                unmapped_qubits[e0] -= 1
            else:
                current_mappings[origin.x][origin.y] = row
                del remaining_destinations[origin]

        # No qubits left to route
        if not remaining_destinations:
            break
    # IDEA: Make the swap steps lazy.
    # This is slightly harder than it looks because `dest` needs to be kept up-to-date
    # and util.swap_permutation consumes the iterable.
    swaps1_parallel = [
        # For each column we compute the translated list of swaps
        [[(Point(x, node_0), Point(x, node_1)) for node_0, node_1 in time_step]
         for time_step in permute_y(current_mappings[x])]
        for x in range(width)
        ]
    # Then we merge the parallel swaps together in their respective time steps.
    swaps1 = list(util.flatten_swaps(swaps1_parallel))
    util.swap_permutation(swaps1, dest, allow_missing_keys=True)
    # Construct, for each row, a partial mapping to the destination columns
    row_mappings: List[Dict[int, int]] = [dict() for y in range(height)]
    for origin, destination in dest.items():
        row_mappings[origin.y][origin.x] = destination.x
    swaps2_paralllel = [
        # For each row we compute the translated list of swaps
        [[(Point(node_0, y), Point(node_1, y)) for node_0, node_1 in time_step]
         # We make permutation dicts for each row
         for time_step in permute_x(row_mappings[y])]
        for y in range(height)
        ]
    # And merge them together into time steps.
    swaps2 = list(util.flatten_swaps(swaps2_paralllel))
    util.swap_permutation(swaps2, dest, allow_missing_keys=True)
    # Construct, for each column, a partial mapping to the destination rows
    column_mappings: List[Dict[int, int]] = [dict() for x in range(width)]
    for origin, destination in dest.items():
        column_mappings[origin.x][origin.y] = destination.y
    swaps3_parallel = [
        # For each column we compute the translated list of swaps
        [[(Point(x, node_0), Point(x, node_1)) for node_0, node_1 in time_step]
         # We make permutation dicts for each column
         for time_step in permute_y(column_mappings[x])]
        for x in range(width)
        ]
    swaps3 = util.flatten_swaps(swaps3_parallel)
    all_swaps = itertools.chain(swaps1, swaps2, swaps3)
    # Map back to integers
    return ([(point0.to_int(width), point1.to_int(width)) for point0, point1 in time_step]
            for time_step in util.optimize_swaps(all_swaps))


def permute_grid(permutation: Permutation[int], height: int) \
        -> Iterator[List[Swap[int]]]:
    """List swaps that implement a permutation on a rectangle graph.
    Rectangle is a cartesian product of two paths.

    Based on the paper "Routing Permutations on Graphs via Matchings"
    by Noga Alon, F. R. K. Chung, and R. L. Graham,
    DOI: https://doi.org/10.1137/S0895480192236628

    :param permutation: A list of destination nodes
    :param height: Length of y graph
    :return: A list describing which matchings to swap at each step.
    """
    return permute_grid_partial(permutation, height, height, trials=1)


def permute_grid_partial(mapping: Mapping[int, int], width: int, height: int,
                         trials: int = 1) \
        -> List[List[Swap[int]]]:
    """List swaps that implement a permutation on a rectangle graph.
    Rectangle is a cartesian product of two paths.

    Based on the paper "Routing Permutations on Graphs via Matchings"
    by Noga Alon, F. R. K. Chung, and R. L. Graham,
    DOI: https://doi.org/10.1137/S0895480192236628

    :param mapping: A (partial) mapping of origin to destination nodes.
    :param height: Length of y graph.
    :param width: Length of the x graph.
    :param trials: Retry doing the permutation this many times, and take the best solution.
    :return: A list describing which matchings to swap at each step.
    """
    permute_x = lambda m: fast_path.permute_path_partial(m, length=width)
    permute_y = lambda m: fast_path.permute_path_partial(m, length=height)
    return permute_cartesian_partial(mapping, width, height, permute_x, permute_y, trials=trials)


def _it_len(it: Iterable[Any]) -> int:
    """Iterate through the iterable and get its length.

    WARNING: May not halt for infinite iterables!
    """
    return sum(1 for _ in it)
