# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Copyright 2019 Andrew M. Childs, Eddie Schoute, Cem M. Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementations for permuting on a cartesian graph. """

import itertools
import logging
import random
from typing import List, Dict, TypeVar, Callable, Iterator, Iterable, Mapping, Any

import networkx as nx

from qiskit.transpiler.routing import util, Permutation, Swap, fast_path

_V = TypeVar('_V')
PartialPermuter = Callable[[Mapping[int, int]], Iterable[List[Swap[int]]]]

LOGGER = logging.getLogger(__name__)


class Point:
    """A node in the cartesian product of graphs.

        Was originally implemented as subclass of NamedTuple (python 3.6+ only)."""

    def __init__(self, x, y):
        """Construct a Point object."""
        self.x = x
        self.y = y

    def __repr__(self):
        return "Point({},{})".format(self.x, self.y)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return other.x == self.x and other.y == self.y
        return False

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    @staticmethod
    def from_int(identifier: int, width: int) -> 'Point':
        """Convert an id and graph width to a Point"""
        x = identifier % width
        y = identifier // width
        return Point(x=x, y=y)

    def to_int(self, width: int) -> int:
        """Convert the Point back to an id"""
        return self.x + width * self.y


class UnmappedQubit:
    """A qubit that has not been mapped by the mapping."""

    def __repr__(self) -> str:
        """A string representation of the UnmappedQubit."""
        return "UnmappedQubit()"


def permute_cartesian_partial(mapping: Mapping[int, int],
                              width: int,
                              height: int,
                              permute_x: PartialPermuter,
                              permute_y: PartialPermuter,
                              trials: int = 1) \
        -> List[List[Swap[int]]]:
    """Compute swaps that implement a mapping on a cartesian product of graphs.

    Based on the paper "Routing Permutations on Graphs via Matchings"
    by Noga Alon, F. R. K. Chung, and R. L. Graham,
    DOI: https://doi.org/10.1137/S0895480192236628,
    and "Circuit Transformations for Quantum Architectures"
    by A.M. Childs, E. Schoute, C.M. Unsal,
    DOI: https://doi.org/10.4230/LIPIcs.TQC.2019.3

    Args:
      mapping: A mapping of origin to destination nodes.
      width: Length of x graph
      height: Length of y graph
      permute_x: Function to permute in the x direction
      permute_y: function to permute in y direction
      trials: Retry doing the permutation this many times, and take the best solution.

    Returns:
      A list describing which matchings to swap at each step.

    Raises:
        ValueError: If an axis in the graph has zero vertices.

    """
    if height == 0 or width == 0:
        if mapping:
            # Mapping not empty
            raise ValueError("The mapping is not empty, but the graph has a zero-length axis:"
                             "width=" + str(width) + ", height=" + str(height))
        return []

    trial_results = iter(list(_partial_cartesian_trial(mapping, width, height,
                                                       permute_x, permute_y))
                         for _ in range(trials))

    # Once we find a zero solution we stop.
    def take_until_zero(results: Iterable[List[_V]]) -> Iterator[List[_V]]:
        """Take results until one is emitted of length zero (and also emit that)."""
        for result in results:
            if not result:
                yield result
                break
            else:
                yield result

    trial_results = take_until_zero(trial_results)
    return min(trial_results, key=util.longest_path)


def _partial_cartesian_trial(mapping: Mapping[int, int],
                             width: int,
                             height: int,
                             permute_x: PartialPermuter,
                             permute_y: PartialPermuter) -> Iterator[List[Swap[int]]]:
    # Reshape mapping to Point(x,y) mapping
    dest = {Point.from_int(orig, width): Point.from_int(dest, width)
            for orig, dest in mapping.items()}

    def right_node(vertex: int) -> int:
        """Compute the node index on the right side of bipartite graph."""
        return width + vertex

    epsilon = 1 / (height ** 2)
    # Each row has a mapping from y coordinates to y coordinates.
    current_mappings = [dict() for x in range(width)]  # type: List[Mapping[int, int]]

    def cost(point: Point,
             destination: Point,
             y: int) -> float:
        """Compute overhead cost induced by routing to this row.

        In the case of an unmapped qubit, the cost is a positive ε value.

        Otherwise, the overhead cost is the cost of implementing the current mapping
        plus the mapping of point.y to y to destination.y,
        minus the cost of just routing directly from point.y to destination.y
        and the cost of implementing the current mapping

        Args:
          point: The origin point.
          destination: The destination point
          y: The current row that is being matched.

        Returns:
          A float that could be negative if cost was actually saved
          over the starting position (by the current mapping).

        """
        current_mapping = current_mappings[point.x]
        return _it_len(permute_y({**current_mapping, **{point.y: y}})) - \
            _it_len(permute_y(current_mapping)) + \
            _it_len(permute_y({y: destination.y})) - \
            _it_len(permute_y({point.y: destination.y}))

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
            for edge0, edge1, data in graph.edges(data=True):
                weight = data["weight"]
                if not (simple_graph.has_edge(edge0, edge1))\
                        or weight < simple_graph[edge0][edge1]["weight"]:
                    simple_graph.add_edge(edge0, edge1, **data)

        # Convert "costs" that need to be minimized to "weights" that need to be maximized.
        # We shrink the range of weights to (-1, 1) by dividing by the absolute maximum cost size+1
        max_cost = max(abs(v[2]) for v in simple_graph.edges(data='weight'))
        if max_cost == 0:  # Avoid divide by zero
            # All costs are 0; set all weights to 1.
            for edge0, edge1 in simple_graph.edges:
                simple_graph[edge0][edge1]["weight"] = 1
        else:
            # High cost means low weight. But it still needs to be positive or it won't be included.
            # Furthermore, we need to guarantee the maximum weighted matching is perfect.
            # Therefore, we divide by height, so that no height-1 weighted edges are larger
            # than height in weight.
            for edge0, edge1 in simple_graph.edges:
                weight = simple_graph[edge0][edge1]["weight"]
                # Calculate the new weight and overwrite the old weight
                simple_graph[edge0][edge1]["weight"] = \
                    (-weight / (max_cost + 1) + 1) / (2 * height) + 1

        matching = nx.algorithms.max_weight_matching(simple_graph, maxcardinality=True)
        if len(matching) != width:
            LOGGER.warning("Routing internal error: The matching was not perfect.")

        for edge0, edge1 in matching:
            origin = simple_graph[edge0][edge1]["origin"]
            if isinstance(origin, UnmappedQubit):
                # Sort edge0 and edge1 since edge0 is on the "left" side of the bipartition
                # and is used for indexing unmapped_qubits.
                edge0, edge1 = sorted((edge0, edge1))
                # We dont map unmapped qubits, but now one less is available in the column
                unmapped_qubits[edge0] -= 1
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
    row_mappings = [dict() for y in range(height)]  # type: List[Dict[int, int]]
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
    column_mappings = [dict() for x in range(width)]  # type: List[Dict[int, int]]
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

    Args:
      permutation: A list of destination nodes
      height: Length of y graph

    Returns:
      A list describing which matchings to swap at each step.

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

    Args:
      mapping: A (partial) mapping of origin to destination nodes.
      height: Length of y graph.
      width: Length of the x graph.
      trials: Retry doing the permutation this many times, and take the best solution.

    Returns:
      A list describing which matchings to swap at each step.

    """
    return permute_cartesian_partial(mapping, width, height,
                                     lambda m: fast_path.permute_path_partial(m, length=width),
                                     lambda m: fast_path.permute_path_partial(m, length=height),
                                     trials=trials)


def _it_len(iterable: Iterable[Any]) -> int:
    """Iterate through the iterable and get its length.

    WARNING: May not halt for infinite iterables!"""
    return sum(1 for _ in iterable)
