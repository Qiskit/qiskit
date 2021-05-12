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

"""Permutation algorithms for general graphs."""

import copy
import logging
from typing import Iterator, Mapping, MutableMapping, MutableSet, List, Iterable, Union

import numpy as np
import retworkx as rx

from .types import Swap, Permutation
from .util import PermutationCircuit, permutation_circuit


logger = logging.getLogger(__name__)


class ApproximateTokenSwapper:
    """A class for computing approximate solutions to the Token Swapping problem.

    Internally caches the graph and associated datastructures for re-use.
    """

    def __init__(
        self, graph: rx.PyGraph, seed: Union[int, np.random.Generator, None] = None
    ) -> None:
        """Construct an ApproximateTokenSwapping object.

        Args:
            graph: Undirected graph represented a coupling map.
            seed: Seed to use for random trials.
        """
        self.graph = graph
        self.shortest_paths = rx.graph_distance_matrix(graph)
        if isinstance(seed, np.random.Generator):
            self.seed = seed
        else:
            self.seed = np.random.default_rng(seed)

    def distance(self, vertex0: int, vertex1: int) -> int:
        """Compute the distance between two nodes in `graph`."""
        return self.shortest_paths[vertex0, vertex1]

    def permutation_circuit(self, permutation: Permutation, trials: int = 4) -> PermutationCircuit:
        """Perform an approximately optimal Token Swapping algorithm to implement the permutation.

        Args:
          permutation: The partial mapping to implement in swaps.
          trials: The number of trials to try to perform the mapping. Minimize over the trials.

        Returns:
          The circuit to implement the permutation
        """
        sequential_swaps = self.map(permutation, trials=trials)
        parallel_swaps = [[swap] for swap in sequential_swaps]
        return permutation_circuit(parallel_swaps)

    def map(self, mapping: Mapping[int, int], trials: int = 4) -> List[Swap[int]]:
        """Perform an approximately optimal Token Swapping algorithm to implement the permutation.

        Supports partial mappings (i.e. not-permutations) for graphs with missing tokens.

        Based on the paper: Approximation and Hardness for Token Swapping by Miltzow et al. (2016)
        ArXiV: https://arxiv.org/abs/1602.05150
        and generalization based on our own work.

        Args:
          mapping: The partial mapping to implement in swaps.
          trials: The number of trials to try to perform the mapping. Minimize over the trials.

        Returns:
          The swaps to implement the mapping
        """
        tokens = dict(mapping)
        digraph = rx.PyDiGraph()
        sub_digraph = rx.PyDiGraph()  # Excludes self-loops in digraph.
        todo_nodes = {node for node, destination in tokens.items() if node != destination}
        for node in self.graph.node_indexes():
            self._add_token_edges(node, tokens, digraph, sub_digraph)

        trial_results = iter(
            list(
                self._trial_map(
                    copy.copy(digraph), copy.copy(sub_digraph), todo_nodes.copy(), tokens.copy()
                )
            )
            for _ in range(trials)
        )

        # Once we find a zero solution we stop.
        def take_until_zero(results: Iterable[List[int]]) -> Iterator[List[int]]:
            """Take results until one is emitted of length zero (and also emit that)."""
            for result in results:
                yield result
                if not result:
                    break

        trial_results = take_until_zero(trial_results)
        return min(trial_results, key=len)

    def _trial_map(
        self,
        digraph: rx.PyDiGraph,
        sub_digraph: rx.PyDiGraph,
        todo_nodes: MutableSet[int],
        tokens: MutableMapping[int, int],
    ) -> Iterator[Swap[int]]:
        """Try to map the tokens to their destinations and minimize the number of swaps."""

        def swap(node0: int, node1: int) -> None:
            """Swap two nodes, maintaining data structures.

            Args:
              node0: The first node
              node1: The second node
            """
            self._swap(node0, node1, tokens, digraph, sub_digraph, todo_nodes)

        # Can't just iterate over todo_nodes, since it may change during iteration.
        steps = 0
        while todo_nodes and steps <= 4 * len(self.graph) ** 2:
            todo_node_id = self.seed.integers(0, len(todo_nodes))
            todo_node = tuple(todo_nodes)[todo_node_id]

            # Try to find a happy swap chain first by searching for a cycle,
            # excluding self-loops.
            # Note that if there are only unhappy swaps involving this todo_node,
            # then an unhappy swap must be performed at some point.
            # So it is not useful to globally search for all happy swap chains first.
            cycle = rx.digraph_find_cycle(sub_digraph, source=todo_node)
            if len(cycle) > 0:
                assert len(cycle) > 1, "The cycle was not happy."
                # We iterate over the cycle in reversed order, starting at the last edge.
                # The first edge is excluded.
                for edge in list(cycle)[-1:0:-1]:
                    yield edge
                    swap(edge[0], edge[1])
                steps += len(cycle) - 1
            else:
                # Try to find a node without a token to swap with.
                try:
                    edge = next(
                        edge
                        for edge in rx.digraph_dfs_edges(sub_digraph, todo_node)
                        if edge[1] not in tokens
                    )
                    # Swap predecessor and successor, because successor does not have a token
                    yield edge
                    swap(edge[0], edge[1])
                    steps += 1
                except StopIteration:
                    # Unhappy swap case
                    cycle = rx.digraph_find_cycle(digraph, source=todo_node)
                    assert len(cycle) == 1, "The cycle was not unhappy."
                    unhappy_node = cycle[0][0]
                    # Find a node that wants to swap with this node.
                    try:
                        predecessor = next(
                            predecessor
                            for predecessor in digraph.predecessor_indices(unhappy_node)
                            if predecessor != unhappy_node
                        )
                    except StopIteration:
                        logger.error(
                            "Unexpected StopIteration raised when getting predecessors"
                            "in unhappy swap case."
                        )
                        return
                    yield unhappy_node, predecessor
                    swap(unhappy_node, predecessor)
                    steps += 1
        if todo_nodes:
            raise RuntimeError("Too many iterations while approximating the Token Swaps.")

    def _add_token_edges(
        self, node: int, tokens: Mapping[int, int], digraph: rx.PyDiGraph, sub_digraph: rx.PyDiGraph
    ) -> None:
        """Add diedges to the graph wherever a token can be moved closer to its destination."""
        if node not in tokens:
            return

        if tokens[node] == node:
            digraph.extend_from_edge_list([(node, node)])
            return

        for neighbor in self.graph.neighbors(node):
            if self.distance(neighbor, tokens[node]) < self.distance(node, tokens[node]):
                digraph.extend_from_edge_list([(node, neighbor)])
                sub_digraph.extend_from_edge_list([(node, neighbor)])

    def _swap(
        self,
        node1: int,
        node2: int,
        tokens: MutableMapping[int, int],
        digraph: rx.PyDiGraph,
        sub_digraph: rx.PyDiGraph,
        todo_nodes: MutableSet[int],
    ) -> None:
        """Swap two nodes, maintaining the data structures."""
        assert self.graph.has_edge(
            node1, node2
        ), "The swap is being performed on a non-existent edge."
        # Swap the tokens on the nodes, taking into account no-token nodes.
        token1 = tokens.pop(node1, None)
        token2 = tokens.pop(node2, None)
        if token2 is not None:
            tokens[node1] = token2
        if token1 is not None:
            tokens[node2] = token1
        # Recompute the edges incident to node 1 and 2
        for node in [node1, node2]:
            digraph.remove_edges_from(
                [(node, successor) for successor in digraph.successor_indices(node)]
            )
            sub_digraph.remove_edges_from(
                [(node, successor) for successor in sub_digraph.successor_indices(node)]
            )
            self._add_token_edges(node, tokens, digraph, sub_digraph)
            if node in tokens and tokens[node] != node:
                todo_nodes.add(node)
            elif node in todo_nodes:
                todo_nodes.remove(node)
