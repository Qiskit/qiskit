"""Test cases for the permutation.util package"""
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

from typing import List, Set, TypeVar
from unittest import TestCase

import networkx as nx
from numpy import random
from qiskit.extensions import SwapGate

from qiskit.transpiler.routing import Swap
from qiskit.transpiler.routing.util import cycles, flatten_swaps, circuit

_V = TypeVar('_V')


class TestUtil(TestCase):
    """The test cases"""

    def test_cycles_simple(self) -> None:
        """A simple test of cycles for a fixed permutation."""
        permutation = {0: 0, 1: 3, 2: 1, 3: 2}

        out = cycles(permutation)

        self.assertCountEqual([{0: 0}, {1: 3, 3: 2, 2: 1}], out)

    def test_cycles_simple2(self) -> None:
        """A simple test of cycles for a fixed permutation."""
        permutation = {0: 2, 1: 3, 2: 4, 3: 1, 4: 0}

        out = cycles(permutation)

        self.assertCountEqual([{0: 2, 2: 4, 4: 0}, {1: 3, 3: 1}], out)

    def test_cycles_big(self) -> None:
        """A randomised test of large size to see if a permutation is decomposed correctly."""
        size = 10 ** 5
        rand_permutation = random.permutation(range(size))
        permutation = {i: rand_permutation[i] for i in range(size)}

        out = cycles(permutation)
        for cycle in out:
            first, mapped_to = cycle.popitem()
            self.assertEqual(mapped_to, permutation[first])

            # Keep popping elements until we presumably reach first again
            while cycle:
                current = mapped_to
                mapped_to = cycle.pop(current)
                self.assertEqual(mapped_to, permutation[current])

            self.assertEqual(first, mapped_to)

    def test_flatten_swaps(self) -> None:
        """Check if swaps in the same timestep are flattened correctly."""
        swaps = [
            [[(0, 1)], [(1, 2)]],
            [[(5, 6)]]
            ]

        flattened = list(flatten_swaps(swaps))
        self.assertListEqual([[(0, 1), (5, 6)], [(1, 2)]], flattened)

    @staticmethod
    def valid_parallel_swaps(tester: TestCase, swaps: List[List[Swap[_V]]]) -> None:
        """
        Tests if the given sequence of swaps does not perform multiple swaps
        on the same node at the same time.
        """
        for step in swaps:
            used = {sw1 for sw1, _ in step} | {sw2 for _, sw2 in step}
            tester.assertEqual(len(used), 2*len(step))

    def test_circuit_simple(self) -> None:
        """Check the circuit function for a simple circuit."""
        swaps = [[(0, 2), (4, 5)], [(1, 2)]]
        result = circuit(swaps)
        dag = result.circuit
        inputmap = result.inputmap

        op_nodes = dag.op_nodes()
        for op_node in op_nodes:
            self.assertIsInstance(op_node.op, SwapGate)
        qargs = {tuple(op_node.qargs) for op_node in op_nodes}
        reversed_inputmap = {b: a for a,b in inputmap.items()}
        qargs = {(reversed_inputmap[qarg0], reversed_inputmap[qarg1]) for qarg0, qarg1 in qargs}

        # We only check that the right nodes are swapped
        # Would be better if we could check that it's done in the same order.
        flattened_swaps = {swap for swap_step in swaps for swap in swap_step}
        self.assertEqual(flattened_swaps, qargs)

    @staticmethod
    def valid_edge_swaps(tester: TestCase, swaps: List[List[Swap[_V]]],
                         valid_graph: nx.Graph) -> None:
        """Check if the swap is in the edge list."""

        for i in swaps:
            for j in i:
                tester.assertTrue(valid_graph.has_edge(*j), 'edge ' + str(j) +' is not valid')
