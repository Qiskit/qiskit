"""Test cases for the permutation.util package"""
from typing import List, TypeVar
from unittest import TestCase

import networkx as nx
from numpy import random

from qiskit.transpiler.passes.extension_mapper.src.permutation import Swap
from qiskit.transpiler.passes.extension_mapper.src.permutation.util \
    import cycles, flatten_swaps, circuit

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
        dag, inputmap = circuit(swaps, allow_swaps=True)

        op_nodes = filter(lambda n: n["type"] == "op",
                          map(lambda n: n[1], dag.multi_graph.nodes(data=True)))
        qargs = list(map(lambda n: n["qargs"], op_nodes))
        names = {k: v[0] for k, v in inputmap.items()}
        self.assertEqual([
            [(names[0], 0), (names[2], 0)],
            [(names[4], 0), (names[5], 0)],
            [(names[1], 0), (names[2], 0)],
            ], qargs)

    @staticmethod
    def valid_edge_swaps(tester: TestCase, swaps: List[List[Swap[_V]]],
                         valid_graph: nx.Graph) -> None:
        """Check if the swap is in the edge list."""

        for i in swaps:
            for j in i:
                tester.assertTrue(valid_graph.has_edge(*j), 'edge is not valid')
