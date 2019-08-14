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

"""Test cases for the permutation.util package"""

from numpy import random

from qiskit.extensions import SwapGate
from qiskit.test import QiskitTestCase
from qiskit.transpiler.routing.util import cycles, flatten_swaps, circuit


class TestUtil(QiskitTestCase):
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
        reversed_inputmap = {b: a for a, b in inputmap.items()}
        qargs = {(reversed_inputmap[qarg0], reversed_inputmap[qarg1]) for qarg0, qarg1 in qargs}

        # We only check that the right nodes are swapped
        # Would be better if we could check that it's done in the same order.
        flattened_swaps = {swap for swap_step in swaps for swap in swap_step}
        self.assertEqual(flattened_swaps, qargs)
