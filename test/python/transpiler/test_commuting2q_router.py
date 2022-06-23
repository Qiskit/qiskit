# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for routing Commuting2qGateGrouper and Commuting2qGateRouter."""

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.circuit.library import TwoLocal
from qiskit.transpiler.passes import Commuting2qGateGrouper, Commuting2qGateRouter
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qiskit.test import QiskitTestCase


class TestCommuting2qGateGrouperRouting(QiskitTestCase):
    """Test the swap strategies on FindCommutingPauliEvolutions and Commuting2qGateRouter passes."""

    def test_idle_qubit(self):
        """Test to route on an op that has an idle qubit."""
        circuit = QuantumCircuit(4)
        circuit.cz(0, 1)
        circuit.cz(0, 2)
        circuit.draw()

        cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (2, 3)])
        swap_strategy = SwapStrategy(cmap, swap_layers=(((0, 1),),))
        pm_ = PassManager([Commuting2qGateGrouper(), Commuting2qGateRouter(swap_strategy)])
        swapped = pm_.run(circuit)

        expected = QuantumCircuit(4)
        expected.cz(0, 1)
        expected.swap(0, 1)
        expected.cz(1, 2)

        self.assertEqual(swapped, expected)

    def test_two_local_t_device(self):
        """Test the swap strategy to route a two_local on a T device.

        The coupling map in this test corresponds to

        .. parsed-literal::

            0 -- 1 -- 2
                 |
                 3
                 |
                 4

        """
        swaps = (
            ((1, 3),),
            ((0, 1), (3, 4)),
            ((1, 3),),
        )

        circ = TwoLocal(5, "ry", "cz", entanglement="full", reps=2).decompose()

        cmap = CouplingMap([[0, 1], [1, 2], [1, 3], [3, 4]])
        cmap.make_symmetric()
        swap_strategy = SwapStrategy(cmap, swaps)
        passmanager = PassManager([Commuting2qGateGrouper(), Commuting2qGateRouter(swap_strategy)])
        swapped = passmanager.run(circ)

        self.assertEqual(swapped.count_ops(), {"cz": 20, "ry": 15, "swap": 8})
