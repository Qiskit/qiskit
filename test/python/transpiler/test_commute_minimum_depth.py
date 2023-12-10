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

"""Test transpiler pass that optimizes depth of quantum circuits by commuting
gates to the left when possible."""

import unittest
from qiskit.test import QiskitTestCase

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommuteMinimumDepth


class TestCommuteMinimumDepth(QiskitTestCase):
    """Test the CommuteMinimumDepth pass."""

    def test_commutative_circuit(self):
        """A simple circuit where the last cnot commutes to the left.

        0:---(+)----------------       0:---(+)--------
              |                              |
        1:----.------(x)--------   =   1:----.-----(x)-
                      |                             |
        2:------------.------.--       2:----.------.--
                             |               |
        3:------------------(+)-       3:---(+)--------
        """
        circuit = QuantumCircuit(4)
        circuit.cx(1, 0)
        circuit.cx(2, 1)
        circuit.cx(2, 3)

        passmanager = PassManager(CommuteMinimumDepth())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(4)
        expected.cx(1, 0)
        expected.cx(2, 3)
        expected.cx(2, 1)
        self.assertEqual(expected, new_circuit)

    def test_non_commutative_gates(self):
        """A simple circuit where the depth can not be reduced.

        0: --.-----(+)--------         0: --.-----(+)--------
             |      |                       |      |
        1: -(+)-----.-----(+)-    =    1: -(+)-----.-----(+)-
                           |                              |
        2: ----------------.--         2: ----------------.--

        """
        circuit = QuantumCircuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(2, 1)
        passmanager = PassManager(CommuteMinimumDepth())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(3)
        expected.cx(0, 1)
        expected.cx(1, 0)
        expected.cx(2, 1)
        self.assertEqual(expected, new_circuit)

    def test_dfs_edges_multiple_times(self):
        """A simple circuit where the same edge has to tested more than once
        to confirm the proper depth and avoid doing impossible commutations.

        0:--.-------------------------
            |
        1:-(+)------------(+)------.--
                           |       |
        2:-(+)------.------.------(+)-
            |       |
        3:--.------(+)----------------

        """
        circuit = QuantumCircuit(4)
        circuit.cx(0, 1)
        circuit.cx(3, 2)
        circuit.cx(2, 3)
        circuit.cx(1, 2)
        circuit.cx(2, 1)
        passmanager = PassManager(CommuteMinimumDepth())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(4)
        expected.cx(0, 1)
        expected.cx(3, 2)
        expected.cx(2, 3)
        expected.cx(1, 2)
        expected.cx(2, 1)
        self.assertEqual(expected, new_circuit)

    def single_source_dfs_edges_multiple_times(self):
        """A circuit where within the same source an edge is tested multiple
        times to find the proper depth."""

        circuit = QuantumCircuit(4)
        circuit.cx(3, 2)
        circuit.cx(1, 3)
        circuit.cx(2, 0)
        circuit.cx(1, 2)
        circuit.cx(2, 1)
        circuit.cx(1, 0)
        passmanager = PassManager(CommuteMinimumDepth())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(4)
        expected.cx(3, 2)
        expected.cx(1, 3)
        expected.cx(2, 0)
        expected.cx(1, 2)
        expected.cx(2, 1)
        expected.cx(1, 0)
        self.assertEqual(expected, new_circuit)


if __name__ == "__main__":
    unittest.main()
