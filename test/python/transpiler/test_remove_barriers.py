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

"""Test the RemoveBarriers pass"""

import unittest
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase


class TestMergeAdjacentBarriers(QiskitTestCase):
    """Test the MergeAdjacentBarriers pass"""

    def test_remove_barriers(self):
        """Remove all barriers"""

        circuit = QuantumCircuit(2)
        circuit.barrier()
        circuit.barrier()

        expected = QuantumCircuit(2)

        pass_ = RemoveBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_remove_barriers_other_gates(self):
        """Remove all barriers, leave other gates intact"""

        circuit = QuantumCircuit(1)
        circuit.barrier()
        circuit.x(0)
        circuit.barrier()
        circuit.h(0)

        expected = QuantumCircuit(1)
        expected.x(0)
        expected.h(0)

        pass_ = RemoveBarriers()
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))


if __name__ == "__main__":
    unittest.main()
