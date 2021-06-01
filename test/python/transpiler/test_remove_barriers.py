# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
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

        pass_ = RemoveBarriers()
        result_dag = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result_dag.size(), 0)

    def test_remove_barriers_other_gates(self):
        """Remove all barriers, leave other gates intact"""

        circuit = QuantumCircuit(1)
        circuit.barrier()
        circuit.x(0)
        circuit.barrier()
        circuit.h(0)

        pass_ = RemoveBarriers()
        result_dag = pass_.run(circuit_to_dag(circuit))

        op_nodes = result_dag.op_nodes()

        self.assertEqual(result_dag.size(), 2)
        for ii, name in enumerate(["x", "h"]):
            self.assertEqual(op_nodes[ii].name, name)


if __name__ == "__main__":
    unittest.main()
