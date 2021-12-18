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

"""Test the CollectLinearFunctions Optimization pass."""

import unittest

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes.optimization import CollectLinearFunctions
from qiskit.test import QiskitTestCase
from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.transpiler import PassManager
from qiskit.quantum_info import Operator


class TestCollectLinearFunctions(QiskitTestCase):
    """Tests to verify that extracting blocks of CX and SWAP gates and
    replacing these blocks by LinearFunctions worke correctly."""

    def test_single_linear_block(self):
        """Test that when all gates in a circuit are either CX or SWAP,
        we end up with a single LinearFunction."""
        circuit = QuantumCircuit(4)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(0, 3)
        circuit.swap(2, 3)
        circuit.cx(0, 1)
        circuit.cx(0, 3)

        pass_manager = PassManager(CollectLinearFunctions())
        optimized_circuit = pass_manager.run(circuit)

        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(LinearFunction(circuit), [0, 1, 2, 3])

        # Check that the two circuits represent the same operator
        self.assertEqual(Operator(optimized_circuit), Operator(expected_circuit))

    def test_two_linear_blocks(self):
        """Test that when we have two blocks of linear gates with one nonlinear gate in the middle,
        we end up with two LinearFunctions."""
        # Create a circuit with a nonlinear gate (h) cleanly separating it into two linear blocks.
        circuit = QuantumCircuit(4)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(0, 3)
        circuit.h(3)
        circuit.swap(2, 3)
        circuit.cx(1, 2)
        circuit.cx(0, 1)

        pass_manager = PassManager(CollectLinearFunctions())
        result_circuit = pass_manager.run(circuit)

        # We expect to see 3 gates (linear, h, linear)
        self.assertTrue(len(result_circuit.data) == 3)
        inst1, qargs1, cargs1 = result_circuit.data[0]
        inst2, qargs2, cargs2 = result_circuit.data[2]
        self.assertTrue(isinstance(inst1, LinearFunction))
        self.assertTrue(isinstance(inst2, LinearFunction))

        # Check that the first linear function represents the subcircuit before h
        resulting_subcircuit1 = QuantumCircuit(4)
        resulting_subcircuit1.append(inst1, qargs1, cargs1)

        expected_subcircuit1 = QuantumCircuit(4)
        expected_subcircuit1.cx(0, 1)
        expected_subcircuit1.cx(0, 2)
        expected_subcircuit1.cx(0, 3)
        self.assertEqual(Operator(resulting_subcircuit1), Operator(expected_subcircuit1))

        # Check that the second linear function represents the subcircuit after h
        resulting_subcircuit2 = QuantumCircuit(4)
        resulting_subcircuit2.append(inst2, qargs2, cargs2)

        expected_subcircuit2 = QuantumCircuit(4)
        expected_subcircuit2.swap(2, 3)
        expected_subcircuit2.cx(1, 2)
        expected_subcircuit2.cx(0, 1)
        self.assertEqual(Operator(resulting_subcircuit2), Operator(expected_subcircuit2))


if __name__ == "__main__":
    unittest.main()
