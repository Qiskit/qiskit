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
        circuit = QuantumCircuit(4)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(0, 3)
        circuit.h(3)
        circuit.swap(2, 3)
        circuit.cx(0, 1)
        circuit.cx(0, 3)

        pass_manager = PassManager(CollectLinearFunctions())
        optimized_circuit = pass_manager.run(circuit)

        linear_functions = [inst for inst, _, _ in optimized_circuit.data if isinstance(inst, LinearFunction)]
        self.assertTrue(len(linear_functions) == 2)

        subcircuit1 = QuantumCircuit(4)
        subcircuit1.cx(0, 1)
        subcircuit1.cx(0, 2)
        subcircuit1.cx(0, 3)

        subcircuit2 = QuantumCircuit(4)
        subcircuit2.swap(2, 3)
        subcircuit2.cx(0, 1)
        subcircuit2.cx(0, 3)

        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(LinearFunction(subcircuit1), [0, 1, 2, 3])
        expected_circuit.h(3)
        expected_circuit.append(LinearFunction(subcircuit2), [0, 1, 2, 3])

        # Check that the two circuits represent the same operator
        self.assertEqual(Operator(optimized_circuit), Operator(expected_circuit))


if __name__ == "__main__":
    unittest.main()
