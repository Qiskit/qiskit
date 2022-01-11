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

"""Test transpiler passes that work with linear functions."""

import unittest

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes.optimization import CollectLinearFunctions
from qiskit.transpiler.passes.synthesis import LinearFunctionsSynthesis
from qiskit.test import QiskitTestCase
from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.transpiler import PassManager
from qiskit.quantum_info import Operator


class TestLinearFunctionsPasses(QiskitTestCase):
    """Tests to verify correctness of the transpiler pass that extracts
    blocks of CX and SWAP gates and replaces these blocks by LinearFunctions,
    and the correctness of the transpiler pass that synthesizes LinearFunctions
    into CX and SWAP gates.
    """

    def test_single_linear_block(self):
        """Test that when all gates in a circuit are either CX or SWAP,
        we end up with a single LinearFunction."""

        # original circuit
        circuit = QuantumCircuit(4)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(0, 3)
        circuit.swap(2, 3)
        circuit.cx(0, 1)
        circuit.cx(0, 3)

        # new circuit with linear functions extracted using transpiler pass
        optimized_circuit = PassManager(CollectLinearFunctions()).run(circuit)

        # check that this circuit consists of a single LinearFunction
        self.assertTrue("linear_function" in optimized_circuit.count_ops().keys())
        self.assertTrue(len(optimized_circuit.data) == 1)
        inst1, _, _ = optimized_circuit.data[0]
        self.assertTrue(isinstance(inst1, LinearFunction))

        # construct a circuit with linear function directly, without the transpiler pass
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(LinearFunction(circuit), [0, 1, 2, 3])

        # check that we have an equivalent circuit
        self.assertEqual(Operator(optimized_circuit), Operator(expected_circuit))

        # now a circuit with linear functions synthesized
        synthesized_circuit = PassManager(LinearFunctionsSynthesis()).run(optimized_circuit)

        # check that there are no LinearFunctions present in synthesized_circuit
        self.assertFalse("linear_function" in synthesized_circuit.count_ops().keys())

        # check that we have an equivalent circuit
        self.assertEqual(Operator(optimized_circuit), Operator(synthesized_circuit))

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

        # new circuit with linear functions extracted using transpiler pass
        result_circuit = PassManager(CollectLinearFunctions()).run(circuit)

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

        # now a circuit with linear functions synthesized
        synthesized_circuit = PassManager(LinearFunctionsSynthesis()).run(result_circuit)

        # check that there are no LinearFunctions present in synthesized_circuit
        self.assertFalse("linear_function" in synthesized_circuit.count_ops().keys())

        # check that we have an equivalent circuit
        self.assertEqual(Operator(result_circuit), Operator(synthesized_circuit))


if __name__ == "__main__":
    unittest.main()
