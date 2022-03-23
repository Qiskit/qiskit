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

"""Test transpiler passes that deal with linear functions."""

import unittest

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes.optimization import CollectLinearFunctions
from qiskit.transpiler.passes.synthesis import (
    LinearFunctionsSynthesis,
    LinearFunctionsToPermutations,
)
from qiskit.test import QiskitTestCase
from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.circuit.library import RealAmplitudes
from qiskit.transpiler import PassManager
from qiskit.quantum_info import Operator


class TestLinearFunctionsPasses(QiskitTestCase):
    """Tests to verify correctness of the transpiler passes that deal with
    linear functions:
    the pass that extracts blocks of CX and SWAP gates and replaces these blocks by LinearFunctions,
    the pass that synthesizes LinearFunctions into CX and SWAP gates,
    and the pass that promotes LinearFunctions to Permutations whenever possible.
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
        self.assertIn("linear_function", optimized_circuit.count_ops().keys())
        self.assertEqual(len(optimized_circuit.data), 1)
        inst1, _, _ = optimized_circuit.data[0]
        self.assertIsInstance(inst1, LinearFunction)

        # construct a circuit with linear function directly, without the transpiler pass
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(LinearFunction(circuit), [0, 1, 2, 3])

        # check that we have an equivalent circuit
        self.assertEqual(Operator(optimized_circuit), Operator(expected_circuit))

        # now a circuit with linear functions synthesized
        synthesized_circuit = PassManager(LinearFunctionsSynthesis()).run(optimized_circuit)

        # check that there are no LinearFunctions present in synthesized_circuit
        self.assertNotIn("linear_function", synthesized_circuit.count_ops().keys())

        # check that we have an equivalent circuit
        self.assertEqual(Operator(optimized_circuit), Operator(synthesized_circuit))

    def test_two_linear_blocks(self):
        """Test that when we have two blocks of linear gates with one nonlinear gate in the middle,
        we end up with two LinearFunctions."""
        # Create a circuit with a nonlinear gate (h) cleanly separating it into two linear blocks.
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(0, 2)
        circuit1.cx(0, 3)
        circuit1.h(3)
        circuit1.swap(2, 3)
        circuit1.cx(1, 2)
        circuit1.cx(0, 1)

        # new circuit with linear functions extracted using transpiler pass
        circuit2 = PassManager(CollectLinearFunctions()).run(circuit1)

        # We expect to see 3 gates (linear, h, linear)
        self.assertEqual(len(circuit2.data), 3)
        inst1, qargs1, cargs1 = circuit2.data[0]
        inst2, qargs2, cargs2 = circuit2.data[2]
        self.assertIsInstance(inst1, LinearFunction)
        self.assertIsInstance(inst2, LinearFunction)

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
        synthesized_circuit = PassManager(LinearFunctionsSynthesis()).run(circuit2)

        # check that there are no LinearFunctions present in synthesized_circuit
        self.assertNotIn("linear_function", synthesized_circuit.count_ops().keys())

        # check that we have an equivalent circuit
        self.assertEqual(Operator(circuit2), Operator(synthesized_circuit))

    def test_to_permutation(self):
        """Test that converting linear functions to permutations works correctly."""

        # Original circuit with two linear blocks; the second block happens to be
        # a permutation
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(0, 2)
        circuit1.cx(0, 3)
        circuit1.h(3)
        circuit1.swap(2, 3)
        circuit1.cx(1, 2)
        circuit1.cx(2, 1)
        circuit1.cx(1, 2)

        # new circuit with linear functions extracted
        circuit2 = PassManager(CollectLinearFunctions()).run(circuit1)

        # check that we have two blocks of linear functions
        self.assertEqual(circuit2.count_ops()["linear_function"], 2)

        # new circuit with linear functions converted to permutations
        # (the first linear function should not be converted, the second should)
        circuit3 = PassManager(LinearFunctionsToPermutations()).run(circuit2)

        # check that there is one linear function and one permutation
        self.assertEqual(circuit3.count_ops()["linear_function"], 1)
        self.assertEqual(circuit3.count_ops()["permutation_[2,0,1]"], 1)

        # check that the final circuit is still equivalent to the original circuit
        self.assertEqual(Operator(circuit1), Operator(circuit3))

    def test_hidden_identity_block(self):
        """Test that extracting linear functions and synthesizing them back
        results in an equivalent circuit when a linear block represents
        the identity matrix."""

        # Create a circuit with multiple non-linear blocks
        circuit1 = QuantumCircuit(3)
        circuit1.h(0)
        circuit1.h(1)
        circuit1.h(2)
        circuit1.swap(0, 2)
        circuit1.swap(0, 2)
        circuit1.h(0)
        circuit1.h(1)
        circuit1.h(2)

        # collect linear functions
        circuit2 = PassManager(CollectLinearFunctions()).run(circuit1)

        # synthesize linear functions
        circuit3 = PassManager(LinearFunctionsSynthesis()).run(circuit2)

        # check that we have an equivalent circuit
        self.assertEqual(Operator(circuit1), Operator(circuit3))

    def test_multiple_non_linear_blocks(self):
        """Test that extracting linear functions and synthesizing them back
        results in an equivalent circuit when there are multiple non-linear blocks."""

        # Create a circuit with multiple non-linear blocks
        circuit1 = QuantumCircuit(3)
        circuit1.h(0)
        circuit1.s(1)
        circuit1.h(0)
        circuit1.cx(0, 1)
        circuit1.cx(0, 2)
        circuit1.swap(1, 2)
        circuit1.h(1)
        circuit1.sdg(2)
        circuit1.cx(1, 0)
        circuit1.cx(1, 2)
        circuit1.h(2)
        circuit1.cx(1, 2)
        circuit1.cx(0, 1)
        circuit1.h(1)
        circuit1.cx(0, 1)
        circuit1.h(1)

        # collect linear functions
        circuit2 = PassManager(CollectLinearFunctions()).run(circuit1)

        # synthesize linear functions
        circuit3 = PassManager(LinearFunctionsSynthesis()).run(circuit2)

        # check that we have an equivalent circuit
        self.assertEqual(Operator(circuit1), Operator(circuit3))

    def test_real_amplitudes_circuit_4q(self):
        """Test that for the 4-qubit real amplitudes circuit
        extracting linear functions produces the expected number of linear blocks,
        and synthesizing these blocks produces an expected number of CNOTs.
        """
        ansatz = RealAmplitudes(4, reps=2)
        circuit1 = ansatz.decompose()

        # collect linear functions
        circuit2 = PassManager(CollectLinearFunctions()).run(circuit1)
        self.assertEqual(circuit2.count_ops()["linear_function"], 2)

        # synthesize linear functions
        circuit3 = PassManager(LinearFunctionsSynthesis()).run(circuit2)
        self.assertEqual(circuit3.count_ops()["cx"], 6)

    def test_real_amplitudes_circuit_5q(self):
        """Test that for the 5-qubit real amplitudes circuit
        extracting linear functions produces the expected number of linear blocks,
        and synthesizing these blocks produces an expected number of CNOTs.
        """
        ansatz = RealAmplitudes(5, reps=2)
        circuit1 = ansatz.decompose()

        # collect linear functions
        circuit2 = PassManager(CollectLinearFunctions()).run(circuit1)
        self.assertEqual(circuit2.count_ops()["linear_function"], 2)

        # synthesize linear functions
        circuit3 = PassManager(LinearFunctionsSynthesis()).run(circuit2)
        self.assertEqual(circuit3.count_ops()["cx"], 8)

    def test_not_collecting_single_gates1(self):
        """Test that extraction of linear functions does not create
        linear functions out of single gates.
        """
        circuit1 = QuantumCircuit(3)
        circuit1.cx(0, 1)
        circuit1.h(1)
        circuit1.cx(1, 2)

        # collect linear functions
        circuit2 = PassManager(CollectLinearFunctions()).run(circuit1)

        # check that there are no LinearFunctions present in synthesized_circuit
        self.assertNotIn("linear_function", circuit2.count_ops().keys())

    def test_not_collecting_single_gates2(self):
        """Test that extraction of linear functions does not create
        linear functions out of single gates.
        """
        circuit1 = QuantumCircuit(3)
        circuit1.h(0)
        circuit1.h(1)
        circuit1.swap(0, 1)
        circuit1.s(1)
        circuit1.swap(1, 2)
        circuit1.h(2)

        # collect linear functions
        circuit2 = PassManager(CollectLinearFunctions()).run(circuit1)

        # check that there are no LinearFunctions present in synthesized_circuit
        self.assertNotIn("linear_function", circuit2.count_ops().keys())

    def test_disconnected_gates1(self):
        """Test that extraction of linear functions does not create
        linear functions out of disconnected gates.
        """
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(2, 3)

        # collect linear functions
        circuit2 = PassManager(CollectLinearFunctions()).run(circuit1)

        # check that there are no LinearFunctions present in synthesized_circuit
        self.assertNotIn("linear_function", circuit2.count_ops().keys())

    def test_disconnected_gates2(self):
        """Test that extraction of linear functions does not create
        linear functions out of disconnected gates.
        """
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(1, 0)
        circuit1.cx(2, 3)

        # collect linear functions
        circuit2 = PassManager(CollectLinearFunctions()).run(circuit1)

        # we expect that the first two CX gates will be combined into
        # a linear function, but the last will not
        self.assertEqual(circuit2.count_ops()["linear_function"], 1)
        self.assertEqual(circuit2.count_ops()["cx"], 1)

    def test_connected_gates(self):
        """Test that extraction of linear functions combines gates
        which become connected later.
        """
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(1, 0)
        circuit1.cx(2, 3)
        circuit1.swap(0, 3)

        # collect linear functions
        circuit2 = PassManager(CollectLinearFunctions()).run(circuit1)

        # we expect that the first two CX gates will be combined into
        # a linear function, but the last will not
        self.assertEqual(circuit2.count_ops()["linear_function"], 1)
        self.assertNotIn("cx", circuit2.count_ops().keys())
        self.assertNotIn("swap", circuit2.count_ops().keys())


if __name__ == "__main__":
    unittest.main()
