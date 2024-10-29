# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests ClassicalFunction as a gate."""

import unittest

from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import XGate
from qiskit.utils.optionals import HAS_TWEEDLEDUM
from test import QiskitTestCase  # pylint: disable=wrong-import-order

if HAS_TWEEDLEDUM:
    from . import examples
    from qiskit.circuit.classicalfunction import classical_function as compile_classical_function


@unittest.skipUnless(HAS_TWEEDLEDUM, "Tweedledum is required for these tests.")
class TestOracleDecomposition(QiskitTestCase):
    # pylint: disable=possibly-used-before-assignment
    """Tests ClassicalFunction.decomposition."""

    def test_grover_oracle(self):
        """grover_oracle.decomposition"""
        oracle = compile_classical_function(examples.grover_oracle)
        quantum_circuit = QuantumCircuit(5)
        quantum_circuit.append(oracle, [2, 1, 0, 3, 4])

        expected = QuantumCircuit(5)
        expected.append(XGate().control(4, ctrl_state="1010"), [2, 1, 0, 3, 4])

        self.assertEqual(quantum_circuit.decompose(), expected)
