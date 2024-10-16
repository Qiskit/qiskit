# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Qiskit's AnnotatedOperation class."""

import ddt
import numpy as np

from qiskit.circuit import QuantumCircuit, twirl_circuit, Gate
from qiskit.circuit.library import CXGate, ECRGate, CZGate, iSwapGate, SwapGate, PermutationGate
from qiskit.circuit.random import random_circuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt.ddt
class TestTwirling(QiskitTestCase):
    """Testing qiskit.circuit.twirl_circuit"""

    @ddt.data(CXGate, ECRGate, CZGate, iSwapGate)
    def test_twirl_circuit_equiv(self, gate):
        """Test the twirled circuit is equivalent."""
        qc = QuantumCircuit(2)
        qc.append(gate(), (0, 1))
        for i in range(100):
            with self.subTest(i):
                res = twirl_circuit(qc, gate, i)
                np.testing.assert_allclose(
                    Operator(qc), Operator(res), err_msg=f"gate: {gate} not equiv to\n{res}"
                )
                self.assertNotEqual(res, qc)
                # Assert we have more than just a 2q gate in the circuit
                self.assertGreater(len(res.count_ops()), 1)

    def test_twirl_circuit_None(self):
        """Test the default twirl all gates."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cz(0, 1)
        qc.ecr(0, 1)
        qc.iswap(0, 1)
        res = twirl_circuit(qc, seed=12345)
        np.testing.assert_allclose(
            Operator(qc), Operator(res), err_msg=f"{qc}\nnot equiv to\n{res}"
        )
        self.assertNotEqual(res, qc)
        self.assertEqual(sum(res.count_ops().values()), 20)

    def test_twirl_circuit_list(self):
        """Test twirling for a circuit list of gates to twirl."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cz(0, 1)
        qc.ecr(0, 1)
        qc.iswap(0, 1)
        res = twirl_circuit(qc, twirling_gate=["cx", iSwapGate], seed=12345)
        np.testing.assert_allclose(
            Operator(qc), Operator(res), err_msg=f"{qc}\nnot equiv to\n{res}"
        )
        self.assertNotEqual(res, qc)
        self.assertEqual(sum(res.count_ops().values()), 12)

    @ddt.data(CXGate, ECRGate, CZGate, iSwapGate)
    def test_many_twirls_equiv(self, gate):
        """Test the twirled circuits are equivalent if num_twirls>1."""
        qc = QuantumCircuit(2)
        qc.append(gate(), (0, 1))
        res = twirl_circuit(qc, gate, seed=424242, num_twirls=1000)
        for twirled_circuit in res:
            np.testing.assert_allclose(
                Operator(qc), Operator(twirled_circuit), err_msg=f"gate: {gate} not equiv to\n{res}"
            )
            self.assertNotEqual(twirled_circuit, qc)

    def test_invalid_gate(self):
        """Test an error is raised with a non-standard gate."""

        class MyGate(Gate):
            """Custom gate."""

            def __init__(self):
                super().__init__("custom", num_qubits=2, params=[])

        qc = QuantumCircuit(2)
        qc.append(MyGate(), (0, 1))

        with self.assertRaises(QiskitError):
            twirl_circuit(qc, twirling_gate=MyGate)

    def test_invalid_standard_gate(self):
        """Test an error is raised with an unsupported standard gate."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        with self.assertRaises(QiskitError):
            twirl_circuit(qc, twirling_gate=SwapGate)

    def test_invalid_string(self):
        """Test an error is raised with an unsupported standard gate."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        with self.assertRaises(QiskitError):
            twirl_circuit(qc, twirling_gate="swap")

    def test_invalid_str_entry_in_list(self):
        """Test an error is raised with an unsupported string gate in list."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        with self.assertRaises(QiskitError):
            twirl_circuit(qc, twirling_gate=[CXGate, "swap"])

    def test_invalid_class_entry_in_list(self):
        """Test an error is raised with an unsupported string gate in list."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        with self.assertRaises(QiskitError):
            twirl_circuit(qc, twirling_gate=[SwapGate, "cx"])

    @ddt.data(CXGate, ECRGate, CZGate, iSwapGate)
    def test_full_circuit(self, gate):
        """Test a circuit with a random assortment of gates."""
        qc = random_circuit(5, 25, seed=12345678942)
        qc.append(PermutationGate([1, 2, 0]), [0, 1, 2])
        res = twirl_circuit(qc)
        np.testing.assert_allclose(
            Operator(qc), Operator(res), err_msg=f"gate: {gate} not equiv to\n{res}"
        )
