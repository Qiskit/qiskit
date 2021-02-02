# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the optimize-1q-gate pass"""

import unittest

import ddt
import numpy as np

from qiskit.circuit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit.library.standard_gates import U3Gate
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler.passes import BasisTranslator
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.quantum_info import Operator
from qiskit.test import QiskitTestCase
from qiskit.circuit import Parameter


@ddt.ddt
class TestOptimize1qGatesDecomposition(QiskitTestCase):
    """Test for 1q gate optimizations. """

    @ddt.data(
        ['cx', 'u3'],
        ['cz', 'u3'],
        ['cx', 'u'],
        ['p', 'sx', 'u', 'cx'],
        ['cz', 'rx', 'rz'],
        ['rxx', 'rx', 'ry'],
        ['iswap', 'rx', 'rz'],
        ['u1', 'rx'],
        ['rz', 'sx'],
        ['p', 'sx'],
        ['r'],
    )
    def test_optimize_h_gates_pass_manager(self, basis):
        """Transpile: qr:--[H]-[H]-[H]--"""
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.h(qr[0])

        expected = QuantumCircuit(qr)
        expected.u(np.pi / 2, 0, np.pi, qr)  # U2(0, pi)

        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertTrue(Operator(circuit).equiv(Operator(result)))

    @ddt.data(
        ['cx', 'u3'],
        ['cz', 'u3'],
        ['cx', 'u'],
        ['p', 'sx', 'u', 'cx'],
        ['cz', 'rx', 'rz'],
        ['rxx', 'rx', 'ry'],
        ['iswap', 'rx', 'rz'],
        ['u1', 'rx'],
        ['rz', 'sx'],
        ['p', 'sx'],
        ['r'],
    )
    def test_ignores_conditional_rotations(self, basis):
        """Conditional rotations should not be considered in the chain."""
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(2, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.p(0.1, qr).c_if(cr, 1)
        circuit.p(0.2, qr).c_if(cr, 3)
        circuit.p(0.3, qr)
        circuit.p(0.4, qr)

        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)

        self.assertTrue(Operator(circuit).equiv(Operator(result)))

    @ddt.data(
        ['cx', 'u3'],
        ['cz', 'u3'],
        ['cx', 'u'],
        ['p', 'sx', 'u', 'cx'],
        ['cz', 'rx', 'rz'],
        ['rxx', 'rx', 'ry'],
        ['iswap', 'rx', 'rz'],
        ['u1', 'rx'],
        ['rz', 'sx'],
        ['p', 'sx'],
        ['r'],
    )
    def test_in_the_back(self, basis):
        """Optimizations can be in the back of the circuit.
        See https://github.com/Qiskit/qiskit-terra/issues/2004.

        qr0:--[U1]-[U1]-[H]--
        """
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.p(0.3, qr)
        circuit.p(0.4, qr)
        circuit.h(qr)

        expected = QuantumCircuit(qr)
        expected.p(0.7, qr)
        expected.h(qr)

        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)

        self.assertTrue(Operator(circuit).equiv(Operator(result)))

    @ddt.data(
        ['cx', 'u3'],
        ['cz', 'u3'],
        ['cx', 'u'],
        ['p', 'sx', 'u', 'cx'],
        ['cz', 'rx', 'rz'],
        ['rxx', 'rx', 'ry'],
        ['iswap', 'rx', 'rz'],
        ['rz', 'sx'],
        ['u1', 'rx'],
        ['p', 'sx'],
    )
    def test_single_parameterized_circuit(self, basis):
        """Parameters should be treated as opaque gates."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta = Parameter('theta')

        qc.p(0.3, qr)
        qc.p(0.4, qr)
        qc.p(theta, qr)
        qc.p(0.1, qr)
        qc.p(0.2, qr)

        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(qc)

        self.assertTrue(
            Operator(qc.bind_parameters({theta: 3.14})).equiv(
                Operator(result.bind_parameters({theta: 3.14}))))

    @ddt.data(
        ['cx', 'u3'],
        ['cz', 'u3'],
        ['cx', 'u'],
        ['p', 'sx', 'u', 'cx'],
        ['cz', 'rx', 'rz'],
        ['rxx', 'rx', 'ry'],
        ['iswap', 'rx', 'rz'],
        ['u1', 'rx'],
        ['rz', 'sx'],
        ['p', 'sx'],
    )
    def test_parameterized_circuits(self, basis):
        """Parameters should be treated as opaque gates."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta = Parameter('theta')

        qc.p(0.3, qr)
        qc.p(0.4, qr)
        qc.p(theta, qr)
        qc.p(0.1, qr)
        qc.p(0.2, qr)
        qc.p(theta, qr)
        qc.p(0.3, qr)
        qc.p(0.2, qr)

        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(qc)

        self.assertTrue(
            Operator(qc.bind_parameters({theta: 3.14})).equiv(
                Operator(result.bind_parameters({theta: 3.14}))))

    @ddt.data(
        ['cx', 'u3'],
        ['cz', 'u3'],
        ['cx', 'u'],
        ['p', 'sx', 'u', 'cx'],
        ['cz', 'rx', 'rz'],
        ['rxx', 'rx', 'ry'],
        ['iswap', 'rx', 'rz'],
        ['u1', 'rx'],
        ['rz', 'sx'],
        ['p', 'sx'],
    )
    def test_parameterized_expressions_in_circuits(self, basis):
        """Expressions of Parameters should be treated as opaque gates."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta = Parameter('theta')
        phi = Parameter('phi')

        sum_ = theta + phi
        product_ = theta * phi
        qc.p(0.3, qr)
        qc.p(0.4, qr)
        qc.p(theta, qr)
        qc.p(phi, qr)
        qc.p(sum_, qr)
        qc.p(product_, qr)
        qc.p(0.3, qr)
        qc.p(0.2, qr)

        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(qc)

        self.assertTrue(
            Operator(qc.bind_parameters({theta: 3.14, phi: 10})).equiv(
                Operator(result.bind_parameters({theta: 3.14, phi: 10}))))

    def test_identity_xyx(self):
        """Test lone identity gates in rx ry basis are removed."""
        circuit = QuantumCircuit(2)
        circuit.rx(0, 1)
        circuit.ry(0, 0)
        basis = ['rxx', 'rx', 'ry']
        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual([], result.data)

    def test_identity_zxz(self):
        """Test lone identity gates in rx rz basis are removed."""
        circuit = QuantumCircuit(2)
        circuit.rx(0, 1)
        circuit.rz(0, 0)
        basis = ['cz', 'rx', 'rz']
        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual([], result.data)

    def test_identity_psx(self):
        """Test lone identity gates in p sx basis are removed."""
        circuit = QuantumCircuit(1)
        circuit.p(0, 0)
        basis = ['cx', 'p', 'sx']
        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual([], result.data)

    def test_identity_u(self):
        """Test lone identity gates in u basis are removed."""
        circuit = QuantumCircuit(1)
        circuit.u(0, 0, 0, 0)
        basis = ['cx', 'u']
        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual([], result.data)

    def test_identity_u3(self):
        """Test lone identity gates in u3 basis are removed."""
        circuit = QuantumCircuit(1)
        circuit.append(U3Gate(0, 0, 0), [0])
        basis = ['cx', 'u3']
        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual([], result.data)

    def test_identity_r(self):
        """Test lone identity gates in r basis are removed."""
        circuit = QuantumCircuit(1)
        circuit.r(0, 0, 0)
        basis = ['r']
        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual([], result.data)

    def test_identity_u1x(self):
        """Test lone identity gates in u1 rx basis are removed."""
        circuit = QuantumCircuit(2)
        circuit.u1(0, 0)
        circuit.rx(0, 1)
        basis = ['cx', 'u1', 'rx']
        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual([], result.data)

    def test_overcomplete_basis(self):
        """Test optimization with an overcomplete basis."""
        circuit = random_circuit(3, 3, seed=42)
        basis = ['rz', 'rxx', 'rx', 'ry', 'p', 'sx', 'u', 'cx']
        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        basis_translated = passmanager.run(circuit)
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result_full = passmanager.run(basis_translated)
        self.assertTrue(Operator(circuit).equiv(Operator(result_full)))
        self.assertGreater(basis_translated.depth(), result_full.depth())

    def test_euler_decomposition_worse(self):
        """Ensure we don't decompose to a deeper circuit."""
        circuit = QuantumCircuit(1)
        circuit.rx(-np.pi / 2, 0)
        circuit.rz(-np.pi / 2, 0)
        basis = ['rx', 'rz']
        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        # decomposition of circuit will result in 3 gates instead of 2
        # assert optimization pass doesn't use it.
        self.assertEqual(result, circuit)


if __name__ == '__main__':
    unittest.main()
