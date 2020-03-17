# -*- coding: utf-8 -*-

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


"""Test hardcoded decomposition rules and matrix definitions for standard gates."""

from qiskit import QuantumCircuit, BasicAer, execute
from qiskit.quantum_info import Operator
from qiskit.test import QiskitTestCase
from qiskit.quantum_info.operators.predicates import matrix_equal


class TestGateDefinitions(QiskitTestCase):
    """Test the decomposition of a gate in terms of other gates
    yields the same matrix as the hardcoded matrix definition."""

    def test_ch_definition(self):  # TODO: expand this to all gates
        """Test ch gate matrix and definition.
        """
        circ = QuantumCircuit(2)
        circ.ch(0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_ccx_definition(self):
        """Test ccx gate matrix and definition.
        """
        circ = QuantumCircuit(3)
        circ.ccx(0, 1, 2)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_crz_definition(self):
        """Test crz gate matrix and definition.
        """
        circ = QuantumCircuit(2)
        circ.crz(1, 0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cry_definition(self):
        """Test cry gate matrix and definition.
        """
        circ = QuantumCircuit(2)
        circ.cry(1, 0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_crx_definition(self):
        """Test crx gate matrix and definition.
        """
        circ = QuantumCircuit(2)
        circ.crx(1, 0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cswap_definition(self):
        """Test cswap gate matrix and definition.
        """
        circ = QuantumCircuit(3)
        circ.cswap(0, 1, 2)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cu1_definition(self):
        """Test cu1 gate matrix and definition.
        """
        circ = QuantumCircuit(2)
        circ.cu1(1, 0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cu3_definition(self):
        """Test cu3 gate matrix and definition.
        """
        circ = QuantumCircuit(2)
        circ.cu3(1, 1, 1, 0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cx_definition(self):
        """Test cx gate matrix and definition.
        """
        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_ryy_matrix_representation(self):
        """Test the matrix representation of the RYY gate.
        """
        from qiskit.extensions.standard.ryy import RYYGate
        theta = 0.991283
        expected = RYYGate(theta).to_matrix()

        circuit = QuantumCircuit(2)
        circuit.ryy(theta, 0, 1)
        backend = BasicAer.get_backend('unitary_simulator')
        simulated = execute(circuit, backend).result().get_unitary()

        self.assertTrue(matrix_equal(expected, simulated))
