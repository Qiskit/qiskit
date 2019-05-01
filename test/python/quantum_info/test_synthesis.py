# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""Tests for quantum synthesis methods."""

import unittest

import numpy as np
from qiskit import execute
from qiskit.circuit import QuantumCircuit
from qiskit.extensions.standard import (HGate, IdGate, SdgGate, SGate, XGate,
                                        YGate, ZGate, U3Gate)
from qiskit.providers.basicaer import UnitarySimulatorPy
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.synthesis import cnot_decompose, euler_angles_1q
from qiskit.test import QiskitTestCase

def make_oneq_cliffords():
    """Make as list of 1q Cliffords"""
    ixyz_list = [g().to_matrix() for g in (IdGate, XGate, YGate, ZGate)]
    ih_list = [g().to_matrix() for g in (IdGate, HGate)]
    irs_list = [IdGate().to_matrix(),
                SdgGate().to_matrix() @ HGate().to_matrix(),
                HGate().to_matrix() @ SGate().to_matrix()]
    oneq_cliffords = [Operator(ixyz @ ih @ irs) for ixyz in ixyz_list
                      for ih in ih_list
                      for irs in irs_list]
    return oneq_cliffords

ONEQ_CLIFFORDS = make_oneq_cliffords()

def make_hard_thetas_oneq(smallest=1e-18, factor=3.2, steps=22, phi=0.7, lam=0.9):
    """Make 1q gates with theta/2 close to 0, pi/2, pi, 3pi/2"""
    return [U3Gate(smallest * factor**i, phi, lam) for i in range(steps)] +
           [U3Gate(-smallest * factor**i, phi, lam) for i in range(steps)] +
           [U3Gate(np.pi/2 + smallest * factor**i, phi, lam) for i in range(steps)] +
           [U3Gate(np.pi/2 - smallest * factor**i, phi, lam) for i in range(steps)] +
           [U3Gate(np.pi + smallest * factor**i, phi, lam) for i in range(steps)] +
           [U3Gate(np.pi - smallest * factor**i, phi, lam) for i in range(steps)] +
           [U3Gate(3*np.pi/2 + smallest * factor**i, phi, lam) for i in range(steps)] +
           [U3Gate(3*np.pi/2 - smallest * factor**i, phi, lam) for i in range(steps)]

HARD_THETA_ONEQS = make_hard_thetas_oneq()

class TestSynthesis(QiskitTestCase):
    """Test synthesis methods."""

    def check_one_qubit_euler_angles(self, operator):
        """Check euler_angles_1q works for the given unitary
        """
        with self.subTest(operator=operator):
            angles = euler_angles_1q(operator.data)
            decomp_circuit = QuantumCircuit(1)
            decomp_circuit.u3(*angles, 0)
            result = execute(decomp_circuit, UnitarySimulatorPy()).result()
            decomp_operator = Operator(result.get_unitary())
            tracedist = np.abs(np.trace(operator.data.T.conj() @ decomp_operator.data))-2
            self.assertTrue(np.abs(tracedist) < 1e-15, f"tr(target^dag.U)-2 = {tracedist}")

    def test_one_qubit_euler_angles_clifford(self):
        """Verify euler_angles_1q produces correct Euler angles for all Cliffords.
        """
        for clifford in ONEQ_CLIFFORDS:
            self.check_one_qubit_euler_angles(clifford)

    def test_one_qubit_euler_angles_random(self, nsamples=100):
        """Verify euler_angles_1q produces correct Euler angles for random  unitaries.
        """
        for _ in range(nsamples):
            unitary = random_unitary(2)
            self.check_one_qubit_euler_angles(unitary)

    def test_two_qubit_kak(self):
        """Verify KAK decomposition for random Haar 4x4 unitaries.
        """
        for _ in range(100):
            unitary = random_unitary(4)
            with self.subTest(unitary=unitary):
                decomp_circuit = cnot_decompose(unitary)
                result = execute(decomp_circuit, UnitarySimulatorPy()).result()
                decomp_unitary = Operator(result.get_unitary())
                equal_up_to_phase = matrix_equal(
                    unitary.data,
                    decomp_unitary.data,
                    ignore_phase=True,
                    atol=1e-7)
                self.assertTrue(equal_up_to_phase)

    def test_two_qubit_kak_from_paulis(self):
        """Verify decomposing Paulis with KAK
        """
        pauli_xz = Pauli(label='XZ')
        unitary = Operator(pauli_xz)
        decomp_circuit = cnot_decompose(unitary)
        result = execute(decomp_circuit, UnitarySimulatorPy()).result()
        decomp_unitary = Operator(result.get_unitary())
        equal_up_to_phase = matrix_equal(
            unitary.data, decomp_unitary.data, ignore_phase=True, atol=1e-7)
        self.assertTrue(equal_up_to_phase)


if __name__ == '__main__':
    unittest.main()
