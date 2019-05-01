# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""Tests for quantum synthesis methods."""

import unittest

import numpy as np
import scipy.linalg as la
from qiskit import execute
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.extensions import UnitaryGate
from qiskit.extensions.standard import (HGate, IdGate, SdgGate, SGate, U3Gate,
                                        XGate, YGate, ZGate)
from qiskit.providers.basicaer import UnitarySimulatorPy
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.synthesis import (cnot_decompose, euler_angles_1q,
                                           two_qubit_decompose)
from qiskit.quantum_info.synthesis.two_qubit_decompose import (TwoQubitWeylDecomposition,
                                                               Ud)
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
    return ([U3Gate(smallest * factor**i, phi, lam) for i in range(steps)] +
            [U3Gate(-smallest * factor**i, phi, lam) for i in range(steps)] +
            [U3Gate(np.pi/2 + smallest * factor**i, phi, lam) for i in range(steps)] +
            [U3Gate(np.pi/2 - smallest * factor**i, phi, lam) for i in range(steps)] +
            [U3Gate(np.pi + smallest * factor**i, phi, lam) for i in range(steps)] +
            [U3Gate(np.pi - smallest * factor**i, phi, lam) for i in range(steps)] +
            [U3Gate(3*np.pi/2 + smallest * factor**i, phi, lam) for i in range(steps)] +
            [U3Gate(3*np.pi/2 - smallest * factor**i, phi, lam) for i in range(steps)])

HARD_THETA_ONEQS = make_hard_thetas_oneq()

class TestSynthesis(QiskitTestCase):
    """Test synthesis methods."""

    def check_one_qubit_euler_angles(self, operator):
        """Check euler_angles_1q works for the given unitary
        """
        with self.subTest(operator=operator):
            target_unitary = operator.data
            angles = euler_angles_1q(target_unitary)
            decomp_circuit = QuantumCircuit(1)
            decomp_circuit.u3(*angles, 0)
            result = execute(decomp_circuit, UnitarySimulatorPy()).result()
            decomp_unitary = result.get_unitary()
            target_unitary *= la.det(target_unitary)**(-0.5)
            decomp_unitary *= la.det(decomp_unitary)**(-0.5)
            maxdist = np.max(np.abs(target_unitary - decomp_unitary))
            if maxdist > 0.1:
                maxdist = np.max(np.abs(target_unitary + decomp_unitary))
            self.assertTrue(np.abs(maxdist) < 1e-14, f"Worst distance {maxdist}")

    def test_one_qubit_euler_angles_clifford(self):
        """Verify euler_angles_1q produces correct Euler angles for all Cliffords.
        """
        for clifford in ONEQ_CLIFFORDS:
            self.check_one_qubit_euler_angles(clifford)

    def test_one_qubit_hard_thetas(self):
        """Verify euler_angles_1q for close-to-degenerate theta"""
        for gate in HARD_THETA_ONEQS:
            self.check_one_qubit_euler_angles(Operator(gate))

    def test_one_qubit_euler_angles_random(self, nsamples=100):
        """Verify euler_angles_1q produces correct Euler angles for random unitaries.
        """
        for _ in range(nsamples):
            unitary = random_unitary(2)
            self.check_one_qubit_euler_angles(unitary)

    def check_two_qubit_weyl_decomposition(self, target_unitary):
        """Check TwoQubitWeylDecomposition works for a given operator"""
        with self.subTest(unitary=target_unitary):
            decomp = TwoQubitWeylDecomposition(target_unitary)
            q = QuantumRegister(2)
            decomp_circuit = QuantumCircuit(q)
            decomp_circuit.append(UnitaryGate(decomp.K2r), [q[0]])
            decomp_circuit.append(UnitaryGate(decomp.K2l), [q[1]])
            decomp_circuit.append(UnitaryGate(Ud(decomp.a, decomp.b, decomp.c)), [q[0], q[1]])
            decomp_circuit.append(UnitaryGate(decomp.K1r), [q[0]])
            decomp_circuit.append(UnitaryGate(decomp.K1l), [q[1]])
            result = execute(decomp_circuit, UnitarySimulatorPy()).result()
            decomp_unitary = result.get_unitary()
            target_unitary *= la.det(target_unitary)**(-0.25)
            decomp_unitary *= la.det(decomp_unitary)**(-0.25)
            maxdists = [np.max(np.abs(target_unitary + phase*decomp_unitary)) for phase in [1, 1j, -1, -1j]]
            maxdist = np.min(maxdists)
            # FIXME: should be possible to set this tolerance tighter after improving the function
            self.assertTrue(np.abs(maxdist) < 1e-7, f"Worst distance {maxdist}")

    def test_two_qubit_weyl_decomposition_a00(self, smallest=1e-18, factor=9.8, steps=11):
        for aaa in ([smallest * factor**i for i in range(steps)] +
                    [np.pi/4  - smallest * factor**i for i in range(steps)] +
                    [np.pi/8, 0.113*np.pi, 0.1972*np.pi]):
            for k1l in ONEQ_CLIFFORDS[:3]:
                for k1r in ONEQ_CLIFFORDS[:3]:
                    for k2l in ONEQ_CLIFFORDS[:3]:
                        for k2r in ONEQ_CLIFFORDS[:3]:
                            k1 = np.kron(k1l.data, k1r.data)
                            k2 = np.kron(k2l.data, k2r.data)
                            a = Ud(aaa, 0, 0)
                            self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

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
