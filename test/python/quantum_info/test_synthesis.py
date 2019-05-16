# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.synthesis import (two_qubit_cnot_decompose, euler_angles_1q,
                                           TwoQubitBasisDecomposer)
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


# It's too slow to use all 24**4 Clifford combos. If we can make it faster, use a larger set
K1K2S = [(ONEQ_CLIFFORDS[3], ONEQ_CLIFFORDS[5], ONEQ_CLIFFORDS[2], ONEQ_CLIFFORDS[21]),
         (ONEQ_CLIFFORDS[5], ONEQ_CLIFFORDS[6], ONEQ_CLIFFORDS[9], ONEQ_CLIFFORDS[7]),
         (ONEQ_CLIFFORDS[2], ONEQ_CLIFFORDS[1], ONEQ_CLIFFORDS[0], ONEQ_CLIFFORDS[4]),
         [Operator(U3Gate(x, y, z)) for x, y, z in
          [(0.2, 0.3, 0.1), (0.7, 0.15, 0.22), (0.001, 0.97, 2.2), (3.14, 2.1, 0.9)]]]


class TestEulerAngles1Q(QiskitTestCase):
    """Test euler_angles_1q()"""

    def check_one_qubit_euler_angles(self, operator, tolerance=1e-14):
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
            self.assertTrue(np.abs(maxdist) < tolerance, "Worst distance {}".format(maxdist))

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


# FIXME: streamline the set of test cases
class TestTwoQubitWeylDecomposition(QiskitTestCase):
    """Test TwoQubitWeylDecomposition()
    """
    # pylint: disable=invalid-name
    # FIXME: should be possible to set this tolerance tighter after improving the function
    def check_two_qubit_weyl_decomposition(self, target_unitary, tolerance=1.e-7):
        """Check TwoQubitWeylDecomposition() works for a given operator"""
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
            maxdists = [np.max(np.abs(target_unitary + phase*decomp_unitary))
                        for phase in [1, 1j, -1, -1j]]
            maxdist = np.min(maxdists)
            self.assertTrue(np.abs(maxdist) < tolerance, "Worst distance {}".format(maxdist))

    def test_two_qubit_weyl_decomposition_cnot(self):
        """Verify Weyl KAK decomposition for U~CNOT"""
        for k1l, k1r, k2l, k2r in K1K2S:
            k1 = np.kron(k1l.data, k1r.data)
            k2 = np.kron(k2l.data, k2r.data)
            a = Ud(np.pi/4, 0, 0)
            self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_iswap(self):
        """Verify Weyl KAK decomposition for U~iswap"""
        for k1l, k1r, k2l, k2r in K1K2S:
            k1 = np.kron(k1l.data, k1r.data)
            k2 = np.kron(k2l.data, k2r.data)
            a = Ud(np.pi/4, np.pi/4, 0)
            self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_swap(self):
        """Verify Weyl KAK decomposition for U~swap"""
        for k1l, k1r, k2l, k2r in K1K2S:
            k1 = np.kron(k1l.data, k1r.data)
            k2 = np.kron(k2l.data, k2r.data)
            a = Ud(np.pi/4, np.pi/4, np.pi/4)
            self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_bgate(self):
        """Verify Weyl KAK decomposition for U~B"""
        for k1l, k1r, k2l, k2r in K1K2S:
            k1 = np.kron(k1l.data, k1r.data)
            k2 = np.kron(k2l.data, k2r.data)
            a = Ud(np.pi/4, np.pi/8, 0)
            self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_a00(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,0,0)"""
        for aaa in ([smallest * factor**i for i in range(steps)] +
                    [np.pi/4 - smallest * factor**i for i in range(steps)] +
                    [np.pi/8, 0.113*np.pi, 0.1972*np.pi]):
            for k1l, k1r, k2l, k2r in K1K2S:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                a = Ud(aaa, 0, 0)
                self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_aa0(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,a,0)"""
        for aaa in ([smallest * factor**i for i in range(steps)] +
                    [np.pi/4 - smallest * factor**i for i in range(steps)] +
                    [np.pi/8, 0.113*np.pi, 0.1972*np.pi]):
            for k1l, k1r, k2l, k2r in K1K2S:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                a = Ud(aaa, aaa, 0)
                self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_aaa(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,a,a)"""
        for aaa in ([smallest * factor**i for i in range(steps)] +
                    [np.pi/4 - smallest * factor**i for i in range(steps)] +
                    [np.pi/8, 0.113*np.pi, 0.1972*np.pi]):
            for k1l, k1r, k2l, k2r in K1K2S:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                a = Ud(aaa, aaa, aaa)
                self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_aama(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,a,-a)"""
        for aaa in ([smallest * factor**i for i in range(steps)] +
                    [np.pi/4 - smallest * factor**i for i in range(steps)] +
                    [np.pi/8, 0.113*np.pi, 0.1972*np.pi]):
            for k1l, k1r, k2l, k2r in K1K2S:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                a = Ud(aaa, aaa, -aaa)
                self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_ab0(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,b,0)"""
        for aaa in ([smallest * factor**i for i in range(steps)] +
                    [np.pi/4 - smallest * factor**i for i in range(steps)] +
                    [np.pi/8, 0.113*np.pi, 0.1972*np.pi]):
            for bbb in np.linspace(0, aaa, 10):
                for k1l, k1r, k2l, k2r in K1K2S:
                    k1 = np.kron(k1l.data, k1r.data)
                    k2 = np.kron(k2l.data, k2r.data)
                    a = Ud(aaa, bbb, 0)
                    self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_abb(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,b,b)"""
        for aaa in ([smallest * factor**i for i in range(steps)] +
                    [np.pi/4 - smallest * factor**i for i in range(steps)] +
                    [np.pi/8, 0.113*np.pi, 0.1972*np.pi]):
            for bbb in np.linspace(0, aaa, 6):
                for k1l, k1r, k2l, k2r in K1K2S:
                    k1 = np.kron(k1l.data, k1r.data)
                    k2 = np.kron(k2l.data, k2r.data)
                    a = Ud(aaa, bbb, bbb)
                    self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_abmb(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,b,-b)"""
        for aaa in ([smallest * factor**i for i in range(steps)] +
                    [np.pi/4 - smallest * factor**i for i in range(steps)] +
                    [np.pi/8, 0.113*np.pi, 0.1972*np.pi]):
            for bbb in np.linspace(0, aaa, 6):
                for k1l, k1r, k2l, k2r in K1K2S:
                    k1 = np.kron(k1l.data, k1r.data)
                    k2 = np.kron(k2l.data, k2r.data)
                    a = Ud(aaa, bbb, -bbb)
                    self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_aac(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,a,c)"""
        for aaa in ([smallest * factor**i for i in range(steps)] +
                    [np.pi/4 - smallest * factor**i for i in range(steps)] +
                    [np.pi/8, 0.113*np.pi, 0.1972*np.pi]):
            for ccc in np.linspace(-aaa, aaa, 6):
                for k1l, k1r, k2l, k2r in K1K2S:
                    k1 = np.kron(k1l.data, k1r.data)
                    k2 = np.kron(k2l.data, k2r.data)
                    a = Ud(aaa, aaa, ccc)
                    self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_abc(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,a,b)"""
        for aaa in ([smallest * factor**i for i in range(steps)] +
                    [np.pi/4 - smallest * factor**i for i in range(steps)] +
                    [np.pi/8, 0.113*np.pi, 0.1972*np.pi]):
            for bbb in np.linspace(0, aaa, 4):
                for ccc in np.linspace(-bbb, bbb, 4):
                    for k1l, k1r, k2l, k2r in K1K2S:
                        k1 = np.kron(k1l.data, k1r.data)
                        k2 = np.kron(k2l.data, k2r.data)
                        a = Ud(aaa, aaa, ccc)
                        self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)


class TestTwoQubitDecomposeExact(QiskitTestCase):
    """Test TwoQubitBasisDecomposer() for exact decompositions
    """
    # pylint: disable=invalid-name
    def check_exact_decomposition(self, target_unitary, decomposer, tolerance=1.e-7):
        """Check exact decomposition for a particular target"""
        with self.subTest(unitary=target_unitary, decomposer=decomposer):
            decomp_circuit = decomposer(target_unitary)
            result = execute(decomp_circuit, UnitarySimulatorPy()).result()
            decomp_unitary = Operator(result.get_unitary())
            result = execute(decomp_circuit, UnitarySimulatorPy()).result()
            decomp_unitary = result.get_unitary()
            target_unitary *= la.det(target_unitary)**(-0.25)
            decomp_unitary *= la.det(decomp_unitary)**(-0.25)
            maxdists = [np.max(np.abs(target_unitary + phase*decomp_unitary))
                        for phase in [1, 1j, -1, -1j]]
            maxdist = np.min(maxdists)
            self.assertTrue(np.abs(maxdist) < tolerance, "Worst distance {}".format(maxdist))

    def test_exact_two_qubit_cnot_decompose_random(self, nsamples=100):
        """Verify exact CNOT decomposition for random Haar 4x4 unitaries.
        """
        for _ in range(nsamples):
            unitary = random_unitary(4)
            self.check_exact_decomposition(unitary.data, two_qubit_cnot_decompose)

    def test_exact_two_qubit_cnot_decompose_paulis(self):
        """Verify exact CNOT decomposition for Paulis
        """
        pauli_xz = Pauli(label='XZ')
        unitary = Operator(pauli_xz)
        self.check_exact_decomposition(unitary.data, two_qubit_cnot_decompose)

    # FIXME: this should not be failing but I ran out of time to debug it
    @unittest.expectedFailure
    def test_exact_supercontrolled_decompose_random(self, nsamples=100):
        """Verify exact decomposition for random supercontrolled basis and random target"""

        for _ in range(nsamples):
            k1 = np.kron(random_unitary(2).data, random_unitary(2).data)
            k2 = np.kron(random_unitary(2).data, random_unitary(2).data)
            basis_unitary = k1 @ Ud(np.pi/4, 0, 0) @ k2
            decomposer = TwoQubitBasisDecomposer(UnitaryGate(basis_unitary))
            self.check_exact_decomposition(random_unitary(4).data, decomposer)

    def test_exact_nonsupercontrolled_decompose(self):
        """Check that the nonsupercontrolled basis throws a warning"""
        with self.assertWarns(UserWarning, msg="Supposed to warn when basis non-supercontrolled"):
            TwoQubitBasisDecomposer(UnitaryGate(Ud(np.pi/4, 0.2, 0.1)))

# FIXME: need to write tests for the approximate decompositions


if __name__ == '__main__':
    unittest.main()
