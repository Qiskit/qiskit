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
from test import combine
from ddt import ddt

import numpy as np
import scipy.linalg as la

from qiskit import execute
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import (HGate, IGate, SdgGate, SGate, U3Gate, UGate,
                                    XGate, YGate, ZGate, CXGate, CZGate,
                                    iSwapGate, RXXGate)
from qiskit.providers.basicaer import UnitarySimulatorPy
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import (TwoQubitWeylDecomposition,
                                                               two_qubit_cnot_decompose,
                                                               TwoQubitBasisDecomposer,
                                                               Ud)
from qiskit.quantum_info.synthesis.ion_decompose import cnot_rxx_decompose
from qiskit.test import QiskitTestCase


def make_oneq_cliffords():
    """Make as list of 1q Cliffords"""
    ixyz_list = [g().to_matrix() for g in (IGate, XGate, YGate, ZGate)]
    ih_list = [g().to_matrix() for g in (IGate, HGate)]
    irs_list = [IGate().to_matrix(),
                SdgGate().to_matrix() @ HGate().to_matrix(),
                HGate().to_matrix() @ SGate().to_matrix()]
    oneq_cliffords = [Operator(ixyz @ ih @ irs) for ixyz in ixyz_list
                      for ih in ih_list
                      for irs in irs_list]
    return oneq_cliffords


ONEQ_CLIFFORDS = make_oneq_cliffords()


def make_hard_thetas_oneq(smallest=1e-18, factor=3.2, steps=22, phi=0.7, lam=0.9):
    """Make 1q gates with theta/2 close to 0, pi/2, pi, 3pi/2"""
    return ([U3Gate(smallest * factor ** i, phi, lam) for i in range(steps)] +
            [U3Gate(-smallest * factor ** i, phi, lam) for i in range(steps)] +
            [U3Gate(np.pi / 2 + smallest * factor ** i, phi, lam) for i in range(steps)] +
            [U3Gate(np.pi / 2 - smallest * factor ** i, phi, lam) for i in range(steps)] +
            [U3Gate(np.pi + smallest * factor ** i, phi, lam) for i in range(steps)] +
            [U3Gate(np.pi - smallest * factor ** i, phi, lam) for i in range(steps)] +
            [U3Gate(3 * np.pi / 2 + smallest * factor ** i, phi, lam) for i in range(steps)] +
            [U3Gate(3 * np.pi / 2 - smallest * factor ** i, phi, lam) for i in range(steps)])


HARD_THETA_ONEQS = make_hard_thetas_oneq()

# It's too slow to use all 24**4 Clifford combos. If we can make it faster, use a larger set
K1K2S = [(ONEQ_CLIFFORDS[3], ONEQ_CLIFFORDS[5], ONEQ_CLIFFORDS[2], ONEQ_CLIFFORDS[21]),
         (ONEQ_CLIFFORDS[5], ONEQ_CLIFFORDS[6], ONEQ_CLIFFORDS[9], ONEQ_CLIFFORDS[7]),
         (ONEQ_CLIFFORDS[2], ONEQ_CLIFFORDS[1], ONEQ_CLIFFORDS[0], ONEQ_CLIFFORDS[4]),
         [Operator(U3Gate(x, y, z)) for x, y, z in
          [(0.2, 0.3, 0.1), (0.7, 0.15, 0.22), (0.001, 0.97, 2.2), (3.14, 2.1, 0.9)]]]


class CheckDecompositions(QiskitTestCase):
    """Implements decomposition checkers."""

    def check_one_qubit_euler_angles(self, operator, basis='U3', tolerance=1e-12,
                                     phase_equal=True):
        """Check OneQubitEulerDecomposer works for the given unitary"""
        target_unitary = operator.data
        if basis is None:
            angles = OneQubitEulerDecomposer().angles(target_unitary)
            decomp_unitary = U3Gate(*angles).to_matrix()
        else:
            decomposer = OneQubitEulerDecomposer(basis)
            decomp_unitary = Operator(decomposer(target_unitary)).data
        # Add global phase to make special unitary
        target_unitary *= la.det(target_unitary) ** (-0.5)
        decomp_unitary *= la.det(decomp_unitary) ** (-0.5)
        maxdist = np.max(np.abs(target_unitary - decomp_unitary))
        if not phase_equal and maxdist > 0.1:
            maxdist = np.max(np.abs(target_unitary + decomp_unitary))
        self.assertTrue(np.abs(maxdist) < tolerance,
                        "Operator {}: Worst distance {}".format(operator, maxdist))

    # FIXME: should be possible to set this tolerance tighter after improving the function
    def check_two_qubit_weyl_decomposition(self, target_unitary, tolerance=1.e-7):
        """Check TwoQubitWeylDecomposition() works for a given operator"""
        # pylint: disable=invalid-name
        decomp = TwoQubitWeylDecomposition(target_unitary)
        op = np.exp(1j * decomp.global_phase) * Operator(np.eye(4))
        for u, qs in (
                (decomp.K2r, [0]),
                (decomp.K2l, [1]),
                (Ud(decomp.a, decomp.b, decomp.c), [0, 1]),
                (decomp.K1r, [0]),
                (decomp.K1l, [1]),
        ):
            op = op.compose(u, qs)
        decomp_unitary = op.data
        maxdist = np.max(np.abs(target_unitary - decomp_unitary))
        self.assertTrue(np.abs(maxdist) < tolerance,
                        "Unitary {}: Worst distance {}".format(target_unitary, maxdist))

    def check_exact_decomposition(self, target_unitary, decomposer, tolerance=1.e-7):
        """Check exact decomposition for a particular target"""
        decomp_circuit = decomposer(target_unitary)
        result = execute(decomp_circuit, UnitarySimulatorPy(), optimization_level=0).result()
        decomp_unitary = result.get_unitary()
        maxdist = np.max(np.abs(target_unitary - decomp_unitary))
        self.assertTrue(np.abs(maxdist) < tolerance,
                        "Unitary {}: Worst distance {}".format(target_unitary, maxdist))


@ddt
class TestEulerAngles1Q(CheckDecompositions):
    """Test euler_angles_1q()"""

    @combine(clifford=ONEQ_CLIFFORDS)
    def test_euler_angles_1q_clifford(self, clifford):
        """Verify euler_angles_1q produces correct Euler angles for all Cliffords."""
        self.check_one_qubit_euler_angles(clifford)

    @combine(gate=HARD_THETA_ONEQS)
    def test_euler_angles_1q_hard_thetas(self, gate):
        """Verify euler_angles_1q for close-to-degenerate theta"""
        self.check_one_qubit_euler_angles(Operator(gate))

    @combine(seed=range(5), name='test_euler_angles_1q_random_{seed}')
    def test_euler_angles_1q_random(self, seed):
        """Verify euler_angles_1q produces correct Euler angles for random_unitary (seed={seed}).
        """
        unitary = random_unitary(2, seed=seed)
        self.check_one_qubit_euler_angles(unitary)


@ddt
class TestOneQubitEulerDecomposer(CheckDecompositions):
    """Test OneQubitEulerDecomposer"""

    def check_one_qubit_euler_angles(self, operator, basis='U3',
                                     tolerance=1e-12,
                                     phase_equal=True):
        """Check euler_angles_1q works for the given unitary"""
        decomposer = OneQubitEulerDecomposer(basis)
        with self.subTest(operator=operator):
            target_unitary = operator.data
            decomp_unitary = Operator(decomposer(target_unitary)).data
            # Add global phase to make special unitary
            target_unitary *= la.det(target_unitary) ** (-0.5)
            decomp_unitary *= la.det(decomp_unitary) ** (-0.5)
            maxdist = np.max(np.abs(target_unitary - decomp_unitary))
            if not phase_equal and maxdist > 0.1:
                maxdist = np.max(np.abs(target_unitary + decomp_unitary))
            self.assertTrue(np.abs(maxdist) < tolerance, "Worst distance {}".format(maxdist))

    @combine(basis=['U3', 'U1X', 'PSX', 'ZSX', 'ZYZ', 'ZXZ', 'XYX', 'RR'],
             name='test_one_qubit_clifford_{basis}_basis')
    def test_one_qubit_clifford_all_basis(self, basis):
        """Verify for {basis} basis and all Cliffords."""
        for clifford in ONEQ_CLIFFORDS:
            self.check_one_qubit_euler_angles(clifford, basis)

    @combine(basis_tolerance=[('U3', 1e-12),
                              ('XYX', 1e-12),
                              ('ZXZ', 1e-12),
                              ('ZYZ', 1e-12),
                              ('U1X', 1e-7),
                              ('PSX', 1e-7),
                              ('ZSX', 1e-7),
                              ('RR', 1e-12)],
             name='test_one_qubit_hard_thetas_{basis_tolerance[0]}_basis')
    # Lower tolerance for U1X test since decomposition since it is
    # less numerically accurate. This is due to it having 5 matrix
    # multiplications and the X90 gates
    def test_one_qubit_hard_thetas_all_basis(self, basis_tolerance):
        """Verify for {basis_tolerance[0]} basis and close-to-degenerate theta."""
        for gate in HARD_THETA_ONEQS:
            self.check_one_qubit_euler_angles(Operator(gate), basis_tolerance[0],
                                              basis_tolerance[1])

    @combine(basis=['U3', 'U1X', 'PSX', 'ZSX', 'ZYZ', 'ZXZ', 'XYX', 'RR'], seed=range(50),
             name='test_one_qubit_random_{basis}_basis_{seed}')
    def test_one_qubit_random_all_basis(self, basis, seed):
        """Verify for {basis} basis and random_unitary (seed={seed})."""
        unitary = random_unitary(2, seed=seed)
        self.check_one_qubit_euler_angles(unitary, basis)

    def test_psx_zsx_special_cases(self):
        """Test decompositions of psx and zsx at special values of parameters"""
        oqed_psx = OneQubitEulerDecomposer(basis='PSX')
        oqed_zsx = OneQubitEulerDecomposer(basis='ZSX')
        theta = np.pi / 3
        phi = np.pi / 5
        lam = np.pi / 7
        test_gates = [UGate(np.pi, phi, lam), UGate(-np.pi, phi, lam),
                      # test abs(lam + phi + theta) near 0
                      UGate(np.pi, np.pi / 3, 2 * np.pi / 3),
                      # test theta=pi/2
                      UGate(np.pi / 2, phi, lam),
                      # test theta=pi/2 and theta+lam=0
                      UGate(np.pi / 2, phi, -np.pi / 2),
                      # test theta close to 3*pi/2 and theta+phi=2*pi
                      UGate(3*np.pi / 2, np.pi / 2, lam),
                      # test theta 0
                      UGate(0, phi, lam),
                      # test phi 0
                      UGate(theta, 0, lam),
                      # test lam 0
                      UGate(theta, phi, 0)]

        for gate in test_gates:
            unitary = gate.to_matrix()
            qc_psx = oqed_psx(unitary)
            qc_zsx = oqed_zsx(unitary)
            self.assertTrue(np.allclose(unitary, Operator(qc_psx).data))
            self.assertTrue(np.allclose(unitary, Operator(qc_zsx).data))


# FIXME: streamline the set of test cases
class TestTwoQubitWeylDecomposition(CheckDecompositions):
    """Test TwoQubitWeylDecomposition()
    """

    def test_two_qubit_weyl_decomposition_cnot(self):
        """Verify Weyl KAK decomposition for U~CNOT"""
        for k1l, k1r, k2l, k2r in K1K2S:
            k1 = np.kron(k1l.data, k1r.data)
            k2 = np.kron(k2l.data, k2r.data)
            a = Ud(np.pi / 4, 0, 0)
            self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_iswap(self):
        """Verify Weyl KAK decomposition for U~iswap"""
        for k1l, k1r, k2l, k2r in K1K2S:
            k1 = np.kron(k1l.data, k1r.data)
            k2 = np.kron(k2l.data, k2r.data)
            a = Ud(np.pi / 4, np.pi / 4, 0)
            self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_swap(self):
        """Verify Weyl KAK decomposition for U~swap"""
        for k1l, k1r, k2l, k2r in K1K2S:
            k1 = np.kron(k1l.data, k1r.data)
            k2 = np.kron(k2l.data, k2r.data)
            a = Ud(np.pi / 4, np.pi / 4, np.pi / 4)
            self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_bgate(self):
        """Verify Weyl KAK decomposition for U~B"""
        for k1l, k1r, k2l, k2r in K1K2S:
            k1 = np.kron(k1l.data, k1r.data)
            k2 = np.kron(k2l.data, k2r.data)
            a = Ud(np.pi / 4, np.pi / 8, 0)
            self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_a00(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,0,0)"""
        for aaa in ([smallest * factor ** i for i in range(steps)] +
                    [np.pi / 4 - smallest * factor ** i for i in range(steps)] +
                    [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]):
            for k1l, k1r, k2l, k2r in K1K2S:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                a = Ud(aaa, 0, 0)
                self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_aa0(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,a,0)"""
        for aaa in ([smallest * factor ** i for i in range(steps)] +
                    [np.pi / 4 - smallest * factor ** i for i in range(steps)] +
                    [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]):
            for k1l, k1r, k2l, k2r in K1K2S:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                a = Ud(aaa, aaa, 0)
                self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_aaa(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,a,a)"""
        for aaa in ([smallest * factor ** i for i in range(steps)] +
                    [np.pi / 4 - smallest * factor ** i for i in range(steps)] +
                    [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]):
            for k1l, k1r, k2l, k2r in K1K2S:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                a = Ud(aaa, aaa, aaa)
                self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_aama(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,a,-a)"""
        for aaa in ([smallest * factor ** i for i in range(steps)] +
                    [np.pi / 4 - smallest * factor ** i for i in range(steps)] +
                    [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]):
            for k1l, k1r, k2l, k2r in K1K2S:
                k1 = np.kron(k1l.data, k1r.data)
                k2 = np.kron(k2l.data, k2r.data)
                a = Ud(aaa, aaa, -aaa)
                self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_ab0(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,b,0)"""
        for aaa in ([smallest * factor ** i for i in range(steps)] +
                    [np.pi / 4 - smallest * factor ** i for i in range(steps)] +
                    [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]):
            for bbb in np.linspace(0, aaa, 10):
                for k1l, k1r, k2l, k2r in K1K2S:
                    k1 = np.kron(k1l.data, k1r.data)
                    k2 = np.kron(k2l.data, k2r.data)
                    a = Ud(aaa, bbb, 0)
                    self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_abb(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,b,b)"""
        for aaa in ([smallest * factor ** i for i in range(steps)] +
                    [np.pi / 4 - smallest * factor ** i for i in range(steps)] +
                    [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]):
            for bbb in np.linspace(0, aaa, 6):
                for k1l, k1r, k2l, k2r in K1K2S:
                    k1 = np.kron(k1l.data, k1r.data)
                    k2 = np.kron(k2l.data, k2r.data)
                    a = Ud(aaa, bbb, bbb)
                    self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_abmb(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,b,-b)"""
        for aaa in ([smallest * factor ** i for i in range(steps)] +
                    [np.pi / 4 - smallest * factor ** i for i in range(steps)] +
                    [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]):
            for bbb in np.linspace(0, aaa, 6):
                for k1l, k1r, k2l, k2r in K1K2S:
                    k1 = np.kron(k1l.data, k1r.data)
                    k2 = np.kron(k2l.data, k2r.data)
                    a = Ud(aaa, bbb, -bbb)
                    self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_aac(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,a,c)"""
        for aaa in ([smallest * factor ** i for i in range(steps)] +
                    [np.pi / 4 - smallest * factor ** i for i in range(steps)] +
                    [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]):
            for ccc in np.linspace(-aaa, aaa, 6):
                for k1l, k1r, k2l, k2r in K1K2S:
                    k1 = np.kron(k1l.data, k1r.data)
                    k2 = np.kron(k2l.data, k2r.data)
                    a = Ud(aaa, aaa, ccc)
                    self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)

    def test_two_qubit_weyl_decomposition_abc(self, smallest=1e-18, factor=9.8, steps=11):
        """Verify Weyl KAK decomposition for U~Ud(a,a,b)"""
        for aaa in ([smallest * factor ** i for i in range(steps)] +
                    [np.pi / 4 - smallest * factor ** i for i in range(steps)] +
                    [np.pi / 8, 0.113 * np.pi, 0.1972 * np.pi]):
            for bbb in np.linspace(0, aaa, 4):
                for ccc in np.linspace(-bbb, bbb, 4):
                    for k1l, k1r, k2l, k2r in K1K2S:
                        k1 = np.kron(k1l.data, k1r.data)
                        k2 = np.kron(k2l.data, k2r.data)
                        a = Ud(aaa, aaa, ccc)
                        self.check_two_qubit_weyl_decomposition(k1 @ a @ k2)


@ddt
class TestTwoQubitDecomposeExact(CheckDecompositions):
    """Test TwoQubitBasisDecomposer() for exact decompositions
    """

    def test_cnot_rxx_decompose(self):
        """Verify CNOT decomposition into RXX gate is correct"""
        cnot = Operator(CXGate())
        decomps = [cnot_rxx_decompose(),
                   cnot_rxx_decompose(plus_ry=True, plus_rxx=True),
                   cnot_rxx_decompose(plus_ry=True, plus_rxx=False),
                   cnot_rxx_decompose(plus_ry=False, plus_rxx=True),
                   cnot_rxx_decompose(plus_ry=False, plus_rxx=False)]
        for decomp in decomps:
            self.assertTrue(cnot.equiv(decomp))

    @combine(seed=range(10), name='test_exact_two_qubit_cnot_decompose_random_{seed}')
    def test_exact_two_qubit_cnot_decompose_random(self, seed):
        """Verify exact CNOT decomposition for random Haar 4x4 unitary (seed={seed}).
        """
        unitary = random_unitary(4, seed=seed)
        self.check_exact_decomposition(unitary.data, two_qubit_cnot_decompose)

    def test_exact_two_qubit_cnot_decompose_paulis(self):
        """Verify exact CNOT decomposition for Paulis
        """
        unitary = Operator.from_label('XZ')
        self.check_exact_decomposition(unitary.data, two_qubit_cnot_decompose)

    @combine(seed=range(10), name='test_exact_supercontrolled_decompose_random_{seed}')
    def test_exact_supercontrolled_decompose_random(self, seed):
        """Exact decomposition for random supercontrolled basis and random target (seed={seed})"""
        k1 = np.kron(random_unitary(2, seed=seed).data, random_unitary(2, seed=seed + 1).data)
        k2 = np.kron(random_unitary(2, seed=seed + 2).data, random_unitary(2, seed=seed + 3).data)
        basis_unitary = k1 @ Ud(np.pi / 4, 0, 0) @ k2
        decomposer = TwoQubitBasisDecomposer(UnitaryGate(basis_unitary))
        self.check_exact_decomposition(random_unitary(4, seed=seed + 4).data, decomposer)

    def test_exact_nonsupercontrolled_decompose(self):
        """Check that the nonsupercontrolled basis throws a warning"""
        with self.assertWarns(UserWarning, msg="Supposed to warn when basis non-supercontrolled"):
            TwoQubitBasisDecomposer(UnitaryGate(Ud(np.pi / 4, 0.2, 0.1)))

    def test_cx_equivalence_0cx(self, seed=0):
        """Check circuits with  0 cx gates locally equivalent to identity
        """
        state = np.random.default_rng(seed)
        rnd = 2 * np.pi * state.random(size=6)

        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)

        qc.u(rnd[0], rnd[1], rnd[2], qr[0])
        qc.u(rnd[3], rnd[4], rnd[5], qr[1])

        sim = UnitarySimulatorPy()
        unitary = execute(qc, sim).result().get_unitary()
        self.assertEqual(two_qubit_cnot_decompose.num_basis_gates(unitary), 0)
        self.assertTrue(Operator(two_qubit_cnot_decompose(unitary)).equiv(unitary))

    def test_cx_equivalence_1cx(self, seed=1):
        """Check circuits with  1 cx gates locally equivalent to a cx
        """
        state = np.random.default_rng(seed)
        rnd = 2 * np.pi * state.random(size=12)

        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)

        qc.u(rnd[0], rnd[1], rnd[2], qr[0])
        qc.u(rnd[3], rnd[4], rnd[5], qr[1])

        qc.cx(qr[1], qr[0])

        qc.u(rnd[6], rnd[7], rnd[8], qr[0])
        qc.u(rnd[9], rnd[10], rnd[11], qr[1])

        sim = UnitarySimulatorPy()
        unitary = execute(qc, sim).result().get_unitary()
        self.assertEqual(two_qubit_cnot_decompose.num_basis_gates(unitary), 1)
        self.assertTrue(Operator(two_qubit_cnot_decompose(unitary)).equiv(unitary))

    def test_cx_equivalence_2cx(self, seed=2):
        """Check circuits with  2 cx gates locally equivalent to some circuit with 2 cx.
        """
        state = np.random.default_rng(seed)
        rnd = 2 * np.pi * state.random(size=18)

        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)

        qc.u(rnd[0], rnd[1], rnd[2], qr[0])
        qc.u(rnd[3], rnd[4], rnd[5], qr[1])

        qc.cx(qr[1], qr[0])

        qc.u(rnd[6], rnd[7], rnd[8], qr[0])
        qc.u(rnd[9], rnd[10], rnd[11], qr[1])

        qc.cx(qr[0], qr[1])

        qc.u(rnd[12], rnd[13], rnd[14], qr[0])
        qc.u(rnd[15], rnd[16], rnd[17], qr[1])

        sim = UnitarySimulatorPy()
        unitary = execute(qc, sim).result().get_unitary()
        self.assertEqual(two_qubit_cnot_decompose.num_basis_gates(unitary), 2)
        self.assertTrue(Operator(two_qubit_cnot_decompose(unitary)).equiv(unitary))

    def test_cx_equivalence_3cx(self, seed=3):
        """Check circuits with 3 cx gates are outside the 0, 1, and 2 qubit regions.
        """
        state = np.random.default_rng(seed)
        rnd = 2 * np.pi * state.random(size=24)

        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)

        qc.u(rnd[0], rnd[1], rnd[2], qr[0])
        qc.u(rnd[3], rnd[4], rnd[5], qr[1])

        qc.cx(qr[1], qr[0])

        qc.u(rnd[6], rnd[7], rnd[8], qr[0])
        qc.u(rnd[9], rnd[10], rnd[11], qr[1])

        qc.cx(qr[0], qr[1])

        qc.u(rnd[12], rnd[13], rnd[14], qr[0])
        qc.u(rnd[15], rnd[16], rnd[17], qr[1])

        qc.cx(qr[1], qr[0])

        qc.u(rnd[18], rnd[19], rnd[20], qr[0])
        qc.u(rnd[21], rnd[22], rnd[23], qr[1])

        sim = UnitarySimulatorPy()
        unitary = execute(qc, sim).result().get_unitary()
        self.assertEqual(two_qubit_cnot_decompose.num_basis_gates(unitary), 3)
        self.assertTrue(Operator(two_qubit_cnot_decompose(unitary)).equiv(unitary))

    def test_seed_289(self):
        """This specific case failed when PR #3585 was applied
        See https://github.com/Qiskit/qiskit-terra/pull/3652"""
        unitary = random_unitary(4, seed=289)
        self.check_exact_decomposition(unitary.data, two_qubit_cnot_decompose)

    @combine(seed=range(10),
             euler_bases=[('U321', ['u3', 'u2', 'u1']), ('U3', ['u3']), ('U', ['u']),
                          ('U1X', ['u1', 'rx']), ('RR', ['r']),
                          ('PSX', ['p', 'sx']), ('ZYZ', ['rz', 'ry']), ('ZXZ', ['rz', 'rx']),
                          ('XYX', ['rx', 'ry']), ('ZSX', ['rz', 'sx'])],
             kak_gates=[(CXGate(), 'cx'), (CZGate(), 'cz'), (iSwapGate(), 'iswap'),
                        (RXXGate(np.pi / 2), 'rxx')],
             name='test_euler_basis_selection_{seed}_{euler_bases[0]}_{kak_gates[1]}')
    def test_euler_basis_selection(self, euler_bases, kak_gates, seed):
        """Verify decomposition uses euler_basis for 1q gates."""
        (euler_basis, oneq_gates) = euler_bases
        (kak_gate, kak_gate_name) = kak_gates

        with self.subTest(euler_basis=euler_basis, kak_gate=kak_gate):
            decomposer = TwoQubitBasisDecomposer(kak_gate, euler_basis=euler_basis)
            unitary = random_unitary(4, seed=seed)
            self.check_exact_decomposition(unitary.data, decomposer)

            decomposition_basis = set(decomposer(unitary).count_ops())
            requested_basis = set(oneq_gates + [kak_gate_name])
            self.assertTrue(
                decomposition_basis.issubset(requested_basis))


# FIXME: need to write tests for the approximate decompositions


if __name__ == '__main__':
    unittest.main()
