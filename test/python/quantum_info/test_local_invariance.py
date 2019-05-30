# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for local invariance routines."""

import unittest

import numpy as np
from qiskit.execute import execute
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.test import QiskitTestCase
from qiskit.providers.basicaer import UnitarySimulatorPy
from qiskit.quantum_info.synthesis.local_invariance import (two_qubit_local_invariants,
                                                            maximally_entangling,
                                                            cx_equivalence)


class TestLocalInvariance(QiskitTestCase):
    """Test local invariance routines"""

    def test_2q_local_invariance_simple(self):
        """Check the local invariance parameters
        for known simple cases.
        """
        sim = UnitarySimulatorPy()

        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)
        U = execute(qc, sim).result().get_unitary()
        vec = two_qubit_local_invariants(U)
        self.assertTrue(np.allclose(vec, [1, 0, 3]))

        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)
        qc.cx(qr[1], qr[0])
        U = execute(qc, sim).result().get_unitary()
        vec = two_qubit_local_invariants(U)
        self.assertTrue(np.allclose(vec, [0, 0, 1]))

        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)
        qc.cx(qr[1], qr[0])
        qc.cx(qr[0], qr[1])
        U = execute(qc, sim).result().get_unitary()
        vec = two_qubit_local_invariants(U)
        self.assertTrue(np.allclose(vec, [0, 0, -1]))

        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)
        qc.swap(qr[1], qr[0])
        U = execute(qc, sim).result().get_unitary()
        vec = two_qubit_local_invariants(U)
        self.assertTrue(np.allclose(vec, [-1, 0, -3]))

    def test_cx_equivalence_0cx_random(self):
        """Check random circuits with  0 cx
        gates locally eqivilent to identity
        """
        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)

        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[0])
        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[1])

        sim = UnitarySimulatorPy()
        U = execute(qc, sim).result().get_unitary()
        self.assertEqual(cx_equivalence(U), 0)

    def test_cx_equivalence_1cx_random(self):
        """Check random circuits with  1 cx
        gates locally eqivilent to a cx
        """
        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)

        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[0])
        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[1])

        qc.cx(qr[1], qr[0])

        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[0])
        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[1])

        sim = UnitarySimulatorPy()
        U = execute(qc, sim).result().get_unitary()
        self.assertEqual(cx_equivalence(U), 1)

    def test_cx_equivalence_2cx_random(self):
        """Check random circuits with  2 cx
        gates locally eqivilent to some
        circuit with 2 cx.
        """
        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)

        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[0])
        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[1])

        qc.cx(qr[1], qr[0])

        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[0])
        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[1])

        qc.cx(qr[0], qr[1])

        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[0])
        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[1])

        sim = UnitarySimulatorPy()
        U = execute(qc, sim).result().get_unitary()
        self.assertEqual(cx_equivalence(U), 2)

    def test_cx_equivalence_3cx_random(self):
        """Check random circuits with 3 cx
        gates are outside the 0, 1, and 2
        qubit regions.
        """
        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)

        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[0])
        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[1])

        qc.cx(qr[1], qr[0])

        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[0])
        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[1])

        qc.cx(qr[0], qr[1])

        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[0])
        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[1])

        qc.cx(qr[1], qr[0])

        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[0])
        rnd = 2*np.pi*np.random.random(size=3)
        qc.u3(rnd[0], rnd[1], rnd[2], qr[1])

        sim = UnitarySimulatorPy()
        U = execute(qc, sim).result().get_unitary()
        self.assertEqual(cx_equivalence(U), 3)

    def test_maximally_entangling_simple(self):
        """Check maximally_entangling for simple
        cases.
        """
        sim = UnitarySimulatorPy()

        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)
        U = execute(qc, sim).result().get_unitary()
        self.assertFalse(maximally_entangling(U))

        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)
        qc.cx(qr[1], qr[0])
        U = execute(qc, sim).result().get_unitary()
        self.assertTrue(maximally_entangling(U))

        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)
        qc.cx(qr[1], qr[0])
        qc.cx(qr[0], qr[1])
        U = execute(qc, sim).result().get_unitary()
        self.assertTrue(maximally_entangling(U))

        qr = QuantumRegister(2, name='q')
        qc = QuantumCircuit(qr)
        qc.swap(qr[1], qr[0])
        U = execute(qc, sim).result().get_unitary()
        self.assertFalse(maximally_entangling(U))


if __name__ == '__main__':
    unittest.main()
