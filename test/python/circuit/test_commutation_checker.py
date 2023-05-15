# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test commutation checker class ."""

import unittest
import numpy as np

from qiskit import ClassicalRegister
from qiskit.test import QiskitTestCase

from qiskit.circuit import QuantumRegister, Parameter, Qubit
from qiskit.circuit import CommutationChecker
from qiskit.circuit.library import (
    ZGate,
    XGate,
    CXGate,
    CCXGate,
    RZGate,
    Measure,
    Barrier,
    Reset,
    LinearFunction,
)


class TestCommutationChecker(QiskitTestCase):
    """Test CommutationChecker class."""

    def test_simple_gates(self):
        """Check simple commutation relations between gates, experimenting with
        different orders of gates, different orders of qubits, different sets of
        qubits over which gates are defined, and so on."""
        comm_checker = CommutationChecker()

        # should commute
        res = comm_checker.commute(ZGate(), [0], [], CXGate(), [0, 1], [])
        self.assertTrue(res)

        # should not commute
        res = comm_checker.commute(ZGate(), [1], [], CXGate(), [0, 1], [])
        self.assertFalse(res)

        # should not commute
        res = comm_checker.commute(XGate(), [0], [], CXGate(), [0, 1], [])
        self.assertFalse(res)

        # should commute
        res = comm_checker.commute(XGate(), [1], [], CXGate(), [0, 1], [])
        self.assertTrue(res)

        # should not commute
        res = comm_checker.commute(XGate(), [1], [], CXGate(), [1, 0], [])
        self.assertFalse(res)

        # should commute
        res = comm_checker.commute(XGate(), [0], [], CXGate(), [1, 0], [])
        self.assertTrue(res)

        # should commute
        res = comm_checker.commute(CXGate(), [1, 0], [], XGate(), [0], [])
        self.assertTrue(res)

        # should not commute
        res = comm_checker.commute(CXGate(), [1, 0], [], XGate(), [1], [])
        self.assertFalse(res)

        # should commute
        res = comm_checker.commute(
            CXGate(),
            [1, 0],
            [],
            CXGate(),
            [1, 0],
            [],
        )
        self.assertTrue(res)

        # should not commute
        res = comm_checker.commute(
            CXGate(),
            [1, 0],
            [],
            CXGate(),
            [0, 1],
            [],
        )
        self.assertFalse(res)

        # should commute
        res = comm_checker.commute(
            CXGate(),
            [1, 0],
            [],
            CXGate(),
            [1, 2],
            [],
        )
        self.assertTrue(res)

        # should not commute
        res = comm_checker.commute(
            CXGate(),
            [1, 0],
            [],
            CXGate(),
            [2, 1],
            [],
        )
        self.assertFalse(res)

        # should commute
        res = comm_checker.commute(
            CXGate(),
            [1, 0],
            [],
            CXGate(),
            [2, 3],
            [],
        )
        self.assertTrue(res)

        res = comm_checker.commute(XGate(), [2], [], CCXGate(), [0, 1, 2], [])
        self.assertTrue(res)

        res = comm_checker.commute(CCXGate(), [0, 1, 2], [], CCXGate(), [0, 2, 1], [])
        self.assertFalse(res)

    def test_passing_quantum_registers(self):
        """Check that passing QuantumRegisters works correctly."""
        comm_checker = CommutationChecker()

        qr = QuantumRegister(4)

        # should commute
        res = comm_checker.commute(CXGate(), [qr[1], qr[0]], [], CXGate(), [qr[1], qr[2]], [])
        self.assertTrue(res)

        # should not commute
        res = comm_checker.commute(CXGate(), [qr[0], qr[1]], [], CXGate(), [qr[1], qr[2]], [])
        self.assertFalse(res)

    def test_caching_positive_results(self):
        """Check that hashing positive results in commutativity checker works as expected."""

        comm_checker = CommutationChecker()
        res = comm_checker.commute(ZGate(), [0], [], CXGate(), [0, 1], [])
        self.assertTrue(res)
        self.assertGreater(len(comm_checker.cache), 0)

    def test_caching_negative_results(self):
        """Check that hashing negative results in commutativity checker works as expected."""

        comm_checker = CommutationChecker()
        res = comm_checker.commute(XGate(), [0], [], CXGate(), [0, 1], [])
        self.assertFalse(res)
        self.assertGreater(len(comm_checker.cache), 0)

    def test_caching_different_qubit_sets(self):
        """Check that hashing same commutativity results over different qubit sets works as expected."""

        comm_checker = CommutationChecker()

        # All the following should be cached in the same way
        # though each relation gets cached twice: (A, B) and (B, A)
        comm_checker.commute(XGate(), [0], [], CXGate(), [0, 1], [])
        comm_checker.commute(XGate(), [10], [], CXGate(), [10, 20], [])
        comm_checker.commute(XGate(), [10], [], CXGate(), [10, 5], [])
        comm_checker.commute(XGate(), [5], [], CXGate(), [5, 7], [])
        self.assertEqual(len(comm_checker.cache), 2)

    def test_gates_with_parameters(self):
        """Check commutativity between (non-parameterized) gates with parameters."""

        comm_checker = CommutationChecker()
        res = comm_checker.commute(RZGate(0), [0], [], XGate(), [0], [])
        self.assertTrue(res)

        res = comm_checker.commute(RZGate(np.pi / 2), [0], [], XGate(), [0], [])
        self.assertFalse(res)

        res = comm_checker.commute(RZGate(np.pi / 2), [0], [], RZGate(0), [0], [])
        self.assertTrue(res)

    def test_parameterized_gates(self):
        """Check commutativity between parameterized gates, both with free and with
        bound parameters."""

        comm_checker = CommutationChecker()

        # gate that has parameters but is not considered parameterized
        rz_gate = RZGate(np.pi / 2)
        self.assertEqual(len(rz_gate.params), 1)
        self.assertFalse(rz_gate.is_parameterized())

        # gate that has parameters and is considered parameterized
        rz_gate_theta = RZGate(Parameter("Theta"))
        rz_gate_phi = RZGate(Parameter("Phi"))
        self.assertEqual(len(rz_gate_theta.params), 1)
        self.assertTrue(rz_gate_theta.is_parameterized())

        # gate that has no parameters and is not considered parameterized
        cx_gate = CXGate()
        self.assertEqual(len(cx_gate.params), 0)
        self.assertFalse(cx_gate.is_parameterized())

        # We should detect that these gates commute
        res = comm_checker.commute(rz_gate, [0], [], cx_gate, [0, 1], [])
        self.assertTrue(res)

        # We should detect that these gates commute
        res = comm_checker.commute(rz_gate, [0], [], rz_gate, [0], [])
        self.assertTrue(res)

        # We should detect that parameterized gates over disjoint qubit subsets commute
        res = comm_checker.commute(rz_gate_theta, [0], [], rz_gate_theta, [1], [])
        self.assertTrue(res)

        # We should detect that parameterized gates over disjoint qubit subsets commute
        res = comm_checker.commute(rz_gate_theta, [0], [], rz_gate_phi, [1], [])
        self.assertTrue(res)

        # We should detect that parameterized gates over disjoint qubit subsets commute
        res = comm_checker.commute(rz_gate_theta, [2], [], cx_gate, [1, 3], [])
        self.assertTrue(res)

        # However, for now commutativity checker should return False when checking
        # commutativity between a parameterized gate and some other gate, when
        # the two gates are over intersecting qubit subsets.
        # This check should be changed if commutativity checker is extended to
        # handle parameterized gates better.
        res = comm_checker.commute(rz_gate_theta, [0], [], cx_gate, [0, 1], [])
        self.assertFalse(res)

        res = comm_checker.commute(rz_gate_theta, [0], [], rz_gate, [0], [])
        self.assertFalse(res)

    def test_measure(self):
        """Check commutativity involving measures."""

        comm_checker = CommutationChecker()

        # Measure is over qubit 0, while gate is over a disjoint subset of qubits
        # We should be able to swap these.
        res = comm_checker.commute(Measure(), [0], [0], CXGate(), [1, 2], [])
        self.assertTrue(res)

        # Measure and gate have intersecting set of qubits
        # We should not be able to swap these.
        res = comm_checker.commute(Measure(), [0], [0], CXGate(), [0, 2], [])
        self.assertFalse(res)

        # Measures over different qubits and clbits
        res = comm_checker.commute(Measure(), [0], [0], Measure(), [1], [1])
        self.assertTrue(res)

        # Measures over different qubits but same classical bit
        # We should not be able to swap these.
        res = comm_checker.commute(Measure(), [0], [0], Measure(), [1], [0])
        self.assertFalse(res)

        # Measures over same qubits but different classical bit
        # ToDo: can we swap these?
        # Currently checker takes the safe approach and returns False.
        res = comm_checker.commute(Measure(), [0], [0], Measure(), [0], [1])
        self.assertFalse(res)

    def test_barrier(self):
        """Check commutativity involving barriers."""

        comm_checker = CommutationChecker()

        # A gate should not commute with a barrier
        # (at least if these are over intersecting qubit sets).
        res = comm_checker.commute(Barrier(4), [0, 1, 2, 3], [], CXGate(), [1, 2], [])
        self.assertFalse(res)

        # Does it even make sense to have a barrier over a subset of qubits?
        # Though in this case, it probably makes sense to say that barrier and gate can be swapped.
        res = comm_checker.commute(Barrier(4), [0, 1, 2, 3], [], CXGate(), [5, 6], [])
        self.assertTrue(res)

    def test_reset(self):
        """Check commutativity involving resets."""

        comm_checker = CommutationChecker()

        # A gate should not commute with reset when the qubits intersect.
        res = comm_checker.commute(Reset(), [0], [], CXGate(), [0, 2], [])
        self.assertFalse(res)

        # A gate should commute with reset when the qubits are disjoint.
        res = comm_checker.commute(Reset(), [0], [], CXGate(), [1, 2], [])
        self.assertTrue(res)

    def test_conditional_gates(self):
        """Check commutativity involving conditional gates."""

        comm_checker = CommutationChecker()

        qr = QuantumRegister(3)
        cr = ClassicalRegister(2)

        # Currently, in all cases commutativity checker should returns False.
        # This is definitely suboptimal.
        res = comm_checker.commute(
            CXGate().c_if(cr[0], 0), [qr[0], qr[1]], [], XGate(), [qr[2]], []
        )
        self.assertFalse(res)

        res = comm_checker.commute(
            CXGate().c_if(cr[0], 0), [qr[0], qr[1]], [], XGate(), [qr[1]], []
        )
        self.assertFalse(res)

        res = comm_checker.commute(
            CXGate().c_if(cr[0], 0), [qr[0], qr[1]], [], CXGate().c_if(cr[0], 0), [qr[0], qr[1]], []
        )
        self.assertFalse(res)

        res = comm_checker.commute(
            XGate().c_if(cr[0], 0), [qr[0]], [], XGate().c_if(cr[0], 1), [qr[0]], []
        )
        self.assertFalse(res)

        res = comm_checker.commute(XGate().c_if(cr[0], 0), [qr[0]], [], XGate(), [qr[0]], [])
        self.assertFalse(res)

    def test_complex_gates(self):
        """Check commutativity involving more complex gates."""

        comm_checker = CommutationChecker()

        lf1 = LinearFunction([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        lf2 = LinearFunction([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        # lf1 is equivalent to swap(0, 1), and lf2 to swap(1, 2).
        # These do not commute.
        res = comm_checker.commute(lf1, [0, 1, 2], [], lf2, [0, 1, 2], [])
        self.assertFalse(res)

        lf3 = LinearFunction([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        lf4 = LinearFunction([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        # lf3 is permutation 1->2, 2->3, 3->1.
        # lf3 is the inverse permutation 1->3, 2->1, 3->2.
        # These commute.
        res = comm_checker.commute(lf3, [0, 1, 2], [], lf4, [0, 1, 2], [])
        self.assertTrue(res)

    def test_c7x_gate(self):
        """Test wide gate works correctly."""
        qargs = [Qubit() for _ in [None] * 8]
        res = CommutationChecker().commute(XGate(), qargs[:1], [], XGate().control(7), qargs, [])
        self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
