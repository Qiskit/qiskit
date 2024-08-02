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
from qiskit.circuit import (
    QuantumRegister,
    Parameter,
    Qubit,
    AnnotatedOperation,
    InverseModifier,
    ControlModifier,
)
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
import qiskit.circuit.library as lib

from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestCommutationChecker(QiskitTestCase):
    """Test CommutationChecker class."""

    def test_simple_gates(self):
        """Check simple commutation relations between gates, experimenting with
        different orders of gates, different orders of qubits, different sets of
        qubits over which gates are defined, and so on."""
        # should commute
        res = scc.commute(lib.ZGate(), [0], [], lib.CXGate(), [0, 1], [])
        self.assertTrue(res)

        # should not commute
        res = scc.commute(lib.ZGate(), [1], [], lib.CXGate(), [0, 1], [])
        self.assertFalse(res)

        # should not commute
        res = scc.commute(lib.XGate(), [0], [], lib.CXGate(), [0, 1], [])
        self.assertFalse(res)

        # should commute
        res = scc.commute(lib.XGate(), [1], [], lib.CXGate(), [0, 1], [])
        self.assertTrue(res)

        # should not commute
        res = scc.commute(lib.XGate(), [1], [], lib.CXGate(), [1, 0], [])
        self.assertFalse(res)

        # should commute
        res = scc.commute(lib.XGate(), [0], [], lib.CXGate(), [1, 0], [])
        self.assertTrue(res)

        # should commute
        res = scc.commute(lib.CXGate(), [1, 0], [], lib.XGate(), [0], [])
        self.assertTrue(res)

        # should not commute
        res = scc.commute(lib.CXGate(), [1, 0], [], lib.XGate(), [1], [])
        self.assertFalse(res)

        # should commute
        res = scc.commute(
            lib.CXGate(),
            [1, 0],
            [],
            lib.CXGate(),
            [1, 0],
            [],
        )
        self.assertTrue(res)

        # should not commute
        res = scc.commute(
            lib.CXGate(),
            [1, 0],
            [],
            lib.CXGate(),
            [0, 1],
            [],
        )
        self.assertFalse(res)

        # should commute
        res = scc.commute(
            lib.CXGate(),
            [1, 0],
            [],
            lib.CXGate(),
            [1, 2],
            [],
        )
        self.assertTrue(res)

        # should not commute
        res = scc.commute(
            lib.CXGate(),
            [1, 0],
            [],
            lib.CXGate(),
            [2, 1],
            [],
        )
        self.assertFalse(res)

        # should commute
        res = scc.commute(
            lib.CXGate(),
            [1, 0],
            [],
            lib.CXGate(),
            [2, 3],
            [],
        )
        self.assertTrue(res)

        res = scc.commute(lib.XGate(), [2], [], lib.CCXGate(), [0, 1, 2], [])
        self.assertTrue(res)

        res = scc.commute(lib.CCXGate(), [0, 1, 2], [], lib.CCXGate(), [0, 2, 1], [])
        self.assertFalse(res)

    def test_passing_quantum_registers(self):
        """Check that passing QuantumRegisters works correctly."""
        qr = QuantumRegister(4)

        # should commute
        res = scc.commute(lib.CXGate(), [qr[1], qr[0]], [], lib.CXGate(), [qr[1], qr[2]], [])
        self.assertTrue(res)

        # should not commute
        res = scc.commute(lib.CXGate(), [qr[0], qr[1]], [], lib.CXGate(), [qr[1], qr[2]], [])
        self.assertFalse(res)

    def test_standard_gates_commutations(self):
        """Check that commutativity checker uses standard gates commutations as expected."""
        scc.clear_cached_commutations()
        scc.clear_cached_commutations()
        res = scc.commute(lib.ZGate(), [0], [], lib.CXGate(), [0, 1], [])
        self.assertTrue(res)
        self.assertEqual(scc.num_cached_entries(), 0)

    def test_caching_positive_results(self):
        """Check that hashing positive results in commutativity checker works as expected."""
        scc.clear_cached_commutations()
        NewGateCX = type("MyClass", (lib.CXGate,), {"content": {}})
        NewGateCX.name = "cx_new"

        res = scc.commute(lib.ZGate(), [0], [], NewGateCX(), [0, 1], [])
        self.assertTrue(res)
        self.assertGreater(len(scc._cached_commutations), 0)

    def test_caching_lookup_with_non_overlapping_qubits(self):
        """Check that commutation lookup with non-overlapping qubits works as expected."""
        scc.clear_cached_commutations()
        res = scc.commute(lib.CXGate(), [0, 2], [], lib.CXGate(), [0, 1], [])
        self.assertTrue(res)
        res = scc.commute(lib.CXGate(), [0, 1], [], lib.CXGate(), [1, 2], [])
        self.assertFalse(res)
        self.assertEqual(len(scc._cached_commutations), 0)

    def test_caching_store_and_lookup_with_non_overlapping_qubits(self):
        """Check that commutations storing and lookup with non-overlapping qubits works as expected."""
        cc_lenm = scc.num_cached_entries()
        NewGateCX = type("MyClass", (lib.CXGate,), {"content": {}})
        NewGateCX.name = "cx_new"
        res = scc.commute(NewGateCX(), [0, 2], [], lib.CXGate(), [0, 1], [])
        self.assertTrue(res)
        res = scc.commute(NewGateCX(), [0, 1], [], lib.CXGate(), [1, 2], [])
        self.assertFalse(res)
        res = scc.commute(NewGateCX(), [1, 4], [], lib.CXGate(), [1, 6], [])
        self.assertTrue(res)
        res = scc.commute(NewGateCX(), [5, 3], [], lib.CXGate(), [3, 1], [])
        self.assertFalse(res)
        self.assertEqual(scc.num_cached_entries(), cc_lenm + 2)

    def test_caching_negative_results(self):
        """Check that hashing negative results in commutativity checker works as expected."""
        scc.clear_cached_commutations()
        NewGateCX = type("MyClass", (lib.CXGate,), {"content": {}})
        NewGateCX.name = "cx_new"

        res = scc.commute(lib.XGate(), [0], [], NewGateCX(), [0, 1], [])
        self.assertFalse(res)
        self.assertGreater(len(scc._cached_commutations), 0)

    def test_caching_different_qubit_sets(self):
        """Check that hashing same commutativity results over different qubit sets works as expected."""
        scc.clear_cached_commutations()
        NewGateCX = type("MyClass", (lib.CXGate,), {"content": {}})
        NewGateCX.name = "cx_new"
        # All the following should be cached in the same way
        # though each relation gets cached twice: (A, B) and (B, A)
        scc.commute(lib.XGate(), [0], [], NewGateCX(), [0, 1], [])
        scc.commute(lib.XGate(), [10], [], NewGateCX(), [10, 20], [])
        scc.commute(lib.XGate(), [10], [], NewGateCX(), [10, 5], [])
        scc.commute(lib.XGate(), [5], [], NewGateCX(), [5, 7], [])
        self.assertEqual(len(scc._cached_commutations), 1)
        self.assertEqual(scc._cache_miss, 1)
        self.assertEqual(scc._cache_hit, 3)

    def test_cache_with_param_gates(self):
        """Check commutativity between (non-parameterized) gates with parameters."""
        scc.clear_cached_commutations()
        res = scc.commute(lib.RZGate(0), [0], [], lib.XGate(), [0], [])
        self.assertTrue(res)

        res = scc.commute(lib.RZGate(np.pi / 2), [0], [], lib.XGate(), [0], [])
        self.assertFalse(res)

        res = scc.commute(lib.RZGate(np.pi / 2), [0], [], lib.RZGate(0), [0], [])
        self.assertTrue(res)

        res = scc.commute(lib.RZGate(np.pi / 2), [1], [], lib.XGate(), [1], [])
        self.assertFalse(res)
        self.assertEqual(scc.num_cached_entries(), 0)

    def test_gates_with_parameters(self):
        """Check commutativity between (non-parameterized) gates with parameters."""
        res = scc.commute(lib.RZGate(0), [0], [], lib.XGate(), [0], [])
        self.assertTrue(res)

        res = scc.commute(lib.RZGate(np.pi / 2), [0], [], lib.XGate(), [0], [])
        self.assertFalse(res)

        res = scc.commute(lib.RZGate(np.pi / 2), [0], [], lib.RZGate(0), [0], [])
        self.assertTrue(res)

    def test_parameterized_gates(self):
        """Check commutativity between parameterized gates, both with free and with
        bound parameters."""
        rz_gate = lib.RZGate(np.pi / 2)
        rz_gate_theta = lib.RZGate(Parameter("Theta"))
        rz_gate_phi = lib.RZGate(Parameter("Phi"))
        cx_gate = lib.CXGate()

        # We should detect that these gates commute
        res = scc.commute(rz_gate, [0], [], cx_gate, [0, 1], [])
        self.assertTrue(res)

        # We should detect that these gates commute
        res = scc.commute(rz_gate, [0], [], rz_gate, [0], [])
        self.assertTrue(res)

        # We should detect that parameterized gates over disjoint qubit subsets commute
        res = scc.commute(rz_gate_theta, [0], [], rz_gate_theta, [1], [])
        self.assertTrue(res)

        # We should detect that parameterized gates over disjoint qubit subsets commute
        res = scc.commute(rz_gate_theta, [0], [], rz_gate_phi, [1], [])
        self.assertTrue(res)

        # We should detect that parameterized gates over disjoint qubit subsets commute
        res = scc.commute(rz_gate_theta, [2], [], cx_gate, [1, 3], [])
        self.assertTrue(res)

        # We should also detect commutation with parameterized gates
        res = scc.commute(rz_gate_theta, [0], [], cx_gate, [0, 1], [])
        self.assertTrue(res)

        res = scc.commute(rz_gate_theta, [0], [], rz_gate, [0], [])
        self.assertTrue(res)

    def test_measure(self):
        """Check commutativity involving lib.Measures."""
        # lib.Measure is over qubit 0, while gate is over a disjoint subset of qubits
        # We should be able to swap these.
        res = scc.commute(lib.Measure(), [0], [0], lib.CXGate(), [1, 2], [])
        self.assertTrue(res)

        # lib.Measure and gate have intersecting set of qubits
        # We should not be able to swap these.
        res = scc.commute(lib.Measure(), [0], [0], lib.CXGate(), [0, 2], [])
        self.assertFalse(res)

        # lib.Measures over different qubits and clbits
        res = scc.commute(lib.Measure(), [0], [0], lib.Measure(), [1], [1])
        self.assertTrue(res)

        # lib.Measures over different qubits but same classical bit
        # We should not be able to swap these.
        res = scc.commute(lib.Measure(), [0], [0], lib.Measure(), [1], [0])
        self.assertFalse(res)

        # lib.Measures over same qubits but different classical bit
        # ToDo: can we swap these?
        # Currently checker takes the safe approach and returns False.
        res = scc.commute(lib.Measure(), [0], [0], lib.Measure(), [0], [1])
        self.assertFalse(res)

    def test_barrier(self):
        """Check commutativity involving lib.Barriers."""
        # A gate should not commute with a lib.Barrier
        # (at least if these are over intersecting qubit sets).
        res = scc.commute(lib.Barrier(4), [0, 1, 2, 3], [], lib.CXGate(), [1, 2], [])
        self.assertFalse(res)

        # Does it even make sense to have a lib.Barrier over a subset of qubits?
        # Though in this case, it probably makes sense to say that lib.Barrier and gate can be swapped.
        res = scc.commute(lib.Barrier(4), [0, 1, 2, 3], [], lib.CXGate(), [5, 6], [])
        self.assertTrue(res)

    def test_reset(self):
        """Check commutativity involving lib.Resets."""
        # A gate should not commute with lib.Reset when the qubits intersect.
        res = scc.commute(lib.Reset(), [0], [], lib.CXGate(), [0, 2], [])
        self.assertFalse(res)

        # A gate should commute with lib.Reset when the qubits are disjoint.
        res = scc.commute(lib.Reset(), [0], [], lib.CXGate(), [1, 2], [])
        self.assertTrue(res)

    def test_conditional_gates(self):
        """Check commutativity involving conditional gates."""
        qr = QuantumRegister(3)
        cr = ClassicalRegister(2)

        # Currently, in all cases commutativity checker should returns False.
        # This is definitely suboptimal.
        res = scc.commute(lib.CXGate().c_if(cr[0], 0), [qr[0], qr[1]], [], lib.XGate(), [qr[2]], [])
        self.assertFalse(res)

        res = scc.commute(lib.CXGate().c_if(cr[0], 0), [qr[0], qr[1]], [], lib.XGate(), [qr[1]], [])
        self.assertFalse(res)

        res = scc.commute(
            lib.CXGate().c_if(cr[0], 0),
            [qr[0], qr[1]],
            [],
            lib.CXGate().c_if(cr[0], 0),
            [qr[0], qr[1]],
            [],
        )
        self.assertFalse(res)

        res = scc.commute(
            lib.XGate().c_if(cr[0], 0), [qr[0]], [], lib.XGate().c_if(cr[0], 1), [qr[0]], []
        )
        self.assertFalse(res)

        res = scc.commute(lib.XGate().c_if(cr[0], 0), [qr[0]], [], lib.XGate(), [qr[0]], [])
        self.assertFalse(res)

    def test_complex_gates(self):
        """Check commutativity involving more complex gates."""
        lf1 = lib.LinearFunction([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        lf2 = lib.LinearFunction([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        # lf1 is equivalent to swap(0, 1), and lf2 to swap(1, 2).
        # These do not commute.
        res = scc.commute(lf1, [0, 1, 2], [], lf2, [0, 1, 2], [])
        self.assertFalse(res)

        lf3 = lib.LinearFunction([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        lf4 = lib.LinearFunction([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        # lf3 is permutation 1->2, 2->3, 3->1.
        # lf3 is the inverse permutation 1->3, 2->1, 3->2.
        # These commute.
        res = scc.commute(lf3, [0, 1, 2], [], lf4, [0, 1, 2], [])
        self.assertTrue(res)

    def test_equal_annotated_operations_commute(self):
        """Check commutativity involving the same annotated operation."""
        op1 = AnnotatedOperation(lib.SGate(), [InverseModifier(), ControlModifier(1)])
        op2 = AnnotatedOperation(lib.SGate(), [InverseModifier(), ControlModifier(1)])
        # the same, so true
        self.assertTrue(scc.commute(op1, [0, 1], [], op2, [0, 1], []))

    def test_annotated_operations_commute_with_unannotated(self):
        """Check commutativity involving annotated operations and unannotated operations."""
        op1 = AnnotatedOperation(lib.SGate(), [InverseModifier(), ControlModifier(1)])
        op2 = AnnotatedOperation(lib.ZGate(), [InverseModifier()])
        op3 = lib.ZGate()
        # all true
        self.assertTrue(scc.commute(op1, [0, 1], [], op2, [1], []))
        self.assertTrue(scc.commute(op1, [0, 1], [], op3, [1], []))
        self.assertTrue(scc.commute(op2, [1], [], op3, [1], []))

    def test_utf8_gate_names(self):
        """Check compatibility of non-ascii quantum gate names."""
        g0 = lib.RXXGate(1.234).to_mutable()
        g0.name = "すみません"

        g1 = lib.RXXGate(2.234).to_mutable()
        g1.name = "ok_0"

        self.assertTrue(scc.commute(g0, [0, 1], [], g1, [1, 0], []))

    def test_annotated_operations_no_commute(self):
        """Check non-commutativity involving annotated operations."""
        op1 = AnnotatedOperation(lib.XGate(), [InverseModifier(), ControlModifier(1)])
        op2 = AnnotatedOperation(lib.XGate(), [InverseModifier()])
        # false
        self.assertFalse(scc.commute(op1, [0, 1], [], op2, [0], []))

    def test_c7x_gate(self):
        """Test wide gate works correctly."""
        qargs = [Qubit() for _ in [None] * 8]
        res = scc.commute(lib.XGate(), qargs[:1], [], lib.XGate().control(7), qargs, [])
        self.assertFalse(res)

    def test_wide_gates_over_nondisjoint_qubits(self):
        """Test that checking wide gates does not lead to memory problems."""
        res = scc.commute(lib.MCXGate(29), list(range(30)), [], lib.XGate(), [0], [])
        self.assertFalse(res)
        res = scc.commute(lib.XGate(), [0], [], lib.MCXGate(29), list(range(30)), [])
        self.assertFalse(res)

    def test_wide_gates_over_disjoint_qubits(self):
        """Test that wide gates still commute when they are over disjoint sets of qubits."""
        res = scc.commute(lib.MCXGate(29), list(range(30)), [], lib.XGate(), [30], [])
        self.assertTrue(res)
        res = scc.commute(lib.XGate(), [30], [], lib.MCXGate(29), list(range(30)), [])
        self.assertTrue(res)

    def test_supported_parameterized_gates(self):
        """Test all supported parameterized gates."""

        gates = {
            lib.RXGate: lib.RXGate(0.41),
            lib.RYGate: lib.YGate(),
            lib.RZGate: lib.TGate(),
            lib.PhaseGate: lib.SdgGate(),
            lib.CRXGate: lib.CXGate(),
            lib.CRYGate: lib.CYGate(),
            lib.CRZGate: lib.CPhaseGate(4.1),
            lib.RXXGate: lib.PauliGate("YY"),
            lib.RYYGate: lib.PauliGate("ZZ"),
            lib.RZZGate: lib.PauliGate("XX"),
            lib.RZXGate: lib.PauliGate("XZ"),
        }

        x = Parameter("x")
        for method, partner in gates.items():
            with self.subTest(method=method):
                gate = method(x)

                qubits = list(range(partner.num_qubits))
                res = scc.commute(gate, qubits, [], partner, qubits, [])
                self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
