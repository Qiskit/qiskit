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
from test import QiskitTestCase  # pylint: disable=wrong-import-order

import numpy as np
from ddt import idata, ddt, data, unpack

from qiskit.circuit import (
    AnnotatedOperation,
    ControlModifier,
    Gate,
    InverseModifier,
    Parameter,
    QuantumRegister,
    Qubit,
    QuantumCircuit,
)
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.circuit.library import (
    Barrier,
    CCXGate,
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CXGate,
    CUGate,
    LinearFunction,
    MCXGate,
    Measure,
    PauliGate,
    PhaseGate,
    Reset,
    RGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZXGate,
    RZZGate,
    SGate,
    XGate,
    YGate,
    ZGate,
    HGate,
    UnitaryGate,
    UGate,
    PauliEvolutionGate,
    PauliProductMeasurement,
)
from qiskit.dagcircuit import DAGOpNode
from qiskit.quantum_info import SparseObservable, SparsePauliOp, Pauli

ROTATION_GATES = [
    RXGate,
    RYGate,
    RZGate,
    PhaseGate,
    RXXGate,
    RYYGate,
    RZZGate,
    RZXGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CPhaseGate,
]


class NewGateCX(Gate):
    """A dummy class containing an cx gate unknown to the commutation checker's library."""

    def __init__(self):
        super().__init__("new_cx", 2, [])

    def to_matrix(self):
        return np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)


class MyEvilRXGate(Gate):
    """A RX gate designed to annoy the caching mechanism (but a realistic gate nevertheless)."""

    def __init__(self, evil_input_not_in_param: float):
        """
        Args:
            evil_input_not_in_param: The RX rotation angle.
        """
        self.value = evil_input_not_in_param
        super().__init__("<evil laugh here>", 1, [])

    def _define(self):
        self.definition = QuantumCircuit(1)
        self.definition.rx(self.value, 0)


@ddt
class TestCommutationChecker(QiskitTestCase):
    """Test CommutationChecker class."""

    def test_simple_gates(self):
        """Check simple commutation relations between gates, experimenting with
        different orders of gates, different orders of qubits, different sets of
        qubits over which gates are defined, and so on."""

        self.assertTrue(scc.commute(HGate(), [0], [], HGate(), [0], []))

        self.assertTrue(scc.commute(ZGate(), [0], [], CXGate(), [0, 1], []))
        self.assertFalse(scc.commute(ZGate(), [1], [], CXGate(), [0, 1], []))

        self.assertFalse(scc.commute(XGate(), [0], [], CXGate(), [0, 1], []))
        self.assertTrue(scc.commute(XGate(), [1], [], CXGate(), [0, 1], []))
        self.assertFalse(scc.commute(XGate(), [1], [], CXGate(), [1, 0], []))
        self.assertTrue(scc.commute(XGate(), [0], [], CXGate(), [1, 0], []))
        self.assertTrue(scc.commute(CXGate(), [1, 0], [], XGate(), [0], []))
        self.assertFalse(scc.commute(CXGate(), [1, 0], [], XGate(), [1], []))

        self.assertTrue(scc.commute(CXGate(), [1, 0], [], CXGate(), [1, 0], []))
        self.assertFalse(scc.commute(CXGate(), [1, 0], [], CXGate(), [0, 1], []))
        self.assertTrue(scc.commute(CXGate(), [1, 0], [], CXGate(), [1, 2], []))
        self.assertFalse(scc.commute(CXGate(), [1, 0], [], CXGate(), [2, 1], []))
        self.assertTrue(scc.commute(CXGate(), [1, 0], [], CXGate(), [2, 3], []))

        self.assertTrue(scc.commute(XGate(), [2], [], CCXGate(), [0, 1, 2], []))
        self.assertFalse(scc.commute(CCXGate(), [0, 1, 2], [], CCXGate(), [0, 2, 1], []))

        # these would commute up to a global phase
        self.assertFalse(scc.commute(HGate(), [0], [], YGate(), [0], []))

    def test_simple_matrices(self):
        """Test simple gates but matrix-based."""
        x = UnitaryGate(XGate())
        had = UnitaryGate(HGate())
        had2 = UnitaryGate(np.kron(HGate(), HGate()))
        cx = UnitaryGate(CXGate())

        self.assertTrue(scc.commute(x, [0], [], x, [0], []))
        self.assertTrue(scc.commute(had, [0], [], had, [0], []))

        self.assertTrue(scc.commute(had2, [0, 1], [], had2, [1, 0], []))
        self.assertFalse(scc.commute(had2, [0, 1], [], cx, [1, 0], []))
        self.assertTrue(scc.commute(cx, [0, 1], [], cx, [0, 1], []))

        self.assertFalse(scc.commute(x, [0], [], cx, [0, 1], []))
        self.assertTrue(scc.commute(x, [1], [], cx, [0, 1], []))

    def test_passing_quantum_registers(self):
        """Check that passing QuantumRegisters works correctly."""
        qr = QuantumRegister(4)
        self.assertTrue(scc.commute(CXGate(), [qr[1], qr[0]], [], CXGate(), [qr[1], qr[2]], []))
        self.assertFalse(scc.commute(CXGate(), [qr[0], qr[1]], [], CXGate(), [qr[1], qr[2]], []))

    def test_standard_gates_commutations(self):
        """Check that commutativity checker uses standard gates commutations as expected."""
        scc.clear_cached_commutations()
        self.assertTrue(scc.commute(ZGate(), [0], [], CXGate(), [0, 1], []))
        self.assertEqual(scc.num_cached_entries(), 0)

    def test_caching_positive_results(self):
        """Check that hashing positive results in commutativity checker works as expected."""
        scc.clear_cached_commutations()
        self.assertTrue(scc.commute(ZGate(), [0], [], CUGate(1, 2, 3, 0), [0, 1], []))
        self.assertGreater(scc.num_cached_entries(), 0)

    def test_caching_lookup_with_non_overlapping_qubits(self):
        """Check that commutation lookup with non-overlapping qubits works as expected."""
        scc.clear_cached_commutations()
        self.assertTrue(scc.commute(CXGate(), [0, 2], [], CXGate(), [0, 1], []))
        self.assertFalse(scc.commute(CXGate(), [0, 1], [], CXGate(), [1, 2], []))
        self.assertEqual(scc.num_cached_entries(), 0)

    def test_caching_store_and_lookup_with_non_overlapping_qubits(self):
        """Check that commutations storing and lookup with non-overlapping qubits works as expected."""
        scc_lenm = scc.num_cached_entries()
        cx_like = CUGate(np.pi, 0, np.pi, 0)
        self.assertTrue(scc.commute(cx_like, [0, 2], [], CXGate(), [0, 1], []))
        self.assertFalse(scc.commute(cx_like, [0, 1], [], CXGate(), [1, 2], []))
        self.assertTrue(scc.commute(cx_like, [1, 4], [], CXGate(), [1, 6], []))
        self.assertFalse(scc.commute(cx_like, [5, 3], [], CXGate(), [3, 1], []))
        self.assertEqual(scc.num_cached_entries(), scc_lenm + 2)

    def test_caching_negative_results(self):
        """Check that hashing negative results in commutativity checker works as expected."""
        scc.clear_cached_commutations()
        self.assertFalse(scc.commute(XGate(), [0], [], CUGate(1, 2, 3, 0), [0, 1], []))
        self.assertGreater(scc.num_cached_entries(), 0)

    def test_caching_different_qubit_sets(self):
        """Check that hashing same commutativity results over different qubit sets works as expected."""
        scc.clear_cached_commutations()
        # All the following should be cached in the same way
        # though each relation gets cached twice: (A, B) and (B, A)
        cx_like = CUGate(np.pi, 0, np.pi, 0)
        scc.commute(XGate(), [0], [], cx_like, [0, 1], [])
        scc.commute(XGate(), [10], [], cx_like, [10, 20], [])
        scc.commute(XGate(), [10], [], cx_like, [10, 5], [])
        scc.commute(XGate(), [5], [], cx_like, [5, 7], [])
        self.assertEqual(scc.num_cached_entries(), 1)

    def test_zero_rotations(self):
        """Check commutativity between (non-parameterized) gates with parameters."""
        self.assertTrue(scc.commute(RZGate(0), [0], [], XGate(), [0], []))
        self.assertTrue(scc.commute(XGate(), [0], [], RZGate(0), [0], []))

    def test_gates_with_parameters(self):
        """Check commutativity between (non-parameterized) gates with parameters."""
        self.assertTrue(scc.commute(RZGate(0), [0], [], XGate(), [0], []))
        self.assertFalse(scc.commute(RZGate(np.pi / 2), [0], [], XGate(), [0], []))
        self.assertTrue(scc.commute(RZGate(np.pi / 2), [0], [], RZGate(0), [0], []))

    def test_parameterized_gates(self):
        """Check commutativity between parameterized gates, both with free and with
        bound parameters."""
        # gate that has parameters but is not considered parameterized
        rz_gate = RZGate(np.pi / 2)
        self.assertEqual(len(rz_gate.params), 1)
        self.assertFalse(rz_gate.is_parameterized())

        # gate that has parameters and is considered parameterized
        rz_gate_theta = RZGate(Parameter("Theta"))
        rx_gate_theta = RXGate(Parameter("Theta"))
        rxx_gate_theta = RXXGate(Parameter("Theta"))
        rz_gate_phi = RZGate(Parameter("Phi"))
        self.assertEqual(len(rz_gate_theta.params), 1)
        self.assertTrue(rz_gate_theta.is_parameterized())

        # gate that has no parameters and is not considered parameterized
        cx_gate = CXGate()
        self.assertEqual(len(cx_gate.params), 0)
        self.assertFalse(cx_gate.is_parameterized())

        # We should detect that these gates commute
        self.assertTrue(scc.commute(rz_gate, [0], [], cx_gate, [0, 1], []))

        # We should detect that these gates commute
        self.assertTrue(scc.commute(rz_gate, [0], [], rz_gate, [0], []))

        # We should detect that parameterized gates over disjoint qubit subsets commute
        self.assertTrue(scc.commute(rz_gate_theta, [0], [], rz_gate_theta, [1], []))

        # We should detect that parameterized gates over disjoint qubit subsets commute
        self.assertTrue(scc.commute(rz_gate_theta, [0], [], rz_gate_phi, [1], []))

        self.assertTrue(scc.commute(rz_gate_theta, [2], [], cx_gate, [1, 3], []))

        # However, for now commutativity checker should return False when checking
        # commutativity between a parameterized gate and some other gate, when
        # the two gates are over intersecting qubit subsets.
        # This check should be changed if commutativity checker is extended to
        # handle parameterized gates better.
        self.assertFalse(scc.commute(rz_gate_theta, [1], [], cx_gate, [0, 1], []))
        self.assertTrue(scc.commute(rz_gate_theta, [0], [], rz_gate, [0], []))
        self.assertTrue(scc.commute(rz_gate_theta, [0], [], rz_gate_phi, [0], []))
        self.assertTrue(scc.commute(rxx_gate_theta, [0, 1], [], rx_gate_theta, [0], []))
        self.assertTrue(scc.commute(rxx_gate_theta, [0, 1], [], XGate(), [0], []))
        self.assertTrue(scc.commute(XGate(), [0], [], rxx_gate_theta, [0, 1], []))
        self.assertTrue(scc.commute(rx_gate_theta, [0], [], rxx_gate_theta, [0, 1], []))
        self.assertTrue(scc.commute(rz_gate_theta, [0], [], cx_gate, [0, 1], []))

    def test_measure(self):
        """Check commutativity involving measures."""
        # Measure is over qubit 0, while gate is over a disjoint subset of qubits
        # We should be able to swap these.
        self.assertTrue(scc.commute(Measure(), [0], [0], CXGate(), [1, 2], []))

        # Measure and gate have intersecting set of qubits
        # We should not be able to swap these.
        self.assertFalse(scc.commute(Measure(), [0], [0], CXGate(), [0, 2], []))

        # Measures over different qubits and clbits
        self.assertTrue(scc.commute(Measure(), [0], [0], Measure(), [1], [1]))

        # Measures over different qubits but same classical bit
        # We should not be able to swap these.
        self.assertFalse(scc.commute(Measure(), [0], [0], Measure(), [1], [0]))

        # Measures over same qubits but different classical bit
        # ToDo: can we swap these?
        # Currently checker takes the safe approach and returns False.
        self.assertFalse(scc.commute(Measure(), [0], [0], Measure(), [0], [1]))

    def test_barrier(self):
        """Check commutativity involving barriers."""
        # A gate should not commute with a barrier
        # (at least if these are over intersecting qubit sets).
        self.assertFalse(scc.commute(Barrier(4), [0, 1, 2, 3], [], CXGate(), [1, 2], []))

        # Does it even make sense to have a barrier over a subset of qubits?
        # Though in this case, it probably makes sense to say that barrier and gate can be swapped.
        self.assertTrue(scc.commute(Barrier(4), [0, 1, 2, 3], [], CXGate(), [5, 6], []))

    def test_reset(self):
        """Check commutativity involving resets."""
        # A gate should not commute with reset when the qubits intersect.
        self.assertFalse(scc.commute(Reset(), [0], [], CXGate(), [0, 2], []))

        # A gate should commute with reset when the qubits are disjoint.
        self.assertTrue(scc.commute(Reset(), [0], [], CXGate(), [1, 2], []))

    def test_complex_gates(self):
        """Check commutativity involving more complex gates."""
        lf1 = LinearFunction([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        lf2 = LinearFunction([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        # lf1 is equivalent to swap(0, 1), and lf2 to swap(1, 2).
        # These do not commute.
        self.assertFalse(scc.commute(lf1, [0, 1, 2], [], lf2, [0, 1, 2], []))

        lf3 = LinearFunction([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        lf4 = LinearFunction([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        # lf3 is permutation 1->2, 2->3, 3->1.
        # lf3 is the inverse permutation 1->3, 2->1, 3->2.
        # These commute.
        self.assertTrue(scc.commute(lf3, [0, 1, 2], [], lf4, [0, 1, 2], []))

    def test_equal_annotated_operations_commute(self):
        """Check commutativity involving the same annotated operation."""
        op1 = AnnotatedOperation(SGate(), [InverseModifier(), ControlModifier(1)])
        op2 = AnnotatedOperation(SGate(), [InverseModifier(), ControlModifier(1)])
        # the same, so true
        self.assertTrue(scc.commute(op1, [0, 1], [], op2, [0, 1], []))

    def test_annotated_operations_commute_with_unannotated(self):
        """Check commutativity involving annotated operations and unannotated operations."""
        op1 = AnnotatedOperation(SGate(), [InverseModifier(), ControlModifier(1)])
        op2 = AnnotatedOperation(ZGate(), [InverseModifier()])
        op3 = ZGate()
        # all true
        self.assertTrue(scc.commute(op1, [0, 1], [], op2, [1], []))
        self.assertTrue(scc.commute(op1, [0, 1], [], op3, [1], []))
        self.assertTrue(scc.commute(op2, [1], [], op3, [1], []))

    def test_utf8_gate_names(self):
        """Check compatibility of non-ascii quantum gate names."""
        g0 = RXXGate(1.234).to_mutable()
        g0.name = "すみません"

        g1 = RXXGate(2.234).to_mutable()
        g1.name = "ok_0"

        self.assertTrue(scc.commute(g0, [0, 1], [], g1, [1, 0], []))

    def test_annotated_operations_no_commute(self):
        """Check non-commutativity involving annotated operations."""
        op1 = AnnotatedOperation(XGate(), [InverseModifier(), ControlModifier(1)])
        op2 = AnnotatedOperation(XGate(), [InverseModifier()])
        # false
        self.assertFalse(scc.commute(op1, [0, 1], [], op2, [0], []))

    def test_c7x_gate(self):
        """Test wide gate works correctly."""
        qargs = [Qubit() for _ in [None] * 8]
        res = scc.commute(XGate(), qargs[:1], [], XGate().control(7, annotated=False), qargs, [])
        self.assertFalse(res)

    def test_wide_gates_over_nondisjoint_qubits(self):
        """Test that checking wide gates does not lead to memory problems."""
        self.assertFalse(scc.commute(MCXGate(29), list(range(30)), [], XGate(), [0], []))

    def test_wide_gates_over_disjoint_qubits(self):
        """Test that wide gates still commute when they are over disjoint sets of qubits."""
        self.assertTrue(scc.commute(MCXGate(29), list(range(30)), [], XGate(), [30], []))
        self.assertTrue(scc.commute(XGate(), [30], [], MCXGate(29), list(range(30)), []))

    def test_serialization(self):
        """Test that the commutation checker is correctly serialized"""
        import pickle

        cx_like = CUGate(np.pi, 0, np.pi, 0)

        scc.clear_cached_commutations()
        self.assertTrue(scc.commute(ZGate(), [0], [], cx_like, [0, 1], []))
        cc2 = pickle.loads(pickle.dumps(scc))
        self.assertEqual(cc2.num_cached_entries(), 1)
        dop1 = DAGOpNode(ZGate(), qargs=[0], cargs=[])
        dop2 = DAGOpNode(cx_like, qargs=[0, 1], cargs=[])
        cc2.commute_nodes(dop1, dop2)
        dop1 = DAGOpNode(ZGate(), qargs=[0], cargs=[])
        dop2 = DAGOpNode(CXGate(), qargs=[0, 1], cargs=[])
        cc2.commute_nodes(dop1, dop2)
        self.assertEqual(cc2.num_cached_entries(), 1)

    @idata(ROTATION_GATES)
    def test_cutoff_angles(self, gate_cls):
        """Check rotations with a small enough angle are cut off."""
        max_power = 30
        from qiskit.circuit.library import DCXGate

        generic_gate = DCXGate()  # gate that does not commute with any rotation gate

        # the cutoff angle depends on the average gate fidelity; i.e. it is the angle
        # for which the average gate fidelity is smaller than 1e-12
        if gate_cls in [CPhaseGate, CRXGate, CRYGate, CRZGate]:
            cutoff_angle = 3.16e-6
        else:
            cutoff_angle = 2.2e-6

        for i in range(1, max_power + 1):
            angle = 2 ** (-i)
            gate = gate_cls(angle)
            qargs = list(range(gate.num_qubits))
            if angle < cutoff_angle:
                self.assertTrue(scc.commute(generic_gate, [0, 1], [], gate, qargs, []))
            else:
                self.assertFalse(scc.commute(generic_gate, [0, 1], [], gate, qargs, []))

    @idata(ROTATION_GATES)
    def test_rotations_pi_multiples(self, gate_cls):
        """Test the rotations modulo 2pi (crx/cry/crz modulo 4pi) commute with any gate."""
        generic_gate = HGate()  # does not commute with any rotation gate
        multiples = np.arange(-6, 7)

        for multiple in multiples:
            with self.subTest(multiple=multiple):
                gate = gate_cls(multiple * np.pi)
                numeric = UnitaryGate(gate.to_matrix())

                # compute a numeric reference, that doesn't go through any special cases and
                # uses a matrix-based commutation check
                expected = scc.commute(
                    generic_gate,
                    [0],
                    [],
                    numeric,
                    list(range(gate.num_qubits)),
                    [],
                    approximation_degree=1 - 1e-5,
                )

                result = scc.commute(generic_gate, [0], [], gate, list(range(gate.num_qubits)), [])
                self.assertEqual(expected, result)

    def test_custom_gate(self):
        """Test a custom gate."""
        my_cx = NewGateCX()

        self.assertTrue(scc.commute(my_cx, [0, 1], [], XGate(), [1], []))
        self.assertFalse(scc.commute(my_cx, [0, 1], [], XGate(), [0], []))
        self.assertTrue(scc.commute(my_cx, [0, 1], [], ZGate(), [0], []))

        self.assertFalse(scc.commute(my_cx, [0, 1], [], my_cx, [1, 0], []))
        self.assertTrue(scc.commute(my_cx, [0, 1], [], my_cx, [0, 1], []))

    def test_custom_gate_caching(self):
        """Test a custom gate is correctly handled on consecutive runs."""

        all_commuter = MyEvilRXGate(0)  # this will commute with anything
        some_rx = MyEvilRXGate(1.6192)  # this should not commute with H

        # the order here is important: we're testing whether the gate that commutes with
        # everything is used after the first commutation check, regardless of the internal
        # gate parameters
        self.assertTrue(scc.commute(all_commuter, [0], [], HGate(), [0], []))
        self.assertFalse(scc.commute(some_rx, [0], [], HGate(), [0], []))

    def test_nonfloat_param(self):
        """Test commutation-checking on a gate that has non-float ``params``."""
        pauli_gate = PauliGate("XX")
        rx_gate_theta = RXGate(Parameter("Theta"))
        self.assertTrue(scc.commute(pauli_gate, [0, 1], [], rx_gate_theta, [0], []))
        self.assertTrue(scc.commute(rx_gate_theta, [0], [], pauli_gate, [0, 1], []))

    def test_2q_pauli_rot_with_non_cached(self):
        """Test the 2q-Pauli rotations with a gate that is not cached."""
        x_equiv = UGate(np.pi, -np.pi / 2, np.pi / 2)
        self.assertTrue(scc.commute(x_equiv, [0], [], RXXGate(np.pi / 2), [0, 1], []))
        self.assertTrue(scc.commute(x_equiv, [1], [], RXXGate(np.pi / 2), [0, 1], []))
        self.assertFalse(scc.commute(x_equiv, [0], [], RYYGate(np.pi), [1, 0], []))
        self.assertFalse(scc.commute(x_equiv, [1], [], RYYGate(np.pi), [1, 0], []))

        something_else = RGate(1, 2)
        self.assertFalse(scc.commute(something_else, [0], [], RXXGate(np.pi / 2), [0, 1], []))
        self.assertFalse(scc.commute(something_else, [1], [], RXXGate(np.pi / 2), [0, 1], []))
        self.assertFalse(scc.commute(something_else, [0], [], RYYGate(np.pi), [1, 0], []))
        self.assertFalse(scc.commute(something_else, [1], [], RYYGate(np.pi), [1, 0], []))

    def test_approximation_degree(self):
        """Test setting the approximation degree."""

        almost_identity = RZGate(1e-5)
        other = HGate()

        self.assertFalse(scc.commute(almost_identity, [0], [], other, [0], []))
        self.assertFalse(
            scc.commute(almost_identity, [0], [], other, [0], [], approximation_degree=1)
        )
        self.assertTrue(
            scc.commute(almost_identity, [0], [], other, [0], [], approximation_degree=1 - 1e-4)
        )

    @data("pauli", "evolution", "measure")
    def test_pauli_based_gates(self, gate_type):
        """Test Pauli-based gates."""
        cases = [
            ("I", [0], "XYZ", list(range(3)), True),
            ("ZZZZ", list(range(4)), "XXXX", list(range(4)), True),
            ("ZZIZ", list(range(4)), "XXIX", list(range(4)), False),
            ("ZZIIIIIIIY", list(range(10)), "YYIIIIIIIY", list(range(10)), True),
            ("ZZIIIIIIIY", list(range(10)), "YYIIIIIIIZ", list(range(10)), False),
            ("ZX", [1, 10], "ZIZYIZXXZXZ", list(range(11)), True),
        ]

        for p1, q1, p2, q2, expected in cases:
            if p1 == "I" and gate_type == "measure":
                continue  # PPM doesn't support all-identity gates

            gate1 = build_pauli_gate(p1, gate_type)
            gate2 = build_pauli_gate(p2, gate_type)
            self.assertEqual(expected, scc.commute(gate1, q1, [], gate2, q2, []))

    @data(
        ("pauli", "measure"),
        ("evolution", "measure"),
        ("evolution", "pauli"),
    )
    @unpack
    def test_mix_pauli_gates(self, gate_type1, gate_type2):
        """Test commutation relations across different Pauli-based gates."""
        cases = [
            ("ZZIIIIIIIY", list(range(10)), "YYIIIIIIIZ", list(range(10)), False),
            ("ZX", [1, 10], "ZIZYIZXXZXZ", list(range(11)), True),
        ]

        for p1, q1, p2, q2, expected in cases:
            gate1 = build_pauli_gate(p1, gate_type1)
            gate2 = build_pauli_gate(p2, gate_type2)
            with self.subTest(p1=p1, p2=p2):
                self.assertEqual(expected, scc.commute(gate1, q1, [], gate2, q2, []))

    def test_pauli_evolution_sums(self):
        """Test PauliEvolutionGate commutations for operators that are sums of Paulis."""
        xxyy = PauliEvolutionGate(SparsePauliOp(["XX", "YY"]))
        zz = PauliEvolutionGate(SparseObservable("ZZ"))
        with self.subTest(left=xxyy, right=zz):
            self.assertTrue(scc.commute(xxyy, [0, 1], [], zz, [0, 1], []))

        swap = PauliEvolutionGate(SparsePauliOp(["II", "XX", "YY", "ZZ"]))
        with self.subTest(left=swap, right=zz):
            self.assertTrue(scc.commute(swap, [0, 1], [], zz, [1, 0], []))

        x = PauliEvolutionGate(Pauli("X"))
        with self.subTest(left=x, right=swap):
            self.assertFalse(scc.commute(x, [1], [], swap, [1, 0], []))

    def test_pauli_evolution_parameterized(self):
        """Test PauliEvolutionGate commutations for parameterized times."""
        z = PauliEvolutionGate(SparsePauliOp([20 * "Z"]), time=Parameter("z"))
        xy = PauliEvolutionGate(SparsePauliOp([10 * "X" + 10 * "Y"]))
        qargs = list(range(20))
        with self.subTest(left=z, right=xy):
            self.assertTrue(scc.commute(z, qargs, [], xy, qargs, []))

        max_qubits = 3
        with self.subTest(left=z, right=xy, max_qubits=max_qubits):
            self.assertFalse(scc.commute(z, qargs, [], xy, qargs, [], max_num_qubits=max_qubits))

        x = PauliEvolutionGate(SparsePauliOp([19 * "X" + "I"]))
        with self.subTest(left=z, right=x):
            self.assertFalse(scc.commute(z, qargs, [], x, qargs, []))


def build_pauli_gate(pauli_string: str, gate_type: str) -> Gate:
    """Build a Pauli-based gate off a Pauli string.

    The gate types are

        * ``"pauli"`` for ``PauliGate``
        * ``"measure"`` for ``PauliProductMeasurement``
        * ``"evolution"`` for ``PauliEvolutionGate``

    Args:
        pauli_string: The Pauli string (e.g. "XZII").
        gate_type: The gate type.

    Returns:
        The gate.
    """
    if gate_type == "pauli":
        return PauliGate(pauli_string)
    if gate_type == "measure":
        return PauliProductMeasurement(Pauli(pauli_string))
    if gate_type == "evolution":
        return PauliEvolutionGate(SparseObservable(pauli_string))

    raise ValueError(f"Invalid gate type: {gate_type}")


if __name__ == "__main__":
    unittest.main()
