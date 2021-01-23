# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the Solovay Kitaev transpilation pass."""

import itertools
import unittest
import math
import numpy as np
import scipy

from hypothesis import given
import hypothesis.strategies as st
from scipy.optimize import minimize
from ddt import ddt, data, unpack

from qiskit.circuit import QuantumCircuit
import qiskit.circuit.library as gates
from qiskit.circuit.library import TGate, TdgGate, RXGate, RYGate, HGate, SGate, SdgGate, IGate, QFT
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import SolovayKitaevDecomposition
from qiskit.test import QiskitTestCase
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes.synthesis.solovay_kitaev import commutator_decompose
from qiskit.transpiler.passes.synthesis.solovay_kitaev_utils import GateSequence

# pylint: disable=invalid-name, missing-class-docstring


def distance(A, B):
    """Find the distance in norm of A and B, ignoring global phase."""

    def objective(global_phase):
        return np.linalg.norm(A - np.exp(1j * global_phase) * B)
    result1 = minimize(objective, [1], bounds=[(-np.pi, np.pi)])
    result2 = minimize(objective, [0.5], bounds=[(-np.pi, np.pi)])
    return min(result1.fun, result2.fun)


def _generate_x_rotation(angle: float) -> np.ndarray:
    return np.array([[1, 0, 0],
                     [0, math.cos(angle), -math.sin(angle)],
                     [0, math.sin(angle), math.cos(angle)]])


def _generate_y_rotation(angle: float) -> np.ndarray:
    return np.array([[math.cos(angle), 0, math.sin(angle)],
                     [0, 1, 0],
                     [-math.sin(angle), 0, math.cos(angle)]])


def _generate_z_rotation(angle: float) -> np.ndarray:
    return np.array([[math.cos(angle), -math.sin(angle), 0],
                     [math.sin(angle), math.cos(angle), 0],
                     [0, 0, 1]])


def _generate_random_rotation() -> np.ndarray:
    return np.array(scipy.stats.special_ortho_group.rvs(3))


def _build_rotation(angle: float, axis: int) -> np.ndarray:
    if axis == 0:
        return _generate_x_rotation(angle)
    elif axis == 1:
        return _generate_y_rotation(angle)
    elif axis == 2:
        return _generate_z_rotation(angle)
    else:
        return _generate_random_rotation()


def _build_axis(axis: int) -> np.ndarray:
    if axis == 0:
        return np.array([1.0, 0.0, 0.0])
    elif axis == 1:
        return np.array([0.0, 1.0, 0.0])
    elif axis == 2:
        return np.array([0.0, 0.0, 1.0])
    else:
        return np.array([1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)])


def _generate_x_su2(angle: float) -> np.ndarray:
    return np.array([[math.cos(angle/2), math.sin(angle/2)*1j],
                     [math.sin(angle/2)*1j, math.cos(angle/2)]], dtype=complex)


def _generate_y_su2(angle: float) -> np.ndarray:
    return np.array([[math.cos(angle/2), math.sin(angle/2)],
                     [-math.sin(angle/2), math.cos(angle/2)]], dtype=complex)


def _generate_z_su2(angle: float) -> np.ndarray:
    return np.array([[np.exp(-(1/2)*angle*1j), 0], [0, np.exp((1/2)*angle*1j)]], dtype=complex)


def _generate_su2(alpha: complex, beta: complex) -> np.ndarray:
    base = np.array([[alpha, beta], [-np.conj(beta), np.conj(alpha)]])
    det = np.linalg.det(base)
    if abs(det) < 1e10:
        return np.array([[1, 0], [0, 1]])
    else:
        return np.linalg.det(base)*base


def _build_unit_vector(a: float, b: float, c: float) -> np.ndarray:
    vector = np.array([a, b, c])
    if a != 0.0 or b != 0.0 or c != 0.0:
        unit_vector = vector/np.linalg.norm(vector)
        return unit_vector
    else:
        return np.array([1, 0, 0])


def is_so3_matrix(array: np.ndarray) -> bool:
    """Check if the input array is a SO(3) matrix."""
    if array.shape != (3, 3):
        return False

    if abs(np.linalg.det(array)-1.0) < 1e-10:
        return False

    if False in np.isreal(array):
        return False

    return True


def are_almost_equal_so3_matrices(a: np.ndarray, b: np.ndarray) -> bool:
    """TODO"""
    for t in itertools.product(range(2), range(2)):
        if abs(a[t[0]][t[1]]-b[t[0]][t[1]]) > 1e-10:
            return False
    return True


@ddt
class TestSolovayKitaev(QiskitTestCase):
    """Test the Solovay Kitaev algorithm and transformation pass."""

    @data(
        [_generate_x_rotation(0.1)],
        [_generate_y_rotation(0.2)],
        [_generate_z_rotation(0.3)],
        [np.dot(_generate_z_rotation(0.5), _generate_y_rotation(0.4))],
        [np.dot(_generate_y_rotation(0.5), _generate_x_rotation(0.4))]
    )
    @unpack
    def test_commutator_decompose_returns_tuple_of_two_so3_gatesequences(self, u_so3: np.ndarray):
        """Test that ``commutator_decompose`` returns two SO(3) gate sequences."""
        actual_result = commutator_decompose(u_so3)
        self.assertTrue(is_so3_matrix(actual_result[0].product))
        self.assertTrue(is_so3_matrix(actual_result[1].product))

    @given(st.builds(_build_rotation, st.floats(max_value=2*math.pi, min_value=0),
                     st.integers(min_value=0, max_value=4)))
    def test_commutator_decompose_returns_tuple_of_two_so3_gatesequences_2(self, u_so3: np.ndarray):
        """Test that ``commutator_decompose`` returns two SO(3) gate sequences."""
        actual_result = commutator_decompose(u_so3)
        self.assertTrue(is_so3_matrix(actual_result[0].product))
        self.assertTrue(is_so3_matrix(actual_result[1].product))

    @data(
        [_generate_x_rotation(0.1)],
        [_generate_y_rotation(0.2)],
        [_generate_z_rotation(0.3)],
        [np.dot(_generate_z_rotation(0.5), _generate_y_rotation(0.4))],
        [np.dot(_generate_y_rotation(0.5), _generate_x_rotation(0.4))]
    )
    @unpack
    def test_commutator_decompose_returns_tuple_whose_commutator_equals_input(self, u_so3):
        """Test that ``commutator_decompose`` exactly decomposes the input."""
        actual_result = commutator_decompose(u_so3)
        first_so3 = actual_result[0].product
        second_so3 = actual_result[1].product
        actual_commutator = np.dot(first_so3, np.dot(second_so3, np.dot(
            np.matrix.getH(first_so3), np.matrix.getH(second_so3))))
        self.assertTrue(are_almost_equal_so3_matrices(
            actual_commutator, u_so3))

    @given(st.builds(_build_rotation, st.floats(max_value=2*math.pi, min_value=0),
                     st.integers(min_value=0, max_value=4)))
    def test_commutator_decompose_returns_tuple_whose_commutator_equals_input_2(self, u_so3):
        """Test that ``commutator_decompose`` exactly decomposes the input."""
        actual_result = commutator_decompose(u_so3)
        first_so3 = actual_result[0].product
        second_so3 = actual_result[1].product
        actual_commutator = np.dot(first_so3, np.dot(second_so3, np.dot(
            np.matrix.getH(first_so3), np.matrix.getH(second_so3))))
        self.assertTrue(are_almost_equal_so3_matrices(
            actual_commutator, u_so3))

    @given(st.builds(_build_rotation, st.floats(max_value=2*math.pi, min_value=0),
                     st.integers(min_value=0, max_value=4)))
    def test_commutator_decompose_returns_tuple_with_first_x_axis_rotation(self, u_so3):
        """Test that ``commutator_decompose`` returns a X-rotation as first element."""
        actual_result = commutator_decompose(u_so3)
        actual_first = actual_result[0]
        actual = actual_first.product
        self.assertAlmostEqual(actual[0][0], 1.0)
        self.assertAlmostEqual(actual[0][1], 0.0)
        self.assertAlmostEqual(actual[0][2], 0.0)
        self.assertAlmostEqual(actual[1][0], 0.0)
        self.assertAlmostEqual(actual[2][0], 0.0)

    @given(st.builds(_build_rotation, st.floats(max_value=2*math.pi, min_value=0),
                     st.integers(min_value=0, max_value=4)))
    def test_commutator_decompose_returns_tuple_with_second_y_axis_rotation(self, u_so3):
        """Test that ``commutator_decompose`` returns a Y-rotation as second element."""
        actual_result = commutator_decompose(u_so3)
        actual_second = actual_result[1]
        actual = actual_second.product
        self.assertAlmostEqual(actual[1][1], 1.0)
        self.assertAlmostEqual(actual[0][1], 0.0)
        self.assertAlmostEqual(actual[1][0], 0.0)
        self.assertAlmostEqual(actual[1][2], 0.0)
        self.assertAlmostEqual(actual[2][1], 0.0)

    def test_solovay_kitaev_basic_gates_on_h_returns_circuit_h(self):
        """Test that ``SolovayKitaevDecomposition`` returns a circuit with an H-gate when
        it approximates the H-gate and the H-gate is in the basic gates."""

        circuit = QuantumCircuit(1)
        circuit.h(0)

        basic_gates = [HGate(), TGate(), SGate(), gates.IGate(), HGate().inverse(), TdgGate(),
                       SdgGate(), RXGate(math.pi), RYGate(math.pi)]
        synth = SolovayKitaevDecomposition(3, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)
        self.assertTrue(circuit == decomposed_circuit)

    def test_solovay_kitaev_basic_gates_on_i_returns_empty_circuit(self):
        """Test that ``SolovayKitaevDecomposition`` returns an empty circuit when
        it approximates the I-gate."""
        circuit = QuantumCircuit(1)
        circuit.i(0)

        basic_gates = [HGate(), TGate(), SGate(), gates.IGate(), HGate().inverse(), TdgGate(),
                       SdgGate(), RXGate(math.pi), RYGate(math.pi)]
        synth = SolovayKitaevDecomposition(3, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)
        self.assertTrue(QuantumCircuit(1) == decomposed_circuit)

    def test_solovay_kitaev_basic_gates_on_t_returns_circuit_t(self):
        """Test that ``SolovayKitaevDecomposition`` returns a circuit with a T-gate when
        it approximates the T-gate and the T-gate is in the basic gates."""
        circuit = QuantumCircuit(1)
        circuit.t(0)

        basic_gates = [HGate(), TGate(), SGate(), gates.IGate(), HGate().inverse(), TdgGate(),
                       SdgGate(), RXGate(math.pi), RYGate(math.pi)]
        synth = SolovayKitaevDecomposition(3, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)
        self.assertTrue(circuit == decomposed_circuit)

    def test_solovay_kitaev_basic_gates_on_s_returns_circuit_s(self):
        """Test that ``SolovayKitaevDecomposition`` returns a circuit with an S-gate when
        it approximates the S-gate and the S-gate is in the basic gates."""
        circuit = QuantumCircuit(1)
        circuit.s(0)

        basic_gates = [HGate(), TGate(), SGate(), gates.IGate(), HGate().inverse(), TdgGate(),
                       SdgGate(), RXGate(math.pi), RYGate(math.pi)]
        synth = SolovayKitaevDecomposition(3, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)
        self.assertTrue(circuit == decomposed_circuit)

    def test_solovay_kitaev_basic_gates_on_rxpi_returns_circuit_rxpi(self):
        """Test that ``SolovayKitaevDecomposition`` returns a circuit with an RX-gate when
        it approximates the RX-gate and the RX-gate is in the basic gates."""
        circuit = QuantumCircuit(1)
        circuit.rx(math.pi, 0)

        basic_gates = [HGate(), TGate(), SGate(), gates.IGate(), HGate().inverse(), TdgGate(),
                       SdgGate(), RXGate(math.pi), RYGate(math.pi)]
        synth = SolovayKitaevDecomposition(3, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)
        self.assertTrue(circuit == decomposed_circuit)

    def test_solovay_kitaev_basic_gates_on_rypi_returns_circuit_rypi(self):
        """Test that ``SolovayKitaevDecomposition`` returns a circuit with an RY-gate when
        it approximates the RY-gate and the RY-gate is in the basic gates."""
        circuit = QuantumCircuit(1)
        circuit.ry(math.pi, 0)

        basic_gates = [HGate(), TGate(), SGate(), gates.IGate(), HGate().inverse(), TdgGate(),
                       SdgGate(), RXGate(math.pi), RYGate(math.pi)]
        synth = SolovayKitaevDecomposition(3, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)
        self.assertTrue(circuit == decomposed_circuit)

    @data(2, 3, 4, 5)
    def test_solovay_kitaev_basic_gates_on_qft_returns_circuit_qft(self, nr_qubits):
        """Test that ``SolovayKitaevDecomposition`` returns a QFT-circuit when
        it approximates the QFT-circuit and the basic gates contain the gates of QFT."""
        circuit = QFT(nr_qubits, 0)
        basic_gates = [HGate(), TGate(), SGate(), gates.IGate(), HGate().inverse(), TdgGate(),
                       SdgGate(), RXGate(math.pi), RYGate(math.pi)]
        synth = SolovayKitaevDecomposition(4, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        self.assertTrue(circuit == decomposed_circuit)

    @data(2, 3, 4, 5)
    def test_solovay_kitaev_on_qft_without_h_in_basic_gates_does_not_return_qft(self, nr_qubits):
        """Test that ``SolovayKitaevDecomposition`` does not return a QFT-circuit when
        it approximates the QFT-circuit and the basic gates do not contain H-gate and inverse"""
        circuit = QFT(nr_qubits, 0)
        basic_gates = [TGate(), SGate(), gates.IGate(), TdgGate(),
                       SdgGate(), RXGate(math.pi), RYGate(math.pi)]
        synth = SolovayKitaevDecomposition(4, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        self.assertFalse(circuit == decomposed_circuit)

    def test_str_basis_gates(self):
        """Test specifying the basis gates by string works."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)

        basis_gates = ['h', 't', 's']
        synth = SolovayKitaevDecomposition(2, basis_gates)

        dag = circuit_to_dag(circuit)
        discretized = dag_to_circuit(synth.run(dag))

        reference = QuantumCircuit(1, global_phase=-np.pi / 8)
        reference.h(0)
        reference.t(0)
        reference.h(0)

        self.assertEqual(discretized, reference)

    @data(2, 3, 4, 5)
    def test_solovay_kitaev_converges(self, depth: int):
        """Test that the SolovayKitaevDecomposition returns a circuit closer to the input gate
        when the depth increaes. """

        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)

        basic_gates = [HGate(), TGate(), SGate(), gates.IGate(), HGate().inverse(), TdgGate(),
                       SdgGate(), RXGate(math.pi), RYGate(math.pi)]
        synth = SolovayKitaevDecomposition(depth, basic_gates)
        synth_plus_one = SolovayKitaevDecomposition(depth+1, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        dag_plus_one = circuit_to_dag(circuit)
        decomposed_dag_plus_one = synth_plus_one.run(dag_plus_one)
        decomposed_circuit_plus_one = dag_to_circuit(decomposed_dag_plus_one)

        self.assertLess(distance(Operator(circuit).data,
                                 Operator(decomposed_circuit_plus_one).data),
                        distance(Operator(circuit).data, Operator(decomposed_circuit).data))


@ddt
class TestSolovayKitaevUtils(QiskitTestCase):
    """Test the algebra utils."""

    @data([GateSequence([IGate()]), IGate(), GateSequence([IGate(), IGate()])])
    @unpack
    def test_append(self, first_value, second_value, third_value):
        """Test append of ``GateSequence``."""
        actual_gate = first_value.append(second_value)
        self.assertTrue(actual_gate == third_value)

    @data(
        [GateSequence([IGate()]), GateSequence([IGate(), IGate()]), 0.0],
        [GateSequence([IGate(), IGate()]), GateSequence(
            [IGate(), IGate(), IGate()]), 0.0],
        [GateSequence([IGate(), RXGate(1)]), GateSequence([RXGate(1)]), 0.0],
        [GateSequence([RXGate(1)]), GateSequence([RXGate(0.99)]), 0.01],
    )
    @unpack
    def test_represents_same_gate_true(self, first_sequence, second_sequence, precision):
        """Test ``represents_same_gate``."""
        self.assertTrue(first_sequence.represents_same_gate(
            second_sequence, precision))

    @data(
        [GateSequence([IGate()]), GateSequence([IGate(), RXGate(1)]), 0.0],
        [GateSequence([RXGate(1), RXGate(1), RXGate(0.5)]),
         GateSequence([RXGate(1)]), 0.0],
        [GateSequence([RXGate(1)]), GateSequence([RXGate(0.5)]), 0.0],
        [GateSequence([RXGate(1)]), GateSequence(
            [RXGate(0.3), RXGate(0.2)]), 0.0],
        [GateSequence([RXGate(0.3), RXGate(0.5), RXGate(1)]),
         GateSequence([RXGate(0.3), RXGate(0.2)]), 0.0],
        [GateSequence([RXGate(0.3), RXGate(0.8), RXGate(1)]),
         GateSequence([RXGate(0.1), RXGate(0.2)]), 0.0],
    )
    @unpack
    def test_represents_same_gate_false(self, first_sequence, second_sequence, precision):
        """Test ``represents_same_gate``."""
        self.assertFalse(first_sequence.represents_same_gate(
            second_sequence, precision))

    @data(
        [GateSequence([IGate(), IGate()]), GateSequence([]), 0.0],
        [GateSequence([IGate(), RXGate(1), IGate()]),
         GateSequence([RXGate(1)]), 0.0],
        [GateSequence([IGate(), RXGate(1), IGate(), RXGate(0.4)]),
         GateSequence([RXGate(1), RXGate(0.4)]), 0.0],
        [GateSequence([IGate(), RXGate(2*math.pi), RXGate(2*math.pi)]),
         GateSequence([]), 1e10],
    )
    @unpack
    def test_simplify(self, original_sequence, expected_sequence, precision):
        """Test the simplify method on ``GateSequence``."""
        actual_sequence = original_sequence.simplify(precision)
        self.assertTrue(actual_sequence == expected_sequence)


if __name__ == '__main__':
    unittest.main()
