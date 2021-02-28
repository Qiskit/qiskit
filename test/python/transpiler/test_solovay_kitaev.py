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

from scipy.optimize import minimize
from ddt import ddt, data

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TGate, TdgGate, HGate, SGate, SdgGate, IGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import SolovayKitaevDecomposition
from qiskit.test import QiskitTestCase
from qiskit.transpiler.passes.synthesis.solovay_kitaev import commutator_decompose
from qiskit.transpiler.passes.synthesis.solovay_kitaev_utils import GateSequence
from qiskit.quantum_info import Operator

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

    if abs(np.linalg.det(array)-1.0) > 1e-10:
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


class TestSolovayKitaev(QiskitTestCase):
    """Test the Solovay Kitaev algorithm and transformation pass."""

    def test_i_returns_empty_circuit(self):
        """Test that ``SolovayKitaevDecomposition`` returns an empty circuit when
        it approximates the I-gate."""
        circuit = QuantumCircuit(1)
        circuit.i(0)

        basic_gates = [HGate(), TGate(), TdgGate()]
        skd = SolovayKitaevDecomposition(3, basic_gates, 3)

        decomposed_circuit = skd(circuit)
        self.assertEqual(QuantumCircuit(1), decomposed_circuit)

    def test_exact_decomposition_acts_trivially(self):
        """Test that the a circuit that can be represented exactly is represented exactly."""
        circuit = QuantumCircuit(1)
        circuit.t(0)
        circuit.h(0)
        circuit.tdg(0)

        basic_gates = [HGate(), TGate(), TdgGate()]
        synth = SolovayKitaevDecomposition(3, basic_gates, 3)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)
        self.assertEqual(circuit, decomposed_circuit)

    def test_str_basis_gates(self):
        """Test specifying the basis gates by string works."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)

        basis_gates = ['h', 't', 's']
        synth = SolovayKitaevDecomposition(2, basis_gates, 3)

        dag = circuit_to_dag(circuit)
        discretized = dag_to_circuit(synth.run(dag))

        reference = QuantumCircuit(1, global_phase=-np.pi / 8)
        reference.h(0)
        reference.t(0)
        reference.h(0)

        self.assertEqual(discretized, reference)


@ddt
class TestGateSequence(QiskitTestCase):
    """Test the ``GateSequence`` class."""

    def test_append(self):
        """Test append."""
        seq = GateSequence([IGate()])
        seq.append(HGate())

        ref = GateSequence([IGate(), HGate()])
        self.assertEqual(seq, ref)

    def test_eq(self):
        """Test equality."""
        base = GateSequence([HGate(), HGate()])
        seq1 = GateSequence([HGate(), HGate()])
        seq2 = GateSequence([IGate()])
        seq3 = GateSequence([HGate(), HGate()])
        seq3.global_phase = 0.12
        seq4 = GateSequence([IGate(), HGate()])

        with self.subTest('equal'):
            self.assertEqual(base, seq1)

        with self.subTest('same product, but different repr (-> false)'):
            self.assertNotEqual(base, seq2)

        with self.subTest('differing global phase (-> false)'):
            self.assertNotEqual(base, seq3)

        with self.subTest('same num gates, but different gates (-> false)'):
            self.assertNotEqual(base, seq4)

    def test_to_circuit(self):
        """Test converting a gate sequence to a circuit."""
        seq = GateSequence([HGate(), HGate(), TGate(), SGate(), SdgGate()])
        ref = QuantumCircuit(1)
        ref.h(0)
        ref.h(0)
        ref.t(0)
        ref.s(0)
        ref.sdg(0)
        # a GateSequence is SU(2), so add the right phase
        z = 1 / np.sqrt(np.linalg.det(Operator(ref)))
        ref.global_phase = np.arctan2(np.imag(z), np.real(z))

        self.assertEqual(seq.to_circuit(), ref)

    def test_adjoint(self):
        """Test adjoint."""
        seq = GateSequence([TGate(), SGate(), HGate(), IGate()])
        inv = GateSequence([IGate(), HGate(), SdgGate(), TdgGate()])

        self.assertEqual(seq.adjoint(), inv)

    def test_copy(self):
        """Test copy."""
        seq = GateSequence([IGate()])
        copied = seq.copy()
        seq.gates.append(HGate())

        self.assertEqual(len(seq.gates), 2)
        self.assertEqual(len(copied.gates), 1)

    @data(0, 1, 10)
    def test_len(self, n):
        """Test __len__."""
        seq = GateSequence([IGate()] * n)
        self.assertEqual(len(seq), n)

    def test_getitem(self):
        """Test __getitem__."""
        seq = GateSequence([IGate(), HGate(), IGate()])
        self.assertEqual(seq[0], IGate())
        self.assertEqual(seq[1], HGate())
        self.assertEqual(seq[2], IGate())

        self.assertEqual(seq[-2], HGate())

    def test_from_su2_matrix(self):
        """Test from_matrix with an SU2 matrix."""
        matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        matrix /= np.sqrt(np.linalg.det(matrix))
        seq = GateSequence.from_matrix(matrix)

        ref = GateSequence([HGate()])

        self.assertEqual(seq.gates, list())
        self.assertTrue(np.allclose(seq.product, ref.product))
        self.assertEqual(seq.global_phase, 0)

    def test_from_so3_matrix(self):
        """Test from_matrix with an SO3 matrix."""
        matrix = np.array([[0, 0, -1],
                           [0, -1, 0],
                           [-1, 0, 0]])
        seq = GateSequence.from_matrix(matrix)

        ref = GateSequence([HGate()])

        self.assertEqual(seq.gates, list())
        self.assertTrue(np.allclose(seq.product, ref.product))
        self.assertEqual(seq.global_phase, 0)

    def test_from_invalid_matrix(self):
        """Test from_matrix with invalid matrices."""
        with self.subTest('2x2 but not SU2'):
            matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            with self.assertRaises(ValueError):
                _ = GateSequence.from_matrix(matrix)

        with self.subTest('not 2x2 or 3x3'):
            with self.assertRaises(ValueError):
                _ = GateSequence.from_matrix(np.array([[1]]))

    def test_dot(self):
        """Test dot."""
        seq1 = GateSequence([HGate()])
        seq2 = GateSequence([TGate(), SGate()])
        composed = seq1.dot(seq2)

        ref = GateSequence([TGate(), SGate(), HGate()])

        # check the product matches
        self.assertTrue(np.allclose(ref.product, composed.product))

        # check the circuit & phases are equivalent
        self.assertTrue(Operator(ref.to_circuit()).equiv(composed.to_circuit()))


@ddt
class TestSolovayKitaevUtils(QiskitTestCase):
    """Test the public functions in the Solovay Kitaev utils."""

    @data(
        _generate_x_rotation(0.1),
        _generate_y_rotation(0.2),
        _generate_z_rotation(0.3),
        np.dot(_generate_z_rotation(0.5), _generate_y_rotation(0.4)),
        np.dot(_generate_y_rotation(0.5), _generate_x_rotation(0.4))
    )
    def test_commutator_decompose_return_type(self, u_so3: np.ndarray):
        """Test that ``commutator_decompose`` returns two SO(3) gate sequences."""
        v, w = commutator_decompose(u_so3)
        self.assertTrue(is_so3_matrix(v.product))
        self.assertTrue(is_so3_matrix(w.product))
        self.assertIsInstance(v, GateSequence)
        self.assertIsInstance(w, GateSequence)

    @data(
        _generate_x_rotation(0.1),
        _generate_y_rotation(0.2),
        _generate_z_rotation(0.3),
        np.dot(_generate_z_rotation(0.5), _generate_y_rotation(0.4)),
        np.dot(_generate_y_rotation(0.5), _generate_x_rotation(0.4))
    )
    def test_commutator_decompose_decomposes_correctly(self, u_so3):
        """Test that ``commutator_decompose`` exactly decomposes the input."""
        v, w = commutator_decompose(u_so3)
        v_so3 = v.product
        w_so3 = w.product
        actual_commutator = np.dot(v_so3, np.dot(w_so3, np.dot(np.conj(v_so3).T, np.conj(w_so3).T)))
        self.assertTrue(np.allclose(actual_commutator, u_so3))


if __name__ == '__main__':
    unittest.main()
