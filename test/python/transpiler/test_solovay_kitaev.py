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

import unittest
import math
import numpy as np
import scipy

from hypothesis import given
import hypothesis.strategies as st
from scipy.optimize import minimize
from scipy.stats import special_ortho_group
from ddt import ddt, data, unpack
import qiskit.circuit.library as gates
import itertools

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import TGate, RXGate, RYGate, HGate, SGate, IGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import SolovayKitaevDecomposition
from qiskit.transpiler.passes.synthesis import commutator_decompose
from qiskit.test import QiskitTestCase
from qiskit.quantum_info import Operator

from qiskit.transpiler.passes.synthesis import GateSequence 

from ddt import ddt, data, unpack
from qiskit.transpiler.passes.synthesis import (
    compute_euler_angles_from_s03, compute_frobenius_norm,
    compute_su2_from_euler_angles, convert_su2_to_so3, 
    _compute_trace_so3, solve_decomposition_angle, 
    compute_rotation_between, _compute_commutator_so3,
    compute_rotation_from_angle_and_axis,
    compute_rotation_axis, convert_so3_to_su2
)


# pylint: disable=invalid-name, missing-class-docstring

class H(Gate):
    def __init__(self):
        super().__init__('H', 1, [])

    def _define(self):
        definition = QuantumCircuit(1)
        definition.h(0)
        definition.global_phase = np.pi / 2
        self.definition = definition

    def to_matrix(self):
        return 1j * gates.HGate().to_matrix()

    def inverse(self):
        return H_dg()


class H_dg(Gate):
    def __init__(self):
        super().__init__('iH_dg', 1, [])

    def _define(self):
        definition = QuantumCircuit(1)
        definition.h(0)
        definition.global_phase = -np.pi / 2
        self.definition = definition

    def to_matrix(self):
        return -1j * gates.HGate().to_matrix()

    def inverse(self):
        return H()


class T(Gate):
    def __init__(self):
        super().__init__('T', 1, [])

    def _define(self):
        definition = QuantumCircuit(1)
        definition.t(0)
        definition.global_phase = -np.pi / 8
        self.definition = definition

    def to_matrix(self):
        return np.exp(-1j * np.pi / 8) * gates.TGate().to_matrix()

    def inverse(self):
        return T_dg()


class T_dg(Gate):
    def __init__(self):
        super().__init__('T_dg', 1, [])

    def _define(self):
        definition = QuantumCircuit(1)
        definition.tdg(0)
        definition.global_phase = np.pi / 8
        self.definition = definition

    def to_matrix(self):
        return np.exp(1j * np.pi / 8) * gates.TdgGate().to_matrix()

    def inverse(self):
        return T()


class S(Gate):
    def __init__(self):
        super().__init__('S', 1, [])

    def _define(self):
        definition = QuantumCircuit(1)
        definition.s(0)
        definition.global_phase = -np.pi / 4
        self.definition = definition

    def to_matrix(self):
        return np.exp(-1j * np.pi / 4) * gates.SGate().to_matrix()

    def inverse(self):
        return S_dg()


class S_dg(Gate):
    def __init__(self):
        super().__init__('S_dg', 1, [])

    def _define(self):
        definition = QuantumCircuit(1)
        definition.sdg(0)
        definition.global_phase = np.pi / 4
        self.definition = definition

    def to_matrix(self):
        return np.exp(1j * np.pi / 4) * gates.SdgGate().to_matrix()

    def inverse(self):
        return S()


def distance(A, B):
    """Find the distance in norm of A and B, ignoring global phase."""

    def objective(global_phase):
        return np.linalg.norm(A - np.exp(1j * global_phase) * B)
    result1 = minimize(objective, [1], bounds=[(-np.pi, np.pi)])
    result2 = minimize(objective, [0.5], bounds=[(-np.pi, np.pi)])
    return min(result1.fun, result2.fun)

def _generate_x_rotation(angle:float) -> np.ndarray:
    return np.array([[1,0,0],[0,math.cos(angle),-math.sin(angle)],[0,math.sin(angle),math.cos(angle)]])

def _generate_y_rotation(angle:float) -> np.ndarray:
    return np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])

def _generate_z_rotation(angle:float) -> np.ndarray:
    return np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])

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
        return np.array([1.0,0.0,0.0])
    elif axis == 1:
        return np.array([0.0,1.0,0.0])
    elif axis == 2:
        return np.array([0.0,0.0,1.0])
    else:
        return np.array([1/math.sqrt(3),1/math.sqrt(3),1/math.sqrt(3)])

def _generate_x_su2(angle:float) -> np.ndarray:
    return np.array([[math.cos(angle/2), math.sin(angle/2)*1j],
                       [math.sin(angle/2)*1j, math.cos(angle/2)]], dtype=complex)

def _generate_y_su2(angle:float) -> np.ndarray:
    return np.array([[math.cos(angle/2), math.sin(angle/2)],
                         [-math.sin(angle/2), math.cos(angle/2)]], dtype=complex)

def _generate_z_su2(angle:float) -> np.ndarray:
    return np.array([[np.exp(-(1/2)*angle*1j), 0], [0, np.exp((1/2)*angle*1j)]], dtype=complex)

def _generate_su2(alpha: complex, beta: complex) -> np.ndarray:
    base = np.array([[alpha,beta],[-np.conj(beta),np.conj(alpha)]])
    det = np.linalg.det(base)
    if abs(det)<1e10:
        return np.array([[1,0],[0,1]])
    else:
        return np.linalg.det(base)*base

def _build_unit_vector(a: float, b: float, c: float) -> np.ndarray:
    vector = np.array([a,b,c])
    if a != 0.0 or b != 0.0 or c!= 0.0:
        unit_vector = vector/np.linalg.norm(vector)
        return unit_vector
    else:
        return np.array([1,0,0])

def is_so3_matrix(array: np.ndarray) -> bool:
    return array.shape == (3,3) and abs(np.linalg.det(array)-1.0)< 1e-10 and not False in np.isreal(array) 

def are_almost_equal_so3_matrices(a: np.ndarray, b: np.ndarray) -> bool:
    for t in itertools.product(range(2),range(2)):
        if abs(a[t[0]][t[1]]-b[t[0]][t[1]])> 1e-10:
            return False
    return True

@ddt
class TestSolovayKitaev(QiskitTestCase):
    """Test the Solovay Kitaev algorithm and transformation pass."""

    @given(st.builds(_build_rotation,st.floats(max_value=2*math.pi,min_value=0),st.integers(min_value=0,max_value=4)))
    def test_commutator_decompose_returns_tuple_of_two_so3_gatesequences(self, u_so3: np.ndarray):        
        actual_result = commutator_decompose(u_so3)
        self.assertTrue(is_so3_matrix(actual_result[0].product))
        self.assertTrue(is_so3_matrix(actual_result[1].product))

    @given(st.builds(_build_rotation,st.floats(max_value=2*math.pi,min_value=0),st.integers(min_value=0,max_value=4)))
    def test_commutator_decompose_returns_tuple_whose_commutator_equals_input(self, u_so3: np.ndarray):        
        actual_result = commutator_decompose(u_so3)
        first_so3 = actual_result[0]
        second_so3 = actual_result[1]
        actual_commutator = np.dot(first_so3,np.dot(second_so3,np.dot(np.matrix.getH(first_so3),np.matrix.getH(second_so3))))
        self.assertAlmostEqual(actual_commutator,u_so3)

    @given(st.builds(_build_rotation,st.floats(max_value=2*math.pi,min_value=0),st.integers(min_value=0,max_value=4)))
    def test_commutator_decompose_returns_tuple_with_first_x_axis_rotation(self, u_so3: np.ndarray):
        actual_result = commutator_decompose(u_so3)
        actual = actual_result[0]
        self.assertAlmostEqual(actual[0][0],1)
        self.assertAlmostEqual(actual[0][1],0)
        self.assertAlmostEqual(actual[0][2],0)
        self.assertAlmostEqual(actual[1][0],0)
        self.assertAlmostEqual(actual[2][0],0)

    @given(st.builds(_build_rotation,st.floats(max_value=2*math.pi,min_value=0),st.integers(min_value=0,max_value=4)))
    def test_commutator_decompose_returns_tuple_with_second_y_axis_rotation(self, u_so3: np.ndarray):
        actual_result = commutator_decompose(u_so3)
        actual = actual_result[1]
        self.assertAlmostEqual(actual[1][1],1)
        self.assertAlmostEqual(actual[0][1],0)
        self.assertAlmostEqual(actual[1][0],0)
        self.assertAlmostEqual(actual[1][2],0)
        self.assertAlmostEqual(actual[2][1],0)
    

    def test_example(self):
        """@Lisa Example to show how to call the pass."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.2, 0)

        basic_gates = [H(), T(), S(), gates.IGate(), H_dg(), T_dg(),
                       S_dg(), RXGate(math.pi), RYGate(math.pi)]
        synth = SolovayKitaevDecomposition(3, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        print(decomposed_circuit.draw())

    def test_example_2(self):
        """@Lisa Example to show how to call the pass."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)

        basic_gates = [H(), T(), S(), T_dg(), S_dg()]
        synth = SolovayKitaevDecomposition(2, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        print(decomposed_circuit.draw())
        print('Original')
        print(Operator(circuit))
        print('Synthesized')
        print(Operator(decomposed_circuit))

    def test_example_non_su2(self):
        """@Lisa Example to show how to call the pass."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)

        basic_gates = [HGate(), TGate(), SGate()]
        synth = SolovayKitaevDecomposition(2, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        print(decomposed_circuit.draw())
        print('Original')
        print(Operator(circuit))
        print('Synthesized')
        print(Operator(decomposed_circuit))
        self.assertLess(distance(Operator(circuit).data, Operator(decomposed_circuit).data), 0.1)


@ddt
class TestSolovayKitaevUtils(QiskitTestCase):
    """Test the algebra utils."""

    @data([GateSequence([IGate()]),IGate(),GateSequence([IGate(),IGate()])])
    @unpack
    def test_append(self,first_value,second_value,third_value):
        actual_gate = first_value.append(second_value)
        self.assertTrue(actual_gate == third_value)

    """
    @data([GateSequence([IGate()]),GateSequence([TGate()]),GateSequence([IGate(),TGate()])],
          [GateSequence([IGate()]),GateSequence([TGate(),IGate()]),GateSequence([IGate(),TGate(),IGate()])],
          [GateSequence([IGate(),TGate(),RXGate(0.1)]),GateSequence([TGate(),IGate()]),GateSequence([IGate(),TGate(),RXGate(0.1),TGate(),IGate()])])
    @unpack
    def test_append(self,first_value,second_value,third_value):
        actual_gate = first_value + second_value
        self.assertTrue(actual_gate == third_value)
    """

    @data(
        [GateSequence([IGate()]),GateSequence([IGate(),IGate()]),0.0],
        [GateSequence([IGate(),IGate()]),GateSequence([IGate(),IGate(),IGate()]),0.0],
        [GateSequence([IGate(),RXGate(1)]),GateSequence([RXGate(1)]),0.0],
        [GateSequence([RXGate(1)]),GateSequence([RXGate(0.99)]),0.01],
          )
    @unpack
    def test_represents_same_gate_true(self,first_sequence: 'GateSequence',second_sequence: 'GateSequence', precision: float):
        self.assertTrue(first_sequence.represents_same_gate(second_sequence, precision))
    
    @data(
        [GateSequence([IGate()]),GateSequence([IGate(),RXGate(1)]),0.0],
        [GateSequence([RXGate(1),RXGate(1),RXGate(0.5)]),GateSequence([RXGate(1)]),0.0],
        [GateSequence([RXGate(1)]),GateSequence([RXGate(0.5)]),0.0],
        [GateSequence([RXGate(1)]),GateSequence([RXGate(0.3),RXGate(0.2)]),0.0],
        [GateSequence([RXGate(0.3),RXGate(0.5),RXGate(1)]),GateSequence([RXGate(0.3),RXGate(0.2)]),0.0],
        [GateSequence([RXGate(0.3),RXGate(0.8),RXGate(1)]),GateSequence([RXGate(0.1),RXGate(0.2)]),0.0],
          )
    @unpack
    def test_represents_same_gate_false(self,first_sequence: 'GateSequence',second_sequence: 'GateSequence', precision: float):
        self.assertFalse(first_sequence.represents_same_gate(second_sequence, precision))

    @data(
        [GateSequence([IGate(),IGate()]),GateSequence([]),0.0],
        [GateSequence([IGate(),RXGate(1),IGate()]),GateSequence([RXGate(1)]),0.0],
        [GateSequence([IGate(),RXGate(1),IGate(),RXGate(0.4)]),GateSequence([RXGate(1),RXGate(0.4)]),0.0],
        [GateSequence([IGate(),RXGate(2*math.pi),RXGate(2*math.pi)]),GateSequence([]),1e10],
    )
    @unpack
    def test_simplify(self,original_sequence: 'GateSequence',expected_sequence: 'GateSequence', precision: float):
        actual_sequence = original_sequence.simplify(precision)
        self.assertTrue(actual_sequence == expected_sequence)


   
@ddt
class AlgebraTest(unittest.TestCase):
    """Test the algebra methods"""
    
    @data([[1],1],[[1,1],np.sqrt(2)],[[1,1,1],np.sqrt(3)])
    @unpack
    def test_compute_frobenius_norm(self,first_data,expected):
        actual = compute_frobenius_norm(first_data)
        self.assertTrue(actual == expected) 
    
    @given(st.lists(st.floats(allow_nan=False),min_size=1))
    def test_compute_frobenius_norm_returns_np_linalg_norm(self,vector):
        self.assertTrue(compute_frobenius_norm(vector) == np.linalg.norm(vector))

    @data(_generate_random_rotation())
    def test_when_compute_euler_angles_from_so3_rotation_then_so3_from_angles_is_again_rotation(self,rotation):
        actual_angles = compute_euler_angles_from_s03(rotation)
        phi = actual_angles[0]
        theta = actual_angles[1]
        psi = actual_angles[2]
        actual_rotation = np.dot(np.dot(_generate_z_rotation(phi),_generate_y_rotation(theta)),_generate_x_rotation(psi))
        self.assertAlmostEqual(actual_rotation[0][0],rotation[0][0])
        self.assertAlmostEqual(actual_rotation[0][1],rotation[0][1])
        self.assertAlmostEqual(actual_rotation[0][2],rotation[0][2])
        self.assertAlmostEqual(actual_rotation[1][0],rotation[1][0])
        self.assertAlmostEqual(actual_rotation[1][1],rotation[1][1])
        self.assertAlmostEqual(actual_rotation[1][2],rotation[1][2])
        self.assertAlmostEqual(actual_rotation[2][0],rotation[2][0])
        self.assertAlmostEqual(actual_rotation[2][1],rotation[2][1])
        self.assertAlmostEqual(actual_rotation[2][2],rotation[2][2])

    @data(
        [_generate_x_rotation(0.1),(0,0,0.1)],
        [_generate_y_rotation(0.2),(0,0.2,0)],
        [_generate_z_rotation(0.3),(0.3,0,0)],
        [np.dot(_generate_z_rotation(0.5),_generate_y_rotation(0.4)),(0.5,0.4,0)],
        [np.dot(_generate_y_rotation(0.5),_generate_x_rotation(0.4)),(0,0.5,0.4)]
    )
    @unpack
    def test_compute_euler_angles_from_so3(self,rotation,expected):
        self.assertAlmostEqual(compute_euler_angles_from_s03(rotation)[0],expected[0])
        self.assertAlmostEqual(compute_euler_angles_from_s03(rotation)[1],expected[1])
        self.assertAlmostEqual(compute_euler_angles_from_s03(rotation)[2],expected[2])

    @data(
        [(0,0,0),np.array([[1,0],[0,1]])],
        [(0.1,0,0),np.array([[np.exp(-0.05j),0],[0,np.exp(0.05j)]])],
        [(0,0.2,0), np.array([[np.cos(0.1),np.sin(0.1)],[-np.sin(0.1),np.cos(0.1)]])],
        [(0,0,0.3), np.array([[np.cos(0.15),np.sin(0.15)*1j],[np.sin(0.15)*1j,np.cos(0.15)]])],
        [(0.1,0.2,0.3), np.dot(_generate_z_su2(0.1),np.dot(_generate_y_su2(0.2),_generate_x_su2(0.3)))]
    )
    @unpack
    def test_compute_su2_from_euler_angles(self,angles,expected):
        actual_su2 = compute_su2_from_euler_angles(angles)
        self.assertAlmostEqual(actual_su2[0][0],expected[0][0])
        self.assertAlmostEqual(actual_su2[0][1],expected[0][1])
        self.assertAlmostEqual(actual_su2[1][0],expected[1][0])
        self.assertAlmostEqual(actual_su2[1][1],expected[1][1])

    @given(st.tuples(st.floats(allow_nan=False, allow_infinity=False),st.floats(allow_nan=False,allow_infinity=False),st.floats(allow_nan=False,allow_infinity=False)))
    def test_compute_su2_from_euler_angles_returns_su2(self,angles):
        actual = compute_su2_from_euler_angles(angles)
        self.assertEqual(actual[0][0],np.conj(actual[1][1]))
        self.assertEqual(actual[0][1],-np.conj(actual[1][0]))
        self.assertAlmostEqual(np.linalg.det(actual),1)

    @data(
        [np.array([[1,0],[0,1]]),np.array([[1,0,0],[0,1,0],[0,0,1]])],
        [_generate_x_su2(0.3),_generate_x_rotation(0.3)],
        [_generate_y_su2(0.5),_generate_y_rotation(0.5)],
        [_generate_z_su2(0.7),_generate_z_rotation(0.7)],
    )
    @unpack
    def test_convert_su2_to_so3(self,su2,expected_so3):
        actual_so3 = convert_su2_to_so3(su2)
        self.assertAlmostEqual(actual_so3[0][0],expected_so3[0][0])
        self.assertAlmostEqual(actual_so3[0][1],expected_so3[0][1])
        self.assertAlmostEqual(actual_so3[0][2],expected_so3[0][2])
        self.assertAlmostEqual(actual_so3[1][0],expected_so3[1][0])
        self.assertAlmostEqual(actual_so3[1][1],expected_so3[1][1])
        self.assertAlmostEqual(actual_so3[1][2],expected_so3[1][2])
        self.assertAlmostEqual(actual_so3[2][0],expected_so3[2][0])
        self.assertAlmostEqual(actual_so3[2][1],expected_so3[2][1])
        self.assertAlmostEqual(actual_so3[2][2],expected_so3[2][2])

    @given(st.builds(_generate_su2,st.complex_numbers(max_magnitude=10),st.complex_numbers(max_magnitude=10)))
    def test_convert_su2_to_so3_returns_so3(self,su2):
        actual = convert_su2_to_so3(su2)
        self.assertAlmostEqual(np.linalg.det(actual),1)

    @data(
        [np.array([[1,0,0],[0,1,0],[0,0,1]]),3.0],
        [np.array([[1,0,0],[0,1,0],[0,0,1+9e-11]]),3.0000000000000],
        [_generate_x_rotation(0.3), 2.910672978251212],
        [_generate_y_rotation(0.5), 2.7551651237807455],
        [_generate_z_rotation(0.7), 2.529684374568977],
    )
    @unpack
    def test_compute_trace_so3(self,so3: np.ndarray,expected_trace: float):
        actual_trace = _compute_trace_so3(so3)
        self.assertEqual(actual_trace,expected_trace)

    @data(
        [np.array([[1,0,0],[0,1,0],[0,0,1]]),0.0],
        [_generate_x_rotation(0.3), 0.3],
        [_generate_y_rotation(0.5), 0.5],
        [_generate_z_rotation(0.7), 0.7],
    )
    @unpack
    def test_solve_decomposition_angle(self,so3: np.ndarray,original_angle: float):
        actual_angle = solve_decomposition_angle(so3)
        expected = math.sin(original_angle/2)
        actual = math.sqrt(1-math.sin(actual_angle/2)**4)*2*math.sin(actual_angle/2)**2
        self.assertAlmostEqual(actual,expected)

    @given(st.builds(_generate_z_rotation,st.floats(max_value=2*math.pi,min_value=0)))
    def test_solve_decomposition_angle_returns_angle_satisfying_equation(self,so3: np.ndarray):
        trace = _compute_trace_so3(so3)
        original_angle = math.acos((1/2)*(trace-1))
        expected = math.sin(original_angle/2)
        actual_angle = solve_decomposition_angle(so3)
        actual = math.sqrt(1-math.sin(actual_angle/2)**4)*2*math.sin(actual_angle/2)**2
        self.assertAlmostEqual(actual,expected)

    @given(st.builds(_build_unit_vector,st.floats(min_value=0,max_value=10),st.floats(min_value=0,max_value=10),st.floats(min_value=0,max_value=10)),
    st.builds(_build_unit_vector,st.floats(min_value=0,max_value=10),st.floats(min_value=0,max_value=10),st.floats(min_value=0,max_value=10)))
    def test_compute_rotation_between(self,from_vector,to_vector):
        actual_rotation = compute_rotation_between(from_vector, to_vector)
        actual_to_vector = np.dot(actual_rotation,from_vector)
        self.assertAlmostEqual(actual_to_vector[0],to_vector[0])
        self.assertAlmostEqual(actual_to_vector[1],to_vector[1])
        self.assertAlmostEqual(actual_to_vector[2],to_vector[2])

    @given(st.builds(_build_unit_vector,st.floats(min_value=0,max_value=10),st.floats(min_value=0,max_value=10),st.floats(min_value=0,max_value=10)),
    st.builds(_build_unit_vector,st.floats(min_value=0,max_value=10),st.floats(min_value=0,max_value=10),st.floats(min_value=0,max_value=10)))
    def test_compute_rotation_between_returns_matrix_with_determinant_1(self,from_vector,to_vector):
        actual_rotation = compute_rotation_between(from_vector, to_vector)
        self.assertAlmostEqual(np.linalg.det(actual_rotation),1.0)

    @given(st.builds(_build_unit_vector,st.floats(min_value=0,max_value=10),st.floats(min_value=0,max_value=10),st.floats(min_value=0,max_value=10)),
    st.builds(_build_unit_vector,st.floats(min_value=0,max_value=10),st.floats(min_value=0,max_value=10),st.floats(min_value=0,max_value=10)))
    def test_compute_rotation_between_returns_matrix_of_shape_3_3(self,from_vector,to_vector):
        actual_rotation = compute_rotation_between(from_vector, to_vector)
        self.assertEqual(actual_rotation.shape,(3,3))

    @given(st.builds(_build_rotation,st.floats(max_value=2*math.pi,min_value=0),st.integers(min_value=0,max_value=4)),
    st.builds(_build_rotation,st.floats(max_value=2*math.pi,min_value=0),st.integers(min_value=0,max_value=4)))
    def test_compute_commutator_so3(self,first_so3: np.ndarray, second_so3: np.ndarray):
        actual_so3 = _compute_commutator_so3(first_so3,second_so3)
        expected_so3 = np.dot(first_so3,np.dot(second_so3,np.dot(np.matrix.getH(first_so3),np.matrix.getH(second_so3))))
        self.assertAlmostEqual(actual_so3[0][0],expected_so3[0][0])
        self.assertAlmostEqual(actual_so3[0][1],expected_so3[0][1])
        self.assertAlmostEqual(actual_so3[0][2],expected_so3[0][2])
        self.assertAlmostEqual(actual_so3[1][0],expected_so3[1][0])
        self.assertAlmostEqual(actual_so3[1][1],expected_so3[1][1])
        self.assertAlmostEqual(actual_so3[1][2],expected_so3[1][2])
        self.assertAlmostEqual(actual_so3[2][0],expected_so3[2][0])
        self.assertAlmostEqual(actual_so3[2][1],expected_so3[2][1])
        self.assertAlmostEqual(actual_so3[2][2],expected_so3[2][2])

    @data(
        [0,np.array([1,0,0]),np.array([[1,0,0],[0,1,0],[0,0,1]])],
        [0.3,np.array([1,0,0]),_generate_x_rotation(0.3)],
        [0.5,np.array([0,1,0]),_generate_y_rotation(0.5)],
        [0.7,np.array([0,0,1]),_generate_z_rotation(0.7)],
    )
    @unpack
    def test_compute_rotation_from_angle_and_axis(self,angle: float, axis: np.ndarray, expected_rotation):
        actual_rotation = compute_rotation_from_angle_and_axis(angle,axis)
        self.assertAlmostEqual(actual_rotation[0][0],expected_rotation[0][0])
        self.assertAlmostEqual(actual_rotation[0][1],expected_rotation[0][1])
        self.assertAlmostEqual(actual_rotation[0][2],expected_rotation[0][2])
        self.assertAlmostEqual(actual_rotation[1][0],expected_rotation[1][0])
        self.assertAlmostEqual(actual_rotation[1][1],expected_rotation[1][1])
        self.assertAlmostEqual(actual_rotation[1][2],expected_rotation[1][2])
        self.assertAlmostEqual(actual_rotation[2][0],expected_rotation[2][0])
        self.assertAlmostEqual(actual_rotation[2][1],expected_rotation[2][1])
        self.assertAlmostEqual(actual_rotation[2][2],expected_rotation[2][2])

    @given(st.floats(max_value=2*math.pi,min_value=0),st.integers(min_value=0,max_value=4))
    def test_compute_rotation_from_angle_and_axis_returns_so3_matrix(self,angle: float, axis_nr: int):
        actual_rotation = compute_rotation_from_angle_and_axis(angle,_build_axis(axis_nr))
        self.assertTrue(is_so3_matrix(actual_rotation))

    @given(st.floats(max_value=2*math.pi,min_value=0),st.integers(min_value=0,max_value=2))
    def test_compute_rotation_from_angle_and_axis_returns_expected_so3_matrix(self,angle: float, axis_nr: int):
        actual_rotation = compute_rotation_from_angle_and_axis(angle,_build_axis(axis_nr))
        expected_rotation = _build_rotation(angle,axis_nr)
        self.assertTrue(are_almost_equal_so3_matrices(actual_rotation, expected_rotation))
    

    @data(
        [0.0,0],
        [0.1,0],
        [0.2,0],
        [0.3,0],
        [math.pi/6,0],
        [math.pi/3,0],
        [math.pi/2,0],
        [math.pi,0],
        [2*math.pi/6,0],
        [0.1,1],
        [0.2,1],
        [0.3,1],
        [math.pi/6,1],
        [math.pi/3,1],
        [math.pi/2,1],
        [2*math.pi/6,1],
        [0.1,2],
        [0.2,2],
        [0.3,2],
        [math.pi/6,2],
        [math.pi/3,2],
        [math.pi/2,2],
        [2*math.pi/6,2],
    )
    @unpack
    def test_compute_rotation_axis(self,angle: float, axis_nr: int):
        rotation = _build_rotation(angle,axis_nr)
        actual_axis = compute_rotation_axis(rotation)
        expected_axis = _build_axis(axis_nr)
        self.assertAlmostEqual(actual_axis[0], expected_axis[0])
        self.assertAlmostEqual(actual_axis[1], expected_axis[1])
        self.assertAlmostEqual(actual_axis[2], expected_axis[2])

    @given(st.floats(max_value=math.pi-0.1,min_value=0.1),st.integers(min_value=0,max_value=2))
    def test_compute_rotation_axis_2(self,angle: float, axis_nr: int):
        rotation = _build_rotation(angle,axis_nr)
        actual_axis = compute_rotation_axis(rotation)
        expected_axis = _build_axis(axis_nr)
        self.assertAlmostEqual(actual_axis[0], expected_axis[0])
        self.assertAlmostEqual(actual_axis[1], expected_axis[1])
        self.assertAlmostEqual(actual_axis[2], expected_axis[2])

    @given(st.floats(max_value=math.pi-0.1,min_value=0.1),st.integers(min_value=0,max_value=2))
    def test_compute_rotation_axis_return_unit_vector_length_3(self,angle: float, axis_nr: int):
        rotation = _build_rotation(angle,axis_nr)
        actual_axis = compute_rotation_axis(rotation)
        self.assertTrue(len(actual_axis) == 3)
        self.assertAlmostEqual(np.linalg.norm(actual_axis),1.0)

    @given(st.builds(_generate_random_rotation))
    def test_convert_so3_to_su2_returns_su2(self,so3):
        actual = convert_so3_to_su2(so3)
        self.assertEqual(actual[0][0],np.conj(actual[1][1]))
        self.assertEqual(actual[0][1],-np.conj(actual[1][0]))
        self.assertAlmostEqual(np.linalg.det(actual),1)

    @data(
        [_generate_x_rotation(0.1),_generate_x_su2(0.1)],
        [_generate_y_rotation(0.2),_generate_y_su2(0.2)],
        [_generate_z_rotation(0.3),_generate_z_su2(0.3)],
        [np.dot(_generate_z_rotation(0.5),_generate_y_rotation(0.4)),np.dot(_generate_z_su2(0.5),_generate_y_su2(0.4))],
        [np.dot(_generate_y_rotation(0.5),_generate_x_rotation(0.4)),np.dot(_generate_y_su2(0.5),_generate_x_su2(0.4))]
    )
    @unpack
    def test_convert_so3_to_su2(self,rotation,expected):
        actual = convert_so3_to_su2(rotation)
        self.assertAlmostEqual(actual[0][0],expected[0][0])
        self.assertAlmostEqual(actual[0][1],expected[0][1])
        self.assertAlmostEqual(actual[1][0],expected[1][0])
        self.assertAlmostEqual(actual[1][1],expected[1][1])

if __name__ == '__main__':
    unittest.main()