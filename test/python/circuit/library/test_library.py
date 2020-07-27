# -*- coding: utf-8 -*-

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

"""Test library of quantum circuits."""

import unittest
from collections import defaultdict
from ddt import ddt, data, unpack
import numpy as np

from qiskit.test.base import QiskitTestCase
from qiskit import BasicAer, execute, transpile
from qiskit.circuit import (QuantumCircuit, QuantumRegister, Parameter, ParameterExpression,
                            ParameterVector)
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import (BlueprintCircuit, Permutation, QuantumVolume, XOR,
                                    InnerProduct, OR, AND, QFT, IQP,
                                    LinearPauliRotations, PolynomialPauliRotations,
                                    IntegerComparator, PiecewiseLinearPauliRotations,
                                    WeightedAdder, Diagonal, NLocal, TwoLocal, RealAmplitudes,
                                    EfficientSU2, ExcitationPreserving, PauliFeatureMap,
                                    ZFeatureMap, ZZFeatureMap, MCMT, MCMTVChain, GMS,
                                    HiddenLinearFunction, GraphState, PhaseEstimation)
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import (XGate, RXGate, RYGate, RZGate, CRXGate, CCXGate, SwapGate,
                                    RXXGate, RYYGate, HGate, ZGate, CXGate, CZGate, CHGate)
from qiskit.quantum_info import Statevector, Operator, Clifford
from qiskit.quantum_info.states import state_fidelity
from qiskit.quantum_info.random import random_unitary


class TestPermutationLibrary(QiskitTestCase):
    """Test library of permutation logic quantum circuits."""

    def test_permutation(self):
        """Test permutation circuit."""
        circuit = Permutation(num_qubits=4, pattern=[1, 0, 3, 2])
        expected = QuantumCircuit(4)
        expected.swap(0, 1)
        expected.swap(2, 3)
        expected = Operator(expected)
        simulated = Operator(circuit)
        self.assertTrue(expected.equiv(simulated))

    def test_permutation_bad(self):
        """Test that [0,..,n-1] permutation is required (no -1 for last element)."""
        self.assertRaises(CircuitError, Permutation, 4, [1, 0, -1, 2])


@ddt
class TestHiddenLinearFunctionLibrary(QiskitTestCase):
    """Test library of Hidden Linear Function circuits."""

    def assertHLFIsCorrect(self, hidden_function, hlf):
        """Assert that the HLF circuit produces the correct matrix.

        Number of qubits is equal to the number of rows (or number of columns)
        of hidden_function.
        """
        num_qubits = len(hidden_function)
        hidden_function = np.asarray(hidden_function)
        simulated = Operator(hlf)

        expected = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
        for i in range(2**num_qubits):
            i_qiskit = int(bin(i)[2:].zfill(num_qubits)[::-1], 2)
            x_vec = np.asarray(list(map(int, bin(i)[2:].zfill(num_qubits)[::-1])))
            expected[i_qiskit, i_qiskit] = 1j**(np.dot(x_vec.transpose(),
                                                       np.dot(hidden_function, x_vec)))

        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits))
        qc = Operator(qc)
        expected = qc.compose(Operator(expected)).compose(qc)
        self.assertTrue(expected.equiv(simulated))

    @data(
        [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    )
    def test_hlf(self, hidden_function):
        """Test if the HLF matrix produces the right matrix."""
        hlf = HiddenLinearFunction(hidden_function)
        self.assertHLFIsCorrect(hidden_function, hlf)

    def test_non_symmetric_raises(self):
        """Test that adjacency matrix is required to be symmetric."""
        with self.assertRaises(CircuitError):
            HiddenLinearFunction([[1, 1, 0], [1, 0, 1], [1, 1, 1]])


@ddt
class TestGraphStateLibrary(QiskitTestCase):
    """Test the graph state circuit."""

    def assertGraphStateIsCorrect(self, adjacency_matrix, graph_state):
        """Check the stabilizers of the graph state against the expected stabilizers.
        Based on https://arxiv.org/pdf/quant-ph/0307130.pdf, Eq. (6).
        """

        stabilizers = Clifford(graph_state).stabilizer.pauli.to_labels()

        expected_stabilizers = []  # keep track of all expected stabilizers
        num_vertices = len(adjacency_matrix)
        for vertex_a in range(num_vertices):
            stabilizer = [None] * num_vertices  # Paulis must be put into right place
            for vertex_b in range(num_vertices):
                if vertex_a == vertex_b:  # self-connection --> 'X'
                    stabilizer[vertex_a] = 'X'
                elif adjacency_matrix[vertex_a][vertex_b] != 0:  # vertices connected --> 'Z'
                    stabilizer[vertex_b] = 'Z'
                else:  # else --> 'I'
                    stabilizer[vertex_b] = 'I'

            # need to reverse for Qiskit's tensoring order
            expected_stabilizers.append(''.join(stabilizer)[::-1])

        self.assertListEqual(expected_stabilizers, stabilizers)

    @data(
        [[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]]
    )
    def test_graph_state(self, adjacency_matrix):
        """Verify the GraphState by checking if the circuit has the expected stabilizers."""
        graph_state = GraphState(adjacency_matrix)
        self.assertGraphStateIsCorrect(adjacency_matrix, graph_state)

    @data(
        [[1, 1, 0], [1, 0, 1], [1, 1, 1]]
    )
    def test_non_symmetric_raises(self, adjacency_matrix):
        """Test that adjacency matrix is required to be symmetric."""
        with self.assertRaises(CircuitError):
            GraphState(adjacency_matrix)


class TestIQPLibrary(QiskitTestCase):
    """Test library of IQP quantum circuits."""

    def test_iqp(self):
        """Test iqp circuit."""
        circuit = IQP(interactions=np.array([[6, 5, 1], [5, 4, 3], [1, 3, 2]]))
        expected = QuantumCircuit(3)
        expected.h([0, 1, 2])
        expected.cu1(5*np.pi/2, 0, 1)
        expected.cu1(3*np.pi/2, 1, 2)
        expected.cu1(1*np.pi/2, 0, 2)
        expected.u1(6*np.pi/8, 0)
        expected.u1(4*np.pi/8, 1)
        expected.u1(2*np.pi/8, 2)
        expected.h([0, 1, 2])
        expected = Operator(expected)
        simulated = Operator(circuit)
        self.assertTrue(expected.equiv(simulated))

    def test_iqp_bad(self):
        """Test that [0,..,n-1] permutation is required (no -1 for last element)."""
        self.assertRaises(CircuitError, IQP, [[6, 5], [2, 4]])


@ddt
class TestGMSLibrary(QiskitTestCase):
    """Test library of Global Mølmer–Sørensen gate."""

    def test_twoq_equivalence(self):
        """Test GMS on 2 qubits is same as RXX."""
        circuit = GMS(num_qubits=2, theta=[[0, np.pi/3], [0, 0]])
        expected = RXXGate(np.pi/3)
        expected = Operator(expected)
        simulated = Operator(circuit)
        self.assertTrue(expected.equiv(simulated))


@ddt
class TestQuantumVolumeLibrary(QiskitTestCase):
    """Test library of quantum volume quantum circuits."""

    def test_qv(self):
        """Test qv circuit."""
        circuit = QuantumVolume(2, 2, seed=2, classical_permutation=False)
        expected = QuantumCircuit(2)
        expected.swap(0, 1)
        expected.append(random_unitary(4, seed=837), [0, 1])
        expected.append(random_unitary(4, seed=262), [0, 1])
        expected = Operator(expected)
        simulated = Operator(circuit)
        self.assertTrue(expected.equiv(simulated))


@ddt
class TestMCMT(QiskitTestCase):
    """Test the multi-controlled multi-target circuit."""

    @data(MCMT, MCMTVChain)
    def test_mcmt_as_normal_control(self, mcmt_class):
        """Test that the MCMT can act as normal control gate."""
        qc = QuantumCircuit(2)
        mcmt = mcmt_class(gate=CHGate(), num_ctrl_qubits=1, num_target_qubits=1)
        qc = qc.compose(mcmt, [0, 1])

        ref = QuantumCircuit(2)
        ref.ch(0, 1)

        self.assertEqual(qc, ref)

    def test_missing_qubits(self):
        """Test that an error is raised if qubits are missing."""
        with self.subTest(msg='no control qubits'):
            with self.assertRaises(AttributeError):
                _ = MCMT(XGate(), num_ctrl_qubits=0, num_target_qubits=1)

        with self.subTest(msg='no target qubits'):
            with self.assertRaises(AttributeError):
                _ = MCMT(ZGate(), num_ctrl_qubits=4, num_target_qubits=0)

    def test_different_gate_types(self):
        """Test the different supported input types for the target gate."""
        x_circ = QuantumCircuit(1)
        x_circ.x(0)
        for input_gate in [x_circ, QuantumCircuit.cx, QuantumCircuit.x, 'cx', 'x', CXGate()]:
            with self.subTest(input_gate=input_gate):
                mcmt = MCMT(input_gate, 2, 2)
                if isinstance(input_gate, QuantumCircuit):
                    self.assertEqual(mcmt.gate.definition[0][0], XGate())
                    self.assertEqual(len(mcmt.gate.definition), 1)
                else:
                    self.assertEqual(mcmt.gate, XGate())

    def test_mcmt_v_chain_ancilla_test(self):
        """Test too few and too many ancillas for the MCMT V-chain mode."""
        with self.subTest(msg='insufficient number of ancillas on gate'):
            qc = QuantumCircuit(5)
            mcmt = MCMTVChain(ZGate(), 3, 1)
            with self.assertRaises(QiskitError):
                qc.append(mcmt, [0, 1, 2, 3, 4])

        with self.subTest(msg='insufficient number of ancillas on method'):
            qc = QuantumCircuit(5)
            mcmt = MCMTVChain(ZGate(), 3, 1)
            with self.assertRaises(QiskitError):
                qc.append(mcmt, [0, 1, 2, 3, 4], [])

        with self.subTest(msg='too many ancillas works on method'):
            qc = QuantumCircuit(8)
            qc.mcmt(CZGate(), [0, 1, 2], 3, [4, 5, 6, 7])

    @data(
        [CZGate(), 1, 1], [CHGate(), 1, 1],
        [CZGate(), 3, 3], [CHGate(), 3, 3],
        [CZGate(), 1, 5], [CHGate(), 1, 5],
        [CZGate(), 5, 1], [CHGate(), 5, 1],
    )
    @unpack
    def test_mcmt_v_chain_simulation(self, cgate, num_controls, num_targets):
        """Test the MCMT V-chain implementation test on a simulation."""
        controls = QuantumRegister(num_controls)
        targets = QuantumRegister(num_targets)

        subsets = [tuple(range(i)) for i in range(num_controls + 1)]
        for subset in subsets:
            qc = QuantumCircuit(targets, controls)
            # Initialize all targets to 1, just to be sure that
            # the generic gate has some effect (f.e. Z gate has no effect
            # on a 0 state)
            qc.x(targets)

            num_ancillas = max(0, num_controls - 1)

            if num_ancillas > 0:
                ancillas = QuantumRegister(num_ancillas)
                qc.add_register(ancillas)
                qubits = controls[:] + targets[:] + ancillas[:]
            else:
                qubits = controls[:] + targets[:]

            for i in subset:
                qc.x(controls[i])

            mcmt = MCMTVChain(cgate, num_controls, num_targets)
            qc.compose(mcmt, qubits, inplace=True)

            for i in subset:
                qc.x(controls[i])

            vec = Statevector.from_label('0' * qc.num_qubits).evolve(qc)

            # target register is initially |11...1>, with length equal to 2**(n_targets)
            vec_exp = np.array([0] * (2**(num_targets) - 1) + [1])

            if isinstance(cgate, CZGate):
                # Z gate flips the last qubit only if it's applied an odd number of times
                if len(subset) == num_controls and (num_controls % 2) == 1:
                    vec_exp[-1] = -1
            elif isinstance(cgate, CHGate):
                # if all the control qubits have been activated,
                # we repeatedly apply the kronecker product of the Hadamard
                # with itself and then multiply the results for the original
                # state of the target qubits
                if len(subset) == num_controls:
                    h_i = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
                    h_tot = np.array([1])
                    for _ in range(num_targets):
                        h_tot = np.kron(h_tot, h_i)
                    vec_exp = np.dot(h_tot, vec_exp)
            else:
                raise ValueError('Test not implement for gate: {}'.format(cgate))

            # append the remaining part of the state
            vec_exp = np.concatenate(
                (vec_exp,
                 [0] * (2**(num_controls + num_ancillas + num_targets) - vec_exp.size))
            )
            f_i = state_fidelity(vec, vec_exp)
            self.assertAlmostEqual(f_i, 1)


@ddt
class TestDiagonalGate(QiskitTestCase):
    """Test diagonal circuit."""

    @data(
        [0, 0],
        [0, 0.8],
        [0, 0, 1, 1],
        [0, 1, 0.5, 1],
        (2 * np.pi * np.random.rand(2 ** 3)),
        (2 * np.pi * np.random.rand(2 ** 4)),
        (2 * np.pi * np.random.rand(2 ** 5))
    )
    def test_diag_gate(self, phases):
        """Test correctness of diagonal decomposition."""
        diag = [np.exp(1j * ph) for ph in phases]
        qc = Diagonal(diag)
        simulated_diag = Statevector(Operator(qc).data.diagonal())
        ref_diag = Statevector(diag)

        self.assertTrue(simulated_diag.equiv(ref_diag))


if __name__ == '__main__':
    unittest.main()
