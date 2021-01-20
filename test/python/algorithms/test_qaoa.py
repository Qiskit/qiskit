# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test QAOA """

import unittest

import math
import networkx as nx
import numpy as np
from ddt import ddt, idata, unpack

from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA, NELDER_MEAD

from qiskit.opflow import I, X, WeightedPauliOperator

from qiskit import BasicAer, QuantumCircuit

from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli
from qiskit.utils import QuantumInstance, aqua_globals
from test.python.algorithms import QiskitAlgorithmsTestCase

W1 = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])
P1 = 1
M1 = (I ^ I ^ I ^ X) + (I ^ I ^ X ^ I) + (I ^ X ^ I ^ I) + (X ^ I ^ I ^ I)
S1 = {'0101', '1010'}


W2 = np.array([
    [0., 8., -9., 0.],
    [8., 0., 7., 9.],
    [-9., 7., 0., -8.],
    [0., 9., -8., 0.],
])
P2 = 1
M2 = None
S2 = {'1011', '0100'}

CUSTOM_SUPERPOSITION = [1/math.sqrt(15)] * 15 + [0]


@ddt
class TestQAOA(QiskitAlgorithmsTestCase):
    """Test QAOA with MaxCut."""
    @idata([
        [W1, P1, M1, S1, False],
        [W2, P2, M2, S2, False],
        [W1, P1, M1, S1, True],
        [W2, P2, M2, S2, True],
    ])
    @unpack
    def test_qaoa(self, w, prob, m, solutions, convert_to_matrix_op):
        """ QAOA test """
        seed = 0
        aqua_globals.random_seed = seed
        self.log.debug('Testing %s-step QAOA with MaxCut on graph\n%s', prob, w)

        backend = BasicAer.get_backend('statevector_simulator')
        optimizer = COBYLA()
        qubit_op, offset = self.get_operator(w)
        qubit_op = qubit_op.to_opflow()
        if convert_to_matrix_op:
            qubit_op = qubit_op.to_matrix_op()

        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
        qaoa = QAOA(optimizer, prob, mixer=m, quantum_instance=quantum_instance)

        # result = qaoa.run(quantum_instance)
        result = qaoa.compute_minimum_eigenvalue(qubit_op)
        x = self.sample_most_likely(result.eigenstate)
        # graph_solution = max_cut.get_graph_solution(x)
        graph_solution = 1 - x
        # self.log.debug('energy:             %s', result.eigenvalue.real)
        # self.log.debug('time:               %s', result.optimizer_time)
        # self.log.debug('maxcut objective:   %s', result.eigenvalue.real + offset)
        # self.log.debug('solution:           %s', graph_solution)
        # self.log.debug('solution objective: %s', max_cut.max_cut_value(x, w))
        self.assertIn(''.join([str(int(i)) for i in graph_solution]), solutions)

    @idata([
        [W1, P1, S1, False],
        [W2, P2, S2, False],
        [W1, P1, S1, True],
        [W2, P2, S2, True],
    ])
    @unpack
    def test_qaoa_qc_mixer(self, w, prob, solutions, convert_to_matrix_op):
        """ QAOA test with a mixer as a parameterized circuit"""
        seed = 0
        aqua_globals.random_seed = seed
        self.log.debug('Testing %s-step QAOA with MaxCut on graph with '
                       'a mixer as a parameterized circuit\n%s', prob, w)

        backend = BasicAer.get_backend('statevector_simulator')
        optimizer = COBYLA()
        qubit_op, _ = self.get_operator(w)
        qubit_op = qubit_op.to_opflow()
        if convert_to_matrix_op:
            qubit_op = qubit_op.to_matrix_op()

        num_qubits = qubit_op.num_qubits
        mixer = QuantumCircuit(num_qubits)
        theta = Parameter('θ')
        mixer.rx(theta, range(num_qubits))

        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
        qaoa = QAOA(optimizer, prob, mixer=mixer, quantum_instance=quantum_instance)

        result = qaoa.compute_minimum_eigenvalue(qubit_op)
        x = self.sample_most_likely(result.eigenstate)
        graph_solution = 1 - x
        self.assertIn(''.join([str(int(i)) for i in graph_solution]), solutions)

    def test_qaoa_qc_mixer_many_parameters(self):
        """ QAOA test with a mixer as a parameterized circuit with the num of parameters > 1. """
        seed = 0
        aqua_globals.random_seed = seed

        optimizer = COBYLA()
        qubit_op, _ = self.get_operator(W1)
        qubit_op = qubit_op.to_opflow()

        num_qubits = qubit_op.num_qubits
        mixer = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            theta = Parameter('θ' + str(i))
            mixer.rx(theta, range(num_qubits))

        backend = BasicAer.get_backend('statevector_simulator')
        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
        qaoa = QAOA(optimizer=optimizer, p=2, mixer=mixer, quantum_instance=quantum_instance)
        result = qaoa.compute_minimum_eigenvalue(qubit_op)
        x = self.sample_most_likely(result.eigenstate)
        print(x)
        graph_solution = 1 - x
        self.assertIn(''.join([str(int(i)) for i in graph_solution]), S1)

    def test_qaoa_qc_mixer_no_parameters(self):
        """ QAOA test with a mixer as a parameterized circuit with zero parameters. """
        seed = 0
        aqua_globals.random_seed = seed

        qubit_op, _ = self.get_operator(W1)
        qubit_op = qubit_op.to_opflow()

        num_qubits = qubit_op.num_qubits
        mixer = QuantumCircuit(num_qubits)
        # just arbitrary circuit
        mixer.rx(np.pi/2, range(num_qubits))

        backend = BasicAer.get_backend('statevector_simulator')
        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
        qaoa = QAOA(optimizer=COBYLA(), p=1, mixer=mixer, quantum_instance=quantum_instance)
        result = qaoa.compute_minimum_eigenvalue(qubit_op)
        # we just assert that we get a result, it is not meaningful.
        self.assertIsNotNone(result.eigenstate)

    def test_change_operator_size(self):
        """ QAOA change operator size test """

        aqua_globals.random_seed = 0
        qubit_op, _ = self.get_operator(
            np.array([
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0]
            ]))
        quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                           seed_simulator=0,
                                           seed_transpiler=0)

        qaoa = QAOA(COBYLA(), 1, quantum_instance=quantum_instance)
        result = qaoa.compute_minimum_eigenvalue(qubit_op.to_opflow())
        x = self.sample_most_likely(result.eigenstate)
        graph_solution = 1 - x
        with self.subTest(msg='QAOA 4x4'):
            self.assertIn(''.join([str(int(i)) for i in graph_solution]), {'0101', '1010'})

        try:
            qubit_op, _ = self.get_operator(
                np.array([
                    [0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0],
                ]))
            qaoa.operator = qubit_op.to_opflow()
        except Exception as ex:  # pylint: disable=broad-except
            self.fail("Failed to change operator. Error: '{}'".format(str(ex)))
            return

        result = qaoa.compute_minimum_eigenvalue(qubit_op.to_opflow())
        x = self.sample_most_likely(result.eigenstate)
        graph_solution = 1 - x
        with self.subTest(msg='QAOA 6x6'):
            self.assertIn(''.join([str(int(i)) for i in graph_solution]), {'010101', '101010'})

    @idata([
        [W2, S2, None],
        [W2, S2, [0.0, 0.0]],
        [W2, S2, [1.0, 0.8]]
    ])
    @unpack
    def test_qaoa_initial_point(self, w, solutions, init_pt):
        """ Check first parameter value used is initial point as expected """
        aqua_globals.random_seed = 10598
        optimizer = COBYLA()
        qubit_op, _ = self.get_operator(w)

        first_pt = []

        def cb_callback(eval_count, parameters, mean, std):
            nonlocal first_pt
            if eval_count == 1:
                first_pt = list(parameters)

        quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        qaoa = QAOA(optimizer, initial_point=init_pt, callback=cb_callback,
                    quantum_instance=quantum_instance)

        result = qaoa.compute_minimum_eigenvalue(qubit_op.to_opflow())
        x = self.sample_most_likely(result.eigenstate)
        graph_solution = 1 - x

        with self.subTest('Initial Point'):
            # If None the preferred random initial point of QAOA variational form
            if init_pt is None:
                np.testing.assert_almost_equal([-0.2398, 0.3378], first_pt, decimal=4)
            else:
                self.assertListEqual(init_pt, first_pt)

        with self.subTest('Solution'):
            self.assertIn(''.join([str(int(i)) for i in graph_solution]), solutions)

    @idata([
        [W2, None],
        [W2, [1.0] + 15*[0.0]],
        [W2, CUSTOM_SUPERPOSITION]
    ])
    @unpack
    def test_qaoa_initial_state(self, w, init_state):
        """ QAOA initial state test """

        optimizer = COBYLA()
        qubit_op, _ = self.get_operator(w)

        init_pt = [0.0, 0.0]  # Avoid generating random initial point

        if init_state is None:
            initial_state = None
        else:
            initial_state = Custom(num_qubits=4, state_vector=init_state)

        quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'))
        qaoa_zero_init_state = QAOA(qubit_op, optimizer, initial_point=init_pt,
                                    initial_state=Zero(qubit_op.num_qubits),
                                    quantum_instance=quantum_instance)
        qaoa = QAOA(qubit_op, optimizer, initial_point=init_pt,
                    initial_state=initial_state, quantum_instance=quantum_instance)

        zero_circuits = qaoa_zero_init_state.construct_circuit(init_pt)
        custom_circuits = qaoa.construct_circuit(init_pt)

        self.assertEqual(len(zero_circuits), len(custom_circuits))

        backend = BasicAer.get_backend('statevector_simulator')
        for zero_circ, custom_circ in zip(zero_circuits, custom_circuits):

            z_length = len(zero_circ.data)
            c_length = len(custom_circ.data)

            self.assertGreaterEqual(c_length, z_length)
            self.assertTrue(zero_circ.data == custom_circ.data[-z_length:])

            custom_init_qc = custom_circ.copy()
            custom_init_qc.data = custom_init_qc.data[0:c_length-z_length]

            if initial_state is None:
                original_init_qc = QuantumCircuit(qubit_op.num_qubits)
                original_init_qc.h(range(qubit_op.num_qubits))
            else:
                original_init_qc = initial_state.construct_circuit()

            job_init_state = execute(original_init_qc, backend)
            job_qaoa_init_state = execute(custom_init_qc, backend)

            statevector_original = job_init_state.result().get_statevector(original_init_qc)
            statevector_custom = job_qaoa_init_state.result().get_statevector(custom_init_qc)

            self.assertEqual(statevector_original.tolist(), statevector_custom.tolist())

    def test_qaoa_random_initial_point(self):
        """ QAOA random initial point """
        aqua_globals.random_seed = 10598
        w = nx.adjacency_matrix(
            nx.fast_gnp_random_graph(5, 0.5, seed=aqua_globals.random_seed)).toarray()
        qubit_op, _ = self.get_operator(w)
        quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed,
                                           shots=4096)

        qaoa = QAOA(NELDER_MEAD(disp=True), 1, quantum_instance=quantum_instance)

        _ = qaoa.compute_minimum_eigenvalue(qubit_op.to_opflow())

        np.testing.assert_almost_equal([-0.8792, 0.3948], qaoa.optimal_params, decimal=4)

    def get_operator(self, weight_matrix):
        """Generate Hamiltonian for the max-cut problem of a graph.

        Args:
            weight_matrix (numpy.ndarray) : adjacency matrix.

        Returns:
            WeightedPauliOperator: operator for the Hamiltonian
            float: a constant shift for the obj function.

        """
        num_nodes = weight_matrix.shape[0]
        pauli_list = []
        shift = 0
        for i in range(num_nodes):
            for j in range(i):
                if weight_matrix[i, j] != 0:
                    x_p = np.zeros(num_nodes, dtype=np.bool)
                    z_p = np.zeros(num_nodes, dtype=np.bool)
                    z_p[i] = True
                    z_p[j] = True
                    pauli_list.append([0.5 * weight_matrix[i, j], Pauli((z_p, x_p))])
                    shift -= 0.5 * weight_matrix[i, j]
        return WeightedPauliOperator(paulis=pauli_list), shift

    def sample_most_likely(self, state_vector):
        """Compute the most likely binary string from state vector.
        Args:
            state_vector (numpy.ndarray or dict): state vector or counts.

        Returns:
            numpy.ndarray: binary string as numpy.ndarray of ints.
        """
        n = int(np.log2(state_vector.shape[0]))
        k = np.argmax(np.abs(state_vector))
        x = np.zeros(n)
        for i in range(n):
            x[i] = k % 2
            k >>= 1
        return x


if __name__ == '__main__':
    unittest.main()
