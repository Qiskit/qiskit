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

"""Test phase estimation"""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import ddt, data
import numpy as np
from qiskit.algorithms.phase_estimators import PhaseEstimation, HamiltonianPhaseEstimation
from qiskit.opflow.evolutions import PauliTrotterEvolution, MatrixEvolution
import qiskit
from qiskit.opflow import (H, X, Y, Z, I, StateFn)


@ddt
class TestHamiltonianPhaseEstimation(QiskitAlgorithmsTestCase):
    """Tests for obtaining eigenvalues from phase estimation"""

    def hamiltonian_pe(self, hamiltonian, state_preparation=None, num_evaluation_qubits=6,
                       backend=None,
                       evolution=None,
                       bound=None):
        """Run HamiltonianPhaseEstimation and return result with all  phases."""
        if backend is None:
            backend = qiskit.BasicAer.get_backend('statevector_simulator')
        quantum_instance = qiskit.utils.QuantumInstance(backend=backend, shots=10000)
        phase_est = HamiltonianPhaseEstimation(
            num_evaluation_qubits=num_evaluation_qubits,
            quantum_instance=quantum_instance)
        result = phase_est.estimate(
            hamiltonian=hamiltonian,
            state_preparation=state_preparation, evolution=evolution,
            bound=bound)
        return result

    @data(MatrixEvolution(), PauliTrotterEvolution('suzuki', 4))
    def test_pauli_sum_1(self, evolution):
        """Two eigenvalues from Pauli sum with X, Z"""
        hamiltonian = 0.5 * X + Z
        state_preparation = StateFn(H.to_circuit())

        result = self.hamiltonian_pe(hamiltonian, state_preparation, evolution=evolution)
        phase_dict = result.filter_phases(0.162, as_float=True)
        phases = list(phase_dict.keys())
        phases.sort()

        self.assertAlmostEqual(phases[0], -1.125, delta=0.001)
        self.assertAlmostEqual(phases[1], 1.125, delta=0.001)

    @data(MatrixEvolution(), PauliTrotterEvolution('suzuki', 3))
    def test_pauli_sum_2(self, evolution):
        """Two eigenvalues from Pauli sum with X, Y, Z"""
        hamiltonian = 0.5 * X + Y + Z
        state_preparation = None

        result = self.hamiltonian_pe(hamiltonian, state_preparation, evolution=evolution)
        phase_dict = result.filter_phases(0.1, as_float=True)
        phases = list(phase_dict.keys())
        phases.sort()

        self.assertAlmostEqual(phases[0], -1.484, delta=0.001)
        self.assertAlmostEqual(phases[1], 1.484, delta=0.001)

    def test_single_pauli_op(self):
        """Two eigenvalues from Pauli sum with X, Y, Z"""
        hamiltonian = Z
        state_preparation = None

        result = self.hamiltonian_pe(hamiltonian, state_preparation, evolution=None)
        eigv = result.most_likely_eigenvalue
        with self.subTest('First eigenvalue'):
            self.assertAlmostEqual(eigv, 1.0, delta=0.001)

        state_preparation = StateFn(X.to_circuit())

        result = self.hamiltonian_pe(hamiltonian, state_preparation, bound=1.05)
        eigv = result.most_likely_eigenvalue
        with self.subTest('Second eigenvalue'):
            self.assertAlmostEqual(eigv, -0.98, delta=0.01)

    def test_H2_hamiltonian(self):
        """Test H2 hamiltonian"""
        hamiltonian = (-1.0523732457728587 * (I ^ I)) + (0.3979374248431802 * (I ^ Z)) \
            + (-0.3979374248431802 * (Z ^ I)) + (-0.011280104256235324 * (Z ^ Z)) \
            + (0.18093119978423147 * (X ^ X))
        state_preparation = StateFn((I ^ H).to_circuit())
        evo = PauliTrotterEvolution(trotter_mode='suzuki', reps=4)
        result = self.hamiltonian_pe(hamiltonian, state_preparation, evolution=evo)
        with self.subTest('Most likely eigenvalues'):
            self.assertAlmostEqual(result.most_likely_eigenvalue, -1.855, delta=.001)
        with self.subTest('Most likely phase'):
            self.assertAlmostEqual(result.most_likely_phase, 0.5937, delta=.001)
        with self.subTest('All eigenvalues'):
            phase_dict = result.filter_phases(0.1)
            phases = list(phase_dict.keys())
            self.assertAlmostEqual(phases[0], -0.8979, delta=0.001)
            self.assertAlmostEqual(phases[1], -1.8551, delta=0.001)
            self.assertAlmostEqual(phases[2], -1.2376, delta=0.001)

    def test_matrix_evolution(self):
        """1Q Hamiltonian with MatrixEvolution"""
        hamiltonian = ((0.5 * X) + (0.6 * Y) + (0.7 * I))
        state_preparation = None
        result = self.hamiltonian_pe(hamiltonian, state_preparation, evolution=MatrixEvolution())
        phase_dict = result.filter_phases(0.2, as_float=True)
        phases = list(phase_dict.keys())
        self.assertAlmostEqual(phases[0], 1.490, delta=0.001)
        self.assertAlmostEqual(phases[1], -0.090, delta=0.001)

    def _setup_from_bound(self, evolution, op_class):
        hamiltonian = 0.5 * X + Y + Z
        state_preparation = None
        bound = 1.2 * sum([abs(hamiltonian.coeff * coeff) for coeff in hamiltonian.coeffs])
        if op_class == 'MatrixOp':
            hamiltonian = hamiltonian.to_matrix_op()
        backend = qiskit.BasicAer.get_backend('statevector_simulator')
        qi = qiskit.utils.QuantumInstance(backend=backend, shots=10000)
        phase_est = HamiltonianPhaseEstimation(num_evaluation_qubits=6,
                                               quantum_instance=qi)
        result = phase_est.estimate(hamiltonian=hamiltonian,
                                    bound=bound,
                                    evolution=evolution,
                                    state_preparation=state_preparation)
        return result

    def test_from_bound(self):
        """HamiltonianPhaseEstimation with bound"""
        for op_class in ('SummedOp', 'MatrixOp'):
            result = self._setup_from_bound(MatrixEvolution(), op_class)
            cutoff = 0.01
            phases = result.filter_phases(cutoff)
            with self.subTest(f'test phases has the correct length: {op_class}'):
                self.assertEqual(len(phases), 2)
                with self.subTest(f'test scaled phases are correct: {op_class}'):
                    self.assertEqual(list(phases.keys()), [1.5, -1.5])
                    phases = result.filter_phases(cutoff, scaled=False)
                with self.subTest(f'test unscaled phases are correct: {op_class}'):
                    self.assertEqual(list(phases.keys()), [0.25, 0.75])

    def test_trotter_from_bound(self):
        """HamiltonianPhaseEstimation with bound and Trotterization"""
        result = self._setup_from_bound(PauliTrotterEvolution(trotter_mode='suzuki', reps=3),
                                        op_class='SummedOp')
        phase_dict = result.filter_phases(0.1)
        phases = list(phase_dict.keys())
        with self.subTest('test phases has the correct length'):
            self.assertEqual(len(phases), 2)
        with self.subTest('test phases has correct values'):
            self.assertAlmostEqual(phases[0], 1.5, delta=0.001)
            self.assertAlmostEqual(phases[1], -1.5, delta=0.001)


@ddt
class TestPhaseEstimation(QiskitAlgorithmsTestCase):
    """Evolution tests."""

    # pylint: disable=invalid-name
    def one_phase(self, unitary_circuit, state_preparation=None, n_eval_qubits=6,
                  backend=None):
        """Run phase estimation with operator, eigenvalue pair `unitary_circuit`,
        `state_preparation`. Return the bit string with the largest amplitude.
        """
        if backend is None:
            backend = qiskit.BasicAer.get_backend('qasm_simulator')
        qi = qiskit.utils.QuantumInstance(backend=backend, shots=10000)
        p_est = PhaseEstimation(num_evaluation_qubits=n_eval_qubits,
                                quantum_instance=qi)
        result = p_est.estimate(unitary=unitary_circuit,
                                state_preparation=state_preparation)
        phase = result.most_likely_phase
        return phase

    @data('qasm_simulator', 'statevector_simulator')
    def test_qpe_Z0(self, backend_type):
        """eigenproblem Z, |0>"""
        backend = qiskit.BasicAer.get_backend(backend_type)

        unitary_circuit = Z.to_circuit()
        state_preparation = None  # prepare |0>
        phase = self.one_phase(unitary_circuit, state_preparation, backend=backend)
        self.assertEqual(phase, 0.0)

    @data('qasm_simulator', 'statevector_simulator')
    def test_qpe_Z1(self, backend_type):
        """eigenproblem Z, |1>"""
        backend = qiskit.BasicAer.get_backend(backend_type)

        unitary_circuit = Z.to_circuit()
        state_preparation = X.to_circuit()  # prepare |1>
        phase = self.one_phase(unitary_circuit, state_preparation, backend=backend)
        self.assertEqual(phase, 0.5)

    @data('plus', 'minus')
    def test_qpe_Xplus(self, state):
        """eigenproblem X, |+>"""
        unitary_circuit = X.to_circuit()
        if state == 'minus':  # prepare |->
            state_preparation = X.to_circuit()
            state_preparation.h(0)
        else:  # prepare |+>
            state_preparation = H.to_circuit()

        phase = self.one_phase(unitary_circuit, state_preparation)
        if state == 'minus':
            self.assertEqual(phase, 0.5)
        else:
            self.assertEqual(phase, 0.0)

    def phase_estimation(self, unitary_circuit, state_preparation=None, num_evaluation_qubits=6,
                         backend=None):
        """Run phase estimation with operator, eigenvalue pair `unitary_circuit`,
        `state_preparation`. Return all results
        """
        if backend is None:
            backend = qiskit.BasicAer.get_backend('statevector_simulator')
        qi = qiskit.utils.QuantumInstance(backend=backend, shots=10000)
        phase_est = PhaseEstimation(num_evaluation_qubits=num_evaluation_qubits,
                                    quantum_instance=qi)
        result = phase_est.estimate(unitary=unitary_circuit,
                                    state_preparation=state_preparation)
        return result

    def test_qpe_Zplus(self):
        """superposition eigenproblem Z, |+>"""
        unitary_circuit = Z.to_circuit()
        state_preparation = H.to_circuit()  # prepare |+>
        result = self.phase_estimation(
            unitary_circuit, state_preparation,
            backend=qiskit.BasicAer.get_backend('statevector_simulator'))

        phases = result.filter_phases(1e-15, as_float=True)
        with self.subTest('test phases has correct values'):
            self.assertEqual(list(phases.keys()), [0.0, 0.5])

        with self.subTest('test phases has correct probabilities'):
            np.testing.assert_allclose(list(phases.values()), [0.5, 0.5])

        with self.subTest('test bitstring representation'):
            phases = result.filter_phases(1e-15, as_float=False)
            self.assertEqual(list(phases.keys()), ['000000', '100000'])


if __name__ == '__main__':
    unittest.main()
