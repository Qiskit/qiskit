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
import numpy as np
from qiskit.algorithms.phase_estimators import PhaseEstimation, HamiltonianPhaseEstimation
from qiskit.opflow.evolutions import PauliTrotterEvolution, MatrixEvolution
import qiskit
from qiskit.opflow import (H, X, Y, Z, I)


class TestHamiltonianPhaseEstimation(QiskitAlgorithmsTestCase):
    """Tests for obtaining eigenvalues from phase estimation"""

    def setUp(self):
        super().setUp()
        self.hamiltonian_1 = ((0.5 * X) + Y + Z).to_pauli_op()

    def hamiltonian_pe(self, hamiltonian, state_preparation=None, num_evaluation_qubits=6,
                       backend=qiskit.BasicAer.get_backend('statevector_simulator'),
                       evolution=MatrixEvolution()):
        """Run HamiltonianPhaseEstimation and return result with all  phases."""
        quantum_instance = qiskit.utils.QuantumInstance(backend=backend, shots=10000)
        phase_est = HamiltonianPhaseEstimation(
            num_evaluation_qubits=num_evaluation_qubits,
            quantum_instance=quantum_instance)
        result = phase_est.estimate(
            hamiltonian=hamiltonian,
            state_preparation=state_preparation, evolution=evolution)
        return result

    # pylint: disable=invalid-name
    def test_pauli_sum_1(self):
        """Two eigenvalues from Pauli sum with X, Z"""
        a1 = 0.5
        a2 = 1.0
        hamiltonian = ((a1 * X) + (a2 * Z)).to_pauli_op()
        state_preparation = H.to_circuit()
        result = self.hamiltonian_pe(hamiltonian, state_preparation)
        phase_dict = result.filter_phases(0.162, as_float=True)
        phases = list(phase_dict.keys())
        self.assertAlmostEqual(phases[0], 1.125, delta=0.001)
        self.assertAlmostEqual(phases[1], -1.125, delta=0.001)
        evo = PauliTrotterEvolution(trotter_mode='suzuki', reps=4)
        result = self.hamiltonian_pe(hamiltonian, state_preparation, evolution=evo)
        phase_dict = result.filter_phases(0.162, as_float=True)
        phases = list(phase_dict.keys())
        phases.sort()
        with self.subTest('Use PauliTrotterEvolution, first phase'):
            self.assertAlmostEqual(phases[0], -1.125, delta=0.001)
        with self.subTest('Use PauliTrotterEvolution, second phase'):
            self.assertAlmostEqual(phases[1], 1.125, delta=0.001)

    def test_pauli_sum_2(self):
        """Two eigenvalues from Pauli sum with X, Y, Z"""
        hamiltonian = self.hamiltonian_1
        state_preparation = None
        result = self.hamiltonian_pe(hamiltonian, state_preparation)
        phase_dict = result.filter_phases(0.1, as_float=True)
        phases = list(phase_dict.keys())
        self.assertAlmostEqual(phases[0], 1.484, delta=0.001)
        self.assertAlmostEqual(phases[1], -1.484, delta=0.001)
        evo = PauliTrotterEvolution(trotter_mode='suzuki', reps=3)
        result = self.hamiltonian_pe(hamiltonian, state_preparation, evolution=evo)
        phase_dict = result.filter_phases(0.1, as_float=True)
        phases = list(phase_dict.keys())
        with self.subTest('Use PauliTrotterEvolution, first phase'):
            self.assertAlmostEqual(phases[0], 1.484, delta=0.001)
        with self.subTest('Use PauliTrotterEvolution, second phase'):
            self.assertAlmostEqual(phases[1], -1.484, delta=0.001)

    def test_H2_hamiltonian(self):
        """Test H2 hamiltonian"""
        hamiltonian = (-1.0523732457728587 * (I ^ I)) + (0.3979374248431802 * (I ^ Z)) \
            + (-0.3979374248431802 * (Z ^ I)) + (-0.011280104256235324 * (Z ^ Z)) \
            + (0.18093119978423147 * (X ^ X))
        state_preparation = (I^H).to_circuit()
        evo = PauliTrotterEvolution(trotter_mode='suzuki', reps=4)
        result = self.hamiltonian_pe(hamiltonian.to_pauli_op(), state_preparation, evolution=evo)
        with self.subTest('Most likely eigenvalues'):
            self.assertAlmostEqual(result.most_likely_eigenvalue, -1.855, delta=.001)
        with self.subTest('All eigenvalues'):
            phase_dict = result.filter_phases(0.1)
            phases = list(phase_dict.keys())
            self.assertAlmostEqual(phases[0], -0.8979, delta=0.001)
            self.assertAlmostEqual(phases[1], -1.8551, delta=0.001)
            self.assertAlmostEqual(phases[2], -1.2376, delta=0.001)

    def test_matrix_evolution(self):
        """1Q Hamiltonian with MatrixEvolution"""
        hamiltonian = ((0.5 * X) + (0.6 * Y) + (0.7 * I)).to_pauli_op()
        state_preparation = None
        result = self.hamiltonian_pe(hamiltonian, state_preparation, evolution=MatrixEvolution())
        phase_dict = result.filter_phases(0.2, as_float=True)
        phases = list(phase_dict.keys())
        self.assertAlmostEqual(phases[0], 1.490, delta=0.001)
        self.assertAlmostEqual(phases[1], -0.090, delta=0.001)

    def _setup_from_bound(self, evolution):
        hamiltonian = self.hamiltonian_1
        state_preparation = None
        bound = 1.2 * sum([abs(hamiltonian.coeff * pauli.coeff) for pauli in hamiltonian])
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
        result = self._setup_from_bound(MatrixEvolution())
        cutoff = 0.01
        phases = result.filter_phases(cutoff)
        with self.subTest('test phases has the correct length'):
            self.assertEqual(len(phases), 2)
        with self.subTest('test scaled phases are correct'):
            self.assertEqual(list(phases.keys()), [1.5, -1.5])
        phases = result.filter_phases(cutoff, scaled=False)
        with self.subTest('test unscaled phases are correct'):
            self.assertEqual(list(phases.keys()), [0.25, 0.75])
        with self.subTest('test most_likely_phase method'):
            self.assertEqual(result.most_likely_eigenvalue, 1.5)
            self.assertEqual(result.most_likely_phase, 0.25)

    def test_trotter_from_bound(self):
        """HamiltonianPhaseEstimation with bound and Trotterization"""
        result = self._setup_from_bound(PauliTrotterEvolution(trotter_mode='suzuki', reps=3))
        phase_dict = result.filter_phases(0.1)
        phases = list(phase_dict.keys())
        with self.subTest('test phases has the correct length'):
            self.assertEqual(len(phases), 2)
        with self.subTest('test phases has correct values'):
            self.assertAlmostEqual(phases[0], 1.5, delta=0.001)
            self.assertAlmostEqual(phases[1], -1.5, delta=0.001)


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

    def test_qpe_Z0(self):
        """eigenproblem Z, |0>"""

        unitary_circuit = Z.to_circuit()
        state_preparation = None  # prepare |0>
        phase = self.one_phase(unitary_circuit, state_preparation)
        self.assertEqual(phase, 0.0)

    def test_qpe_Z0_statevector(self):
        """eigenproblem Z, |0>, statevector simulator"""

        unitary_circuit = Z.to_circuit()
        state_preparation = None  # prepare |0>
        phase = self.one_phase(unitary_circuit, state_preparation,
                               backend=qiskit.BasicAer.get_backend('statevector_simulator'))
        self.assertEqual(phase, 0.0)

    def test_qpe_Z1(self):
        """eigenproblem Z, |1>"""
        unitary_circuit = Z.to_circuit()
        state_preparation = X.to_circuit()  # prepare |1>
        phase = self.one_phase(unitary_circuit, state_preparation)
        self.assertEqual(phase, 0.5)

    def test_qpe_Z1_estimate(self):
        """eigenproblem Z, |1>, estimate interface"""
        unitary_circuit = Z.to_circuit()
        state_preparation = X.to_circuit()  # prepare |1>
        backend = qiskit.BasicAer.get_backend('statevector_simulator')
        qi = qiskit.utils.QuantumInstance(backend=backend)
        num_evaluation_qubits = 6
        pe = PhaseEstimation(num_evaluation_qubits, quantum_instance=qi)
        result = pe.estimate(unitary=unitary_circuit, state_preparation=state_preparation)
        phase = result.most_likely_phase
        self.assertEqual(phase, 0.5)

    def test_qpe_Xplus(self):
        """eigenproblem X, |+>"""
        unitary_circuit = X.to_circuit()
        state_preparation = H.to_circuit()  # prepare |+>
        phase = self.one_phase(unitary_circuit, state_preparation)
        self.assertEqual(phase, 0.0)

    def test_qpe_Xminus(self):
        """eigenproblem X, |->"""
        unitary_circuit = X.to_circuit()
        state_preparation = X.to_circuit()
        state_preparation.append(H.to_circuit(), [0])  # prepare |->
        phase = self.one_phase(unitary_circuit, state_preparation)
        self.assertEqual(phase, 0.5)

    def phase_estimation(self, unitary_circuit, state_preparation=None, num_evaluation_qubits=6,
                         backend=qiskit.BasicAer.get_backend('qasm_simulator')):
        """Run phase estimation with operator, eigenvalue pair `unitary_circuit`,
        `state_preparation`. Return all results
        """
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

    def test_qpe_Zplus_strings(self):
        """superposition eigenproblem Z, |+>, bitstrings"""
        unitary_circuit = Z.to_circuit()
        state_preparation = H.to_circuit()  # prepare |+>
        result = self.phase_estimation(
            unitary_circuit, state_preparation,
            backend=qiskit.BasicAer.get_backend('statevector_simulator'))
        phases = result.filter_phases(1e-15, as_float=False)
        self.assertEqual(list(phases.keys()), ['000000', '100000'])


if __name__ == '__main__':
    unittest.main()
