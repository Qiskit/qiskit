# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for PVQD."""

from ddt import ddt, data
import numpy as np

from qiskit.test import QiskitTestCase

from qiskit import BasicAer as Aer
from qiskit.algorithms.evolvers import EvolutionProblem
from qiskit.algorithms.evolvers.pvqd import PVQD
from qiskit.algorithms.optimizers import L_BFGS_B, GradientDescent, SPSA
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import X, Z, I, MatrixExpectation


@ddt
class TestPVQD(QiskitTestCase):
    """Tests for the pVQD algorithm."""

    def setUp(self):
        super().setUp()
        self.backend = Aer.get_backend("statevector_simulator")
        self.expectation = MatrixExpectation()
        self.hamiltonian = 0.1 * (Z ^ Z) + (I ^ X) + (X ^ I)
        self.observable = Z ^ Z
        self.ansatz = EfficientSU2(2, reps=1)
        self.initial_parameters = np.zeros(self.ansatz.num_parameters)

    @data(True, False)
    def test_pvqd(self, gradient):
        """Test a simple evolution."""
        time = 0.02

        if gradient:
            optimizer = GradientDescent(maxiter=1)
        else:
            optimizer = L_BFGS_B(maxiter=1)

        # run pVQD keeping track of the energy and the magnetization
        pvqd = PVQD(
            self.ansatz,
            self.initial_parameters,
            timestep=0.01,
            optimizer=optimizer,
            quantum_instance=self.backend,
            expectation=self.expectation,
        )
        problem = EvolutionProblem(
            self.hamiltonian, time, aux_operators=[self.hamiltonian, self.observable]
        )
        result = pvqd.evolve(problem)

        self.assertTrue(len(result.fidelities) == 3)
        self.assertTrue(np.all(result.times == np.array([0.0, 0.01, 0.02])))
        self.assertTrue(result.observables.shape == (3, 2))
        num_parameters = self.ansatz.num_parameters
        self.assertTrue(
            len(result.parameters) == 3
            and np.all([len(params) == num_parameters for params in result.parameters])
        )

    def test_invalid_timestep(self):
        """Test raises if the timestep is larger than the evolution time."""
        pvqd = PVQD(
            self.ansatz,
            self.initial_parameters,
            timestep=1,
            optimizer=L_BFGS_B(),
            quantum_instance=self.backend,
            expectation=self.expectation,
        )
        problem = EvolutionProblem(
            self.hamiltonian, time=0.01, aux_operators=[self.hamiltonian, self.observable]
        )

        with self.assertRaises(ValueError):
            _ = pvqd.evolve(problem)

    def test_initial_guess_and_observables(self):
        """Test doing no optimizations stays at initial guess."""
        initial_guess = np.zeros(self.ansatz.num_parameters)

        pvqd = PVQD(
            self.ansatz,
            self.initial_parameters,
            timestep=0.01,
            optimizer=SPSA(maxiter=0, learning_rate=0.1, perturbation=0.01),
            initial_guess=initial_guess,
            quantum_instance=self.backend,
            expectation=self.expectation,
        )
        problem = EvolutionProblem(
            self.hamiltonian, time=0.1, aux_operators=[self.hamiltonian, self.observable]
        )

        result = pvqd.evolve(problem)

        observables = result.aux_ops_evaluated
        print(result.evolved_state.draw())
        self.assertEqual(observables[0], 0.1)  # expected energy
        self.assertEqual(observables[1], 1)  # expected magnetization
