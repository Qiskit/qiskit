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

from qiskit.test import QiskitTestCase

import numpy as np
import scipy as sc

from qiskit import Aer
from qiskit.algorithms import PVQD
from qiskit.algorithms.optimizers import SPSA, L_BFGS_B, GradientDescent, COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import X, Z, I, MatrixExpectation, Gradient
from qiskit.quantum_info import Statevector

import matplotlib.pyplot as plt


class TestPVQD(QiskitTestCase):
    """Tests for the pVQD algorithm."""

    def setUp(self):
        super().setUp()
        self.backend = Aer.get_backend("statevector_simulator")
        self.expectation = MatrixExpectation()
        self.hamiltonian = 0.1 * (Z ^ Z) + (I ^ X) + (X ^ I)
        self.observable = Z ^ Z
        self.ansatz = EfficientSU2(2, reps=2)
        self.initial_parameters = np.zeros(self.ansatz.num_parameters)
        self.initial_parameters[-2] = np.pi / 2
        self.initial_parameters[-4] = np.pi / 2

    def exact(self, final_time, dt, hamiltonian, observable, initial_state):
        """Get the exact values for energy and the observable."""
        energies = []  # list of energies evaluated at timesteps dt
        obs = []  # list of observables
        ts = []  # list of timepoints at which energy/obs are evaluated
        t = 0
        while t <= final_time:
            # get exact state at time t
            exact_state = initial_state.evolve(sc.linalg.expm(-1j * t * hamiltonian.to_matrix()))

            # store observables and time
            ts.append(t)
            energies.append(exact_state.expectation_value(hamiltonian.to_matrix()))
            obs.append(exact_state.expectation_value(observable.to_matrix()))

            # next timestep
            t += dt

        return ts, energies, obs

    def test_pvqd(self):
        """Test a simple evolution."""
        time = 0.02
        dt = 0.01

        optimizer = L_BFGS_B()

        # run pVQD keeping track of the energy and the magnetization
        pvqd = PVQD(
            self.ansatz, self.initial_parameters, optimizer, quantum_instance=self.backend,
            expectation=self.expectation
        )
        result = pvqd.evolve(self.hamiltonian, time, dt, observables=[self.hamiltonian, self.observable])

        self.assertTrue(len(result.fidelities) == 3)
        self.assertTrue(np.all(result.times == np.array([0.0, 0.01, 0.02])))
        self.assertTrue(result.observables.shape == (3, 2))
        num_parameters = self.ansatz.num_parameters
        self.assertTrue(len(result.parameters) == 3 and np.all(
            [len(params) == num_parameters for params in result.parameters]))

    def test_gradients(self):
        """Test the calculation of gradients with the gradient framework."""
        time = 0.01
        dt = 0.01

        optimizer = GradientDescent(learning_rate=0.01)
        gradient = Gradient()

        # run pVQD keeping track of the energy and the magnetization
        pvqd = PVQD(
            self.ansatz, self.initial_parameters, optimizer, gradient=gradient,
            quantum_instance=self.backend, expectation=self.expectation
        )
        result = pvqd.evolve(self.hamiltonian, time, dt)

        print(result)
