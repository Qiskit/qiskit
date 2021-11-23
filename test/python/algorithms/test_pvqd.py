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

        backend = Aer.get_backend("statevector_simulator")
        expectation = MatrixExpectation()
        hamiltonian = 0.1 * (Z ^ Z) + (I ^ X) + (X ^ I)
        observable = Z ^ Z

        time = 1
        dt = 0.01

        ansatz = EfficientSU2(2, reps=2)
        optimizer = SPSA(maxiter=300, learning_rate=0.1, perturbation=0.01)
        # optimizer = SPSA()
        # optimizer = COBYLA()
        # optimizer = GradientDescent(learning_rate=0.01)
        # optimizer = L_BFGS_B()
        initial_parameters = np.zeros(ansatz.num_parameters)
        initial_parameters[-2] = np.pi / 2
        initial_parameters[-4] = np.pi / 2

        # run pVQD keeping track of the energy and the magnetization
        pvqd = PVQD(ansatz, initial_parameters, optimizer,
                    quantum_instance=backend, expectation=expectation)
        result = pvqd.evolve(hamiltonian, time, dt, observables=[hamiltonian, observable])

        # get reference results
        initial_state = Statevector(ansatz.bind_parameters(initial_parameters))
        ref_t, ref_energy, ref_magn = self.exact(time, dt, hamiltonian, observable, initial_state)

        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_title("Energy")
        ax1.plot(result.times, result.observables[:, 0], label="pVQD")
        ax1.plot(ref_t, ref_energy, label="exact")
        ax2.set_title("Magnetization")
        ax2.plot(result.times, result.observables[:, 1], label="pVQD")
        ax2.plot(ref_t, ref_magn, label="exact")
        ax2.set_xlabel(r"time $t$")
        plt.tight_layout()
        plt.show()
        print(result)
