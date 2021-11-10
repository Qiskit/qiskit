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

from qiskit import Aer
from qiskit.algorithms import PVQD
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import X, Z, MatrixExpectation

import matplotlib.pyplot as plt


class TestPVQD(QiskitTestCase):
    """Tests for the pVQD algorithm."""

    def test_pvqd(self):
        """Test a simple evolution."""

        backend = Aer.get_backend("statevector_simulator")
        expectation = MatrixExpectation()
        hamiltonian = X ^ X
        observable = Z ^ Z

        time = 0.1
        dt = 0.01

        ansatz = EfficientSU2(2, reps=3)
        optimizer = SPSA(maxiter=100, learning_rate=0.01, perturbation=0.01)
        initial_parameters = np.zeros(ansatz.num_parameters)
        initial_parameters[-2] = np.pi / 2
        initial_parameters[-4] = np.pi / 2

        pvqd = PVQD(ansatz, initial_parameters, optimizer, backend, expectation)
        result = pvqd.evolve(hamiltonian, time, dt, observables=[hamiltonian, observable])

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(result.times, result.observables[:, 0])
        ax2.plot(result.times, result.observables[:, 1])
        plt.show()
        print(result)
