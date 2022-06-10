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

"""Test iCANS."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

from qiskit.algorithms import ICANS
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import X, I, Z
from qiskit.providers.basicaer import QasmSimulatorPy


class TestICANS(QiskitAlgorithmsTestCase):
    """Test iCANS"""

    def setUp(self):
        super().setUp()
        self.backend = QasmSimulatorPy()

    def test_simple_run(self):
        """Test a simple iCANS run."""

        hamiltonian = (X ^ I) + (I ^ X) - 0.1 * (Z ^ Z)
        ansatz = EfficientSU2(num_qubits=2, reps=1)

        history = []

        def store_history(*args):
            print(args)
            history.append(args)

        icans = ICANS(ansatz, min_shots=10, quantum_instance=self.backend, callback=store_history)
        result = icans.compute_minimum_eigenvalue(hamiltonian)

        print(result)


if __name__ == "__main__":
    unittest.main()
