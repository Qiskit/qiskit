# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import unittest

from qiskit.algorithms.quantum_time_evolution.variational.principles.metric_tensor_calculator import \
    build
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z
from test.python.algorithms import QiskitAlgorithmsTestCase


class TestMetricTensorBuilder(QiskitAlgorithmsTestCase):

    def test_build(self):

        observable = SummedOp([0.2252 * (I ^ I), 0.5716 * (Z ^ Z), 0.3435 * (I ^ Z),
                               -0.4347 * (Z ^ I), 0.091 * (Y ^ Y),
                               0.091 * (X ^ X)]).reduce()

        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = ansatz.ordered_parameters
        metric_tensor = build(observable, ansatz, parameters)
        print(metric_tensor)


if __name__ == "__main__":
    unittest.main()
