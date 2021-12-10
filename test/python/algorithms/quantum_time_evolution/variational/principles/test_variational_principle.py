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

"""Test variational principle."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np

from qiskit.algorithms.quantum_time_evolution.variational.principles.imaginary.implementations.imaginary_mc_lachlan_variational_principle import (
    ImaginaryMcLachlanVariationalPrinciple,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z


class TestVariationalPrinciple(QiskitAlgorithmsTestCase):
    """Test variational principle."""

    def test_lazy_init(self):
        observable = SummedOp(
            [
                0.2252 * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        )

        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = ansatz.ordered_parameters

        var_principle = ImaginaryMcLachlanVariationalPrinciple()

        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)

        np.testing.assert_equal(var_principle._hamiltonian, observable)
        np.testing.assert_equal(var_principle._ansatz, ansatz)

        assert var_principle._operator is not None
        assert var_principle._raw_metric_tensor is not None
        assert var_principle._raw_evolution_grad is not None


if __name__ == "__main__":
    unittest.main()
