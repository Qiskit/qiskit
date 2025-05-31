# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Clifford+T optimization pass"""

import numpy as np

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes import SolovayKitaev, OptimizeCliffordT
from qiskit.quantum_info import Operator
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestOptimizeCliffordT(QiskitTestCase):
    """Test the OptimizeCliffordT pass."""

    angles = np.linspace(0, 2 * np.pi, 10)

    @data(*angles)
    def test_solovay_kitaev_rx(self, angle):
        """Test optimization of circuits coming out of the Solovay-Kitaev pass."""
        qc = QuantumCircuit(1)
        qc.rx(angle, 0)

        # Run Solovay-Kitaev pass on qc
        transpiled = SolovayKitaev()(qc)
        self.assertLessEqual(set(transpiled.count_ops()), {"h", "t", "tdg"})

        # Run Clifford+T optimization pass on the transpiled circuit
        optimized = OptimizeCliffordT()(transpiled)

        self.assertTrue(Operator(transpiled), Operator(optimized))
