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
"""
Tests AQC plugin.
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.converters import dag_to_circuit
from qiskit.quantum_info import Operator
from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.aqc.aqc_plugin import AQCSynthesisPlugin


class TestAQCSynthesisPlugin(QiskitTestCase):
    """Basic tests of the AQC synthesis plugin."""

    def _compare_circuits_up_global_phase(
        self, target_matrix: np.ndarray, approx_matrix: np.ndarray
    ):
        # Hilbert–Schmidt inner product
        hs_product = np.trace(np.dot(approx_matrix.conj().T, target_matrix))

        alpha = np.angle(hs_product)
        target_matrix *= np.exp(-1j * alpha)

        return 0.5 * (np.linalg.norm(approx_matrix - target_matrix, "fro") ** 2)

    def test_aqc_plugin(self):
        """Basic test of the plugin."""
        qc = QuantumCircuit(3)
        qc.mcx(
            [
                0,
                1,
            ],
            2,
        )
        target_unitary = Operator(qc).data

        plugin = AQCSynthesisPlugin()
        dag = plugin.run(target_unitary, approximation_degree=0.001)

        approx_circuit = dag_to_circuit(dag)
        approx_unitary = Operator(approx_circuit).data

        error = self._compare_circuits_up_global_phase(target_unitary, approx_unitary)
        self.assertLess(error, 0.001)
