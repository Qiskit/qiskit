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
    """Tests AQC synthesis plugin"""

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
        unitary = Operator(qc).data

        plugin = AQCSynthesisPlugin()
        dag = plugin.run(unitary, approximation_degree=0.001, thetas=None)
        unitary2 = Operator(dag_to_circuit(dag)).data
        error = 0.5 * (np.linalg.norm(unitary2 - unitary, "fro") ** 2)
        self.assertLess(error, 0.7)
