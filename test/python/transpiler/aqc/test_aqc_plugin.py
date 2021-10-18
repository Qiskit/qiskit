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
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.quantum_info import Operator
from qiskit.test import QiskitTestCase
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.synthesis.aqc.aqc_plugin import AQCSynthesisPlugin


class TestAQCSynthesisPlugin(QiskitTestCase):
    """Basic tests of the AQC synthesis plugin."""

    def setUp(self):
        super().setUp()
        self._qc = QuantumCircuit(3)
        self._qc.mcx(
            [
                0,
                1,
            ],
            2,
        )

        self._target_unitary = Operator(self._qc).data

    def test_aqc_plugin(self):
        """Basic test of the plugin."""
        plugin = AQCSynthesisPlugin()
        dag = plugin.run(self._target_unitary)

        approx_circuit = dag_to_circuit(dag)
        approx_unitary = Operator(approx_circuit).data

        np.testing.assert_array_almost_equal(self._target_unitary, approx_unitary, 3)

    def test_plugin_setup(self):
        """Tests the plugin via unitary synthesis pass"""
        transpiler_pass = UnitarySynthesis(basis_gates=["rx", "ry", "rz", "cx"], method="aqc")

        dag = circuit_to_dag(self._qc)
        dag = transpiler_pass.run(dag)

        approx_circuit = dag_to_circuit(dag)
        approx_unitary = Operator(approx_circuit).data

        np.testing.assert_array_almost_equal(self._target_unitary, approx_unitary, 3)
