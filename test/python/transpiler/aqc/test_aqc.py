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
Tests AQC framework using hardcoded and randomly generated circuits.
"""
import unittest

from test.python.transpiler.aqc.sample_data import ORIGINAL_CIRCUIT, INITIAL_THETAS

import numpy as np

from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.quantum_info import Operator
from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.aqc.aqc import AQC
from qiskit.transpiler.synthesis.aqc.cnot_structures import make_cnot_network
from qiskit.transpiler.synthesis.aqc.cnot_unit_circuit import CNOTUnitCircuit
from qiskit.transpiler.synthesis.aqc.cnot_unit_objective import DefaultCNOTUnitObjective


class TestAqc(QiskitTestCase):
    """Main tests of approximate quantum compiler."""

    def test_aqc(self):
        """Tests AQC on a hardcoded circuit/matrix."""

        seed = 12345

        num_qubits = int(round(np.log2(np.array(ORIGINAL_CIRCUIT).shape[0])))
        cnots = make_cnot_network(
            num_qubits=num_qubits, network_layout="spin", connectivity_type="full", depth=0
        )

        optimizer = L_BFGS_B(maxiter=200)

        aqc = AQC(optimizer=optimizer, seed=seed)

        target_matrix = np.array(ORIGINAL_CIRCUIT)
        approximate_circuit = CNOTUnitCircuit(num_qubits, cnots)
        approximating_objective = DefaultCNOTUnitObjective(num_qubits, cnots)
        # todo: approximating_objective = DefaultCNOTUnitObjective(approximate_circuit)

        aqc.compile_unitary(
            target_matrix=target_matrix,
            approximate_circuit=approximate_circuit,
            approximating_objective=approximating_objective,
            initial_point=np.array(INITIAL_THETAS),
        )

        approx_matrix = Operator(approximate_circuit).data
        error = 0.5 * (np.linalg.norm(approx_matrix - ORIGINAL_CIRCUIT, "fro") ** 2)
        self.assertTrue(error < 1e-3)


if __name__ == "__main__":
    unittest.main()
