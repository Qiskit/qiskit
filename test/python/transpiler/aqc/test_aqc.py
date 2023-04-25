# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
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
from qiskit.transpiler.synthesis.aqc.fast_gradient.fast_gradient import FastCNOTUnitObjective


class TestAqc(QiskitTestCase):
    """Main tests of approximate quantum compiler."""

    def test_aqc(self):
        """Tests AQC on a hardcoded circuit/matrix."""

        seed = 12345

        num_qubits = int(round(np.log2(ORIGINAL_CIRCUIT.shape[0])))
        cnots = make_cnot_network(
            num_qubits=num_qubits, network_layout="spin", connectivity_type="full", depth=0
        )

        optimizer = L_BFGS_B(maxiter=200)

        aqc = AQC(optimizer=optimizer, seed=seed)

        target_matrix = ORIGINAL_CIRCUIT
        approximate_circuit = CNOTUnitCircuit(num_qubits, cnots)
        approximating_objective = DefaultCNOTUnitObjective(num_qubits, cnots)

        aqc.compile_unitary(
            target_matrix=target_matrix,
            approximate_circuit=approximate_circuit,
            approximating_objective=approximating_objective,
            initial_point=INITIAL_THETAS,
        )

        approx_matrix = Operator(approximate_circuit).data
        error = 0.5 * (np.linalg.norm(approx_matrix - ORIGINAL_CIRCUIT, "fro") ** 2)
        self.assertTrue(error < 1e-3)

    def test_aqc_fastgrad(self):
        """
        Tests AQC on a MCX circuit/matrix with random initial guess using
        the accelerated implementation.
        """
        seed = 12345
        np.random.seed(seed)

        num_qubits = int(3)
        cnots = make_cnot_network(
            num_qubits=num_qubits, network_layout="spin", connectivity_type="full", depth=0
        )

        optimizer = L_BFGS_B(maxiter=200)
        aqc = AQC(optimizer=optimizer, seed=seed)

        # Make multi-control CNOT gate matrix.
        # Another option: target_matrix = ORIGINAL_CIRCUIT
        target_matrix = np.eye(2**num_qubits, dtype=np.cfloat)
        target_matrix[-2:, -2:] = [[0, 1], [1, 0]]

        circ = CNOTUnitCircuit(num_qubits, cnots)
        objv = FastCNOTUnitObjective(num_qubits, cnots)

        aqc.compile_unitary(
            target_matrix=target_matrix,
            approximate_circuit=circ,
            approximating_objective=objv,
            initial_point=2 * np.pi * np.random.rand(objv.num_thetas),
        )

        approx_matrix = Operator(circ).data
        error = 0.5 * (np.linalg.norm(approx_matrix - target_matrix, "fro") ** 2)
        self.assertTrue(error < 1e-3)

    def test_aqc_determinant_minus_one(self):
        """
        Tests AQC on a matrix with determinant −1.
        """
        seed = 12345

        num_qubits = int(3)
        cnots = make_cnot_network(
            num_qubits=num_qubits, network_layout="spin", connectivity_type="full", depth=0
        )

        optimizer = L_BFGS_B(maxiter=200)
        aqc = AQC(optimizer=optimizer, seed=seed)

        target_matrix = np.eye(2**num_qubits, dtype=int)
        # Make a unitary with determinant −1 by swapping any two columns
        target_matrix[:, 2], target_matrix[:, 3] = target_matrix[:, 3], target_matrix[:, 2].copy()

        approximate_circuit = CNOTUnitCircuit(num_qubits, cnots)
        approximating_objective = DefaultCNOTUnitObjective(num_qubits, cnots)

        aqc.compile_unitary(
            target_matrix=target_matrix,
            approximate_circuit=approximate_circuit,
            approximating_objective=approximating_objective,
            initial_point=INITIAL_THETAS,
        )

        approx_matrix = Operator(approximate_circuit).data

        error = 0.5 * (np.linalg.norm(approx_matrix - target_matrix, "fro") ** 2)
        self.assertTrue(error < 1e-3)


if __name__ == "__main__":
    unittest.main()
