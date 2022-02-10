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
import sys
import unittest
import numpy as np
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.quantum_info import Operator
from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.aqc.aqc import AQC
from qiskit.transpiler.synthesis.aqc.cnot_structures import make_cnot_network
from qiskit.transpiler.synthesis.aqc.cnot_unit_circuit import CNOTUnitCircuit
from qiskit.transpiler.synthesis.aqc.cnot_unit_objective import DefaultCNOTUnitObjective
from qiskit.transpiler.synthesis.aqc.fast_gradient.fast_gradient import FastCNOTUnitObjective
from test.python.transpiler.aqc.sample_data import ORIGINAL_CIRCUIT, INITIAL_THETAS

__glo_verbose__ = False


class TestAqc(QiskitTestCase):
    """Main tests of approximate quantum compiler."""

    @staticmethod
    def _print_result_info(target_matrix: np.ndarray, approx_matrix: np.ndarray):
        if __glo_verbose__:
            diff = approx_matrix - target_matrix
            fro_err = 0.5 * (np.linalg.norm(diff, "fro") ** 2)
            sin_err = np.linalg.norm(diff, 2)
            print("\nApproximation misfit:")
            print(f"Cost function based on Frobenius norm: {fro_err:0.8f}")
            print(f"Max. singular value of (V - U): {sin_err:0.8f}")

    def test_aqc(self):
        """
        Tests AQC on a hardcoded circuit/matrix using default implementation
        of Frobenius norm minimizer: min(||U - V||_F^2).
        """

        seed = 12345
        num_qubits = int(round(np.log2(np.array(ORIGINAL_CIRCUIT).shape[0])))
        cnots = make_cnot_network(
            num_qubits=num_qubits, network_layout="spin", connectivity_type="full", depth=0
        )

        optimizer = L_BFGS_B(maxiter=200)
        aqc = AQC(optimizer=optimizer, seed=seed)

        target_matrix = np.array(ORIGINAL_CIRCUIT)
        circ = CNOTUnitCircuit(num_qubits, cnots)
        objv = DefaultCNOTUnitObjective(num_qubits, cnots)

        aqc.compile_unitary(
            target_matrix=target_matrix,
            approximate_circuit=circ,
            approximating_objective=objv,
            initial_point=np.array(INITIAL_THETAS),
        )

        approx_matrix = Operator(circ).data
        error = 0.5 * (np.linalg.norm(approx_matrix - target_matrix, "fro") ** 2)
        self._print_result_info(target_matrix, approx_matrix)
        self.assertTrue(error < 1e-3)

    def test_aqc_fastgrad(self):
        """
        Tests AQC on a MCX circuit/matrix with random initial guess using
        'fast gradient' implementation of Frobenius norm minimizer:
        min(||U - V||_F^2).
        """
        seed = 12345
        num_qubits = int(3)
        cnots = make_cnot_network(
            num_qubits=num_qubits, network_layout="spin",
            connectivity_type="full", depth=0
        )

        optimizer = L_BFGS_B(maxiter=200)
        aqc = AQC(optimizer=optimizer, seed=seed)

        # Make multi-control CNOT gate matrix.
        target_matrix = np.eye(2 ** num_qubits, dtype=np.cfloat)
        target_matrix[-2:, -2:] = [[0, 1], [1, 0]]
        # target_matrix = np.array(ORIGINAL_CIRCUIT)

        circ = CNOTUnitCircuit(num_qubits, cnots)
        objv = FastCNOTUnitObjective(num_qubits, cnots)

        aqc.compile_unitary(
            target_matrix=target_matrix,
            approximate_circuit=circ,
            approximating_objective=objv,
            initial_point=2 * np.pi * np.array(np.random.rand(objv.num_thetas))
        )

        approx_matrix = Operator(circ).data
        error = 0.5 * (np.linalg.norm(approx_matrix - target_matrix, "fro") ** 2)
        self._print_result_info(target_matrix, approx_matrix)
        self.assertTrue(error < 1e-3)


if __name__ == "__main__":
    __glo_verbose__ = ("-v" in sys.argv) or ("--verbose" in sys.argv)
    unittest.main()
