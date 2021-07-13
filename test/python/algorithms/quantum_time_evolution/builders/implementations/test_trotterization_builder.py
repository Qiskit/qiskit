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

""" Test Trotterization Evolution Builder. """
import unittest

import numpy as np
import scipy.linalg
from qiskit import quantum_info

from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterization_builder import (
    TrotterizationBuilder,
)
from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations.trotter_mode_enum import (
    TrotterModeEnum,
)
from test.python.opflow import QiskitOpflowTestCase
import qiskit
from qiskit.circuit import ParameterVector
from qiskit.opflow import (
    CX,
    EvolvedOp,
    H,
    I,
    ListOp,
    X,
    Y,
    Z,
    Zero,
)


class TestTrotterizationBuilder(QiskitOpflowTestCase):
    """Trotterization Evolution Builder tests."""

    def test_trotter_with_identity(self):
        """trotterization of operator with identity term"""
        op = (2.0 * I ^ I) + (Z ^ Y)
        exact_matrix = scipy.linalg.expm(-1j * op.to_matrix())
        evo = TrotterizationBuilder(trotter_mode=TrotterModeEnum.SUZUKI, reps=2)
        with self.subTest("all PauliOp terms"):
            circ_op = evo.build(EvolvedOp(op))
            circuit_matrix = quantum_info.Operator(circ_op.to_circuit()).data
            np.testing.assert_array_almost_equal(exact_matrix, circuit_matrix)

        with self.subTest("MatrixOp identity term"):
            op = (2.0 * I ^ I).to_matrix_op() + (Z ^ Y)
            circ_op = evo.build(EvolvedOp(op))
            circuit_matrix = quantum_info.Operator(circ_op.to_circuit()).data
            np.testing.assert_array_almost_equal(exact_matrix, circuit_matrix)

        with self.subTest("CircuitOp identity term"):
            op = (2.0 * I ^ I).to_circuit_op() + (Z ^ Y)
            circ_op = evo.build(EvolvedOp(op))
            circuit_matrix = quantum_info.Operator(circ_op.to_circuit()).data
            np.testing.assert_array_almost_equal(exact_matrix, circuit_matrix)

    def test_parameterized_evolution(self):
        """parameterized evolution test"""
        thetas = ParameterVector("θ", length=7)
        op = (
            (thetas[0] * I ^ I)
            + (thetas[1] * I ^ Z)
            + (thetas[2] * X ^ X)
            + (thetas[3] * Z ^ I)
            + (thetas[4] * Y ^ Z)
            + (thetas[5] * Z ^ Z)
        )
        op = op * thetas[6]
        evolution = TrotterizationBuilder(trotter_mode=TrotterModeEnum.TROTTER, reps=1)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        mean = evolution.build(wf)
        circuit = mean.to_circuit()
        # Check that all parameters are in the circuit
        for p in thetas:
            self.assertIn(p, circuit.parameters)
        # Check that the identity-parameters only exist as global phase
        self.assertNotIn(thetas[0], circuit._parameter_table.get_keys())

    def test_bind_parameters(self):
        """bind parameters test"""
        thetas = ParameterVector("θ", length=6)
        op = (
            (thetas[1] * I ^ Z)
            + (thetas[2] * X ^ X)
            + (thetas[3] * Z ^ I)
            + (thetas[4] * Y ^ Z)
            + (thetas[5] * Z ^ Z)
        )
        op = thetas[0] * op
        evolution = TrotterizationBuilder(trotter_mode=TrotterModeEnum.TROTTER, reps=1)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        wf = wf.assign_parameters({thetas: np.arange(10, 16)})
        mean = evolution.build(wf)
        circuit_params = mean.to_circuit().parameters
        # Check that the no parameters are in the circuit
        for p in thetas[1:]:
            self.assertNotIn(p, circuit_params)

    def test_bind_circuit_parameters(self):
        """bind circuit parameters test"""
        thetas = ParameterVector("θ", length=6)
        op = (
            (thetas[1] * I ^ Z)
            + (thetas[2] * X ^ X)
            + (thetas[3] * Z ^ I)
            + (thetas[4] * Y ^ Z)
            + (thetas[5] * Z ^ Z)
        )
        op = thetas[0] * op
        evolution = TrotterizationBuilder(trotter_mode=TrotterModeEnum.TROTTER, reps=1)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        evo = evolution.build(wf)
        mean = evo.assign_parameters({thetas: np.arange(10, 16)})
        # Check that the no parameters are in the circuit
        for p in thetas[1:]:
            self.assertNotIn(p, mean.to_circuit().parameters)
        # Check that original circuit is unchanged
        for p in thetas:
            self.assertIn(p, evo.to_circuit().parameters)

    # TODO test with other Op types than CircuitStateFn
    def test_bind_parameter_list(self):
        """bind parameters list test"""
        thetas = ParameterVector("θ", length=6)
        op = (
            (thetas[1] * I ^ Z)
            + (thetas[2] * X ^ X)
            + (thetas[3] * Z ^ I)
            + (thetas[4] * Y ^ Z)
            + (thetas[5] * Z ^ Z)
        )
        op = thetas[0] * op
        evolution = TrotterizationBuilder(trotter_mode=TrotterModeEnum.TROTTER, reps=1)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        evo = evolution.build(wf)
        param_list = np.transpose([np.arange(10, 16), np.arange(2, 8), np.arange(30, 36)]).tolist()
        means = evo.assign_parameters({thetas: param_list})
        self.assertIsInstance(means, ListOp)
        # Check that the no parameters are in the circuit
        for p in thetas[1:]:
            for circop in means.oplist:
                self.assertNotIn(p, circop.to_circuit().parameters)
        # Check that original circuit is unchanged
        for p in thetas:
            self.assertIn(p, evo.to_circuit().parameters)

    def test_mixed_evolution(self):
        """bind parameters test"""
        thetas = ParameterVector("θ", length=6)
        op = (
            (thetas[1] * (I ^ Z).to_matrix_op())
            + (thetas[2] * (X ^ X)).to_matrix_op()
            + (thetas[3] * Z ^ I)
            + (thetas[4] * Y ^ Z).to_circuit_op()
            + (thetas[5] * (Z ^ I).to_circuit_op())
        )
        op = thetas[0] * op
        evolution = TrotterizationBuilder(trotter_mode=TrotterModeEnum.TROTTER, reps=1)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        wf = wf.assign_parameters({thetas: np.arange(10, 16)})
        mean = evolution.build(wf)
        circuit_params = mean.to_circuit().parameters
        # Check that the no parameters are in the circuit
        for p in thetas[1:]:
            self.assertNotIn(p, circuit_params)


if __name__ == "__main__":
    unittest.main()
