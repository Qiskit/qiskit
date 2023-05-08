# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Evolution"""

import unittest
from test.python.opflow import QiskitOpflowTestCase
import numpy as np
import scipy.linalg

import qiskit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.extensions import UnitaryGate
from qiskit.opflow import (
    CX,
    CircuitOp,
    EvolutionFactory,
    EvolvedOp,
    H,
    I,
    ListOp,
    PauliTrotterEvolution,
    QDrift,
    SummedOp,
    Suzuki,
    Trotter,
    X,
    Y,
    Z,
    Zero,
)


class TestEvolution(QiskitOpflowTestCase):
    """Evolution tests."""

    def test_exp_i(self):
        """exponential of Pauli test"""
        op = Z.exp_i()
        gate = op.to_circuit().data[0].operation
        self.assertIsInstance(gate, qiskit.circuit.library.RZGate)
        self.assertEqual(gate.params[0], 2)

    def test_trotter_with_identity(self):
        """trotterization of operator with identity term"""
        op = (2.0 * I ^ I) + (Z ^ Y)
        exact_matrix = scipy.linalg.expm(-1j * op.to_matrix())
        evo = PauliTrotterEvolution(trotter_mode="suzuki", reps=2)
        with self.subTest("all PauliOp terms"):
            circ_op = evo.convert(EvolvedOp(op))
            circuit_matrix = qiskit.quantum_info.Operator(circ_op.to_circuit()).data
            np.testing.assert_array_almost_equal(exact_matrix, circuit_matrix)

        with self.subTest("MatrixOp identity term"):
            op = (2.0 * I ^ I).to_matrix_op() + (Z ^ Y)
            circ_op = evo.convert(EvolvedOp(op))
            circuit_matrix = qiskit.quantum_info.Operator(circ_op.to_circuit()).data
            np.testing.assert_array_almost_equal(exact_matrix, circuit_matrix)

        with self.subTest("CircuitOp identity term"):
            op = (2.0 * I ^ I).to_circuit_op() + (Z ^ Y)
            circ_op = evo.convert(EvolvedOp(op))
            circuit_matrix = qiskit.quantum_info.Operator(circ_op.to_circuit()).data
            np.testing.assert_array_almost_equal(exact_matrix, circuit_matrix)

    def test_pauli_evolution(self):
        """pauli evolution test"""
        op = (
            (-1.052373245772859 * I ^ I)
            + (0.39793742484318045 * I ^ Z)
            + (0.18093119978423156 * X ^ X)
            + (-0.39793742484318045 * Z ^ I)
            + (-0.01128010425623538 * Z ^ Z)
        )
        evolution = EvolutionFactory.build(operator=op)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = ((np.pi / 2) * op).exp_i() @ CX @ (H ^ I) @ Zero
        mean = evolution.convert(wf)
        self.assertIsNotNone(mean)

    def test_summedop_pauli_evolution(self):
        """SummedOp[PauliOp] evolution test"""
        op = SummedOp(
            [
                (-1.052373245772859 * I ^ I),
                (0.39793742484318045 * I ^ Z),
                (0.18093119978423156 * X ^ X),
                (-0.39793742484318045 * Z ^ I),
                (-0.01128010425623538 * Z ^ Z),
            ]
        )
        evolution = EvolutionFactory.build(operator=op)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = ((np.pi / 2) * op).exp_i() @ CX @ (H ^ I) @ Zero
        mean = evolution.convert(wf)
        self.assertIsNotNone(mean)

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
        evolution = PauliTrotterEvolution(trotter_mode="trotter", reps=1)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        mean = evolution.convert(wf)
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
        evolution = PauliTrotterEvolution(trotter_mode="trotter", reps=1)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        wf = wf.assign_parameters({thetas: np.arange(10, 16)})
        mean = evolution.convert(wf)
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
        evolution = PauliTrotterEvolution(trotter_mode="trotter", reps=1)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        evo = evolution.convert(wf)
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
        evolution = PauliTrotterEvolution(trotter_mode="trotter", reps=1)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        evo = evolution.convert(wf)
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

    def test_bind_parameters_complex(self):
        """bind parameters with a complex value test"""
        th1 = Parameter("th1")
        th2 = Parameter("th2")

        operator = th1 * X + th2 * Y
        bound_operator = operator.bind_parameters({th1: 3j, th2: 2})

        expected_bound_operator = SummedOp([3j * X, (2 + 0j) * Y])
        self.assertEqual(bound_operator, expected_bound_operator)

    def test_qdrift(self):
        """QDrift test"""
        op = (2 * Z ^ Z) + (3 * X ^ X) - (4 * Y ^ Y) + (0.5 * Z ^ I)
        trotterization = QDrift().convert(op)
        self.assertGreater(len(trotterization.oplist), 150)
        last_coeff = None
        # Check that all types are correct and all coefficients are equals
        for op in trotterization.oplist:
            self.assertIsInstance(op, (EvolvedOp, CircuitOp))
            if isinstance(op, EvolvedOp):
                if last_coeff:
                    self.assertEqual(op.primitive.coeff, last_coeff)
                else:
                    last_coeff = op.primitive.coeff

    def test_qdrift_summed_op(self):
        """QDrift test for SummedOp"""
        op = SummedOp(
            [
                (2 * Z ^ Z),
                (3 * X ^ X),
                (-4 * Y ^ Y),
                (0.5 * Z ^ I),
            ]
        )
        trotterization = QDrift().convert(op)
        self.assertGreater(len(trotterization.oplist), 150)
        last_coeff = None
        # Check that all types are correct and all coefficients are equals
        for op in trotterization.oplist:
            self.assertIsInstance(op, (EvolvedOp, CircuitOp))
            if isinstance(op, EvolvedOp):
                if last_coeff:
                    self.assertEqual(op.primitive.coeff, last_coeff)
                else:
                    last_coeff = op.primitive.coeff

    def test_matrix_op_evolution(self):
        """MatrixOp evolution test"""
        op = (
            (-1.052373245772859 * I ^ I)
            + (0.39793742484318045 * I ^ Z)
            + (0.18093119978423156 * X ^ X)
            + (-0.39793742484318045 * Z ^ I)
            + (-0.01128010425623538 * Z ^ Z) * np.pi / 2
        )
        exp_mat = op.to_matrix_op().exp_i().to_matrix()
        ref_mat = scipy.linalg.expm(-1j * op.to_matrix())
        np.testing.assert_array_almost_equal(ref_mat, exp_mat)

    def test_log_i(self):
        """MatrixOp.log_i() test"""
        op = (
            (-1.052373245772859 * I ^ I)
            + (0.39793742484318045 * I ^ Z)
            + (0.18093119978423156 * X ^ X)
            + (-0.39793742484318045 * Z ^ I)
            + (-0.01128010425623538 * Z ^ Z) * np.pi / 2
        )
        # Test with CircuitOp
        log_exp_op = op.to_matrix_op().exp_i().log_i().to_pauli_op()
        np.testing.assert_array_almost_equal(op.to_matrix(), log_exp_op.to_matrix())

        # Test with MatrixOp
        log_exp_op = op.to_matrix_op().exp_i().to_matrix_op().log_i().to_pauli_op()
        np.testing.assert_array_almost_equal(op.to_matrix(), log_exp_op.to_matrix())

        # Test with PauliOp
        log_exp_op = op.to_matrix_op().exp_i().to_pauli_op().log_i().to_pauli_op()
        np.testing.assert_array_almost_equal(op.to_matrix(), log_exp_op.to_matrix())

        # Test with EvolvedOp
        log_exp_op = op.exp_i().to_pauli_op().log_i().to_pauli_op()
        np.testing.assert_array_almost_equal(op.to_matrix(), log_exp_op.to_matrix())

        # Test with proper ListOp
        op = ListOp(
            [
                (0.39793742484318045 * I ^ Z),
                (0.18093119978423156 * X ^ X),
                (-0.39793742484318045 * Z ^ I),
                (-0.01128010425623538 * Z ^ Z) * np.pi / 2,
            ]
        )
        log_exp_op = op.to_matrix_op().exp_i().to_matrix_op().log_i().to_pauli_op()
        np.testing.assert_array_almost_equal(op.to_matrix(), log_exp_op.to_matrix())

    def test_matrix_op_parameterized_evolution(self):
        """parameterized MatrixOp evolution test"""
        theta = Parameter("θ")
        op = (
            (-1.052373245772859 * I ^ I)
            + (0.39793742484318045 * I ^ Z)
            + (0.18093119978423156 * X ^ X)
            + (-0.39793742484318045 * Z ^ I)
            + (-0.01128010425623538 * Z ^ Z)
        )
        op = op * theta
        wf = (op.to_matrix_op().exp_i()) @ CX @ (H ^ I) @ Zero
        self.assertIn(theta, wf.to_circuit().parameters)

        op = op.assign_parameters({theta: 1})
        exp_mat = op.to_matrix_op().exp_i().to_matrix()
        ref_mat = scipy.linalg.expm(-1j * op.to_matrix())
        np.testing.assert_array_almost_equal(ref_mat, exp_mat)

        wf = wf.assign_parameters({theta: 3})
        self.assertNotIn(theta, wf.to_circuit().parameters)

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
        evolution = PauliTrotterEvolution(trotter_mode="trotter", reps=1)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        wf = wf.assign_parameters({thetas: np.arange(10, 16)})
        mean = evolution.convert(wf)
        circuit_params = mean.to_circuit().parameters
        # Check that the no parameters are in the circuit
        for p in thetas[1:]:
            self.assertNotIn(p, circuit_params)

    def test_reps(self):
        """Test reps and order params in Trotterization"""
        reps = 7
        trotter = Trotter(reps=reps)
        self.assertEqual(trotter.reps, reps)

        order = 5
        suzuki = Suzuki(reps=reps, order=order)
        self.assertEqual(suzuki.reps, reps)
        self.assertEqual(suzuki.order, order)

        qdrift = QDrift(reps=reps)
        self.assertEqual(qdrift.reps, reps)

    def test_suzuki_directly(self):
        """Test for Suzuki converter"""
        operator = X + Z

        evo = Suzuki()
        evolution = evo.convert(operator)

        matrix = np.array(
            [[0.29192658 - 0.45464871j, -0.84147098j], [-0.84147098j, 0.29192658 + 0.45464871j]]
        )
        np.testing.assert_array_almost_equal(evolution.to_matrix(), matrix)

    def test_evolved_op_to_instruction(self):
        """Test calling `to_instruction` on a plain EvolvedOp.

        Regression test of Qiskit/qiskit-terra#8025.
        """
        op = EvolvedOp(0.5 * X)
        circuit = op.to_instruction()

        unitary = scipy.linalg.expm(-0.5j * X.to_matrix())
        expected = UnitaryGate(unitary)

        self.assertEqual(circuit, expected)


if __name__ == "__main__":
    unittest.main()
