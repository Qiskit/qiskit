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

""" Test Evolution """

import unittest
from test.aqua import QiskitAquaTestCase

import numpy as np
import scipy.linalg

import qiskit
from qiskit.circuit import ParameterVector, Parameter

from qiskit.aqua.operators import (X, Y, Z, I, CX, H, ListOp, CircuitOp, Zero, EvolutionFactory,
                                   EvolvedOp, PauliTrotterEvolution, QDrift, Trotter, Suzuki)

# pylint: disable=invalid-name


class TestEvolution(QiskitAquaTestCase):
    """Evolution tests."""

    def test_exp_i(self):
        """ exponential of Pauli test """
        op = Z.exp_i()
        gate = op.to_circuit().data[0][0]
        self.assertIsInstance(gate, qiskit.circuit.library.RZGate)
        self.assertEqual(gate.params[0], 2)

    def test_trotter_with_identity(self):
        """ trotterization of operator with identity term """
        op = (2.0 * I ^ I) + (Z ^ Y)
        exact_matrix = scipy.linalg.expm(-1j * op.to_matrix())
        evo = PauliTrotterEvolution(trotter_mode='suzuki', reps=2)
        with self.subTest('all PauliOp terms'):
            circ_op = evo.convert(EvolvedOp(op))
            circuit_matrix = qiskit.quantum_info.Operator(circ_op.to_circuit()).data
            np.testing.assert_array_almost_equal(exact_matrix, circuit_matrix)

        with self.subTest('MatrixOp identity term'):
            op = (2.0 * I ^ I).to_matrix_op() + (Z ^ Y)
            circ_op = evo.convert(EvolvedOp(op))
            circuit_matrix = qiskit.quantum_info.Operator(circ_op.to_circuit()).data
            np.testing.assert_array_almost_equal(exact_matrix, circuit_matrix)

        with self.subTest('CircuitOp identity term'):
            op = (2.0 * I ^ I).to_circuit_op() + (Z ^ Y)
            circ_op = evo.convert(EvolvedOp(op))
            circuit_matrix = qiskit.quantum_info.Operator(circ_op.to_circuit()).data
            np.testing.assert_array_almost_equal(exact_matrix, circuit_matrix)

    def test_pauli_evolution(self):
        """ pauli evolution test """
        op = (-1.052373245772859 * I ^ I) + \
             (0.39793742484318045 * I ^ Z) + \
             (0.18093119978423156 * X ^ X) + \
             (-0.39793742484318045 * Z ^ I) + \
             (-0.01128010425623538 * Z ^ Z)
        evolution = EvolutionFactory.build(operator=op)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = ((np.pi / 2) * op).exp_i() @ CX @ (H ^ I) @ Zero
        mean = evolution.convert(wf)
        self.assertIsNotNone(mean)

    def test_parameterized_evolution(self):
        """ parameterized evolution test """
        thetas = ParameterVector('θ', length=7)
        op = (thetas[0] * I ^ I) + \
             (thetas[1] * I ^ Z) + \
             (thetas[2] * X ^ X) + \
             (thetas[3] * Z ^ I) + \
             (thetas[4] * Y ^ Z) + \
             (thetas[5] * Z ^ Z)
        op = op * thetas[6]
        evolution = PauliTrotterEvolution(trotter_mode='trotter', reps=1)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        mean = evolution.convert(wf)
        circuit_params = mean.to_circuit().parameters
        # Check that the non-identity parameters are in the circuit
        for p in thetas[1:]:
            self.assertIn(p, circuit_params)
        self.assertNotIn(thetas[0], circuit_params)

    def test_bind_parameters(self):
        """ bind parameters test """
        thetas = ParameterVector('θ', length=6)
        op = (thetas[1] * I ^ Z) + \
             (thetas[2] * X ^ X) + \
             (thetas[3] * Z ^ I) + \
             (thetas[4] * Y ^ Z) + \
             (thetas[5] * Z ^ Z)
        op = thetas[0] * op
        evolution = PauliTrotterEvolution(trotter_mode='trotter', reps=1)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        wf = wf.assign_parameters({thetas: np.arange(10, 16)})
        mean = evolution.convert(wf)
        circuit_params = mean.to_circuit().parameters
        # Check that the no parameters are in the circuit
        for p in thetas[1:]:
            self.assertNotIn(p, circuit_params)

    def test_bind_circuit_parameters(self):
        """ bind circuit parameters test """
        thetas = ParameterVector('θ', length=6)
        op = (thetas[1] * I ^ Z) + \
             (thetas[2] * X ^ X) + \
             (thetas[3] * Z ^ I) + \
             (thetas[4] * Y ^ Z) + \
             (thetas[5] * Z ^ Z)
        op = thetas[0] * op
        evolution = PauliTrotterEvolution(trotter_mode='trotter', reps=1)
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
        """ bind parameters list test """
        thetas = ParameterVector('θ', length=6)
        op = (thetas[1] * I ^ Z) + \
             (thetas[2] * X ^ X) + \
             (thetas[3] * Z ^ I) + \
             (thetas[4] * Y ^ Z) + \
             (thetas[5] * Z ^ Z)
        op = thetas[0] * op
        evolution = PauliTrotterEvolution(trotter_mode='trotter', reps=1)
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

    def test_qdrift(self):
        """ QDrift test """
        op = (2 * Z ^ Z) + (3 * X ^ X) - (4 * Y ^ Y) + (.5 * Z ^ I)
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
        """ MatrixOp evolution test """
        # pylint: disable=no-member
        op = (-1.052373245772859 * I ^ I) + \
             (0.39793742484318045 * I ^ Z) + \
             (0.18093119978423156 * X ^ X) + \
             (-0.39793742484318045 * Z ^ I) + \
             (-0.01128010425623538 * Z ^ Z) * np.pi/2
        exp_mat = op.to_matrix_op().exp_i().to_matrix()
        ref_mat = scipy.linalg.expm(-1j * op.to_matrix())
        np.testing.assert_array_almost_equal(ref_mat, exp_mat)

    def test_log_i(self):
        """ MatrixOp.log_i() test """
        op = (-1.052373245772859 * I ^ I) + \
             (0.39793742484318045 * I ^ Z) + \
             (0.18093119978423156 * X ^ X) + \
             (-0.39793742484318045 * Z ^ I) + \
             (-0.01128010425623538 * Z ^ Z) * np.pi/2
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
        op = ListOp([(0.39793742484318045 * I ^ Z),
                     (0.18093119978423156 * X ^ X),
                     (-0.39793742484318045 * Z ^ I),
                     (-0.01128010425623538 * Z ^ Z) * np.pi / 2])
        log_exp_op = op.to_matrix_op().exp_i().to_matrix_op().log_i().to_pauli_op()
        np.testing.assert_array_almost_equal(op.to_matrix(), log_exp_op.to_matrix())

    def test_matrix_op_parameterized_evolution(self):
        """ parameterized MatrixOp evolution test """
        # pylint: disable=no-member
        theta = Parameter('θ')
        op = (-1.052373245772859 * I ^ I) + \
             (0.39793742484318045 * I ^ Z) + \
             (0.18093119978423156 * X ^ X) + \
             (-0.39793742484318045 * Z ^ I) + \
             (-0.01128010425623538 * Z ^ Z)
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
        """ bind parameters test """
        thetas = ParameterVector('θ', length=6)
        op = (thetas[1] * (I ^ Z).to_matrix_op()) + \
             (thetas[2] * (X ^ X)).to_matrix_op() + \
             (thetas[3] * Z ^ I) + \
             (thetas[4] * Y ^ Z).to_circuit_op() + \
             (thetas[5] * (Z ^ I).to_circuit_op())
        op = thetas[0] * op
        evolution = PauliTrotterEvolution(trotter_mode='trotter', reps=1)
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


if __name__ == '__main__':
    unittest.main()
