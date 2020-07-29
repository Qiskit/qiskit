# -*- coding: utf-8 -*-

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

""" Test Operator construction, including OpPrimitives and singletons. """

import unittest
from test.aqua import QiskitAquaTestCase

import numpy

from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import StateFn, Zero, One, H, X, I, Z, Plus, Minus, CircuitSampler


# pylint: disable=invalid-name

class TestStateOpMeasEvals(QiskitAquaTestCase):
    """Tests of evals of Meas-Operator-StateFn combos."""

    def test_statefn_overlaps(self):
        """ state functions overlaps test """
        wf = (4 * StateFn({'101010': .5, '111111': .3})) + ((3 + .1j) * (Zero ^ 6))
        wf_vec = StateFn(wf.to_matrix())
        self.assertAlmostEqual(wf.adjoint().eval(wf), 14.45)
        self.assertAlmostEqual(wf_vec.adjoint().eval(wf_vec), 14.45)
        self.assertAlmostEqual(wf_vec.adjoint().eval(wf), 14.45)
        self.assertAlmostEqual(wf.adjoint().eval(wf_vec), 14.45)

    def test_wf_evals_x(self):
        """ wf evals x test """
        qbits = 4
        wf = ((Zero ^ qbits) + (One ^ qbits)) * (1 / 2 ** .5)
        # Note: wf = Plus^qbits fails because TensoredOp can't handle it.
        wf_vec = StateFn(wf.to_matrix())
        op = X ^ qbits
        # op = I^6
        self.assertAlmostEqual(wf.adjoint().eval(op.eval(wf)), 1)
        self.assertAlmostEqual(wf_vec.adjoint().eval(op.eval(wf)), 1)
        self.assertAlmostEqual(wf.adjoint().eval(op.eval(wf_vec)), 1)
        self.assertAlmostEqual(wf_vec.adjoint().eval(op.eval(wf_vec)), 1)

        # op = (H^X^Y)^2
        op = H ^ 6
        wf = ((Zero ^ 6) + (One ^ 6)) * (1 / 2 ** .5)
        wf_vec = StateFn(wf.to_matrix())
        # print(wf.adjoint().to_matrix() @ op.to_matrix() @ wf.to_matrix())
        self.assertAlmostEqual(wf.adjoint().eval(op.eval(wf)), .25)
        self.assertAlmostEqual(wf_vec.adjoint().eval(op.eval(wf)), .25)
        self.assertAlmostEqual(wf.adjoint().eval(op.eval(wf_vec)), .25)
        self.assertAlmostEqual(wf_vec.adjoint().eval(op.eval(wf_vec)), .25)

    def test_coefficients_correctly_propagated(self):
        """Test that the coefficients in SummedOp and states are correctly used."""
        with self.subTest('zero coeff in SummedOp'):
            op = 0 * (I + Z)
            state = Plus
            self.assertEqual((~StateFn(op) @ state).eval(), 0j)

        backend = Aer.get_backend('qasm_simulator')
        q_instance = QuantumInstance(backend, seed_simulator=97, seed_transpiler=97)
        op = I
        with self.subTest('zero coeff in summed StateFn and CircuitSampler'):
            state = 0 * (Plus + Minus)
            sampler = CircuitSampler(q_instance).convert(~StateFn(op) @ state)
            self.assertEqual(sampler.eval(), 0j)

        with self.subTest('coeff gets squared in CircuitSampler shot-based readout'):
            state = (Plus + Minus) / numpy.sqrt(2)
            sampler = CircuitSampler(q_instance).convert(~StateFn(op) @ state)
            self.assertAlmostEqual(sampler.eval(), 1+0j)


if __name__ == '__main__':
    unittest.main()
