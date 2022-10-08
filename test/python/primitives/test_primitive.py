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

"""Tests for BasePrimitive."""

from ddt import ddt, data, unpack

from numpy import array

from qiskit.circuit.random import random_circuit
from qiskit.primitives.base_primitive import BasePrimitive
from qiskit.test import QiskitTestCase


@ddt
class TestCircuitValidation(QiskitTestCase):
    @data(
        (random_circuit(2, 2, seed=0), (random_circuit(2, 2, seed=0),)),
        (
            [random_circuit(2, 2, seed=0), random_circuit(2, 2, seed=1)],
            (random_circuit(2, 2, seed=0), random_circuit(2, 2, seed=1)),
        ),
    )
    @unpack
    def test_validate_circuits(self, circuits, expected):
        assert BasePrimitive._validate_circuits(circuits) == expected

    @data(None, "ERROR")
    def test_type_error(self, circuits):
        with self.assertRaises(TypeError):
            BasePrimitive._validate_circuits(circuits)

    @data((), [], "")
    def test_value_error(self, circuits):
        with self.assertRaises(ValueError):
            BasePrimitive._validate_circuits(circuits)


@ddt
class TestParameterValuesValidation(QiskitTestCase):
    @data(
        # (float("nan"), ((float("nan"),),)),  # TODO: should be disallowed
        (float("inf"), ((float("inf"),),)),  # TODO: should be disallowed
        (float("-inf"), ((float("-inf"),),)),  # TODO: should be disallowed
        ((), ((),)),
        ([], ((),)),
        (0, ((0,),)),
        (1.2, ((1.2,),)),
        ((0,), ((0,),)),
        ([0], ((0,),)),
        ([1.2], ((1.2,),)),
        ((0, 1), ((0, 1),)),
        ([0, 1], ((0, 1),)),
        ([0, 1.2], ((0, 1.2),)),
        ([0.3, 1.2], ((0.3, 1.2),)),
        (((0, 1)), ((0, 1),)),
        (([0, 1]), ((0, 1),)),
        ([(0, 1)], ((0, 1),)),
        ([[0, 1]], ((0, 1),)),
        ([[0, 1.2]], ((0, 1.2),)),
        ([[0.3, 1.2]], ((0.3, 1.2),)),
    )
    @unpack
    def test_validate_parameter_values(self, parameter_values, expected):
        for parameter_values in [parameter_values, array(parameter_values)]:
            assert BasePrimitive._validate_parameter_values(parameter_values) == expected
            assert (
                BasePrimitive._validate_parameter_values(None, default=parameter_values) == expected
            )

    @data("ERROR", ("E", "R", "R", "O", "R"), (["E", "R", "R"], ["O", "R"]), 1j, (1j,), ((1j,),))
    def test_type_error(self, parameter_values):
        with self.assertRaises(TypeError):
            BasePrimitive._validate_parameter_values(parameter_values)

    def test_value_error(self):
        with self.assertRaises(ValueError):
            BasePrimitive._validate_parameter_values(None)
