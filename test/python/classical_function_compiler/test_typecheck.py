# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests classicalfunction compiler type checker."""
from qiskit.test import QiskitTestCase
from qiskit.circuit.classicalfunction import ClassicalFunctionCompilerTypeError
from qiskit.circuit.classicalfunction import classical_function as compile_classical_function

from . import examples, bad_examples


class TestTypeCheck(QiskitTestCase):
    """Tests classicalfunction compiler type checker (good examples)."""

    def test_id(self):
        """Tests examples.identity type checking"""
        network = compile_classical_function(examples.identity)
        self.assertEqual(network.args, ["a"])
        self.assertEqual(network.types, [{"Int1": "type", "a": "Int1", "return": "Int1"}])

    def test_bool_not(self):
        """Tests examples.bool_not type checking"""
        network = compile_classical_function(examples.bool_not)
        self.assertEqual(network.args, ["a"])
        self.assertEqual(network.types, [{"Int1": "type", "a": "Int1", "return": "Int1"}])

    def test_id_assign(self):
        """Tests examples.id_assing type checking"""
        network = compile_classical_function(examples.id_assing)
        self.assertEqual(network.args, ["a"])
        self.assertEqual(
            network.types, [{"Int1": "type", "a": "Int1", "b": "Int1", "return": "Int1"}]
        )

    def test_bit_and(self):
        """Tests examples.bit_and type checking"""
        network = compile_classical_function(examples.bit_and)
        self.assertEqual(network.args, ["a", "b"])
        self.assertEqual(
            network.types, [{"Int1": "type", "a": "Int1", "b": "Int1", "return": "Int1"}]
        )

    def test_bit_or(self):
        """Tests examples.bit_or type checking"""
        network = compile_classical_function(examples.bit_or)
        self.assertEqual(network.args, ["a", "b"])
        self.assertEqual(
            network.types, [{"Int1": "type", "a": "Int1", "b": "Int1", "return": "Int1"}]
        )

    def test_bool_or(self):
        """Tests examples.bool_or type checking"""
        network = compile_classical_function(examples.bool_or)
        self.assertEqual(network.args, ["a", "b"])
        self.assertEqual(
            network.types, [{"Int1": "type", "a": "Int1", "b": "Int1", "return": "Int1"}]
        )


class TestTypeCheckFail(QiskitTestCase):
    """Tests classicalfunction compiler type checker (bad examples)."""

    def assertExceptionMessage(self, context, message):
        """Asserts the message of an exception context"""
        self.assertTrue(message in context.exception.args[0])

    def test_bit_not(self):
        """Int1wise not does not work on bit (aka bool)
        ~True   # -2
        ~False  # -1
        """
        with self.assertRaises(ClassicalFunctionCompilerTypeError) as context:
            compile_classical_function(bad_examples.bit_not)
        self.assertExceptionMessage(context, "does not operate with Int1 type")
