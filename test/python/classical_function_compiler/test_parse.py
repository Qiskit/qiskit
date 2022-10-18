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

"""Tests the classicalfunction parser."""
import unittest

from qiskit.test import QiskitTestCase
from qiskit.utils.optionals import HAS_TWEEDLEDUM

if HAS_TWEEDLEDUM:
    from . import bad_examples as examples
    from qiskit.circuit.classicalfunction import ClassicalFunctionParseError
    from qiskit.circuit.classicalfunction import classical_function as compile_classical_function


@unittest.skipUnless(HAS_TWEEDLEDUM, "Tweedledum is required for these tests.")
class TestParseFail(QiskitTestCase):
    """Tests bad_examples with the classicalfunction parser."""

    def assertExceptionMessage(self, context, message):
        """Asserts the message of an exception context"""
        self.assertTrue(message in context.exception.args[0])

    def test_id_bad_return(self):
        """Trying to parse examples.id_bad_return raises ClassicalFunctionParseError"""
        with self.assertRaises(ClassicalFunctionParseError) as context:
            compile_classical_function(examples.id_bad_return)
        self.assertExceptionMessage(context, "return type error")

    def test_id_no_type_arg(self):
        """Trying to parse examples.id_no_type_arg raises ClassicalFunctionParseError"""
        with self.assertRaises(ClassicalFunctionParseError) as context:
            compile_classical_function(examples.id_no_type_arg)
        self.assertExceptionMessage(context, "argument type is needed")

    def test_id_no_type_return(self):
        """Trying to parse examples.id_no_type_return raises ClassicalFunctionParseError"""
        with self.assertRaises(ClassicalFunctionParseError) as context:
            compile_classical_function(examples.id_no_type_return)
        self.assertExceptionMessage(context, "return type is needed")

    def test_out_of_scope(self):
        """Trying to parse examples.out_of_scope raises ClassicalFunctionParseError"""
        with self.assertRaises(ClassicalFunctionParseError) as context:
            compile_classical_function(examples.out_of_scope)
        self.assertExceptionMessage(context, "out of scope: c")
