# -*- coding: utf-8 -*-

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

import unittest

from qiskit.circuit.oracle_compiler import compile_oracle, ParseError
from . import bad_examples as examples


class TestParseFail(unittest.TestCase):
    def assertExceptionMessage(self, context, message):
        self.assertTrue(message in context.exception.args[0])

    def test_id_bad_return(self):
        with self.assertRaises(ParseError) as context:
            compile_oracle(examples.id_bad_return)
        self.assertExceptionMessage(context, 'return type error')

    def test_id_no_type_arg(self):
        with self.assertRaises(ParseError) as context:
            compile_oracle(examples.id_no_type_arg)
        self.assertExceptionMessage(context, 'argument type is needed')

    def test_id_no_type_return(self):
        with self.assertRaises(ParseError) as context:
            compile_oracle(examples.id_no_type_return)
        self.assertExceptionMessage(context, 'return type is needed')

    def test_out_of_scope(self):
        with self.assertRaises(ParseError) as context:
            compile_oracle(examples.out_of_scope)
        self.assertExceptionMessage(context, 'out of scope: c')


if __name__ == '__main__':
    unittest.main()
