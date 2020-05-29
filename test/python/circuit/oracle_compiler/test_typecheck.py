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

from qiskit.circuit.oracle_compiler import compile_oracle
from . import examples, bad_examples


class TestTypeCheck(unittest.TestCase):
    def test_id(self):
        network = compile_oracle(examples.id)
        self.assertEqual(network.types, [{'Bit': 'type', 'a': 'Bit', 'return': 'Bit'}])

    def test_bool_not(self):
        network = compile_oracle(examples.bool_not)
        self.assertEqual(network.types, [{'Bit': 'type', 'a': 'Bit', 'return': 'Bit'}])

    def test_id_assign(self):
        network = compile_oracle(examples.id_assing)
        self.assertEqual(network.types, [{'Bit': 'type', 'a': 'Bit', 'b': 'Bit', 'return': 'Bit'}])

    def test_bit_and(self):
        network = compile_oracle(examples.bit_and)
        self.assertEqual(network.types, [{'Bit': 'type', 'a': 'Bit', 'b': 'Bit', 'return': 'Bit'}])

    def test_bit_or(self):
        network = compile_oracle(examples.bit_or)
        self.assertEqual(network.types, [{'Bit': 'type', 'a': 'Bit', 'b': 'Bit', 'return': 'Bit'}])

    def test_bool_or(self):
        network = compile_oracle(examples.bool_or)
        self.assertEqual(network.types, [{'Bit': 'type', 'a': 'Bit', 'b': 'Bit', 'return': 'Bit'}])


class TestTypeCheckFail(unittest.TestCase):
    def assertExceptionMessage(self, context, message):
        self.assertTrue(message in context.exception.args[0])

    def test_bit_not(self):
        """bitwise not does not work on bit (aka bool)
          ~True   # -2
          ~False  # -1
        """
        with self.assertRaises(TypeError) as context:
            compile_oracle(bad_examples.bit_not)
        self.assertExceptionMessage(context, 'does not operate with Bit type')


if __name__ == '__main__':
    unittest.main()
