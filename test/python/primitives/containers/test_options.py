# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Unit tests for Options."""


from qiskit.primitives.containers import BasePrimitiveOptions
from qiskit.test import QiskitTestCase


class TestBasePrimitiveOptions(QiskitTestCase):
    """Test the BasePrimitiveOptions class."""

    def test_update(self):
        """Test update method"""
        options = BasePrimitiveOptions()
        options.update({"test": 1})
        self.assertEqual(options.test, 1)
        options.update(test2=True)
        self.assertEqual(options.test2, True)

        options2 = BasePrimitiveOptions()
        options2.update(options)
        self.assertEqual(options.test, 1)
        self.assertEqual(options.test2, True)
