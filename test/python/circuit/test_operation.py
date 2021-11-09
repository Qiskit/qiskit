# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Qiskit's Operation class."""

import unittest
from qiskit.circuit import Operation
from qiskit.circuit.exceptions import CircuitError
from qiskit.test import QiskitTestCase


class TestOperationClass(QiskitTestCase):
    """Testing qiskit.circuit.Operation"""

    def test_can_not_instantiate_directly(self):
        """Test that we cannot instantiate an object of class Operation directly."""

        with self.assertRaises(CircuitError) as exc:
            Operation("my_operation", 2, 4, [])
        self.assertIn("should not be instantiated directly", exc.exception.message)


if __name__ == "__main__":
    unittest.main()
