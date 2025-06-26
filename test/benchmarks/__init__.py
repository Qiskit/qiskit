import sys
import warnings
import unittest
from unittest import mock
from qiskit import __init__ as qiskit_init
# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

class TestDeprecationWarnings(unittest.TestCase):
    @mock.patch('sys.version_info', new=(3, 9))
    def test_deprecation_warning_for_python_3_9(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            # Re-trigger the import to check for warnings after mocking
            import qiskit.__init__ as qiskit_init
            self.assertTrue(any(issubclass(warn.category, DeprecationWarning) for warn in w))
