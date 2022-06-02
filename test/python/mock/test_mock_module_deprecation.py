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

"""Test for deprecation of qiskit.test.mock module."""
from qiskit.test import QiskitTestCase


class MockModuleDeprecationTest(QiskitTestCase):
    """Test for deprecation of qiskit.test.mock module."""

    def test_deprecated_mock_module(self):
        """Test that the mock module is deprecated."""
        with self.assertWarns(DeprecationWarning):
            import qiskit.test.mock # pylint: disable=unused-import
