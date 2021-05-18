# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Test the BackendStatus.
"""

from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeLondon

from qiskit.providers.models import BackendStatus


class TestBackendConfiguration(QiskitTestCase):
    """Test the methods on the BackendConfiguration class."""

    def setUp(self):
        super().setUp()
        self.backend_status = BackendStatus("my_backend", "1.0", True, 2, "online")

    def test_repr(self):
        self.assertIsInstance(self.backend_status.__repr__(), str)
        repr_html = self.backend_status._repr_html_()
        self.assertIsInstance(repr_html, str)
        self.assertIn(f"0x{id(self.backend_status):016X}", repr_html)
        self.assertIn(self.backend_status.backend_name, repr_html)

    def test_fake_backend_status(self):
        fake_backend = FakeLondon()
        backend_status = fake_backend.status()
        self.assertIsInstance(backend_status, BackendStatus)


if __name__ == "__main__":
    import unittest

    unittest.main()
