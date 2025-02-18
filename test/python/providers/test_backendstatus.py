# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2024.
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

from qiskit.providers.fake_provider import Fake5QV1
from qiskit.providers.models.backendstatus import BackendStatus
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestBackendConfiguration(QiskitTestCase):
    """Test the BackendStatus class."""

    def setUp(self):
        """Test backend status for one of the fake backends"""
        super().setUp()

    def test_repr(self):
        """Test representation methods of BackendStatus"""
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="``qiskit.providers.models.backendstatus.BackendStatus`` is deprecated ",
        ):
            backend_status = BackendStatus("my_backend", "1.0", True, 2, "online")

        self.assertIsInstance(backend_status.__repr__(), str)
        repr_html = backend_status._repr_html_()
        self.assertIsInstance(repr_html, str)
        self.assertIn(backend_status.backend_name, repr_html)

    def test_fake_backend_status(self):
        """Test backend status for one of the fake backends"""
        with self.assertWarns(DeprecationWarning):
            fake_backend = Fake5QV1()
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="``qiskit.providers.models.backendstatus.BackendStatus`` is deprecated ",
        ):
            backend_status = fake_backend.status()
        self.assertIsInstance(backend_status, BackendStatus)


if __name__ == "__main__":
    import unittest

    unittest.main()
