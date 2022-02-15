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

"""Test BaseBackend methods."""

from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q, FakeMelbourne


class TestBackendAttrs(QiskitTestCase):
    """Test the backend methods."""

    def setUp(self):
        super().setUp()
        self.pulse_backend = FakeOpenPulse2Q()
        self.backend = FakeMelbourne()

    def test_name(self):
        """Test that name can be extracted."""
        self.assertEqual(self.pulse_backend.name(), "fake_openpulse_2q")
        self.assertEqual(self.backend.name(), "fake_melbourne")

    def test_version(self):
        """Test that name can be extracted."""
        self.assertEqual(self.pulse_backend.version, 1)
        self.assertEqual(self.backend.version, 1)

    def test_str_and_repr(self):
        """Test the custom __str__ and __repr__ methods."""
        self.assertEqual(str(self.pulse_backend), "fake_openpulse_2q")
        self.assertEqual(str(self.backend), "fake_melbourne")
        self.assertEqual(repr(self.pulse_backend), "<FakeOpenPulse2Q('fake_openpulse_2q')>")
