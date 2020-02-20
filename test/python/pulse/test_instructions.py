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

# pylint: disable=invalid-name,unexpected-keyword-arg

"""Unit tests for pulse instructions."""

from qiskit.pulse import Snapshot
from qiskit.test import QiskitTestCase


class TestSnapshot(QiskitTestCase):
    """Snapshot tests."""

    def test_default(self):
        """Test default snapshot."""
        snapshot = Snapshot(label='test_name', snapshot_type='state')

        self.assertEqual(snapshot.name, "test_name")
        self.assertEqual(snapshot.type, "state")
        self.assertEqual(snapshot.duration, 0)
