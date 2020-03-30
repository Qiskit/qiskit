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

"""Testing a Faulty Backend (1Q)."""

from qiskit.test.mock import FakeOurenseFaultyQ1
from qiskit.test import QiskitTestCase


class BackendpropertiesTestCase(QiskitTestCase):
    """Test usability methods of backend.properties() with FakeOurenseFaultyQ1,
    which is like FakeOurense but with a faulty 1Q"""

    backend = FakeOurenseFaultyQ1()

    def test_operational_false(self):
        """Test operation status of the qubit. Q1 is non-operational """
        self.assertFalse(self.backend.properties().operational(1))

    def test_coupling_map(self):
        """Test coupling map with a faulty qubit."""
        coupling_map = self.backend.configuration().coupling_map
        self.assertEqual(coupling_map, [[3, 4], [4, 3]])
