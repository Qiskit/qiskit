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

from qiskit.pulse import Delay, DriveChannel
from qiskit.test import QiskitTestCase


class TestDelayCommand(QiskitTestCase):
    """Delay tests."""

    def test_delay(self):
        """Test delay."""
        delay = Delay(10, DriveChannel(0), name='test_name')

        self.assertEqual(delay.name, "test_name")
        self.assertEqual(delay.duration, 10)
        self.assertEqual(delay.operands, [10, DriveChannel(0)])
