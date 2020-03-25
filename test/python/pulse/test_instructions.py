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

"""Unit tests for pulse instructions."""

from qiskit.pulse import DriveChannel, pulse_lib
from qiskit.pulse import Delay, Play, ShiftPhase, Snapshot
from qiskit.test import QiskitTestCase


class TestDelay(QiskitTestCase):
    """Delay tests."""

    def test_delay(self):
        """Test delay."""
        delay = Delay(10, DriveChannel(0), name='test_name')

        self.assertEqual(delay.name, "test_name")
        self.assertEqual(delay.duration, 10)
        self.assertEqual(delay.operands, [10, DriveChannel(0)])


class TestShiftPhase(QiskitTestCase):
    """Test the instruction construction."""

    def test_default(self):
        """Test basic ShiftPhase."""
        shift_phase = ShiftPhase(1.57, DriveChannel(0))

        self.assertEqual(shift_phase.phase, 1.57)
        self.assertEqual(shift_phase.duration, 0)
        self.assertTrue(shift_phase.name.startswith('shiftphase'))


class TestSnapshot(QiskitTestCase):
    """Snapshot tests."""

    def test_default(self):
        """Test default snapshot."""
        snapshot = Snapshot(label='test_name', snapshot_type='state')

        self.assertEqual(snapshot.name, "test_name")
        self.assertEqual(snapshot.type, "state")
        self.assertEqual(snapshot.duration, 0)


class TestPlay(QiskitTestCase):
    """Play tests."""

    def test_play(self):
        """Test basic play instruction."""
        duration = 64
        pulse = pulse_lib.SamplePulse([1.0] * duration, name='test')
        play = Play(pulse, DriveChannel(1))
        self.assertEqual(play.name, pulse.name)
        self.assertEqual(play.duration, duration)
