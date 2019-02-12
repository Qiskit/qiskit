# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Test cases for the pulse command group."""

import unittest
import numpy as np

from qiskit.pulse.commands import (Acquire,
                                   FrameChange,
                                   FunctionalPulse,
                                   PersistentValue,
                                   Snapshot)
from qiskit.test import QiskitTestCase


class TestAcquire(QiskitTestCase):
    """Acquisition tests."""

    def test_default(self):
        """Test default discriminator and kernel
        """
        acq_comm = Acquire(qubit=0, memory_slot=0, register_slot=0)

        self.assertEqual(acq_comm.qubit, 0)
        self.assertEqual(acq_comm.memory_slot, 0)
        self.assertEqual(acq_comm.register_slot, 0)
        self.assertEqual(acq_comm.discriminator.name, "max_1Q_fidelity")
        self.assertEqual(acq_comm.discriminator.params, [0, 0])
        self.assertEqual(acq_comm.kernel.name, "boxcar")
        self.assertEqual(acq_comm.kernel.params, [])
        self.assertEqual(acq_comm.duration, 0)


class TestFrameChange(QiskitTestCase):
    """FrameChange tests."""

    def test_default(self):
        """Test default frame change
        """
        fc_comm = FrameChange(phase=1.57-0.785j)

        self.assertEqual(fc_comm.phase, 1.57-0.785j)
        self.assertEqual(fc_comm.duration, 0)


class TestFunctionalPulse(QiskitTestCase):
    """FunctionalPulse tests."""

    def test_gaussian(self):
        """Test gaussian pulse
        """

        @FunctionalPulse
        def gaussian(duration, amp, t0, sig):
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)

        pulse_instance = gaussian(10, amp=1, t0=5, sig=1)
        _y = 1 * np.exp(-(np.linspace(0, 9, 10) - 5)**2 / 1**2)

        self.assertListEqual(list(pulse_instance.sample), list(_y))

        # Parameter update (complex pulse)
        pulse_instance.params = {'amp': 0.5-0.5j}
        _y = (0.5-0.5j) * np.exp(-(np.linspace(0, 9, 10) - 5)**2 / 1**2)

        self.assertListEqual(list(pulse_instance.sample), list(_y))

        # check duration
        self.assertEqual(pulse_instance.duration, 10)

    def test_square(self):
        """Test square pulse
        """

        @FunctionalPulse
        def square(duration, amp, t0, t1):
            x = np.linspace(0, duration - 1, duration)
            return np.where((x >= t0) & (x <= t1), amp, 0)

        pulse_instance = square(10, amp=1, t0=2, t1=8)
        _y = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0]

        self.assertListEqual(list(pulse_instance.sample), list(_y))

        # Parameter update (complex pulse)
        pulse_instance.params = {'amp': 0.5-0.5j}
        _y = [0, 0, 0.5 - 0.5j, 0.5 - 0.5j, 0.5 - 0.5j,
              0.5 - 0.5j, 0.5 - 0.5j, 0.5 - 0.5j, 0.5 - 0.5j, 0]

        self.assertEqual(list(pulse_instance.sample), list(_y))

        # check duration
        self.assertEqual(pulse_instance.duration, 10)


class TestPersistentValue(QiskitTestCase):
    """PersistentValue tests."""

    def test_default(self):
        """Test default persistent value
        """
        pv_comm = PersistentValue(value=0.5-0.5j)

        self.assertEqual(pv_comm.value, 0.5-0.5j)
        self.assertEqual(pv_comm.duration, 0)


class TestSnapshot(QiskitTestCase):
    """Snapshot tests."""

    def test_default(self):
        """Test default snapshot
        """
        snap_comm = Snapshot(label='test_label')

        self.assertEqual(snap_comm.label, "test_label")
        self.assertEqual(snap_comm.duration, 0)


if __name__ == '__main__':
    unittest.main()
