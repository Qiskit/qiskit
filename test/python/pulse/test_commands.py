# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,unexpected-keyword-arg

"""Test cases for the pulse command group."""

import unittest
import numpy as np

from qiskit.pulse import (Acquire, FrameChange, PersistentValue,
                          Snapshot, Kernel, Discriminator, function)
from qiskit.test import QiskitTestCase


class TestAcquire(QiskitTestCase):
    """Acquisition tests."""

    def test_default(self):
        """Test default discriminator and kernel.
        """

        kernel_opts = {
            'start_window': 0,
            'stop_window': 10
        }
        kernel = Kernel(name='boxcar', **kernel_opts)

        discriminator_opts = {
            'neighborhoods': [{'qubits': 1, 'channels': 1}],
            'cal': 'coloring',
            'resample': False
        }
        discriminator = Discriminator(name='linear_discriminator', **discriminator_opts)

        acq_command = Acquire(duration=10, kernel=kernel, discriminator=discriminator)

        self.assertEqual(acq_command.duration, 10)
        self.assertEqual(acq_command.discriminator.name, 'linear_discriminator')
        self.assertEqual(acq_command.discriminator.params, discriminator_opts)
        self.assertEqual(acq_command.kernel.name, 'boxcar')
        self.assertEqual(acq_command.kernel.params, kernel_opts)


class TestFrameChange(QiskitTestCase):
    """FrameChange tests."""

    def test_default(self):
        """Test default frame change.
        """
        fc_command = FrameChange(phase=1.57-0.785j)

        self.assertEqual(fc_command.phase, 1.57-0.785j)
        self.assertEqual(fc_command.duration, 0)


class TestFunction(QiskitTestCase):
    """SamplePulse tests."""

    def test_gaussian(self):
        """Test gaussian pulse.
        """

        @function
        def gaussian(duration, amp, t0, sig):
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)

        pulse_command = gaussian(name='gaussian', duration=10, amp=1, t0=5, sig=1)
        _y = 1 * np.exp(-(np.linspace(0, 9, 10) - 5)**2 / 1**2)

        self.assertListEqual(list(pulse_command.samples), list(_y))

        # check duration
        self.assertEqual(pulse_command.duration, 10)

        # check name
        self.assertEqual(pulse_command.name, 'gaussian')


class TestPersistentValue(QiskitTestCase):
    """PersistentValue tests."""

    def test_default(self):
        """Test default persistent value.
        """
        pv_command = PersistentValue(value=0.5-0.5j)

        self.assertEqual(pv_command.value, 0.5-0.5j)
        self.assertEqual(pv_command.duration, 0)


class TestSnapshot(QiskitTestCase):
    """Snapshot tests."""

    def test_default(self):
        """Test default snapshot.
        """
        snap_command = Snapshot(label='test_label', snap_type='state')

        self.assertEqual(snap_command.label, "test_label")
        self.assertEqual(snap_command.type, "state")
        self.assertEqual(snap_command.duration, 0)


if __name__ == '__main__':
    unittest.main()
