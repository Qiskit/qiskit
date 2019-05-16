# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,unexpected-keyword-arg

"""Test cases for the pulse command group."""

import unittest
import numpy as np

from qiskit.pulse import (Acquire, FrameChange, PersistentValue,
                          Snapshot, Kernel, Discriminator, functional_pulse)
from qiskit.test import QiskitTestCase


class TestAcquire(QiskitTestCase):
    """Acquisition tests."""

    def test_can_construct_valid_acquire_command(self):
        """Test if valid acquire command can be constructed.
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

    def test_can_construct_acquire_command_with_default_values(self):
        """Test if an acquire command can be constructed with default discriminator and kernel.
        """
        acq_command = Acquire(duration=10)

        self.assertEqual(acq_command.duration, 10)
        self.assertEqual(acq_command.discriminator, None)
        self.assertEqual(acq_command.kernel, None)


class TestFrameChange(QiskitTestCase):
    """FrameChange tests."""

    def test_default(self):
        """Test default frame change.
        """
        fc_command = FrameChange(phase=1.57 - 0.785j)

        self.assertEqual(fc_command.phase, 1.57-0.785j)
        self.assertEqual(fc_command.duration, 0)


class TestFunctionalPulse(QiskitTestCase):
    """SamplePulse tests."""

    def test_gaussian(self):
        """Test gaussian pulse.
        """

        @functional_pulse
        def gaussian(duration, amp, t0, sig):
            x = np.linspace(0, duration - 1, duration)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)

        pulse_command = gaussian(duration=10, name='test_pulse', amp=1, t0=5, sig=1)
        _y = 1 * np.exp(-(np.linspace(0, 9, 10) - 5)**2 / 1**2)

        self.assertListEqual(list(pulse_command.samples), list(_y))

        # check name
        self.assertEqual(pulse_command.name, 'test_pulse')

        # check duration
        self.assertEqual(pulse_command.duration, 10)


class TestPersistentValue(QiskitTestCase):
    """PersistentValue tests."""

    def test_default(self):
        """Test default persistent value.
        """
        pv_command = PersistentValue(value=0.5 - 0.5j)

        self.assertEqual(pv_command.value, 0.5-0.5j)
        self.assertEqual(pv_command.duration, 0)


class TestSnapshot(QiskitTestCase):
    """Snapshot tests."""

    def test_default(self):
        """Test default snapshot.
        """
        snap_command = Snapshot(name='test_name', snap_type='state')

        self.assertEqual(snap_command.name, "test_name")
        self.assertEqual(snap_command.type, "state")
        self.assertEqual(snap_command.duration, 0)


class TestKernel(QiskitTestCase):
    """Kernel tests."""

    def test_can_construct_kernel_with_default_values(self):
        """Test if Kernel can be constructed with default name and params.
        """
        kernel = Kernel()

        self.assertEqual(kernel.name, None)
        self.assertEqual(kernel.params, {})


if __name__ == '__main__':
    unittest.main()
