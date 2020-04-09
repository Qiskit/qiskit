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

"""Test cases for the pulse command group."""

import unittest

from qiskit.pulse import (Acquire, FrameChange, PersistentValue, Snapshot, Kernel, Discriminator,
                          Delay)
from qiskit.test import QiskitTestCase


class TestAcquire(QiskitTestCase):
    """Acquisition tests."""

    def test_can_construct_valid_acquire_command(self):
        """Test if valid acquire command can be constructed."""
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

        with self.assertWarns(DeprecationWarning):
            acq = Acquire(duration=10, kernel=kernel, discriminator=discriminator)

        self.assertEqual(acq.duration, 10)
        self.assertEqual(acq.discriminator.name, 'linear_discriminator')
        self.assertEqual(acq.discriminator.params, discriminator_opts)
        self.assertEqual(acq.kernel.name, 'boxcar')
        self.assertEqual(acq.kernel.params, kernel_opts)

    def test_can_construct_acquire_command_with_default_values(self):
        """Test if an acquire command can be constructed with default discriminator and kernel.
        """
        with self.assertWarns(DeprecationWarning):
            acq_a = Acquire(duration=10)

        self.assertEqual(acq_a.duration, 10)
        self.assertEqual(acq_a.discriminator, None)
        self.assertEqual(acq_a.kernel, None)
        self.assertEqual(acq_a.name, None)


class TestFrameChangeCommand(QiskitTestCase):
    """FrameChange tests. Deprecated."""

    def test_default(self):
        """Test default frame change.
        """
        with self.assertWarns(DeprecationWarning):
            fc_command = FrameChange(phase=1.57)

        self.assertEqual(fc_command.phase, 1.57)
        self.assertEqual(fc_command.duration, 0)
        self.assertTrue(fc_command.name.startswith('fc'))


class TestPersistentValueCommand(QiskitTestCase):
    """PersistentValue tests."""

    def test_default(self):
        """Test default persistent value.
        """
        with self.assertWarns(DeprecationWarning):
            pv_command = PersistentValue(value=0.5 - 0.5j)

        self.assertEqual(pv_command.value, 0.5-0.5j)
        self.assertEqual(pv_command.duration, 0)
        self.assertTrue(pv_command.name.startswith('pv'))


class TestSnapshotCommand(QiskitTestCase):
    """Snapshot tests."""

    def test_default(self):
        """Test default snapshot.
        """
        snap_command = Snapshot(label='test_name', snapshot_type='state')

        self.assertEqual(snap_command.name, "test_name")
        self.assertEqual(snap_command.type, "state")
        self.assertEqual(snap_command.duration, 0)


class TestDelayCommand(QiskitTestCase):
    """Delay tests."""

    def test_delay(self):
        """Test delay."""
        with self.assertWarns(DeprecationWarning):
            delay_command = Delay(10, name='test_name')

        self.assertEqual(delay_command.name, "test_name")
        self.assertEqual(delay_command.duration, 10)


class TestKernel(QiskitTestCase):
    """Kernel tests."""

    def test_can_construct_kernel_with_default_values(self):
        """Test if Kernel can be constructed with default name and params."""
        kernel = Kernel()

        self.assertEqual(kernel.name, None)
        self.assertEqual(kernel.params, {})


class TestDiscriminator(QiskitTestCase):
    """Discriminator tests."""

    def test_can_construct_discriminator_with_default_values(self):
        """Test if Discriminator can be constructed with default name and params."""
        discriminator = Discriminator()

        self.assertEqual(discriminator.name, None)
        self.assertEqual(discriminator.params, {})


if __name__ == '__main__':
    unittest.main()
