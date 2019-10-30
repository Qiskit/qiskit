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

"""Test cases for the pulse channel group."""

import unittest

from qiskit.pulse.channels import AcquireChannel, MemorySlot, RegisterSlot, SnapshotChannel
from qiskit.pulse.channels import PulseChannelSpec, Qubit
from qiskit.pulse.channels import DriveChannel, ControlChannel, MeasureChannel
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q


class TestAcquireChannel(QiskitTestCase):
    """AcquireChannel tests."""

    def test_default(self):
        """Test default acquire channel.
        """
        acquire_channel = AcquireChannel(123)

        self.assertEqual(acquire_channel.index, 123)
        self.assertEqual(acquire_channel.name, 'a123')


class TestMemorySlot(QiskitTestCase):
    """AcquireChannel tests."""

    def test_default(self):
        """Test default memory slot.
        """
        memory_slot = MemorySlot(123)

        self.assertEqual(memory_slot.index, 123)
        self.assertEqual(memory_slot.name, 'm123')


class TestRegisterSlot(QiskitTestCase):
    """RegisterSlot tests."""

    def test_default(self):
        """Test default register slot.
        """
        register_slot = RegisterSlot(123)

        self.assertEqual(register_slot.index, 123)
        self.assertEqual(register_slot.name, 'c123')


class TestSnapshotChannel(QiskitTestCase):
    """SnapshotChannel tests."""

    def test_default(self):
        """Test default snapshot channel.
        """
        snapshot_channel = SnapshotChannel()

        self.assertEqual(snapshot_channel.index, 0)
        self.assertEqual(snapshot_channel.name, 's0')


class TestDriveChannel(QiskitTestCase):
    """DriveChannel tests."""

    def test_default(self):
        """Test default drive channel.
        """
        drive_channel = DriveChannel(123)

        self.assertEqual(drive_channel.index, 123)
        self.assertEqual(drive_channel.name, 'd123')


class TestControlChannel(QiskitTestCase):
    """ControlChannel tests."""

    def test_default(self):
        """Test default control channel.
        """
        control_channel = ControlChannel(123)

        self.assertEqual(control_channel.index, 123)
        self.assertEqual(control_channel.name, 'u123')


class TestMeasureChannel(QiskitTestCase):
    """MeasureChannel tests."""

    def test_default(self):
        """Test default measure channel.
        """
        measure_channel = MeasureChannel(123)

        self.assertEqual(measure_channel.index, 123)
        self.assertEqual(measure_channel.name, 'm123')


class TestQubit(QiskitTestCase):
    """Qubit tests."""

    def test_default(self):
        """Test default qubit.
        """
        qubit = Qubit(1, DriveChannel(2), MeasureChannel(4), AcquireChannel(5),
                      control_channels=[ControlChannel(3)])

        self.assertEqual(qubit.drive, DriveChannel(2))
        self.assertEqual(qubit.controls[0], ControlChannel(3))
        self.assertEqual(qubit.measure, MeasureChannel(4))
        self.assertEqual(qubit.acquire, AcquireChannel(5))


class TestPulseSpecification(QiskitTestCase):
    """PulseSpecification tests."""

    def test_default(self):
        """Test default device specification.
        """
        spec = PulseChannelSpec(n_qubits=2, n_control=0, n_registers=2)

        self.assertEqual(spec.drives[0], DriveChannel(0))
        self.assertEqual(spec.acquires[1], AcquireChannel(1))
        self.assertEqual(spec.memoryslots[0], MemorySlot(0))
        self.assertEqual(spec.registers[1], RegisterSlot(1))

    def test_creation_from_backend_with_zero_u_channels(self):
        """Test creation of device specification from backend.
        """
        backend = FakeOpenPulse2Q()

        device = PulseChannelSpec.from_backend(backend)

        self.assertEqual(device.drives[0], DriveChannel(0))
        self.assertEqual(device.controls[0], ControlChannel(0))
        self.assertEqual(device.measures[0], MeasureChannel(0))
        self.assertEqual(device.acquires[0], AcquireChannel(0))
        self.assertEqual(device.registers[0], RegisterSlot(0))
        self.assertEqual(device.memoryslots[0], MemorySlot(0))


if __name__ == '__main__':
    unittest.main()
