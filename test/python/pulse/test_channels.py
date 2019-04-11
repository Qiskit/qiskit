# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test cases for the pulse channel group."""

import unittest

from qiskit.pulse.channels import AcquireChannel, MemorySlot, RegisterSlot, SnapshotChannel
from qiskit.pulse.channels import DeviceSpecification, Qubit
from qiskit.pulse.channels import DriveChannel, ControlChannel, MeasureChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.test import QiskitTestCase


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
        qubit = Qubit(1,
                      drive_channels=[DriveChannel(2, 1.2)],
                      control_channels=[ControlChannel(3)],
                      measure_channels=[MeasureChannel(4)],
                      acquire_channels=[AcquireChannel(5)])

        self.assertEqual(qubit.drive, DriveChannel(2, 1.2))
        self.assertEqual(qubit.control, ControlChannel(3))
        self.assertEqual(qubit.measure, MeasureChannel(4))
        self.assertEqual(qubit.acquire, AcquireChannel(5))


class TestDeviceSpecification(QiskitTestCase):
    """DeviceSpecification tests."""

    def test_default(self):
        """Test default device specification.
        """
        qubits = [
            Qubit(0, drive_channels=[DriveChannel(0, 1.2)], acquire_channels=[AcquireChannel(0)]),
            Qubit(1, drive_channels=[DriveChannel(1, 3.4)], acquire_channels=[AcquireChannel(1)])
        ]
        registers = [RegisterSlot(i) for i in range(2)]
        mem_slots = [MemorySlot(i) for i in range(2)]
        spec = DeviceSpecification(qubits, registers, mem_slots)

        self.assertEqual(spec.q[0].drive, DriveChannel(0, 1.2))
        self.assertEqual(spec.q[1].acquire, AcquireChannel(1))
        self.assertEqual(spec.mem[0], MemorySlot(0))
        self.assertEqual(spec.c[1], RegisterSlot(1))

    def test_creation_from_backend_with_zero_u_channels(self):
        """Test creation of device specification from backend with u_channels == 0.
        """

        class DummyBackend:
            """Dummy backend"""
            def configuration(self):
                # pylint: disable=missing-docstring
                class DummyConfig:
                    @property
                    def n_qubits(self):
                        return 2

                    @property
                    def n_registers(self):
                        return 2

                    @property
                    def n_uchannels(self):
                        return 0

                    @property
                    def defaults(self):
                        return {'qubit_freq_est': [1.2, 3.4],
                                'meas_freq_est': [1.2, 3.4]}

                return DummyConfig()

        device = DeviceSpecification.create_from(DummyBackend())

        self.assertEqual(device.q[0].drive, DriveChannel(0, 1.2))
        with self.assertRaises(PulseError):
            _ = device.q[0].control


if __name__ == '__main__':
    unittest.main()
