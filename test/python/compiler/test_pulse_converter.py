# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Converter Test."""

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.qobj import PulseQobjInstruction
from qiskit.qobj.converters.pulse_instruction import PulseQobjConverter
from qiskit.pulse.commands import SamplePulse, FrameChange, PersistentValue, Snapshot, Acquire
from qiskit.pulse.channels import (DeviceSpecification, Qubit, AcquireChannel, DriveChannel,
                                   RegisterSlot, MemorySlot)


class TestPulseConverter(QiskitTestCase):
    """Pulse converter tests."""

    def setUp(self):
        self.device = DeviceSpecification(
            qubits=[
                Qubit(0,
                      drive_channels=[DriveChannel(0, 1.2)],
                      acquire_channels=[AcquireChannel(0)])
            ],
            registers=[
                RegisterSlot(0)
            ],
            mem_slots=[
                MemorySlot(0)
            ]
        )

        self.converter = PulseQobjConverter(PulseQobjInstruction, meas_level=2)

    def test_drive_instruction(self):
        """Test converted qobj from DriveInstruction."""
        command = SamplePulse(np.arange(0, 0.01), name='linear')
        instruction = command(self.device.q[0].drive)

        valid_qobj = PulseQobjInstruction(
            name='linear',
            ch='d0',
            t0=0
        )

        self.assertEqual(self.converter(instruction), valid_qobj)

    def test_frame_change(self):
        """Test converted qobj from FrameChangeInstruction."""
        command = FrameChange(phase=0.1)
        instruction = command(self.device.q[0].drive)

        valid_qobj = PulseQobjInstruction(
            name='fc',
            ch='d0',
            t0=0,
            phase=0.1
        )

        self.assertEqual(self.converter(instruction), valid_qobj)

    def test_persistent_value(self):
        """Test converted qobj from PersistentValueInstruction."""
        command = PersistentValue(value=0.1j)
        instruction = command(self.device.q[0].drive)

        valid_qobj = PulseQobjInstruction(
            name='pv',
            ch='d0',
            t0=0,
            val=0.1j
        )

        self.assertEqual(self.converter(instruction), valid_qobj)

    def test_acquire(self):
        """Test converted qobj from AcquireInstruction."""
        command = Acquire(duration=10)
        instruction = command(self.device.q, self.device.mem, self.device.c)

        valid_qobj = PulseQobjInstruction(
            name='acquire',
            t0=0,
            duration=10,
            qubits=[0],
            memory_slot=[0],
            register_slot=[0]
        )

        self.assertEqual(self.converter(instruction), valid_qobj)

    def test_snapshot(self):
        """Test converted qobj from SnapShot."""
        instruction = Snapshot(label='label', snap_type='type')

        valid_qobj = PulseQobjInstruction(
            name='snapshot',
            t0=0,
            label='label',
            type='type'
        )

        self.assertEqual(self.converter(instruction), valid_qobj)
