# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Converter Test."""

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.qobj import PulseQobjInstruction, PulseQobjExperimentConfig
from qiskit.qobj.converters import PulseQobjConverter, LoConfigConverter
from qiskit.pulse.commands import SamplePulse, FrameChange, PersistentValue, Snapshot, Acquire
from qiskit.pulse.channels import (DeviceSpecification, Qubit, AcquireChannel, DriveChannel,
                                   MeasureChannel, RegisterSlot, MemorySlot)
from qiskit.pulse import LoConfig


class TestInstructionConverter(QiskitTestCase):
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

    def test_drive_instruction(self):
        """Test converted qobj from DriveInstruction."""
        converter = PulseQobjConverter(PulseQobjInstruction, meas_level=2)
        command = SamplePulse(np.arange(0, 0.01), name='linear')
        instruction = command(self.device.q[0].drive)

        valid_qobj = PulseQobjInstruction(
            name='linear',
            ch='d0',
            t0=0
        )

        self.assertEqual(converter(0, instruction), valid_qobj)

    def test_frame_change(self):
        """Test converted qobj from FrameChangeInstruction."""
        converter = PulseQobjConverter(PulseQobjInstruction, meas_level=2)
        command = FrameChange(phase=0.1)
        instruction = command(self.device.q[0].drive)

        valid_qobj = PulseQobjInstruction(
            name='fc',
            ch='d0',
            t0=0,
            phase=0.1
        )

        self.assertEqual(converter(0, instruction), valid_qobj)

    def test_persistent_value(self):
        """Test converted qobj from PersistentValueInstruction."""
        converter = PulseQobjConverter(PulseQobjInstruction, meas_level=2)
        command = PersistentValue(value=0.1j)
        instruction = command(self.device.q[0].drive)

        valid_qobj = PulseQobjInstruction(
            name='pv',
            ch='d0',
            t0=0,
            val=0.1j
        )

        self.assertEqual(converter(0, instruction), valid_qobj)

    def test_acquire(self):
        """Test converted qobj from AcquireInstruction."""
        converter = PulseQobjConverter(PulseQobjInstruction, meas_level=2)
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

        self.assertEqual(converter(0, instruction), valid_qobj)

    def test_snapshot(self):
        """Test converted qobj from SnapShot."""
        converter = PulseQobjConverter(PulseQobjInstruction, meas_level=2)
        instruction = Snapshot(label='label', snap_type='type')

        valid_qobj = PulseQobjInstruction(
            name='snapshot',
            t0=0,
            label='label',
            type='type'
        )

        self.assertEqual(converter(0, instruction), valid_qobj)


class TestLoConverter(QiskitTestCase):
    """LO converter tests."""

    def setUp(self):
        self.device = DeviceSpecification(
            qubits=[
                Qubit(0,
                      drive_channels=[DriveChannel(0, 1.2)],
                      measure_channels=[MeasureChannel(0, 3.4)],
                      acquire_channels=[AcquireChannel(0)])
            ],
            registers=[
                RegisterSlot(0)
            ],
            mem_slots=[
                MemorySlot(0)
            ]
        )

    def test_qubit_los(self):
        """Test qubit channel configuration."""
        user_lo_config = LoConfig({self.device.q[0].drive: 1.3})
        converter = LoConfigConverter(PulseQobjExperimentConfig,
                                      [1.2], [3.4])

        valid_qobj = PulseQobjExperimentConfig(
            qubit_lo_freq=[1.3]
        )

        self.assertEqual(converter(0, user_lo_config), valid_qobj)

    def test_meas_los(self):
        """Test measurement channel configuration."""
        user_lo_config = LoConfig({self.device.q[0].measure: 3.5})
        converter = LoConfigConverter(PulseQobjExperimentConfig,
                                      [1.2], [3.4])

        valid_qobj = PulseQobjExperimentConfig(
            meas_lo_freq=[3.5]
        )

        self.assertEqual(converter(0, user_lo_config), valid_qobj)
