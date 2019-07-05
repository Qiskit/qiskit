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

"""Converter Test."""

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.qobj import (PulseQobjInstruction, PulseQobjExperimentConfig, PulseLibraryItem,
                         QobjMeasurementOption)
from qiskit.qobj.converters import (InstructionToQobjConverter, QobjToInstructionConverter,
                                    LoConfigConverter)
from qiskit.pulse.commands import (SamplePulse, FrameChange, PersistentValue, Snapshot, Acquire,
                                   Discriminator, Kernel)
from qiskit.pulse.channels import (DeviceSpecification, PulseChannelSpec, Qubit, AcquireChannel,
                                   DriveChannel, ControlChannel, MeasureChannel, RegisterSlot,
                                   MemorySlot)
from qiskit.pulse.schedule import ParameterizedSchedule
from qiskit.pulse import LoConfig


class TestInstructionToQobjConverter(QiskitTestCase):
    """Pulse converter tests."""

    def setUp(self):
        self.device = PulseChannelSpec(n_qubits=1, n_control=0, n_registers=1)

    def test_drive_instruction(self):
        """Test converted qobj from PulseInstruction."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        command = SamplePulse(np.arange(0, 0.01), name='linear')
        instruction = command(self.device.drives[0])

        valid_qobj = PulseQobjInstruction(
            name='linear',
            ch='d0',
            t0=0
        )

        self.assertEqual(converter(0, instruction), valid_qobj)

    def test_frame_change(self):
        """Test converted qobj from FrameChangeInstruction."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        command = FrameChange(phase=0.1)
        instruction = command(self.device.drives[0])

        valid_qobj = PulseQobjInstruction(
            name='fc',
            ch='d0',
            t0=0,
            phase=0.1
        )

        self.assertEqual(converter(0, instruction), valid_qobj)

    def test_persistent_value(self):
        """Test converted qobj from PersistentValueInstruction."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        command = PersistentValue(value=0.1j)
        instruction = command(self.device.drives[0])

        valid_qobj = PulseQobjInstruction(
            name='pv',
            ch='d0',
            t0=0,
            val=0.1j
        )

        self.assertEqual(converter(0, instruction), valid_qobj)

    def test_acquire(self):
        """Test converted qobj from AcquireInstruction."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        command = Acquire(duration=10)
        instruction = command(self.device.acquires,
                              self.device.memoryslots,
                              self.device.registers)

        valid_qobj = PulseQobjInstruction(
            name='acquire',
            t0=0,
            duration=10,
            qubits=[0],
            memory_slot=[0],
            register_slot=[0]
        )

        self.assertEqual(converter(0, instruction), valid_qobj)

        # test without register
        instruction = command(self.device.acquires, self.device.memoryslots)

        valid_qobj = PulseQobjInstruction(
            name='acquire',
            t0=0,
            duration=10,
            qubits=[0],
            memory_slot=[0]
        )

        self.assertEqual(converter(0, instruction), valid_qobj)

    def test_snapshot(self):
        """Test converted qobj from SnapShot."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        instruction = Snapshot(label='label', snapshot_type='type')

        valid_qobj = PulseQobjInstruction(
            name='snapshot',
            t0=0,
            label='label',
            type='type'
        )

        self.assertEqual(converter(0, instruction), valid_qobj)


class TestQobjToInstructionConverter(QiskitTestCase):
    """Pulse converter tests."""

    def setUp(self):
        self.linear = SamplePulse(np.arange(0, 0.01), name='linear')
        self.pulse_library = [PulseLibraryItem(name=self.linear.name,
                                               samples=self.linear.samples.tolist())]

        self.converter = QobjToInstructionConverter(self.pulse_library, buffer=0)

        self.device = PulseChannelSpec(n_qubits=2, n_control=0, n_registers=2)

    def test_drive_instruction(self):
        """Test converted qobj from PulseInstruction."""
        cmd = self.linear
        instruction = cmd(DriveChannel(0)) << 10

        qobj = PulseQobjInstruction(name='linear', ch='d0', t0=10)
        converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.timeslots, instruction.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1].command, cmd)

    def test_frame_change(self):
        """Test converted qobj from FrameChangeInstruction."""
        cmd = FrameChange(phase=0.1)
        instruction = cmd(MeasureChannel(0))

        qobj = PulseQobjInstruction(name='fc', ch='m0', t0=0, phase=0.1)
        converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.timeslots, instruction.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1].command, cmd)

    def test_persistent_value(self):
        """Test converted qobj from PersistentValueInstruction."""
        cmd = PersistentValue(value=0.1j)
        instruction = cmd(ControlChannel(1))

        qobj = PulseQobjInstruction(name='pv', ch='u1', t0=0, val=0.1j)
        converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.timeslots, instruction.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1].command, cmd)

    def test_acquire(self):
        """Test converted qobj from AcquireInstruction."""
        cmd = Acquire(10, Discriminator(name='test_disc', params={'test_params': 1.0}),
                      Kernel(name='test_kern', params={'test_params': 'test'}))
        instruction = cmd(self.device.acquires,
                          self.device.memoryslots,
                          self.device.registers)

        qobj = PulseQobjInstruction(name='acquire', t0=0, duration=10, qubits=[0, 1],
                                    memory_slot=[0, 1], register_slot=[0, 1],
                                    kernels=[QobjMeasurementOption(
                                        name='test_kern', params={'test_params': 'test'})],
                                    discriminators=[QobjMeasurementOption(
                                        name='test_disc', params={'test_params': 1.0})])
        converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.timeslots, instruction.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1].command, cmd)

    def test_snapshot(self):
        """Test converted qobj from SnapShot."""
        cmd = Snapshot(label='label', snapshot_type='type')
        instruction = cmd << 10

        qobj = PulseQobjInstruction(name='snapshot', t0=10, label='label', type='type')
        converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.timeslots, instruction.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1], cmd)

    def test_parameterized_frame_change(self):
        """Test converted qobj from FrameChangeInstruction."""
        cmd = FrameChange(phase=4.)
        instruction = cmd(MeasureChannel(0)) << 10

        qobj = PulseQobjInstruction(name='fc', ch='m0', t0=10, phase='P1**2')
        converted_instruction = self.converter(qobj)

        self.assertIsInstance(converted_instruction, ParameterizedSchedule)

        evaluated_instruction = converted_instruction.bind_parameters(2.)

        self.assertEqual(evaluated_instruction.timeslots, instruction.timeslots)
        self.assertEqual(evaluated_instruction.instructions[0][-1].command, cmd)

    def test_parameterized_persistent_value(self):
        """Test converted qobj from PersistentValueInstruction."""
        cmd = PersistentValue(value=0.5+0.j)
        instruction = cmd(ControlChannel(1)) << 10

        qobj = PulseQobjInstruction(name='pv', ch='u1', t0=10, val='P1*cos(np.pi*P2)')
        converted_instruction = self.converter(qobj)

        self.assertIsInstance(converted_instruction, ParameterizedSchedule)

        evaluated_instruction = converted_instruction.bind_parameters(P1=0.5, P2=0.)

        self.assertEqual(evaluated_instruction.timeslots, instruction.timeslots)
        self.assertEqual(evaluated_instruction.instructions[0][-1].command, cmd)


class TestLoConverter(QiskitTestCase):
    """LO converter tests."""

    def setUp(self):
        self.device = PulseChannelSpec(n_qubits=1, n_control=0, n_registers=1)

    def test_qubit_los(self):
        """Test qubit channel configuration."""
        user_lo_config = LoConfig({self.device.drives[0]: 1.3})
        converter = LoConfigConverter(PulseQobjExperimentConfig,
                                      [1.2], [3.4], [(0., 5.)], [(0., 5.)])

        valid_qobj = PulseQobjExperimentConfig(qubit_lo_freq=[1.3])

        self.assertEqual(converter(user_lo_config), valid_qobj)

    def test_meas_los(self):
        """Test measurement channel configuration."""
        user_lo_config = LoConfig({self.device.measures[0]: 3.5})
        converter = LoConfigConverter(PulseQobjExperimentConfig,
                                      [1.2], [3.4], [(0., 5.)], [(0., 5.)])

        valid_qobj = PulseQobjExperimentConfig(meas_lo_freq=[3.5])

        self.assertEqual(converter(user_lo_config), valid_qobj)


class TestInstructionToQobjConverterWithDeviceSpecification(QiskitTestCase):
    """Pulse converter tests."""
    # TODO: This test will be deprecated in future update.

    def setUp(self):
        self.device = DeviceSpecification(
            qubits=[
                Qubit(0, DriveChannel(0), MeasureChannel(0), AcquireChannel(0))
            ],
            registers=[
                RegisterSlot(0)
            ],
            mem_slots=[
                MemorySlot(0)
            ]
        )

    def test_drive_instruction(self):
        """Test converted qobj from PulseInstruction."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
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
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
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
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
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
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
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

        # test without register
        instruction = command(self.device.q, self.device.mem)

        valid_qobj = PulseQobjInstruction(
            name='acquire',
            t0=0,
            duration=10,
            qubits=[0],
            memory_slot=[0]
        )

        self.assertEqual(converter(0, instruction), valid_qobj)

    def test_snapshot(self):
        """Test converted qobj from SnapShot."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        instruction = Snapshot(label='label', snapshot_type='type', name='snapshot')

        valid_qobj = PulseQobjInstruction(
            name='snapshot',
            t0=0,
            label='label',
            type='type'
        )

        self.assertEqual(converter(0, instruction), valid_qobj)


class TestQobjToInstructionConverterWithDeviceSpecification(QiskitTestCase):
    """Pulse converter tests."""
    # TODO: This test will be deprecated in future update.

    def setUp(self):
        self.linear = SamplePulse(np.arange(0, 0.01), name='linear')
        self.pulse_library = [PulseLibraryItem(name=self.linear.name,
                                               samples=self.linear.samples.tolist())]

        self.converter = QobjToInstructionConverter(self.pulse_library, buffer=0)

        self.device = DeviceSpecification(
            qubits=[
                Qubit(0, DriveChannel(0), MeasureChannel(0), AcquireChannel(0)),
                Qubit(1, DriveChannel(1), MeasureChannel(1), AcquireChannel(1)),
            ],
            registers=[
                RegisterSlot(0),
                RegisterSlot(1)
            ],
            mem_slots=[
                MemorySlot(0),
                MemorySlot(1)
            ]
        )

    def test_drive_instruction(self):
        """Test converted qobj from PulseInstruction."""
        cmd = self.linear
        instruction = cmd(DriveChannel(0)) << 10

        qobj = PulseQobjInstruction(name='linear', ch='d0', t0=10)
        converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.timeslots, instruction.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1].command, cmd)

    def test_frame_change(self):
        """Test converted qobj from FrameChangeInstruction."""
        cmd = FrameChange(phase=0.1)
        instruction = cmd(MeasureChannel(0))

        qobj = PulseQobjInstruction(name='fc', ch='m0', t0=0, phase=0.1)
        converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.timeslots, instruction.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1].command, cmd)

    def test_persistent_value(self):
        """Test converted qobj from PersistentValueInstruction."""
        cmd = PersistentValue(value=0.1j)
        instruction = cmd(ControlChannel(1))

        qobj = PulseQobjInstruction(name='pv', ch='u1', t0=0, val=0.1j)
        converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.timeslots, instruction.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1].command, cmd)

    def test_acquire(self):
        """Test converted qobj from AcquireInstruction."""
        cmd = Acquire(10, Discriminator(name='test_disc', params={'test_params': 1.0}),
                      Kernel(name='test_kern', params={'test_params': 'test'}))
        instruction = cmd(self.device.q, self.device.mem, self.device.c)

        qobj = PulseQobjInstruction(name='acquire', t0=0, duration=10, qubits=[0, 1],
                                    memory_slot=[0, 1], register_slot=[0, 1],
                                    kernels=[QobjMeasurementOption(
                                        name='test_kern', params={'test_params': 'test'})],
                                    discriminators=[QobjMeasurementOption(
                                        name='test_disc', params={'test_params': 1.0})])
        converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.timeslots, instruction.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1].command, cmd)

    def test_snapshot(self):
        """Test converted qobj from SnapShot."""
        cmd = Snapshot(label='label', snapshot_type='type', name='snapshot')
        instruction = cmd << 10

        qobj = PulseQobjInstruction(name='snapshot', t0=10, label='label', type='type')
        converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.timeslots, instruction.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1], cmd)

    def test_parameterized_frame_change(self):
        """Test converted qobj from FrameChangeInstruction."""
        cmd = FrameChange(phase=4.)
        instruction = cmd(MeasureChannel(0)) << 10

        qobj = PulseQobjInstruction(name='fc', ch='m0', t0=10, phase='P1**2')
        converted_instruction = self.converter(qobj)

        self.assertIsInstance(converted_instruction, ParameterizedSchedule)

        evaluated_instruction = converted_instruction.bind_parameters(2.)

        self.assertEqual(evaluated_instruction.timeslots, instruction.timeslots)
        self.assertEqual(evaluated_instruction.instructions[0][-1].command, cmd)

    def test_parameterized_persistent_value(self):
        """Test converted qobj from PersistentValueInstruction."""
        cmd = PersistentValue(value=0.5+0.j)
        instruction = cmd(ControlChannel(1)) << 10

        qobj = PulseQobjInstruction(name='pv', ch='u1', t0=10, val='P1*cos(np.pi*P2)')
        converted_instruction = self.converter(qobj)

        self.assertIsInstance(converted_instruction, ParameterizedSchedule)

        evaluated_instruction = converted_instruction.bind_parameters(P1=0.5, P2=0.)

        self.assertEqual(evaluated_instruction.timeslots, instruction.timeslots)
        self.assertEqual(evaluated_instruction.instructions[0][-1].command, cmd)


class TestLoConverterWithDeviceSpecification(QiskitTestCase):
    """LO converter tests."""
    # TODO: This test will be deprecated in future update.

    def setUp(self):
        self.device = DeviceSpecification(
            qubits=[
                Qubit(0, DriveChannel(0), MeasureChannel(0), AcquireChannel(0))
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
                                      [1.2], [3.4], [(0., 5.)], [(0., 5.)])

        valid_qobj = PulseQobjExperimentConfig(qubit_lo_freq=[1.3])

        self.assertEqual(converter(user_lo_config), valid_qobj)

    def test_meas_los(self):
        """Test measurement channel configuration."""
        user_lo_config = LoConfig({self.device.q[0].measure: 3.5})
        converter = LoConfigConverter(PulseQobjExperimentConfig,
                                      [1.2], [3.4], [(0., 5.)], [(0., 5.)])

        valid_qobj = PulseQobjExperimentConfig(meas_lo_freq=[3.5])

        self.assertEqual(converter(user_lo_config), valid_qobj)
