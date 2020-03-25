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
                                   Gaussian, GaussianSquare, ConstantPulse, Drag)
from qiskit.pulse.instructions import ShiftPhase, SetFrequency
from qiskit.pulse.channels import (DriveChannel, ControlChannel, MeasureChannel, AcquireChannel,
                                   MemorySlot, RegisterSlot)
from qiskit.pulse.schedule import ParameterizedSchedule, Schedule
from qiskit.pulse import LoConfig, Kernel, Discriminator


class TestInstructionToQobjConverter(QiskitTestCase):
    """Pulse converter tests."""

    def test_drive_instruction(self):
        """Test converted qobj from PulseInstruction."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        command = SamplePulse(np.arange(0, 0.01), name='linear')
        instruction = command(DriveChannel(0))

        valid_qobj = PulseQobjInstruction(
            name='linear',
            ch='d0',
            t0=0
        )

        self.assertEqual(converter(0, instruction), valid_qobj)

    def test_gaussian_pulse_instruction(self):
        """Test that parametric pulses are correctly converted to PulseQobjInstructions."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        instruction = Gaussian(duration=25, sigma=15, amp=-0.5 + 0.2j)(DriveChannel(0))

        valid_qobj = PulseQobjInstruction(
            name='parametric_pulse',
            pulse_shape='gaussian',
            ch='d0',
            t0=0,
            parameters={'duration': 25, 'sigma': 15, 'amp': -0.5 + 0.2j})
        self.assertEqual(converter(0, instruction), valid_qobj)

    def test_gaussian_square_pulse_instruction(self):
        """Test that parametric pulses are correctly converted to PulseQobjInstructions."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        instruction = GaussianSquare(duration=1500, sigma=15,
                                     amp=-0.5 + 0.2j, width=1300)(MeasureChannel(1))

        valid_qobj = PulseQobjInstruction(
            name='parametric_pulse',
            pulse_shape='gaussian_square',
            ch='m1',
            t0=10,
            parameters={'duration': 1500, 'sigma': 15, 'amp': -0.5 + 0.2j, 'width': 1300})
        self.assertEqual(converter(10, instruction), valid_qobj)

    def test_constant_pulse_instruction(self):
        """Test that parametric pulses are correctly converted to PulseQobjInstructions."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        instruction = ConstantPulse(duration=25, amp=1)(ControlChannel(2))

        valid_qobj = PulseQobjInstruction(
            name='parametric_pulse',
            pulse_shape='constant',
            ch='u2',
            t0=20,
            parameters={'duration': 25, 'amp': 1})
        self.assertEqual(converter(20, instruction), valid_qobj)

    def test_drag_pulse_instruction(self):
        """Test that parametric pulses are correctly converted to PulseQobjInstructions."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        instruction = Drag(duration=25, sigma=15, amp=-0.5 + 0.2j, beta=0.5)(DriveChannel(0))

        valid_qobj = PulseQobjInstruction(
            name='parametric_pulse',
            pulse_shape='drag',
            ch='d0',
            t0=30,
            parameters={'duration': 25, 'sigma': 15, 'amp': -0.5 + 0.2j, 'beta': 0.5})
        self.assertEqual(converter(30, instruction), valid_qobj)

    def test_frame_change(self):
        """Test converted qobj from FrameChangeInstruction."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        command = FrameChange(phase=0.1)
        instruction = command(DriveChannel(0))

        valid_qobj = PulseQobjInstruction(
            name='fc',
            ch='d0',
            t0=0,
            phase=0.1
        )

        self.assertEqual(converter(0, instruction), valid_qobj)
        instruction = ShiftPhase(0.1, DriveChannel(0))
        self.assertEqual(converter(0, instruction), valid_qobj)

    def test_set_frequency(self):
        """Test converted qobj from SetFrequencyInstruction."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        instruction = SetFrequency(8.0, DriveChannel(0))

        valid_qobj = PulseQobjInstruction(
            name='sf',
            ch='d0',
            t0=0,
            frequency=8.0
        )

        self.assertEqual(converter(0, instruction), valid_qobj)

    def test_persistent_value(self):
        """Test converted qobj from PersistentValueInstruction."""
        converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        command = PersistentValue(value=0.1j)
        instruction = command(DriveChannel(0))

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
        instruction = command(AcquireChannel(0),
                              MemorySlot(0),
                              RegisterSlot(0))

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
        instruction = command(AcquireChannel(0), MemorySlot(0))

        valid_qobj = PulseQobjInstruction(
            name='acquire',
            t0=0,
            duration=10,
            qubits=[0],
            memory_slot=[0]
        )

        self.assertEqual(converter(0, instruction), valid_qobj)

    def test_snapshot(self):
        """Test converted qobj from Snapshot."""
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
        self.n_qubits = 2

    def test_drive_instruction(self):
        """Test converted qobj from PulseInstruction."""
        instruction = self.linear(DriveChannel(0))

        qobj = PulseQobjInstruction(name='linear', ch='d0', t0=10)
        converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.instructions[0][-1], instruction)

    def test_parametric_pulses(self):
        """Test converted qobj from ParametricInstruction."""
        instruction = Gaussian(duration=25, sigma=15, amp=-0.5 + 0.2j)(DriveChannel(0))
        qobj = PulseQobjInstruction(
            name='parametric_pulse',
            pulse_shape='gaussian',
            ch='d0',
            t0=0,
            parameters={'duration': 25, 'sigma': 15, 'amp': -0.5 + 0.2j})
        converted_instruction = self.converter(qobj)
        self.assertEqual(converted_instruction.timeslots, instruction.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1], instruction)

    def test_frame_change(self):
        """Test converted qobj from FrameChangeInstruction."""
        qobj = PulseQobjInstruction(name='fc', ch='m0', t0=0, phase=0.1)
        converted_instruction = self.converter(qobj)

        instruction = ShiftPhase(0.1, MeasureChannel(0))
        self.assertEqual(converted_instruction.timeslots, instruction.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1], instruction)

    def test_set_frequency(self):
        """Test converted qobj from FrameChangeInstruction."""
        instruction = SetFrequency(8.0, DriveChannel(0))

        qobj = PulseQobjInstruction(name='sf', ch='d0', t0=0, frequency=8.0)
        converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.timeslots, instruction.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1], instruction)
        self.assertTrue('frequency' in qobj.to_dict())

    def test_persistent_value(self):
        """Test converted qobj from PersistentValueInstruction."""
        cmd = PersistentValue(value=0.1j)
        instruction = cmd(ControlChannel(1))

        qobj = PulseQobjInstruction(name='pv', ch='u1', t0=0, val=0.1j)
        converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.timeslots, instruction.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1], instruction)

    def test_acquire(self):
        """Test converted qobj from AcquireInstruction."""
        schedule = Schedule()
        for i in range(self.n_qubits):
            schedule |= Acquire(10, AcquireChannel(i), MemorySlot(i), RegisterSlot(i),
                                kernel=Kernel(name='test_kern', test_params='test'),
                                discriminator=Discriminator(name='test_disc',
                                                            test_params=1.0))

        qobj = PulseQobjInstruction(name='acquire', t0=0, duration=10, qubits=[0, 1],
                                    memory_slot=[0, 1], register_slot=[0, 1],
                                    kernels=[QobjMeasurementOption(
                                        name='test_kern', params={'test_params': 'test'})],
                                    discriminators=[QobjMeasurementOption(
                                        name='test_disc', params={'test_params': 1.0})])
        converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.timeslots, schedule.timeslots)
        self.assertEqual(converted_instruction.instructions[0][-1].duration, 10)
        self.assertEqual(converted_instruction.instructions[0][-1].kernel.params,
                         {'test_params': 'test'})
        self.assertEqual(converted_instruction.instructions[1][-1].channel, AcquireChannel(1))

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
        instruction = ShiftPhase(4., MeasureChannel(0))

        qobj = PulseQobjInstruction(name='fc', ch='m0', t0=10, phase='P1**2')
        converted_instruction = self.converter(qobj)

        self.assertIsInstance(converted_instruction, ParameterizedSchedule)

        evaluated_instruction = converted_instruction.bind_parameters(2.)

        self.assertEqual(evaluated_instruction.timeslots, instruction.timeslots)
        self.assertEqual(evaluated_instruction.instructions[0][-1], instruction)

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

    def test_qubit_los(self):
        """Test qubit channel configuration."""
        user_lo_config = LoConfig({DriveChannel(0): 1.3e9})
        converter = LoConfigConverter(PulseQobjExperimentConfig,
                                      [1.2e9], [3.4e9], [(0., 5e9)], [(0., 5e9)])

        valid_qobj = PulseQobjExperimentConfig(qubit_lo_freq=[1.3])

        self.assertEqual(converter(user_lo_config), valid_qobj)

    def test_meas_los(self):
        """Test measurement channel configuration."""
        user_lo_config = LoConfig({MeasureChannel(0): 3.5e9})
        converter = LoConfigConverter(PulseQobjExperimentConfig,
                                      [1.2e9], [3.4e9], [(0., 5e9)], [(0., 5e9)])

        valid_qobj = PulseQobjExperimentConfig(meas_lo_freq=[3.5])

        self.assertEqual(converter(user_lo_config), valid_qobj)
