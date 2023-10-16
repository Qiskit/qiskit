# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test IR instructions"""

from qiskit.test import QiskitTestCase
from qiskit.pulse import (
    Gaussian,
    Frame,
    GenericFrame,
    QubitFrame,
    MeasurementFrame,
    Qubit,
    PulseError,
    MemorySlot,
)
from qiskit.pulse.ir import (
    GenericInstruction,
    AcquireInstruction,
)


class TestInstructions(QiskitTestCase):
    """Test Instruction base class

    This module tests features which are common to all subclasses."""

    frame = GenericFrame("a")
    logical_element = Qubit(1)
    gaus_pulse = Gaussian(duration=80, amp=0.5, sigma=40)
    waveform = gaus_pulse.get_waveform()

    def test_instruction_creation_invalid_initial_time(self):
        """Test that instructions can't be constructed with invalid initial_time"""

        with self.assertRaises(PulseError):
            GenericInstruction(
                "Play",
                operand=self.gaus_pulse,
                logical_element=self.logical_element,
                frame=self.frame,
                initial_time=-1,
            )
        with self.assertRaises(PulseError):
            GenericInstruction(
                "Delay",
                operand=20,
                logical_element=self.logical_element,
                frame=self.frame,
                initial_time=0.5,
            )

    def test_initial_time_update(self):
        """Test that initial_time update is validated correctly"""
        new_initial_time = 50
        inst = GenericInstruction("Delay", operand=20, logical_element=self.logical_element)
        inst.initial_time = new_initial_time
        self.assertEqual(inst.initial_time, new_initial_time)

        with self.assertRaises(PulseError):
            inst.initial_time = -10
        with self.assertRaises(PulseError):
            inst.initial_time = 0.5

    def test_instruction_final_time(self):
        """Test that final time is returned correctly"""
        initial_time = 10
        duration = 20
        inst = GenericInstruction("Delay", operand=duration, logical_element=self.logical_element)
        inst.initial_time = initial_time
        self.assertEqual(inst.final_time, initial_time + duration)

    def test_instruction_shift_initial_time(self):
        """Test that shifting initial_time is done correctly"""
        initial_time = 10
        shift = 15
        inst = GenericInstruction(
            "Delay", operand=10, logical_element=self.logical_element, initial_time=initial_time
        )
        inst.shift_initial_time(shift)
        self.assertEqual(inst.initial_time, initial_time + shift)

        with self.assertRaises(PulseError):
            inst.shift_initial_time(0.5)

        inst = GenericInstruction("Delay", operand=10, logical_element=self.logical_element)
        # Can't shift un-scheduled instruction
        with self.assertRaises(PulseError):
            inst.shift_initial_time(shift)


class TestGenericInstructions(QiskitTestCase):
    """Test PulseIR generic instructions"""

    frame = Frame("a")
    logical_element = Qubit(1)
    gaus_pulse = Gaussian(duration=80, amp=0.5, sigma=40)
    waveform = gaus_pulse.get_waveform()

    def test_unsupported_instructions(self):
        """Test that unsupported instructions can't be constructed"""

        with self.assertRaises(PulseError):
            GenericInstruction("not-supported", operand=None, logical_element=self.logical_element)

    def test_play_instruction_creation(self):
        """Test that Play instruction is created and validated correctly"""
        inst = GenericInstruction(
            "Play", operand=self.gaus_pulse, logical_element=self.logical_element, frame=self.frame
        )

        self.assertEqual(inst.instruction_type, "Play")
        self.assertEqual(inst.duration, self.gaus_pulse.duration)
        self.assertEqual(inst.frame, self.frame)
        self.assertEqual(inst.logical_element, self.logical_element)
        self.assertEqual(inst.operand, self.gaus_pulse)
        self.assertEqual(inst.initial_time, None)

        inst = GenericInstruction(
            "Play",
            operand=self.waveform,
            logical_element=self.logical_element,
            frame=self.frame,
            initial_time=100,
        )

        self.assertEqual(inst.duration, self.waveform.duration)
        self.assertEqual(inst.initial_time, 100)

        # Play instruction needs both logical element and frame, and pulse type operand.
        with self.assertRaises(PulseError):
            GenericInstruction(
                "Play", operand=self.gaus_pulse, logical_element=self.logical_element
            )
        with self.assertRaises(PulseError):
            GenericInstruction("Play", operand=self.gaus_pulse, frame=self.frame)
        # Operand has to be a pulse
        with self.assertRaises(PulseError):
            GenericInstruction(
                "Play", operand=1, logical_element=self.logical_element, frame=self.frame
            )

    def test_delay_instruction_creation(self):
        """Test that Delay instruction is created and validated correctly"""
        inst = GenericInstruction("Delay", operand=20, logical_element=self.logical_element)
        self.assertEqual(inst.instruction_type, "Delay")
        self.assertEqual(inst.duration, 20)
        self.assertEqual(inst.logical_element, self.logical_element)
        self.assertEqual(inst.operand, 20)
        self.assertEqual(inst.initial_time, None)

        inst = GenericInstruction(
            "Delay",
            operand=20,
            logical_element=self.logical_element,
            frame=self.frame,
            initial_time=100,
        )
        self.assertEqual(inst.frame, self.frame)
        self.assertEqual(inst.initial_time, 100)

        with self.assertRaises(PulseError):
            GenericInstruction("Delay", operand=0.5, logical_element=self.logical_element)
        with self.assertRaises(PulseError):
            GenericInstruction("Delay", operand=-1, logical_element=self.logical_element)
        with self.assertRaises(PulseError):
            GenericInstruction("Delay", operand=10, frame=self.frame)

    def test_frame_instruction_creation(self):
        """Test that frame instruction is created and validated correctly"""
        inst = GenericInstruction("ShiftPhase", operand=100, frame=self.frame)
        self.assertEqual(inst.duration, 0)
        self.assertEqual(inst.instruction_type, "ShiftPhase")
        self.assertEqual(inst.operand, 100)
        self.assertEqual(inst.frame, self.frame)

        inst = GenericInstruction(
            "ShiftFrequency", operand=100.5, frame=self.frame, logical_element=self.logical_element
        )
        self.assertEqual(inst.instruction_type, "ShiftFrequency")
        self.assertEqual(inst.operand, 100.5)
        self.assertEqual(inst.logical_element, self.logical_element)

        inst = GenericInstruction("SetFrequency", operand=100.5, frame=self.frame, initial_time=100)
        self.assertEqual(inst.initial_time, 100)

        GenericInstruction("SetPhase", operand=100.5, frame=self.frame)

    def test_generic_instruction_comparison(self):
        """Test comparison of generic instructions"""
        inst1 = GenericInstruction("Delay", operand=10, logical_element=Qubit(0))

        inst = GenericInstruction("Delay", operand=10, logical_element=Qubit(0))
        self.assertEqual(inst1, inst)
        inst = GenericInstruction("Delay", operand=100, logical_element=Qubit(0))
        self.assertNotEqual(inst1, inst)
        inst = GenericInstruction("Delay", operand=10, logical_element=Qubit(1))
        self.assertNotEqual(inst1, inst)
        inst = GenericInstruction("Delay", operand=10, logical_element=Qubit(1), initial_time=100)
        self.assertNotEqual(inst1, inst)
        inst = GenericInstruction(
            "Delay", operand=10, logical_element=Qubit(0), frame=QubitFrame(1)
        )
        self.assertNotEqual(inst1, inst)
        inst = GenericInstruction(
            "Play", operand=self.gaus_pulse, logical_element=Qubit(1), frame=QubitFrame(1)
        )
        self.assertNotEqual(inst1, inst)

        inst1 = GenericInstruction("ShiftPhase", operand=0.1, frame=QubitFrame(0))
        inst = GenericInstruction("ShiftFrequency", operand=0.1, frame=QubitFrame(0))
        self.assertNotEqual(inst1, inst)

    def test_generic_instruction_repr(self):
        """Test instruction __repr__ method"""
        inst = GenericInstruction("Delay", operand=10, logical_element=Qubit(0))
        ref = "Delay(operand=10,logical_element=Qubit(0),duration=10)"
        self.assertEqual(str(inst), ref)

        inst = GenericInstruction("ShiftFrequency", operand=10.3, frame=QubitFrame(0))
        ref = "ShiftFrequency(operand=10.3,frame=QubitFrame(0),duration=0)"
        self.assertEqual(str(inst), ref)

        inst = GenericInstruction(
            "ShiftPhase", operand=0.25, frame=MeasurementFrame(0), logical_element=Qubit(1)
        )
        ref = (
            "ShiftPhase(operand=0.25,logical_element=Qubit(1),frame=MeasurementFrame(0),duration=0)"
        )
        self.assertEqual(str(inst), ref)

        inst = GenericInstruction(
            "Play",
            operand=self.gaus_pulse,
            logical_element=Qubit(0),
            frame=QubitFrame(1),
            initial_time=100,
        )
        ref = "Play(operand=" + str(self.gaus_pulse)
        ref += (
            f",logical_element=Qubit(0),frame=QubitFrame(1)"
            f",duration={self.gaus_pulse.duration},initial_time=100)"
        )
        self.assertEqual(str(inst), ref)


class TestAcquireInstruction(QiskitTestCase):
    """Test PulseIR acquire instructions"""

    def test_creation(self):
        """Test that acquire instructions are instantiated correctly"""
        inst = AcquireInstruction(Qubit(1), MemorySlot(3), 100, 10)
        self.assertEqual(inst.qubit, Qubit(1))
        self.assertEqual(inst.memory_slot, MemorySlot(3))
        self.assertEqual(inst.duration, 100)
        self.assertEqual(inst.initial_time, 10)
        self.assertEqual(inst.final_time, 110)

    def test_parameter_validation(self):
        """Test that parameters are validated correctly when instantiating acquire instruction"""
        with self.assertRaises(PulseError):
            AcquireInstruction(Qubit(1), MemorySlot(3), 100.5)
        with self.assertRaises(PulseError):
            AcquireInstruction(Qubit(1), MemorySlot(3), -10)
        with self.assertRaises(PulseError):
            AcquireInstruction(Qubit(1), MemorySlot(3), 10, 10.5)
        with self.assertRaises(PulseError):
            AcquireInstruction(Qubit(1), MemorySlot(3), 10, -1)

    def test_acquire_instruction_comparison(self):
        """Test that acquire instruction comparison is done correctly"""
        ref = AcquireInstruction(Qubit(1), MemorySlot(3), 100, 10)
        self.assertEqual(ref, AcquireInstruction(Qubit(1), MemorySlot(3), 100, 10))
        self.assertNotEqual(ref, AcquireInstruction(Qubit(1), MemorySlot(3), 100))
        self.assertNotEqual(ref, AcquireInstruction(Qubit(1), MemorySlot(3), 200, 10))
        self.assertNotEqual(ref, AcquireInstruction(Qubit(1), MemorySlot(2), 100, 10))
        self.assertNotEqual(ref, AcquireInstruction(Qubit(2), MemorySlot(3), 100, 10))

    def test_acquire_instruction_repr(self):
        """Test the acquire instruction repr function"""
        inst = AcquireInstruction(Qubit(1), MemorySlot(2), 100, 10)
        ref = "Acquire(qubit=Qubit(1),memory_slot=MemorySlot(2),duration=100,initial_time=10)"
        self.assertEqual(str(inst), ref)
        inst = AcquireInstruction(Qubit(3), MemorySlot(1), 50)
        ref = "Acquire(qubit=Qubit(3),memory_slot=MemorySlot(1),duration=50,initial_time=None)"
        self.assertEqual(str(inst), ref)
