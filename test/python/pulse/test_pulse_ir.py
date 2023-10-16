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

"""Test pulse IR"""
import copy

from qiskit.test import QiskitTestCase
from qiskit.pulse import (
    Gaussian,
    Constant,
    Frame,
    MixedFrame,
    QubitFrame,
    MeasurementFrame,
    Qubit,
    MemorySlot,
    PulseError,
)
from qiskit.pulse.ir import (
    GenericInstruction,
    AcquireInstruction,
    PulseIR,
)

from qiskit.pulse.transforms import AlignSequential, AlignLeft


class TestPulseIR(QiskitTestCase):
    """Test PulseIR objects"""

    pulse_ir_example = PulseIR()
    pulse_ir_example.add_element(GenericInstruction("ShiftFrequency", 100.2, frame=QubitFrame(1)))
    pulse_ir_example.add_element(GenericInstruction("Delay", 80, logical_element=Qubit(2)))
    pulse_ir_example.add_element(
        GenericInstruction(
            "Play", Gaussian(160, 0.5, 40), logical_element=Qubit(2), frame=QubitFrame(2)
        )
    )
    pulse_ir_example.add_element(
        GenericInstruction(
            "Play", Gaussian(160, 0.5, 40), logical_element=Qubit(2), frame=QubitFrame(1)
        )
    )

    block1 = PulseIR()
    block1.add_element(GenericInstruction("SetFrequency", 10.2, Qubit(3), QubitFrame(3)))
    block1.add_element(GenericInstruction("ShiftPhase", 0.2, Qubit(3), QubitFrame(4)))
    block1.add_element(GenericInstruction("Play", Constant(60, 0.3), Qubit(3), QubitFrame(4)))
    pulse_ir_example.add_element(block1)

    block2 = PulseIR()
    block1.add_element(GenericInstruction("Play", Constant(60, 0.3), Qubit(2), QubitFrame(2)))
    block1.add_element(GenericInstruction("Play", Constant(60, 0.3), Qubit(2), QubitFrame(3)))

    block3 = PulseIR()
    block3.add_element(
        GenericInstruction(
            "Play", Constant(120, 0.3), logical_element=Qubit(1), frame=MeasurementFrame(1)
        )
    )
    block3.add_element(AcquireInstruction(qubit=Qubit(1), memory_slot=MemorySlot(1), duration=1200))
    block3.add_element(AcquireInstruction(qubit=Qubit(2), memory_slot=MemorySlot(2), duration=1200))

    block2.add_element(block3)
    pulse_ir_example.add_element(block2)
    pulse_ir_example.add_element(
        AcquireInstruction(qubit=Qubit(3), memory_slot=MemorySlot(5), duration=1200)
    )

    def ir_creation(self):
        """Test ir creation"""
        ir_example = PulseIR()
        self.assertEqual(ir_example.alignment, AlignSequential)

        ir_example = PulseIR(alignment=AlignLeft)
        self.assertEqual(ir_example.alignment, AlignLeft)

    def test_add_instruction(self):
        """Test adding single instruction"""

        pulse_ir = PulseIR()
        inst = GenericInstruction("Delay", operand=100, logical_element=Qubit(0))
        pulse_ir.add_element(inst)
        self.assertEqual(pulse_ir.elements[0], inst)
        inst = GenericInstruction("Delay", operand=100, logical_element=Qubit(1))
        pulse_ir.add_element(inst)
        self.assertEqual(pulse_ir.elements[1], inst)

    def test_add_instruction_list(self):
        """Test adding instruction list"""

        pulse_ir = PulseIR()
        inst_list = [
            GenericInstruction("Delay", operand=100, logical_element=Qubit(0)),
            GenericInstruction(
                "Play", operand=Constant(100, 0.2), logical_element=Qubit(0), frame=Frame("a")
            ),
            AcquireInstruction(Qubit(2), MemorySlot(3), 1000),
        ]

        pulse_ir.add_element(inst_list)

        self.assertEqual(pulse_ir.elements, inst_list)

    def test_add_sub_block(self):
        """Test adding sub block"""

        pulse_ir = PulseIR()
        pulse_ir.add_element(GenericInstruction("Delay", operand=100, logical_element=Qubit(0)))

        block = PulseIR()
        block.add_element(GenericInstruction("Delay", operand=100, logical_element=Qubit(1)))
        pulse_ir.add_element(block)

        self.assertEqual(pulse_ir.elements[1], block)

    def test_get_initial_time(self):
        """Test initial_time is returned correctly"""

        pulse_ir = PulseIR()
        # Empty IR defaults to None
        self.assertEqual(pulse_ir.initial_time, None)

        pulse_ir.add_element(
            GenericInstruction("Delay", operand=100, logical_element=Qubit(0), initial_time=1000)
        )
        pulse_ir.add_element(
            GenericInstruction("Delay", operand=100, logical_element=Qubit(1), initial_time=500)
        )
        self.assertEqual(pulse_ir.initial_time, 500)

        # Test recursion initial_time
        block = PulseIR()
        block.add_element(
            GenericInstruction(
                "Play",
                Constant(100, 0.3),
                logical_element=Qubit(1),
                frame=QubitFrame(1),
                initial_time=100,
            )
        )
        pulse_ir.add_element(block)
        self.assertEqual(pulse_ir.initial_time, 100)

        # If any instruction is not scheduled, initial_time is none
        pulse_ir.add_element(GenericInstruction("Delay", operand=100, logical_element=Qubit(0)))
        self.assertEqual(pulse_ir.initial_time, None)

    def test_shift_initial_time(self):
        """Test shift initial_time"""
        pulse_ir = PulseIR()
        pulse_ir.add_element(GenericInstruction("Delay", operand=100, logical_element=Qubit(0)))

        # Can't shift initial_time of IR with unscheduled instructions.
        with self.assertRaises(PulseError):
            pulse_ir.shift_initial_time(100)

        pulse_ir.elements[0].initial_time = 1000
        pulse_ir.add_element(
            GenericInstruction("Delay", operand=100, logical_element=Qubit(1), initial_time=500)
        )
        pulse_ir.shift_initial_time(100)
        self.assertEqual(pulse_ir.initial_time, 600)
        self.assertEqual(pulse_ir.elements[0].initial_time, 1100)
        self.assertEqual(pulse_ir.elements[1].initial_time, 600)

        with self.assertRaises(PulseError):
            pulse_ir.shift_initial_time(0.5)

    def test_shift_initial_time_with_starting_index(self):
        """Test shift initial_time"""
        pulse_ir = PulseIR()
        pulse_ir.add_element(
            GenericInstruction("Delay", operand=100, logical_element=Qubit(0), initial_time=500)
        )
        pulse_ir.add_element(
            GenericInstruction("Delay", operand=100, logical_element=Qubit(1), initial_time=1000)
        )

        pulse_ir.shift_initial_time(100, start_ind=1)
        self.assertEqual(pulse_ir.initial_time, 500)
        self.assertEqual(pulse_ir.elements[0].initial_time, 500)
        self.assertEqual(pulse_ir.elements[1].initial_time, 1100)

    def test_get_final_time(self):
        """Test final time is returned correctly"""

        pulse_ir = PulseIR()
        # Empty IR defaults to None
        self.assertEqual(pulse_ir.final_time, None)

        pulse_ir.add_element(
            GenericInstruction("Delay", operand=100, logical_element=Qubit(0), initial_time=1000)
        )
        pulse_ir.add_element(
            GenericInstruction("Delay", operand=100, logical_element=Qubit(1), initial_time=500)
        )
        self.assertEqual(pulse_ir.final_time, 1100)

        # Recursion final time
        block = PulseIR()
        block.add_element(
            GenericInstruction(
                "Play",
                Constant(100, 0.3),
                logical_element=Qubit(1),
                frame=QubitFrame(1),
                initial_time=2000,
            )
        )
        pulse_ir.add_element(block)
        self.assertEqual(pulse_ir.final_time, 2100)

        # If any instruction is not scheduled, final time is none
        pulse_ir.add_element(GenericInstruction("Delay", operand=100, logical_element=Qubit(0)))
        self.assertEqual(pulse_ir.initial_time, None)

    def test_get_logical_elements(self):
        """Test that logical elements are returned correctly"""

        logical_elements = self.pulse_ir_example.logical_elements()
        ref = {Qubit(1), Qubit(2), Qubit(3)}
        self.assertEqual(logical_elements, ref)

    def test_get_frames(self):
        """Test that frames are returned correctly"""

        frames = self.pulse_ir_example.frames()
        ref = {QubitFrame(1), QubitFrame(2), QubitFrame(3), QubitFrame(4), MeasurementFrame(1)}
        self.assertEqual(frames, ref)

    def test_get_mixed_frames(self):
        """Test that mixed frames are returned correctly"""

        mixed_frames = self.pulse_ir_example.mixed_frames()
        ref = {
            MixedFrame(Qubit(2), QubitFrame(2)),
            MixedFrame(Qubit(2), QubitFrame(1)),
            MixedFrame(Qubit(3), QubitFrame(3)),
            MixedFrame(Qubit(3), QubitFrame(4)),
            MixedFrame(Qubit(2), QubitFrame(3)),
            MixedFrame(Qubit(1), MeasurementFrame(1)),
        }
        self.assertEqual(mixed_frames, ref)

    def test_get_instructions_by_mixed_frames_recursion(self):
        """Test that instructions are returned correctly by mixed frame with recursion"""
        for mixed_frame in self.pulse_ir_example.mixed_frames():
            inst_list = self.pulse_ir_example.get_instructions_by_mixed_frame(mixed_frame)
            self.assertTrue(len(inst_list) >= 1)

        q2qf2 = self.pulse_ir_example.get_instructions_by_mixed_frame(
            MixedFrame(Qubit(2), QubitFrame(2))
        )
        self.assertEqual(len(q2qf2), 2)

        q2qf1 = self.pulse_ir_example.get_instructions_by_mixed_frame(
            MixedFrame(Qubit(2), QubitFrame(1))
        )
        ref = GenericInstruction(
            "Play", Gaussian(160, 0.5, 40), logical_element=Qubit(2), frame=QubitFrame(1)
        )
        self.assertEqual(q2qf1[0], self.pulse_ir_example.elements[3])
        self.assertEqual(q2qf1[0], ref)

        q10qf2 = self.pulse_ir_example.get_instructions_by_mixed_frame(
            MixedFrame(Qubit(10), QubitFrame(2))
        )
        self.assertEqual(q10qf2, [])

        q3qf4 = self.pulse_ir_example.get_instructions_by_mixed_frame(
            MixedFrame(Qubit(3), QubitFrame(4))
        )
        ref1 = GenericInstruction("ShiftPhase", 0.2, Qubit(3), QubitFrame(4))
        ref2 = GenericInstruction("Play", Constant(60, 0.3), Qubit(3), QubitFrame(4))
        self.assertEqual(q3qf4[0], ref1)
        self.assertEqual(q3qf4[1], ref2)

    def test_get_instructions_by_mixed_frames_no_recursion(self):
        """Test that instructions are returned correctly by mixed frame without recursion"""
        pulse_ir = PulseIR()
        pulse_ir.add_element(GenericInstruction("ShiftFrequency", 100.2, Qubit(1), QubitFrame(1)))

        block1 = PulseIR()
        block1.add_element(GenericInstruction("SetFrequency", 10.2, Qubit(1), QubitFrame(1)))
        block1.add_element(GenericInstruction("ShiftPhase", 0.2, Qubit(2), QubitFrame(2)))
        pulse_ir.add_element(block1)

        q1qf1 = pulse_ir.get_instructions_by_mixed_frame(MixedFrame(Qubit(1), QubitFrame(1)))
        self.assertEqual(len(q1qf1), 2)
        self.assertEqual(q1qf1[0], pulse_ir.elements[0])
        self.assertEqual(q1qf1[1], pulse_ir.elements[1].elements[0])

        q1qf1_no_recursion = pulse_ir.get_instructions_by_mixed_frame(
            MixedFrame(Qubit(1), QubitFrame(1)),
            recursive=False,
        )
        self.assertEqual(len(q1qf1_no_recursion), 1)
        self.assertEqual(q1qf1_no_recursion[0], pulse_ir.elements[0])

        q2qf2 = pulse_ir.get_instructions_by_mixed_frame(MixedFrame(Qubit(2), QubitFrame(2)))
        self.assertEqual(len(q2qf2), 1)
        self.assertEqual(q2qf2[0], pulse_ir.elements[1].elements[1])

        q2qf2_no_recursion = pulse_ir.get_instructions_by_mixed_frame(
            MixedFrame(Qubit(2), QubitFrame(2)),
            recursive=False,
        )
        self.assertEqual(q2qf2_no_recursion, [])

    def test_get_acquire_instructions(self):
        """Test that acquire instructions are returned correctly"""
        # All instructions
        acquire_instructions = self.pulse_ir_example.get_acquire_instructions()
        self.assertEqual(len(acquire_instructions), 3)

        # per qubit
        qubit1_acquire = self.pulse_ir_example.get_acquire_instructions(qubit=Qubit(1))
        ref = AcquireInstruction(qubit=Qubit(1), memory_slot=MemorySlot(1), duration=1200)
        self.assertEqual(len(qubit1_acquire), 1)
        self.assertEqual(qubit1_acquire[0], ref)

        qubit5_acquire = self.pulse_ir_example.get_acquire_instructions(qubit=Qubit(5))
        self.assertEqual(qubit5_acquire, [])

        # recursion
        no_recursion = self.pulse_ir_example.get_acquire_instructions(recursive=False)
        ref = AcquireInstruction(qubit=Qubit(3), memory_slot=MemorySlot(5), duration=1200)
        self.assertEqual(len(qubit1_acquire), 1)
        self.assertEqual(no_recursion[0], ref)

    def test_has_child(self):
        """Test that has_child_IR method works correctly"""
        pulse_ir = PulseIR()
        self.assertFalse(pulse_ir.has_child_ir())

        pulse_ir.add_element(
            GenericInstruction("ShiftFrequency", 100.2, Qubit(1), QubitFrame(1)),
        )
        self.assertFalse(pulse_ir.has_child_ir())

        block = PulseIR()
        pulse_ir.add_element(block)
        self.assertTrue(pulse_ir.has_child_ir())

    def test_sort_unscheduled_inst(self):
        """Test that the sort method raises an error if any instruction is not scheduled."""
        pulse_ir = PulseIR()
        pulse_ir.add_element(
            GenericInstruction("ShiftFrequency", 100.2, Qubit(1), QubitFrame(1)),
        )
        with self.assertRaises(PulseError):
            pulse_ir.sort_by_initial_time()

        pulse_ir.add_element(
            GenericInstruction("SetFrequency", 100.2, Qubit(1), QubitFrame(2), initial_time=10)
        )
        with self.assertRaises(PulseError):
            pulse_ir.sort_by_initial_time()

    def test_sort(self):
        """Test that the sort method works correctly with"""
        ir_example = PulseIR()
        times = [100, 50, 120, 400]
        inst = [
            GenericInstruction("SetPhase", 1, frame=QubitFrame(1), initial_time=time)
            for time in times
        ]
        ir_example.add_element(inst)
        ir_example.sort_by_initial_time()
        times.sort()
        for i, time in enumerate(times):
            self.assertEqual(ir_example.elements[i].initial_time, time)

        block = PulseIR()
        block.add_element(
            GenericInstruction("Delay", 10, logical_element=Qubit(0), initial_time=10)
        )
        ir_example.add_element(block)
        ir_example.sort_by_initial_time()
        self.assertEqual(ir_example.elements[0], block)

    def test_flatten_unscheduled_inst(self):
        """Test that the flatten method raises an error if any instruction is not scheduled."""
        pulse_ir = PulseIR()
        pulse_ir.add_element(
            GenericInstruction("ShiftFrequency", 100.2, Qubit(1), QubitFrame(1)),
        )
        with self.assertRaises(PulseError):
            pulse_ir.flatten()

        pulse_ir.add_element(
            GenericInstruction("SetFrequency", 100.2, Qubit(1), QubitFrame(2), initial_time=10)
        )
        with self.assertRaises(PulseError):
            pulse_ir.flatten()

    def test_flatten_no_sub_blocks(self):
        """Test that flatten doesn't do anything with no sub blocks"""
        pulse_ir = PulseIR()
        pulse_ir.add_element(
            [
                GenericInstruction("SetFrequency", 100.2, Qubit(1), QubitFrame(2), initial_time=10),
                GenericInstruction("Delay", 10, Qubit(1), QubitFrame(2), initial_time=100),
            ]
        )
        ref = copy.deepcopy(pulse_ir)
        pulse_ir.flatten()
        self.assertEqual(pulse_ir, ref)

    def test_flatten_with_sub_blocks(self):
        """Test that the flatten method works correctly with sub blocks"""
        pulse_ir = PulseIR()
        pulse_ir.add_element(
            [
                GenericInstruction("SetFrequency", 100.2, Qubit(1), QubitFrame(2), initial_time=10),
                GenericInstruction("Delay", 10, Qubit(1), QubitFrame(2), initial_time=100),
            ]
        )
        block = PulseIR()
        block.add_element(
            GenericInstruction("ShiftFrequency", 100.2, Qubit(1), QubitFrame(2), initial_time=50)
        )
        pulse_ir.add_element(block)
        pulse_ir.flatten()
        self.assertFalse(pulse_ir.has_child_ir())
        self.assertEqual(pulse_ir.elements[-1], block.elements[0])

    def test_ir_comparison_no_sub_blocks(self):
        """Test that ir is compared correctly with no sub blocks"""
        pulse_ir = PulseIR()
        pulse_ir.add_element(
            [
                GenericInstruction("SetFrequency", 100.2, Qubit(1), QubitFrame(2), initial_time=10),
                GenericInstruction("Delay", 10, Qubit(1), QubitFrame(2), initial_time=100),
            ]
        )
        ref = copy.deepcopy(pulse_ir)
        self.assertEqual(pulse_ir, ref)

        # One element is different
        ref.elements[0] = GenericInstruction(
            "ShiftFrequency", 10, Qubit(1), QubitFrame(2), initial_time=100
        )
        self.assertNotEqual(pulse_ir, ref)

        # Extra element
        ref = copy.deepcopy(pulse_ir)
        ref.add_element(
            GenericInstruction("ShiftFrequency", 10, Qubit(1), QubitFrame(2), initial_time=100)
        )
        self.assertNotEqual(pulse_ir, ref)

        # Different alignment
        ref = copy.deepcopy(pulse_ir)
        ref._alignment = AlignLeft
        self.assertNotEqual(pulse_ir, ref)

    def test_ir_comparison_with_sub_blocks(self):
        """Test that ir is compared correctly with sub blocks"""
        pulse_ir = PulseIR()
        pulse_ir.add_element(
            [
                GenericInstruction("SetFrequency", 100.2, Qubit(1), QubitFrame(2), initial_time=10),
                GenericInstruction("Delay", 10, Qubit(1), QubitFrame(2), initial_time=100),
            ]
        )
        block = PulseIR()
        pulse_ir.add_element(block)

        # empty sub block
        ref = copy.deepcopy(pulse_ir)
        self.assertEqual(pulse_ir, ref)

        # sub block has extra element
        ref.elements[2].add_element(
            GenericInstruction("ShiftFrequency", 10, Qubit(1), QubitFrame(2), initial_time=100)
        )
        self.assertNotEqual(pulse_ir, ref)

        # sub block is identical
        pulse_ir.elements[2].add_element(
            GenericInstruction("ShiftFrequency", 10, Qubit(1), QubitFrame(2), initial_time=100)
        )
        self.assertEqual(pulse_ir, ref)

    def test_instruction_removal_from_ir(self):
        """Test that instructions are removed correctly from IR"""
        inst1 = GenericInstruction("SetFrequency", 100.2, Qubit(1), QubitFrame(2), initial_time=10)
        inst2 = GenericInstruction("Delay", 10, Qubit(1), QubitFrame(2), initial_time=100)

        pulse_ir = PulseIR()
        pulse_ir.add_element([inst1, inst2])
        pulse_ir.remove_instruction(inst1)

        self.assertEqual(pulse_ir.elements[0], inst2)
        pulse_ir.remove_instruction(inst2)
        self.assertEqual(pulse_ir.elements, [])

        # Verify that removing a non-existent instruction doesn't raise an error
        pulse_ir.remove_instruction(inst1)

    def test_unscheduled_instruction_removal(self):
        """Test that unscheduled instructions can not be removed"""
        inst1 = GenericInstruction("SetFrequency", 100.2, Qubit(1), QubitFrame(2))
        inst2 = GenericInstruction("Delay", 10, Qubit(1), QubitFrame(2))

        pulse_ir = PulseIR()
        pulse_ir.add_element([inst1, inst2])
        with self.assertRaises(PulseError):
            pulse_ir.remove_instruction(inst1)

    def test_remove_partial_instructions(self):
        """Test removal of partial instructions"""
        pulse_ir = PulseIR()
        pulse_ir.add_element(GenericInstruction("SetFrequency", 100.2, Qubit(1), QubitFrame(2)))

        pulse_ir.remove_partial_instructions()
        self.assertEqual(len(pulse_ir), 1)

        partial_inst_1 = GenericInstruction("Delay", 10, Qubit(1))
        partial_inst_2 = GenericInstruction("SetFrequency", 10.3, frame=QubitFrame(1))
        pulse_ir.add_element([partial_inst_1, partial_inst_2])
        pulse_ir.remove_partial_instructions()
        self.assertEqual(len(pulse_ir), 1)

        # with recursion
        sub_pulse_ir = PulseIR()
        sub_pulse_ir.add_element(GenericInstruction("SetFrequency", 100.2, Qubit(3), QubitFrame(3)))
        sub_pulse_ir.add_element([partial_inst_1, partial_inst_2])
        pulse_ir.add_element(sub_pulse_ir)
        pulse_ir.remove_partial_instructions(recursive=True)
        self.assertEqual(len(pulse_ir), 2)
        self.assertEqual(len(pulse_ir.elements[1]), 1)

    def test_get_partial_instruction_info(self):
        """Test retrieval of partial instructions info"""
        pulse_ir = PulseIR()
        pulse_ir.add_element(GenericInstruction("SetFrequency", 100.2, Qubit(1), QubitFrame(2)))
        pulse_ir.add_element(GenericInstruction("Delay", 10, Qubit(1), QubitFrame(2)))

        self.assertEqual(pulse_ir._get_partial_instruction_info(), set())

        pulse_ir.add_element(GenericInstruction("SetFrequency", 100.2, frame=QubitFrame(3)))
        pulse_ir.add_element(AcquireInstruction(Qubit(1), MemorySlot(2), 100))

        self.assertEqual(pulse_ir._get_partial_instruction_info(), {QubitFrame(3)})

        pulse_ir.add_element(GenericInstruction("Delay", 10, Qubit(1)))
        self.assertEqual(pulse_ir._get_partial_instruction_info(), {QubitFrame(3), Qubit(1)})

        # Test recursion
        sub_pulse_ir = PulseIR()
        sub_pulse_ir.add_element(GenericInstruction("Delay", 10, Qubit(4)))

        pulse_ir.add_element(sub_pulse_ir)
        self.assertEqual(
            pulse_ir._get_partial_instruction_info(), {QubitFrame(3), Qubit(1), Qubit(4)}
        )

    def test_get_broadcasting_info(self):
        """Test broadcasting info creation"""
        pulse_ir = PulseIR()
        pulse_ir.add_element(GenericInstruction("SetFrequency", 100.2, Qubit(1), QubitFrame(2)))
        pulse_ir.add_element(GenericInstruction("SetFrequency", 100.2, Qubit(2), QubitFrame(2)))
        pulse_ir.add_element(GenericInstruction("SetFrequency", 100.2, Qubit(2), QubitFrame(3)))
        pulse_ir.add_element(AcquireInstruction(Qubit(1), MemorySlot(2), 100))

        self.assertEqual(len(pulse_ir.get_broadcasting_info()), 0)

        pulse_ir.add_element(GenericInstruction("Delay", 100, Qubit(2)))
        pulse_ir.add_element(GenericInstruction("SetFrequency", 100.2, frame=QubitFrame(2)))

        broadcasting_info = pulse_ir.get_broadcasting_info()
        self.assertEqual(broadcasting_info.keys(), pulse_ir._get_partial_instruction_info())
        self.assertEqual(
            broadcasting_info[Qubit(2)],
            {MixedFrame(Qubit(2), QubitFrame(2)), MixedFrame(Qubit(2), QubitFrame(3))},
        )
        self.assertEqual(
            broadcasting_info[QubitFrame(2)],
            {MixedFrame(Qubit(2), QubitFrame(2)), MixedFrame(Qubit(1), QubitFrame(2))},
        )

        # Test recursion
        sub_pulse_ir = PulseIR()
        sub_pulse_ir.add_element(GenericInstruction("Delay", 100, Qubit(1)))
        sub_pulse_ir.add_element(GenericInstruction("SetFrequency", 100.2, Qubit(2), QubitFrame(4)))

        pulse_ir.add_element(sub_pulse_ir)
        broadcasting_info = pulse_ir.get_broadcasting_info()
        self.assertEqual(broadcasting_info[Qubit(1)], {MixedFrame(Qubit(1), QubitFrame(2))})
        self.assertTrue(MixedFrame(Qubit(2), QubitFrame(4)) in broadcasting_info[Qubit(2)])

    def test_get_mixed_frames_with_broadcasting_info(self):
        """Test MixedFrames identification when broadcasting info is provided"""
        pulse_ir = PulseIR()
        pulse_ir.add_element(GenericInstruction("SetFrequency", 100.2, Qubit(1), QubitFrame(2)))
        pulse_ir.add_element(GenericInstruction("SetFrequency", 100.2, Qubit(2), QubitFrame(2)))
        pulse_ir.add_element(GenericInstruction("SetFrequency", 100.2, frame=QubitFrame(3)))
        pulse_ir.add_element(GenericInstruction("Delay", 100, Qubit(3)))
        pulse_ir.add_element(AcquireInstruction(Qubit(1), MemorySlot(2), 100))

        broadcasting_info = {
            Qubit(3): {MixedFrame(Qubit(3), QubitFrame(4))},
            QubitFrame(3): {MixedFrame(Qubit(4), QubitFrame(3))},
        }

        mixed_frames = pulse_ir.mixed_frames(broadcasting_info=broadcasting_info)
        self.assertEqual(len(mixed_frames), 4)
        self.assertTrue(MixedFrame(Qubit(3), QubitFrame(4)) in mixed_frames)
        self.assertTrue(MixedFrame(Qubit(4), QubitFrame(3)) in mixed_frames)

        # With recursion
        broadcasting_info[Qubit(5)] = {MixedFrame(Qubit(5), QubitFrame(5))}

        sub_pulse_ir = PulseIR()
        sub_pulse_ir.add_element(GenericInstruction("Delay", 100, Qubit(5)))
        pulse_ir.add_element(sub_pulse_ir)
        mixed_frames = pulse_ir.mixed_frames(broadcasting_info=broadcasting_info)
        self.assertEqual(len(mixed_frames), 5)
        self.assertTrue(MixedFrame(Qubit(5), QubitFrame(5)) in mixed_frames)

    def test_pulse_ir_repr(self):
        """Test PulseIR representation"""
        pulse_ir = PulseIR(alignment=AlignLeft)
        ref = "PulseIR[alignment=AlignLeft,[]]"
        self.assertEqual(str(pulse_ir), ref)

        pulse_ir = PulseIR()
        pulse_ir.add_element(GenericInstruction("SetFrequency", 100.2, Qubit(1), QubitFrame(2)))
        ref = (
            "PulseIR[alignment=AlignSequential,[SetFrequency(operand=100.2,"
            "logical_element=Qubit(1),frame=QubitFrame(2),duration=0)]]"
        )
        self.assertEqual(str(pulse_ir), ref)

        pulse_ir.add_element(GenericInstruction("Delay", 10, Qubit(2), initial_time=100))
        ref = (
            "PulseIR[alignment=AlignSequential,[SetFrequency(operand=100.2,"
            "logical_element=Qubit(1),frame=QubitFrame(2),duration=0),"
            " Delay(operand=10,logical_element=Qubit(2),duration=10,initial_time=100)]]"
        )
        self.assertEqual(str(pulse_ir), ref)
