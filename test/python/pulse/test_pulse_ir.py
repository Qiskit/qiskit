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

from test import QiskitTestCase
from qiskit.pulse import (
    Constant,
    MemorySlot,
    PulseError,
    Play,
    Delay,
    Acquire,
    ShiftPhase,
    DriveChannel,
    AcquireChannel,
)

from qiskit.pulse.ir import (
    IrInstruction,
    IrBlock,
)

from qiskit.pulse.transforms import AlignSequential, AlignLeft


class TestIrInstruction(QiskitTestCase):
    """Test IR Instruction"""

    _play_inst = Play(Constant(100, 0.5), channel=DriveChannel(1))

    def test_instruction_creation(self):
        """Test ir instruction creation"""
        ir_inst = IrInstruction(self._play_inst)
        self.assertEqual(ir_inst.initial_time, None)
        self.assertEqual(ir_inst.duration, 100)
        self.assertEqual(ir_inst.duration, self._play_inst.duration)
        self.assertEqual(ir_inst.instruction, self._play_inst)

        ir_inst = IrInstruction(self._play_inst, initial_time=100)
        self.assertEqual(ir_inst.initial_time, 100)
        self.assertEqual(ir_inst.final_time, 200)

    def test_instruction_creation_invalid_initial_time(self):
        """Test that instructions can't be constructed with invalid initial_time"""
        with self.assertRaises(PulseError):
            IrInstruction(self._play_inst, initial_time=0.5)

        with self.assertRaises(PulseError):
            IrInstruction(self._play_inst, initial_time=-1)

    def test_initial_time_update(self):
        """Test that initial_time update is done and validated correctly"""
        ir_inst = IrInstruction(self._play_inst)

        ir_inst.initial_time = 50
        self.assertEqual(ir_inst.initial_time, 50)

        with self.assertRaises(PulseError):
            ir_inst.initial_time = -10
        with self.assertRaises(PulseError):
            ir_inst.initial_time = 0.5

    def test_instruction_shift_initial_time(self):
        """Test that shifting initial_time is done correctly"""
        initial_time = 10
        shift = 15

        ir_inst = IrInstruction(self._play_inst)

        with self.assertRaises(PulseError):
            ir_inst.shift_initial_time(shift)

        ir_inst = IrInstruction(self._play_inst, initial_time=initial_time)
        ir_inst.shift_initial_time(shift)

        self.assertEqual(ir_inst.initial_time, initial_time + shift)

        with self.assertRaises(PulseError):
            ir_inst.shift_initial_time(0.5)

    def test_final_time_unscheduled(self):
        """Test that final time is returned as None if unscheduled"""
        ir_inst = IrInstruction(self._play_inst)
        self.assertEqual(ir_inst.final_time, None)

    def test_repr(self):
        """Test repr"""
        ir_inst = IrInstruction(self._play_inst)
        self.assertEqual(
            str(ir_inst),
            "IrInstruction(Play(Constant(duration=100, amp=0.5, angle=0.0), DriveChannel(1)), t0=None)",
        )

        ir_inst = IrInstruction(self._play_inst, 100)
        self.assertEqual(
            str(ir_inst),
            "IrInstruction(Play(Constant(duration=100, amp=0.5, angle=0.0), DriveChannel(1)), t0=100)",
        )


class TestIrBlock(QiskitTestCase):
    """Test IrBlock objects"""

    _delay_inst = Delay(50, channel=DriveChannel(0))
    _play_inst = Play(Constant(100, 0.5), channel=DriveChannel(1))
    _shift_phase_inst = ShiftPhase(0.1, channel=DriveChannel(3))
    _acquire_inst = Acquire(200, channel=AcquireChannel(2), mem_slot=MemorySlot(4))

    def ir_creation(self):
        """Test ir creation"""
        ir_example = IrBlock(AlignSequential())
        self.assertEqual(ir_example.alignment, AlignSequential())
        self.assertEqual(len(ir_example), 0)

    def test_add_instruction(self):
        """Test adding single instruction"""

        pulse_ir = IrBlock(AlignLeft())
        ir_inst = IrInstruction(self._play_inst)
        pulse_ir.add_element(ir_inst)
        self.assertEqual(pulse_ir.elements[0], ir_inst)
        ir_inst = IrInstruction(self._delay_inst)
        pulse_ir.add_element(ir_inst)
        self.assertEqual(pulse_ir.elements[1], ir_inst)
        self.assertEqual(len(pulse_ir), 2)

    def test_add_instruction_list(self):
        """Test adding instruction list"""

        pulse_ir = IrBlock(AlignLeft())
        inst_list = [
            IrInstruction(self._play_inst),
            IrInstruction(self._delay_inst),
            IrInstruction(self._acquire_inst),
        ]

        pulse_ir.add_element(inst_list)

        self.assertEqual(pulse_ir.elements, inst_list)
        self.assertEqual(len(pulse_ir), 3)

    def test_add_sub_block(self):
        """Test adding sub block"""

        pulse_ir = IrBlock(AlignLeft())
        pulse_ir.add_element(IrInstruction(self._play_inst))

        block = IrBlock(AlignLeft())
        block.add_element(IrInstruction(self._delay_inst))
        pulse_ir.add_element(block)

        self.assertEqual(pulse_ir.elements[1], block)
        self.assertEqual(len(pulse_ir), 2)

    def test_get_initial_time(self):
        """Test initial_time is returned correctly"""

        pulse_ir = IrBlock(AlignLeft())
        # Empty IR defaults to None
        self.assertEqual(pulse_ir.initial_time, None)

        pulse_ir.add_element(IrInstruction(self._play_inst, initial_time=100))
        pulse_ir.add_element(IrInstruction(self._delay_inst, initial_time=50))
        self.assertEqual(pulse_ir.initial_time, 50)

        # Test recursion initial_time
        block = IrBlock(AlignLeft())
        block.add_element(IrInstruction(self._shift_phase_inst, initial_time=20))
        pulse_ir.add_element(block)
        self.assertEqual(pulse_ir.initial_time, 20)

        # If any instruction is not scheduled, initial_time is none
        pulse_ir.add_element(IrInstruction(self._acquire_inst))
        self.assertEqual(pulse_ir.initial_time, None)

    def test_shift_initial_time(self):
        """Test shift initial_time"""
        pulse_ir = IrBlock(AlignLeft())
        pulse_ir.add_element(IrInstruction(self._play_inst))

        # Can't shift initial_time of IR with unscheduled instructions.
        with self.assertRaises(PulseError):
            pulse_ir.shift_initial_time(100)

        pulse_ir.elements[0].initial_time = 1000
        pulse_ir.add_element(IrInstruction(self._delay_inst, initial_time=500))
        pulse_ir.shift_initial_time(100)
        self.assertEqual(pulse_ir.initial_time, 600)
        self.assertEqual(pulse_ir.elements[0].initial_time, 1100)
        self.assertEqual(pulse_ir.elements[1].initial_time, 600)

        with self.assertRaises(PulseError):
            pulse_ir.shift_initial_time(0.5)

    def test_shift_initial_time_with_sub_blocks(self):
        """Test shift initial_time with sub blocks"""
        pulse_ir = IrBlock(AlignLeft())
        pulse_ir.add_element(IrInstruction(self._play_inst, 100))
        pulse_ir.add_element(IrInstruction(self._play_inst, 200))
        block = copy.deepcopy(pulse_ir)
        block.shift_initial_time(1000)
        pulse_ir.add_element(block)
        pulse_ir.shift_initial_time(100)
        self.assertEqual(pulse_ir.elements[0].initial_time, 200)
        self.assertEqual(pulse_ir.elements[1].initial_time, 300)
        self.assertEqual(pulse_ir.elements[2].elements[0].initial_time, 1200)
        self.assertEqual(pulse_ir.elements[2].elements[1].initial_time, 1300)

    def test_get_final_time(self):
        """Test final time is returned correctly"""

        pulse_ir = IrBlock(AlignLeft())
        # Empty IR defaults to None
        self.assertEqual(pulse_ir.final_time, None)

        pulse_ir.add_element(IrInstruction(self._delay_inst, initial_time=500))
        pulse_ir.add_element(IrInstruction(self._play_inst, initial_time=1000))
        self.assertEqual(pulse_ir.final_time, 1000 + self._play_inst.duration)

        # Recursion final time
        block = IrBlock(AlignLeft())
        block.add_element(IrInstruction(self._shift_phase_inst, initial_time=2000))
        pulse_ir.add_element(block)
        self.assertEqual(pulse_ir.final_time, 2000)

        # If any instruction is not scheduled, final time is none
        pulse_ir.add_element(IrInstruction(self._shift_phase_inst))
        self.assertEqual(pulse_ir.initial_time, None)

    def test_get_duration(self):
        """Test that duration is calculated and returned correctly"""
        pulse_ir = IrBlock(AlignLeft())
        pulse_ir.add_element(
            [
                IrInstruction(self._delay_inst, initial_time=10),
                IrInstruction(self._play_inst, initial_time=100),
            ]
        )
        self.assertEqual(pulse_ir.duration, 190)

    def test_get_duration_with_sub_blocks(self):
        """Test that duration is calculated and returned correctly with sub blocks"""
        pulse_ir = IrBlock(AlignLeft())
        pulse_ir.add_element(
            [
                IrInstruction(self._delay_inst, initial_time=10),
                IrInstruction(self._play_inst, initial_time=100),
            ]
        )
        block = IrBlock(AlignLeft())
        block.add_element(IrInstruction(self._play_inst, 200))
        pulse_ir.add_element(block)
        self.assertEqual(pulse_ir.duration, 290)

    def test_get_duration_unscheduled(self):
        """Test that duration is returned as None if something is not scheduled"""
        pulse_ir = IrBlock(AlignLeft())
        pulse_ir.add_element(
            [
                IrInstruction(self._delay_inst, initial_time=10),
                IrInstruction(self._play_inst),
            ]
        )
        self.assertEqual(pulse_ir.duration, None)

    def test_has_child(self):
        """Test that has_child_IR method works correctly"""
        pulse_ir = IrBlock(AlignLeft())
        self.assertFalse(pulse_ir.has_child_ir())

        pulse_ir.add_element(IrInstruction(self._shift_phase_inst, initial_time=2000))
        self.assertFalse(pulse_ir.has_child_ir())

        block = IrBlock(AlignLeft())
        pulse_ir.add_element(block)
        self.assertTrue(pulse_ir.has_child_ir())

    def test_ir_comparison_no_sub_blocks(self):
        """Test that ir is compared correctly with no sub blocks"""
        pulse_ir = IrBlock(AlignLeft())
        pulse_ir.add_element(
            [
                IrInstruction(self._delay_inst, initial_time=10),
                IrInstruction(self._play_inst, initial_time=100),
            ]
        )
        ref = copy.deepcopy(pulse_ir)
        self.assertEqual(pulse_ir, ref)

        # One element is different
        ref.elements[0] = IrInstruction(self._shift_phase_inst, initial_time=50)
        self.assertNotEqual(pulse_ir, ref)

        # Extra element
        ref = copy.deepcopy(pulse_ir)
        ref.add_element(IrInstruction(self._shift_phase_inst, initial_time=50))
        self.assertNotEqual(pulse_ir, ref)

        # Different alignment
        ref = copy.deepcopy(pulse_ir)
        ref._alignment = AlignLeft
        self.assertNotEqual(pulse_ir, ref)

    def test_ir_comparison_with_sub_blocks(self):
        """Test that ir is compared correctly with sub blocks"""
        pulse_ir = IrBlock(AlignLeft())
        pulse_ir.add_element(
            [
                IrInstruction(self._delay_inst, initial_time=10),
                IrInstruction(self._play_inst, initial_time=100),
            ]
        )
        block = IrBlock(AlignLeft())
        pulse_ir.add_element(block)

        # empty sub block
        ref = copy.deepcopy(pulse_ir)
        self.assertEqual(pulse_ir, ref)

        # sub block has extra element
        ref.elements[2].add_element(IrInstruction(self._shift_phase_inst, initial_time=500))
        self.assertNotEqual(pulse_ir, ref)

        # sub block is identical
        pulse_ir.elements[2].add_element(IrInstruction(self._shift_phase_inst, initial_time=500))
        self.assertEqual(pulse_ir, ref)

    def test_repr(self):
        """Test repr"""
        pulse_ir = IrBlock(AlignLeft())
        pulse_ir.add_element(
            [
                IrInstruction(self._play_inst),
                IrInstruction(self._play_inst),
                IrInstruction(self._play_inst),
            ]
        )
        self.assertEqual(str(pulse_ir), "IrBlock(AlignLeft(), 3 IrInstructions)")
        block = IrBlock(AlignLeft())
        block.add_element(IrInstruction(self._delay_inst, 100))
        pulse_ir.add_element(block)
        self.assertEqual(str(pulse_ir), "IrBlock(AlignLeft(), 3 IrInstructions, 1 IrBlocks)")
        pulse_ir.elements[0].initial_time = 10
        pulse_ir.elements[1].initial_time = 20
        pulse_ir.elements[2].initial_time = 30
        self.assertEqual(
            str(pulse_ir),
            "IrBlock(AlignLeft(), 3 IrInstructions, 1 IrBlocks, initial_time=10, duration=140)",
        )
