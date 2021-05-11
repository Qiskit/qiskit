# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Test cases for the pulse schedule block."""

from qiskit import pulse, circuit
from qiskit.pulse import transforms
from qiskit.pulse.exceptions import PulseError
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q


class BaseTestBlock(QiskitTestCase):
    """ScheduleBlock tests."""

    def setUp(self):
        super().setUp()

        self.backend = FakeOpenPulse2Q()

        self.test_waveform0 = pulse.Constant(100, 0.1)
        self.test_waveform1 = pulse.Constant(200, 0.1)

        self.d0 = pulse.DriveChannel(0)
        self.d1 = pulse.DriveChannel(1)

        self.left_context = transforms.AlignLeft()
        self.right_context = transforms.AlignRight()
        self.sequential_context = transforms.AlignSequential()
        self.equispaced_context = transforms.AlignEquispaced(duration=1000)

        def _align_func(j):
            return {1: 0.1, 2: 0.25, 3: 0.7, 4: 0.85}.get(j)

        self.func_context = transforms.AlignFunc(duration=1000, func=_align_func)

    def assertScheduleEqual(self, target, reference):
        """Check if two block are equal schedule representation."""
        self.assertEqual(transforms.target_qobj_transform(target), reference)


class TestTransformation(BaseTestBlock):
    """Test conversion of ScheduleBlock to Schedule."""

    def test_left_alignment(self):
        """Test left alignment context."""
        block = pulse.ScheduleBlock(alignment_context=self.left_context)
        block = block.append(pulse.Play(self.test_waveform0, self.d0))
        block = block.append(pulse.Play(self.test_waveform1, self.d1))

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform1, self.d1))

        self.assertScheduleEqual(block, ref_sched)

    def test_right_alignment(self):
        """Test right alignment context."""
        block = pulse.ScheduleBlock(alignment_context=self.right_context)
        block = block.append(pulse.Play(self.test_waveform0, self.d0))
        block = block.append(pulse.Play(self.test_waveform1, self.d1))

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(100, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform1, self.d1))

        self.assertScheduleEqual(block, ref_sched)

    def test_sequential_alignment(self):
        """Test sequential alignment context."""
        block = pulse.ScheduleBlock(alignment_context=self.sequential_context)
        block = block.append(pulse.Play(self.test_waveform0, self.d0))
        block = block.append(pulse.Play(self.test_waveform1, self.d1))

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(100, pulse.Play(self.test_waveform1, self.d1))

        self.assertScheduleEqual(block, ref_sched)

    def test_equispace_alignment(self):
        """Test equispace alignment context."""
        block = pulse.ScheduleBlock(alignment_context=self.equispaced_context)
        for _ in range(4):
            block = block.append(pulse.Play(self.test_waveform0, self.d0))

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(300, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(600, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(900, pulse.Play(self.test_waveform0, self.d0))

        self.assertScheduleEqual(block, ref_sched)

    def test_func_alignment(self):
        """Test func alignment context."""
        block = pulse.ScheduleBlock(alignment_context=self.func_context)
        for _ in range(4):
            block = block.append(pulse.Play(self.test_waveform0, self.d0))

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(50, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(200, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(650, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(800, pulse.Play(self.test_waveform0, self.d0))

        self.assertScheduleEqual(block, ref_sched)

    def test_nested_alignment(self):
        """Test nested block scheduling."""
        block_sub = pulse.ScheduleBlock(alignment_context=self.right_context)
        block_sub = block_sub.append(pulse.Play(self.test_waveform0, self.d0))
        block_sub = block_sub.append(pulse.Play(self.test_waveform1, self.d1))

        block_main = pulse.ScheduleBlock(alignment_context=self.sequential_context)
        block_main = block_main.append(block_sub)
        block_main = block_main.append(pulse.Delay(10, self.d0))
        block_main = block_main.append(block_sub)

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform1, self.d1))
        ref_sched = ref_sched.insert(100, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(200, pulse.Delay(10, self.d0))
        ref_sched = ref_sched.insert(210, pulse.Play(self.test_waveform1, self.d1))
        ref_sched = ref_sched.insert(310, pulse.Play(self.test_waveform0, self.d0))

        self.assertScheduleEqual(block_main, ref_sched)


class TestBlockOperation(BaseTestBlock):
    """Test fundamental operation on schedule block.

    Because ScheduleBlock adapts to the lazy scheduling, no uniitest for
    overlap constraints is necessary. Test scheme becomes simpler than the schedule.

    Some tests have dependency on schedule conversion.
    This operation should be tested in `test.python.pulse.test_block.TestTransformation`.
    """

    def setUp(self):
        super().setUp()

        self.test_blocks = [
            pulse.Play(self.test_waveform0, self.d0),
            pulse.Play(self.test_waveform1, self.d1),
            pulse.Delay(50, self.d0),
            pulse.Play(self.test_waveform1, self.d0),
        ]

    def test_append_an_instruction_to_empty_block(self):
        """Test append instructions to an empty block."""
        block = pulse.ScheduleBlock()
        block = block.append(pulse.Play(self.test_waveform0, self.d0))

        self.assertEqual(block.blocks[0], pulse.Play(self.test_waveform0, self.d0))

    def test_append_an_instruction_to_empty_block_sugar(self):
        """Test append instructions to an empty block with syntax sugar."""
        block = pulse.ScheduleBlock()
        block += pulse.Play(self.test_waveform0, self.d0)

        self.assertEqual(block.blocks[0], pulse.Play(self.test_waveform0, self.d0))

    def test_append_an_instruction_to_empty_block_inplace(self):
        """Test append instructions to an empty block with inplace."""
        block = pulse.ScheduleBlock()
        block.append(pulse.Play(self.test_waveform0, self.d0), inplace=True)

        self.assertEqual(block.blocks[0], pulse.Play(self.test_waveform0, self.d0))

    def test_append_a_block_to_empty_block(self):
        """Test append another ScheduleBlock to empty block."""
        block = pulse.ScheduleBlock()
        block.append(pulse.Play(self.test_waveform0, self.d0), inplace=True)

        block_main = pulse.ScheduleBlock()
        block_main = block_main.append(block)

        self.assertEqual(block_main.blocks[0], block)

    def test_append_an_instruction_to_block(self):
        """Test append instructions to a non-empty block."""
        block = pulse.ScheduleBlock()
        block = block.append(pulse.Delay(100, self.d0))

        block = block.append(pulse.Delay(100, self.d0))

        self.assertEqual(len(block.blocks), 2)

    def test_append_an_instruction_to_block_inplace(self):
        """Test append instructions to a non-empty block with inplace."""
        block = pulse.ScheduleBlock()
        block = block.append(pulse.Delay(100, self.d0))

        block.append(pulse.Delay(100, self.d0), inplace=True)

        self.assertEqual(len(block.blocks), 2)

    def test_duration(self):
        """Test if correct duration is returned with implicit scheduling."""
        block = pulse.ScheduleBlock()
        for inst in self.test_blocks:
            block.append(inst)

        self.assertEqual(block.duration, 350)

    def test_timeslots(self):
        """Test if correct timeslot is returned with implicit scheduling."""
        block = pulse.ScheduleBlock()
        for inst in self.test_blocks:
            block.append(inst)

        ref_slots = {self.d0: [(0, 100), (100, 150), (150, 350)], self.d1: [(0, 200)]}

        self.assertDictEqual(block.timeslots, ref_slots)

    def test_start_time(self):
        """Test if correct schedule start time is returned with implicit scheduling."""
        block = pulse.ScheduleBlock()
        for inst in self.test_blocks:
            block.append(inst)

        self.assertEqual(block.start_time, 0)

    def test_stop_time(self):
        """Test if correct schedule stop time is returned with implicit scheduling."""
        block = pulse.ScheduleBlock()
        for inst in self.test_blocks:
            block.append(inst)

        self.assertEqual(block.stop_time, 350)

    def test_channels(self):
        """Test if all channels are returned."""
        block = pulse.ScheduleBlock()
        for inst in self.test_blocks:
            block.append(inst)

        self.assertEqual(len(block.channels), 2)

    def test_instructions(self):
        """Test if all instructions are returned."""
        block = pulse.ScheduleBlock()
        for inst in self.test_blocks:
            block.append(inst)

        self.assertEqual(block.blocks, tuple(self.test_blocks))

    def test_channel_duraction(self):
        """Test if correct durations is calculated for each channel."""
        block = pulse.ScheduleBlock()
        for inst in self.test_blocks:
            block.append(inst)

        self.assertEqual(block.ch_duration(self.d0), 350)
        self.assertEqual(block.ch_duration(self.d1), 200)

    def test_channel_start_time(self):
        """Test if correct start time is calculated for each channel."""
        block = pulse.ScheduleBlock()
        for inst in self.test_blocks:
            block.append(inst)

        self.assertEqual(block.ch_start_time(self.d0), 0)
        self.assertEqual(block.ch_start_time(self.d1), 0)

    def test_channel_stop_time(self):
        """Test if correct stop time is calculated for each channel."""
        block = pulse.ScheduleBlock()
        for inst in self.test_blocks:
            block.append(inst)

        self.assertEqual(block.ch_stop_time(self.d0), 350)
        self.assertEqual(block.ch_stop_time(self.d1), 200)

    def test_cannot_insert(self):
        """Test insert is not supported."""
        block = pulse.ScheduleBlock()

        with self.assertRaises(PulseError):
            block.insert(0, pulse.Delay(10, self.d0))

    def test_cannot_shift(self):
        """Test shift is not supported."""
        block = pulse.ScheduleBlock()
        for inst in self.test_blocks:
            block.append(inst)

        with self.assertRaises(PulseError):
            block.shift(10, inplace=True)

    def test_cannot_append_schedule(self):
        """Test schedule cannot be appended. Schedule should be input as Call instruction."""
        block = pulse.ScheduleBlock()

        sched = pulse.Schedule()
        sched += pulse.Delay(10, self.d0)

        with self.assertRaises(PulseError):
            block.append(sched)

    def test_replace(self):
        """Test replacing specific instruction."""
        block = pulse.ScheduleBlock()
        for inst in self.test_blocks:
            block.append(inst)

        replaced = pulse.Play(pulse.Constant(300, 0.1), self.d1)
        target = pulse.Delay(50, self.d0)

        block_replaced = block.replace(target, replaced, inplace=False)

        # original schedule is not destroyed
        self.assertListEqual(list(block.blocks), self.test_blocks)

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform1, self.d1))
        ref_sched = ref_sched.insert(200, replaced)
        ref_sched = ref_sched.insert(100, pulse.Play(self.test_waveform1, self.d0))

        self.assertScheduleEqual(block_replaced, ref_sched)

    def test_replace_inplace(self):
        """Test replacing specific instruction with inplace."""
        block = pulse.ScheduleBlock()
        for inst in self.test_blocks:
            block.append(inst)

        replaced = pulse.Play(pulse.Constant(300, 0.1), self.d1)
        target = pulse.Delay(50, self.d0)

        block.replace(target, replaced, inplace=True)

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform1, self.d1))
        ref_sched = ref_sched.insert(200, replaced)
        ref_sched = ref_sched.insert(100, pulse.Play(self.test_waveform1, self.d0))

        self.assertScheduleEqual(block, ref_sched)

    def test_replace_block_by_instruction(self):
        """Test replacing block with instruction."""
        sub_block1 = pulse.ScheduleBlock()
        sub_block1 = sub_block1.append(pulse.Delay(50, self.d0))
        sub_block1 = sub_block1.append(pulse.Play(self.test_waveform0, self.d0))

        sub_block2 = pulse.ScheduleBlock()
        sub_block2 = sub_block2.append(pulse.Delay(50, self.d0))
        sub_block2 = sub_block2.append(pulse.Play(self.test_waveform1, self.d1))

        main_block = pulse.ScheduleBlock()
        main_block = main_block.append(pulse.Delay(50, self.d0))
        main_block = main_block.append(pulse.Play(self.test_waveform0, self.d0))
        main_block = main_block.append(sub_block1)
        main_block = main_block.append(sub_block2)
        main_block = main_block.append(pulse.Play(self.test_waveform0, self.d1))

        replaced = main_block.replace(sub_block1, pulse.Delay(100, self.d0))

        ref_blocks = [
            pulse.Delay(50, self.d0),
            pulse.Play(self.test_waveform0, self.d0),
            pulse.Delay(100, self.d0),
            sub_block2,
            pulse.Play(self.test_waveform0, self.d1),
        ]

        self.assertListEqual(list(replaced.blocks), ref_blocks)

    def test_replace_instruction_by_block(self):
        """Test replacing instruction with block."""
        sub_block1 = pulse.ScheduleBlock()
        sub_block1 = sub_block1.append(pulse.Delay(50, self.d0))
        sub_block1 = sub_block1.append(pulse.Play(self.test_waveform0, self.d0))

        sub_block2 = pulse.ScheduleBlock()
        sub_block2 = sub_block2.append(pulse.Delay(50, self.d0))
        sub_block2 = sub_block2.append(pulse.Play(self.test_waveform1, self.d1))

        main_block = pulse.ScheduleBlock()
        main_block = main_block.append(pulse.Delay(50, self.d0))
        main_block = main_block.append(pulse.Play(self.test_waveform0, self.d0))
        main_block = main_block.append(pulse.Delay(100, self.d0))
        main_block = main_block.append(sub_block2)
        main_block = main_block.append(pulse.Play(self.test_waveform0, self.d1))

        replaced = main_block.replace(pulse.Delay(100, self.d0), sub_block1)

        ref_blocks = [
            pulse.Delay(50, self.d0),
            pulse.Play(self.test_waveform0, self.d0),
            sub_block1,
            sub_block2,
            pulse.Play(self.test_waveform0, self.d1),
        ]

        self.assertListEqual(list(replaced.blocks), ref_blocks)

    def test_len(self):
        """Test __len__ method"""
        block = pulse.ScheduleBlock()
        self.assertEqual(len(block), 0)

        for j in range(1, 10):
            block = block.append(pulse.Delay(10, self.d0))
            self.assertEqual(len(block), j)

    def test_inherit_from(self):
        """Test creating schedule with another schedule."""
        ref_metadata = {"test": "value"}
        ref_name = "test"

        base_sched = pulse.ScheduleBlock(name=ref_name, metadata=ref_metadata)
        new_sched = pulse.ScheduleBlock.initialize_from(base_sched)

        self.assertEqual(new_sched.name, ref_name)
        self.assertDictEqual(new_sched.metadata, ref_metadata)


class TestBlockEquality(BaseTestBlock):
    """Test equality of blocks.

    Equality of instruction ordering is compared on DAG representation.
    This should be tested for each transform.
    """

    def test_different_channels(self):
        """Test equality is False if different channels."""
        block1 = pulse.ScheduleBlock()
        block1 += pulse.Delay(10, self.d0)

        block2 = pulse.ScheduleBlock()
        block2 += pulse.Delay(10, self.d1)

        self.assertNotEqual(block1, block2)

    def test_different_transform(self):
        """Test equality is False if different transforms."""
        block1 = pulse.ScheduleBlock(alignment_context=self.left_context)
        block1 += pulse.Delay(10, self.d0)

        block2 = pulse.ScheduleBlock(alignment_context=self.right_context)
        block2 += pulse.Delay(10, self.d0)

        self.assertNotEqual(block1, block2)

    def test_different_transform_opts(self):
        """Test equality is False if different transform options."""
        context1 = transforms.AlignEquispaced(duration=100)
        context2 = transforms.AlignEquispaced(duration=500)

        block1 = pulse.ScheduleBlock(alignment_context=context1)
        block1 += pulse.Delay(10, self.d0)

        block2 = pulse.ScheduleBlock(alignment_context=context2)
        block2 += pulse.Delay(10, self.d0)

        self.assertNotEqual(block1, block2)

    def test_instruction_out_of_order_left(self):
        """Test equality is True if two blocks have instructions in different order."""
        block1 = pulse.ScheduleBlock(alignment_context=self.left_context)
        block1 += pulse.Play(self.test_waveform0, self.d0)
        block1 += pulse.Play(self.test_waveform0, self.d1)

        block2 = pulse.ScheduleBlock(alignment_context=self.left_context)
        block2 += pulse.Play(self.test_waveform0, self.d1)
        block2 += pulse.Play(self.test_waveform0, self.d0)

        self.assertEqual(block1, block2)

    def test_instruction_in_order_left(self):
        """Test equality is True if two blocks have instructions in same order."""
        block1 = pulse.ScheduleBlock(alignment_context=self.left_context)
        block1 += pulse.Play(self.test_waveform0, self.d0)
        block1 += pulse.Play(self.test_waveform0, self.d1)

        block2 = pulse.ScheduleBlock(alignment_context=self.left_context)
        block2 += pulse.Play(self.test_waveform0, self.d0)
        block2 += pulse.Play(self.test_waveform0, self.d1)

        self.assertEqual(block1, block2)

    def test_instruction_out_of_order_right(self):
        """Test equality is True if two blocks have instructions in different order."""
        block1 = pulse.ScheduleBlock(alignment_context=self.right_context)
        block1 += pulse.Play(self.test_waveform0, self.d0)
        block1 += pulse.Play(self.test_waveform0, self.d1)

        block2 = pulse.ScheduleBlock(alignment_context=self.right_context)
        block2 += pulse.Play(self.test_waveform0, self.d1)
        block2 += pulse.Play(self.test_waveform0, self.d0)

        self.assertEqual(block1, block2)

    def test_instruction_in_order_right(self):
        """Test equality is True if two blocks have instructions in same order."""
        block1 = pulse.ScheduleBlock(alignment_context=self.right_context)
        block1 += pulse.Play(self.test_waveform0, self.d0)
        block1 += pulse.Play(self.test_waveform0, self.d1)

        block2 = pulse.ScheduleBlock(alignment_context=self.right_context)
        block2 += pulse.Play(self.test_waveform0, self.d0)
        block2 += pulse.Play(self.test_waveform0, self.d1)

        self.assertEqual(block1, block2)

    def test_instruction_out_of_order_sequential(self):
        """Test equality is False if two blocks have instructions in different order."""
        block1 = pulse.ScheduleBlock(alignment_context=self.sequential_context)
        block1 += pulse.Play(self.test_waveform0, self.d0)
        block1 += pulse.Play(self.test_waveform0, self.d1)

        block2 = pulse.ScheduleBlock(alignment_context=self.sequential_context)
        block2 += pulse.Play(self.test_waveform0, self.d1)
        block2 += pulse.Play(self.test_waveform0, self.d0)

        self.assertNotEqual(block1, block2)

    def test_instruction_in_order_sequential(self):
        """Test equality is True if two blocks have instructions in same order."""
        block1 = pulse.ScheduleBlock(alignment_context=self.sequential_context)
        block1 += pulse.Play(self.test_waveform0, self.d0)
        block1 += pulse.Play(self.test_waveform0, self.d1)

        block2 = pulse.ScheduleBlock(alignment_context=self.sequential_context)
        block2 += pulse.Play(self.test_waveform0, self.d0)
        block2 += pulse.Play(self.test_waveform0, self.d1)

        self.assertEqual(block1, block2)

    def test_instruction_out_of_order_equispaced(self):
        """Test equality is False if two blocks have instructions in different order."""
        block1 = pulse.ScheduleBlock(alignment_context=self.equispaced_context)
        block1 += pulse.Play(self.test_waveform0, self.d0)
        block1 += pulse.Play(self.test_waveform0, self.d1)

        block2 = pulse.ScheduleBlock(alignment_context=self.equispaced_context)
        block2 += pulse.Play(self.test_waveform0, self.d1)
        block2 += pulse.Play(self.test_waveform0, self.d0)

        self.assertNotEqual(block1, block2)

    def test_instruction_in_order_equispaced(self):
        """Test equality is True if two blocks have instructions in same order."""
        block1 = pulse.ScheduleBlock(alignment_context=self.equispaced_context)
        block1 += pulse.Play(self.test_waveform0, self.d0)
        block1 += pulse.Play(self.test_waveform0, self.d1)

        block2 = pulse.ScheduleBlock(alignment_context=self.equispaced_context)
        block2 += pulse.Play(self.test_waveform0, self.d0)
        block2 += pulse.Play(self.test_waveform0, self.d1)

        self.assertEqual(block1, block2)

    def test_instruction_out_of_order_func(self):
        """Test equality is False if two blocks have instructions in different order."""
        block1 = pulse.ScheduleBlock(alignment_context=self.func_context)
        block1 += pulse.Play(self.test_waveform0, self.d0)
        block1 += pulse.Play(self.test_waveform0, self.d1)

        block2 = pulse.ScheduleBlock(alignment_context=self.func_context)
        block2 += pulse.Play(self.test_waveform0, self.d1)
        block2 += pulse.Play(self.test_waveform0, self.d0)

        self.assertNotEqual(block1, block2)

    def test_instruction_in_order_func(self):
        """Test equality is True if two blocks have instructions in same order."""
        block1 = pulse.ScheduleBlock(alignment_context=self.func_context)
        block1 += pulse.Play(self.test_waveform0, self.d0)
        block1 += pulse.Play(self.test_waveform0, self.d1)

        block2 = pulse.ScheduleBlock(alignment_context=self.func_context)
        block2 += pulse.Play(self.test_waveform0, self.d0)
        block2 += pulse.Play(self.test_waveform0, self.d1)

        self.assertEqual(block1, block2)

    def test_instrution_in_oder_but_different_node(self):
        """Test equality is False if two blocks have different instructions."""
        block1 = pulse.ScheduleBlock(alignment_context=self.left_context)
        block1 += pulse.Play(self.test_waveform0, self.d0)
        block1 += pulse.Play(self.test_waveform1, self.d1)

        block2 = pulse.ScheduleBlock(alignment_context=self.left_context)
        block2 += pulse.Play(self.test_waveform0, self.d0)
        block2 += pulse.Play(self.test_waveform0, self.d1)

        self.assertNotEqual(block1, block2)

    def test_instruction_out_of_order_complex_equal(self):
        """Test complex schedule equality can be correctly evaluated."""
        block1_a = pulse.ScheduleBlock(alignment_context=self.left_context)
        block1_a += pulse.Delay(10, self.d0)
        block1_a += pulse.Play(self.test_waveform1, self.d1)
        block1_a += pulse.Play(self.test_waveform0, self.d0)

        block1_b = pulse.ScheduleBlock(alignment_context=self.left_context)
        block1_b += pulse.Play(self.test_waveform1, self.d1)
        block1_b += pulse.Delay(10, self.d0)
        block1_b += pulse.Play(self.test_waveform0, self.d0)

        block2_a = pulse.ScheduleBlock(alignment_context=self.right_context)
        block2_a += block1_a
        block2_a += block1_b
        block2_a += block1_a

        block2_b = pulse.ScheduleBlock(alignment_context=self.right_context)
        block2_b += block1_a
        block2_b += block1_a
        block2_b += block1_b

        self.assertEqual(block2_a, block2_b)

    def test_instruction_out_of_order_complex_not_equal(self):
        """Test complex schedule equality can be correctly evaluated."""
        block1_a = pulse.ScheduleBlock(alignment_context=self.left_context)
        block1_a += pulse.Play(self.test_waveform0, self.d0)
        block1_a += pulse.Play(self.test_waveform1, self.d1)
        block1_a += pulse.Delay(10, self.d0)

        block1_b = pulse.ScheduleBlock(alignment_context=self.left_context)
        block1_b += pulse.Play(self.test_waveform1, self.d1)
        block1_b += pulse.Delay(10, self.d0)
        block1_b += pulse.Play(self.test_waveform0, self.d0)

        block2_a = pulse.ScheduleBlock(alignment_context=self.right_context)
        block2_a += block1_a
        block2_a += block1_b
        block2_a += block1_a

        block2_b = pulse.ScheduleBlock(alignment_context=self.right_context)
        block2_b += block1_a
        block2_b += block1_a
        block2_b += block1_b

        self.assertNotEqual(block2_a, block2_b)


class TestParametrizedBlockOperation(BaseTestBlock):
    """Test fundamental operation with parametrization."""

    def setUp(self):
        super().setUp()

        self.amp0 = circuit.Parameter("amp0")
        self.amp1 = circuit.Parameter("amp1")
        self.dur0 = circuit.Parameter("dur0")
        self.dur1 = circuit.Parameter("dur1")

        self.test_par_waveform0 = pulse.Constant(self.dur0, self.amp0)
        self.test_par_waveform1 = pulse.Constant(self.dur1, self.amp1)

    def test_report_parameter_assignment(self):
        """Test duration assignment check."""
        block = pulse.ScheduleBlock()
        block += pulse.Play(self.test_par_waveform0, self.d0)

        # check parameter evaluation mechanism
        self.assertTrue(block.is_parameterized())
        self.assertFalse(block.is_schedulable())

        # assign duration
        block = block.assign_parameters({self.dur0: 200})
        self.assertTrue(block.is_parameterized())
        self.assertTrue(block.is_schedulable())

    def test_cannot_get_duration_if_not_assigned(self):
        """Test raise error when duration is not assigned."""
        block = pulse.ScheduleBlock()
        block += pulse.Play(self.test_par_waveform0, self.d0)

        with self.assertRaises(PulseError):
            #  pylint: disable=pointless-statement
            block.duration

    def test_get_assigend_duration(self):
        """Test duration is correctly evaluated."""
        block = pulse.ScheduleBlock()
        block += pulse.Play(self.test_par_waveform0, self.d0)
        block += pulse.Play(self.test_waveform0, self.d0)

        block = block.assign_parameters({self.dur0: 300})

        self.assertEqual(block.duration, 400)

    def test_nested_parametrized_instructions(self):
        """Test parameters of nested schedule can be assigned."""
        test_waveform = pulse.Constant(100, self.amp0)

        param_sched = pulse.Schedule(pulse.Play(test_waveform, self.d0))
        call_inst = pulse.instructions.Call(param_sched)

        sub_block = pulse.ScheduleBlock()
        sub_block += call_inst

        block = pulse.ScheduleBlock()
        block += sub_block

        self.assertTrue(block.is_parameterized())

        # assign durations
        block = block.assign_parameters({self.amp0: 0.1})
        self.assertFalse(block.is_parameterized())

    def test_equality_of_parametrized_channels(self):
        """Test check equality of blocks involving parametrized channels."""
        par_ch = circuit.Parameter("ch")

        block1 = pulse.ScheduleBlock(alignment_context=self.left_context)
        block1 += pulse.Play(self.test_waveform0, pulse.DriveChannel(par_ch))
        block1 += pulse.Play(self.test_par_waveform0, self.d0)

        block2 = pulse.ScheduleBlock(alignment_context=self.left_context)
        block2 += pulse.Play(self.test_par_waveform0, self.d0)
        block2 += pulse.Play(self.test_waveform0, pulse.DriveChannel(par_ch))

        self.assertEqual(block1, block2)

        block1_assigned = block1.assign_parameters({par_ch: 1})
        block2_assigned = block2.assign_parameters({par_ch: 1})
        self.assertEqual(block1_assigned, block2_assigned)

    def test_replace_parametrized_instruction(self):
        """Test parametrized instruction can updated with parameter table."""
        block = pulse.ScheduleBlock()
        block += pulse.Play(self.test_par_waveform0, self.d0)
        block += pulse.Delay(100, self.d0)
        block += pulse.Play(self.test_waveform0, self.d0)

        replaced = block.replace(
            pulse.Play(self.test_par_waveform0, self.d0),
            pulse.Play(self.test_par_waveform1, self.d0),
        )
        self.assertTrue(replaced.is_parameterized())

        # check assign parameters
        replaced_assigned = replaced.assign_parameters({self.dur1: 100, self.amp1: 0.1})
        self.assertFalse(replaced_assigned.is_parameterized())

    def test_parametrized_context(self):
        """Test parametrize context parameter."""
        duration = circuit.Parameter("dur")
        param_context = transforms.AlignEquispaced(duration=duration)

        block = pulse.ScheduleBlock(alignment_context=param_context)
        block += pulse.Delay(10, self.d0)
        block += pulse.Delay(10, self.d0)
        block += pulse.Delay(10, self.d0)
        block += pulse.Delay(10, self.d0)
        self.assertTrue(block.is_parameterized())
        self.assertFalse(block.is_schedulable())

        block.assign_parameters({duration: 100}, inplace=True)
        self.assertFalse(block.is_parameterized())
        self.assertTrue(block.is_schedulable())

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(0, pulse.Delay(10, self.d0))
        ref_sched = ref_sched.insert(30, pulse.Delay(10, self.d0))
        ref_sched = ref_sched.insert(60, pulse.Delay(10, self.d0))
        ref_sched = ref_sched.insert(90, pulse.Delay(10, self.d0))

        self.assertScheduleEqual(block, ref_sched)
