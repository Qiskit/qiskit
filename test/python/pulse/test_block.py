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

"""Test cases for the pulse schedule block."""

from qiskit import pulse
from qiskit.pulse.transforms import block_to_schedule
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

    def assertScheduleEqual(self, target, reference):
        """Check if two block are equal schedule representation."""
        self.assertEqual(block_to_schedule(target), reference)


class TestTransformation(BaseTestBlock):
    """Test conversion of ScheduleBlock to Schedule."""

    def test_left_alignment(self):
        """Test left alignment context."""
        block = pulse.ScheduleBlock(transform=pulse.transforms.AlignmentKind.left.name)
        block = block.append(pulse.Play(self.test_waveform0, self.d0))
        block = block.append(pulse.Play(self.test_waveform1, self.d1))

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform1, self.d1))

        self.assertScheduleEqual(block, ref_sched)

    def test_right_alignment(self):
        """Test right alignment context."""
        block = pulse.ScheduleBlock(transform=pulse.transforms.AlignmentKind.right.name)
        block = block.append(pulse.Play(self.test_waveform0, self.d0))
        block = block.append(pulse.Play(self.test_waveform1, self.d1))

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(100, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform1, self.d1))

        self.assertScheduleEqual(block, ref_sched)

    def test_sequential_alignment(self):
        """Test sequential alignment context."""
        block = pulse.ScheduleBlock(transform=pulse.transforms.AlignmentKind.sequential.name)
        block = block.append(pulse.Play(self.test_waveform0, self.d0))
        block = block.append(pulse.Play(self.test_waveform1, self.d1))

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(100, pulse.Play(self.test_waveform1, self.d1))

        self.assertScheduleEqual(block, ref_sched)

    def test_equispace_alignment(self):
        """Test equispace alignment context."""
        block = pulse.ScheduleBlock(transform=pulse.transforms.AlignmentKind.equispaced.name,
                                    duration=1000)
        for _ in range(4):
            block = block.append(pulse.Play(self.test_waveform0, self.d0))

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(100, pulse.Delay(200, self.d0))
        ref_sched = ref_sched.insert(300, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(400, pulse.Delay(200, self.d0))
        ref_sched = ref_sched.insert(600, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(700, pulse.Delay(200, self.d0))
        ref_sched = ref_sched.insert(900, pulse.Play(self.test_waveform0, self.d0))

        self.assertScheduleEqual(block, ref_sched)

    def test_func_alignment(self):
        """Test func alignment context."""
        def align_func(j):
            return {1: 0.1, 2: 0.25, 3: 0.7, 4: 0.85}.get(j)

        block = pulse.ScheduleBlock(transform=pulse.transforms.AlignmentKind.func.name,
                                    duration=1000, func=align_func)
        for _ in range(4):
            block = block.append(pulse.Play(self.test_waveform0, self.d0))

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(0, pulse.Delay(50, self.d0))
        ref_sched = ref_sched.insert(50, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(150, pulse.Delay(50, self.d0))
        ref_sched = ref_sched.insert(200, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(300, pulse.Delay(350, self.d0))
        ref_sched = ref_sched.insert(650, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(750, pulse.Delay(50, self.d0))
        ref_sched = ref_sched.insert(800, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(900, pulse.Delay(100, self.d0))

        self.assertScheduleEqual(block, ref_sched)

    def test_nested_alignment(self):
        """Test nested block scheduling."""
        block_sub = pulse.ScheduleBlock(transform=pulse.transforms.AlignmentKind.right.name)
        block_sub = block_sub.append(pulse.Play(self.test_waveform0, self.d0))
        block_sub = block_sub.append(pulse.Play(self.test_waveform1, self.d1))

        block_main = pulse.ScheduleBlock(transform=pulse.transforms.AlignmentKind.sequential.name)
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
            pulse.Play(self.test_waveform1, self.d0)
        ]

    def test_append_an_instruction_to_empty_block(self):
        """Test append instructions to an empty block."""
        block = pulse.ScheduleBlock()
        block = block.append(pulse.Play(self.test_waveform0, self.d0))

        self.assertEqual(block.instructions[0], pulse.Play(self.test_waveform0, self.d0))

    def test_append_an_instruction_to_empty_block_sugar(self):
        """Test append instructions to an empty block with syntax sugar."""
        block = pulse.ScheduleBlock()
        block += pulse.Play(self.test_waveform0, self.d0)

        self.assertEqual(block.instructions[0], pulse.Play(self.test_waveform0, self.d0))

    def test_append_an_instruction_to_empty_block_inplace(self):
        """Test append instructions to an empty block with inplace."""
        block = pulse.ScheduleBlock()
        block.append(pulse.Play(self.test_waveform0, self.d0), inplace=True)

        self.assertEqual(block.instructions[0], pulse.Play(self.test_waveform0, self.d0))

    def test_append_a_block_to_empty_block(self):
        """Test append another ScheduleBlock to empty block."""
        block = pulse.ScheduleBlock()
        block.append(pulse.Play(self.test_waveform0, self.d0), inplace=True)

        block_main = pulse.ScheduleBlock()
        block_main = block_main.append(block)

        self.assertEqual(block_main.instructions[0], block)

    def test_append_an_instruction_to_block(self):
        """Test append instructions to a non-empty block."""
        block = pulse.ScheduleBlock()
        block = block.append(pulse.Delay(100, self.d0))

        block = block.append(pulse.Delay(100, self.d0))

        self.assertEqual(len(block.instructions), 2)

    def test_append_an_instruction_to_block_inplace(self):
        """Test append instructions to a non-empty block with inplace."""
        block = pulse.ScheduleBlock()
        block = block.append(pulse.Delay(100, self.d0))

        block.append(pulse.Delay(100, self.d0), inplace=True)

        self.assertEqual(len(block.instructions), 2)

    def test_duration(self):
        """Test if correct duration is returned with implicit scheduling."""
        block = pulse.ScheduleBlock(*self.test_blocks)

        self.assertEqual(block.duration, 350)

    def test_timeslots(self):
        """Test if correct timeslot is returned with implicit scheduling."""
        block = pulse.ScheduleBlock(*self.test_blocks)

        ref_slots = {
            self.d0: [(0, 100), (100, 150), (150, 350)],
            self.d1: [(0, 200)]
        }

        self.assertDictEqual(block.timeslots, ref_slots)

    def test_start_time(self):
        """Test if correct schedule start time is returned with implicit scheduling."""
        block = pulse.ScheduleBlock(*self.test_blocks)

        self.assertEqual(block.start_time, 0)

    def test_stop_time(self):
        """Test if correct schedule stop time is returned with implicit scheduling."""
        block = pulse.ScheduleBlock(*self.test_blocks)

        self.assertEqual(block.stop_time, 350)

    def test_channels(self):
        """Test if all channels are returned."""
        block = pulse.ScheduleBlock(*self.test_blocks)

        self.assertEqual(len(block.channels), 2)

    def test_instructions(self):
        """Test if all instructions are returned."""
        block = pulse.ScheduleBlock(*self.test_blocks)

        self.assertEqual(block.instructions, tuple(self.test_blocks))

    def test_channel_duraction(self):
        """Test if correct durations is calculated for each channel."""
        block = pulse.ScheduleBlock(*self.test_blocks)

        self.assertEqual(block.ch_duration(self.d0), 350)
        self.assertEqual(block.ch_duration(self.d1), 200)

    def test_channel_start_time(self):
        """Test if correct start time is calculated for each channel."""
        block = pulse.ScheduleBlock(*self.test_blocks)

        self.assertEqual(block.ch_start_time(self.d0), 0)
        self.assertEqual(block.ch_start_time(self.d1), 0)

    def test_channel_stop_time(self):
        """Test if correct stop time is calculated for each channel."""
        block = pulse.ScheduleBlock(*self.test_blocks)

        self.assertEqual(block.ch_stop_time(self.d0), 350)
        self.assertEqual(block.ch_stop_time(self.d1), 200)

    def test_cannot_insert(self):
        """Test insert is not supported."""
        block = pulse.ScheduleBlock()

        with self.assertRaises(PulseError):
            block.insert(0, pulse.Delay(10, self.d0))

    def test_cannot_shift(self):
        """Test shift is not supported."""
        block = pulse.ScheduleBlock(*self.test_blocks)

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
        block = pulse.ScheduleBlock(*self.test_blocks)

        replaced = pulse.Play(pulse.Constant(300, 0.1), self.d1)
        target = pulse.Delay(50, self.d0)

        block_replaced = block.replace(target, replaced)

        # original schedule is not destroyed
        self.assertListEqual(list(block.instructions), self.test_blocks)

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform1, self.d1))
        ref_sched = ref_sched.insert(200, replaced)
        ref_sched = ref_sched.insert(100, pulse.Play(self.test_waveform1, self.d0))

        self.assertScheduleEqual(block_replaced, ref_sched)

    def test_replace_inplace(self):
        """Test replacing specific instruction with inplace."""
        block = pulse.ScheduleBlock(*self.test_blocks)

        replaced = pulse.Play(pulse.Constant(300, 0.1), self.d1)
        target = pulse.Delay(50, self.d0)

        block.replace(target, replaced, inplace=True)

        ref_sched = pulse.Schedule()
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform0, self.d0))
        ref_sched = ref_sched.insert(0, pulse.Play(self.test_waveform1, self.d1))
        ref_sched = ref_sched.insert(200, replaced)
        ref_sched = ref_sched.insert(100, pulse.Play(self.test_waveform1, self.d0))

        self.assertScheduleEqual(block, ref_sched)

    def test_len(self):
        """Test __len__ method"""
        block = pulse.ScheduleBlock()
        self.assertEqual(len(block), 0)

        for j in range(1, 10):
            block = block.append(pulse.Delay(10, self.d0))
            self.assertEqual(len(block), j)


class TestBlockEquality(BaseTestBlock):
    """Test equality of blocks."""

    def test_different_channels(self):
        """Test equality is False if different channels."""
        self.assertNotEqual(pulse.ScheduleBlock(pulse.Delay(10, self.d0)),
                            pulse.ScheduleBlock(pulse.Delay(10, self.d1)))

    def test_different_transform(self):
        """Test equality is False if different transforms."""
        self.assertNotEqual(pulse.ScheduleBlock(pulse.Delay(10, self.d0),
                                                transform='left'),
                            pulse.ScheduleBlock(pulse.Delay(10, self.d0),
                                                transform='right'))

    def test_different_transform_opts(self):
        """Test equality is False if different transform options."""
        self.assertNotEqual(pulse.ScheduleBlock(pulse.Delay(10, self.d0),
                                                transform='equispaced',
                                                duration=100),
                            pulse.ScheduleBlock(pulse.Delay(10, self.d0),
                                                transform='equispaced',
                                                duration=500))

    def test_instruction_out_of_order(self):
        """Test equality is False if instructions are out of order."""
        self.assertNotEqual(pulse.ScheduleBlock(pulse.Delay(10, self.d0),
                                                pulse.Play(self.test_waveform0, self.d1)),
                            pulse.ScheduleBlock(pulse.Delay(10, )))












