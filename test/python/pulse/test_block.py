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

import unittest

from qiskit import pulse
from qiskit.pulse.transforms import block_to_schedule
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q


class BaseTestBlock(QiskitTestCase):
    """ScheduleBlock tests."""

    def setup(self):
        super().setup()

        self.backend = FakeOpenPulse2Q()

        self.test_waveform = pulse.Constant(100, 0.1)
        self.d0 = pulse.DriveChannel(0)
        self.d1 = pulse.DriveChannel(1)

        self.play_d0 = pulse.Play(self.test_waveform, self.d0)
        self.play_d1 = pulse.Play(self.test_waveform, self.d1)

    def assertScheduleEqual(self, target, reference):
        """Check if two block are equal schedule representation."""
        if not isinstance(target, pulse.Schedule):
            target = block_to_schedule(target)

        if not isinstance(reference, pulse.Schedule):
            reference = block_to_schedule(reference)

        self.assertEqual(target, reference)


class TestBlockBuilding(BaseTestBlock):
    """Test construction of block."""

    def test_append_an_instruction_to_empty_block(self):
        """Test append instructions to an empty block."""
        block = pulse.ScheduleBlock()
        block = block.append(self.play_d0)

        self.assertEqual(block.instructions[0], self.play_d0)



class TestReplace(BaseTestBlock):
    """Test block replacement."""
    pass


class TestDelay(BaseTestBlock):
    """Test Delay Instruction"""
    pass


class TestBlockFilter(BaseTestBlock):
    """Test Schedule filtering methods"""
    pass


class TestBlockEquality(BaseTestBlock):
    """Test equality of blocks."""
    pass


class TestTransformation(BaseTestBlock):
    """Test block transformation."""
    pass


if __name__ == '__main__':
    unittest.main()
