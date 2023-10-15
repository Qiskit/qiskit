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

"""Test pulse logical elements and frames"""

from qiskit.pulse import (
    Qubit,
    GenericFrame,
    MixedFrame,
)
from qiskit.test import QiskitTestCase


class TestMixedFrames(QiskitTestCase):
    """Test mixed frames."""

    def test_mixed_frame_initialization(self):
        """Test that MixedFrame objects are created correctly"""
        frame = GenericFrame("frame1")
        qubit = Qubit(1)
        mixed_frame = MixedFrame(qubit, frame)
        self.assertEqual(mixed_frame.logical_element, qubit)
        self.assertEqual(mixed_frame.frame, frame)

    def test_mixed_frames_comparison(self):
        """Test the comparison of various mixed frames"""
        self.assertEqual(
            MixedFrame(Qubit(1), GenericFrame("a")),
            MixedFrame(Qubit(1), GenericFrame("a")),
        )

        self.assertNotEqual(
            MixedFrame(Qubit(1), GenericFrame("a")),
            MixedFrame(Qubit(2), GenericFrame("a")),
        )
        self.assertNotEqual(
            MixedFrame(Qubit(1), GenericFrame("a")),
            MixedFrame(Qubit(1), GenericFrame("b")),
        )

    def test_mixed_frame_repr(self):
        """Test MixedFrame __repr__"""
        frame = GenericFrame("frame1")
        qubit = Qubit(1)
        mixed_frame = MixedFrame(qubit, frame)
        self.assertEqual(str(mixed_frame), f"MixedFrame({qubit},{frame})")
