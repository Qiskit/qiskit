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
    PulseError,
    GenericFrame,
    QubitFrame,
    MeasurementFrame,
)
from qiskit.test import QiskitTestCase


class TestFrames(QiskitTestCase):
    """Test frames."""

    def test_generic_frame_initialization(self):
        """Test that Frame objects are created correctly"""
        frame = GenericFrame(name="frame1")
        self.assertEqual(frame.name, "frame1")
        self.assertEqual(str(frame), "GenericFrame(frame1)")

    def test_generic_frame_comparison(self):
        """Test that GenericFrame objects are compared correctly"""
        frame1 = GenericFrame(name="frame1")

        self.assertEqual(frame1, GenericFrame(name="frame1"))
        self.assertNotEqual(frame1, GenericFrame(name="frame2"))
        self.assertNotEqual(frame1, QubitFrame(3))

    def test_qubit_frame_initialization(self):
        """Test that QubitFrame type frames are created and validated correctly"""
        frame = QubitFrame(2)
        self.assertEqual(frame.index, 2)
        self.assertEqual(str(frame), "QubitFrame(2)")

        with self.assertRaises(PulseError):
            QubitFrame(0.5)
        with self.assertRaises(PulseError):
            QubitFrame(-0.5)
        with self.assertRaises(PulseError):
            QubitFrame(-1)

    def test_qubit_frame_comparison(self):
        """Test the comparison of QubitFrame"""
        self.assertEqual(QubitFrame(0), QubitFrame(0))
        self.assertNotEqual(QubitFrame(0), QubitFrame(1))
        self.assertNotEqual(MeasurementFrame(0), QubitFrame(0))

    def test_measurement_frame_initialization(self):
        """Test that MeasurementFrame type frames are created and validated correctly"""
        frame = MeasurementFrame(2)
        self.assertEqual(frame.index, 2)
        self.assertEqual(str(frame), "MeasurementFrame(2)")

        with self.assertRaises(PulseError):
            MeasurementFrame(0.5)
        with self.assertRaises(PulseError):
            MeasurementFrame(-0.5)
        with self.assertRaises(PulseError):
            MeasurementFrame(-1)

    def test_measurement_frame_comparison(self):
        """Test the comparison of measurement frames"""
        self.assertEqual(MeasurementFrame(0), MeasurementFrame(0))
        self.assertNotEqual(MeasurementFrame(0), MeasurementFrame(1))
        self.assertNotEqual(MeasurementFrame(0), QubitFrame(0))
