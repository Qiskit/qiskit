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
    Qubit,
    Coupler,
    GenericFrame,
    QubitFrame,
    MeasurementFrame,
    MixedFrame,
    CRMixedFrame,
)
from qiskit.test import QiskitTestCase


class TestLogicalElements(QiskitTestCase):
    """Test logical elements."""

    def test_qubit_initialization(self):
        """Test that Qubit type logical elements are created and validated correctly"""
        qubit = Qubit(0)
        self.assertEqual(qubit.index, 0)
        self.assertEqual(qubit.name, "Q0")

        with self.assertRaises(PulseError):
            Qubit(0.5)
        with self.assertRaises(PulseError):
            Qubit(-0.5)
        with self.assertRaises(PulseError):
            Qubit(-1)

    def test_coupler_initialization(self):
        """Test that Coupler type logical elements are created and validated correctly"""
        coupler = Coupler(0, 3)
        self.assertEqual(coupler.index, (0, 3))
        self.assertEqual(coupler.name, "Coupler(0, 3)")

        with self.assertRaises(PulseError):
            Coupler(-1, 0)
        with self.assertRaises(PulseError):
            Coupler(2, -0.5)
        with self.assertRaises(PulseError):
            Coupler(3, -1)

    def test_logical_elements_comparison(self):
        """Test the comparison of various logical elements"""
        self.assertEqual(Qubit(0), Qubit(0))
        self.assertNotEqual(Qubit(0), Qubit(1))

        self.assertEqual(Coupler(0, 1), Coupler(0, 1))
        self.assertNotEqual(Coupler(0, 1), Coupler(0, 2))


class TestFrames(QiskitTestCase):
    """Test frames."""

    def test_generic_frame_initialization(self):
        """Test that Frame objects are created correctly"""
        frame = GenericFrame(name="frame1", frequency=100.2, phase=1.3)
        self.assertEqual(frame.name, "GenericFrame(frame1)")
        self.assertEqual(frame.frequency, 100.2)
        self.assertEqual(frame.phase, 1.3)

        frame = GenericFrame(name="frame1", frequency=100.2)
        self.assertEqual(frame.phase, 0)

    def test_generic_frame_comparison(self):
        """Test that GenericFrame objects are compared correctly"""
        frame1 = GenericFrame(name="frame1", frequency=100.2, phase=1.3)

        self.assertEqual(frame1, GenericFrame(name="frame1", frequency=100.2, phase=1.3))
        self.assertNotEqual(frame1, GenericFrame(name="frame2", frequency=100.2, phase=1.3))
        self.assertNotEqual(frame1, GenericFrame(name="frame1", frequency=50.2, phase=1.3))
        self.assertNotEqual(frame1, GenericFrame(name="frame1", frequency=100.2))
        self.assertNotEqual(frame1, QubitFrame(3))

    def test_qubit_frame_initialization(self):
        """Test that QubitFrame type frames are created and validated correctly"""
        frame = QubitFrame(2)
        self.assertEqual(frame.qubit_index, 2)
        self.assertEqual(frame.name, "QubitFrame2")

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
        self.assertEqual(frame.qubit_index, 2)
        self.assertEqual(frame.name, "MeasurementFrame2")

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


class TestMixedFrames(QiskitTestCase):
    """Test mixed frames."""

    def test_mixed_frame_initialization(self):
        """Test that MixedFrame objects are created correctly"""
        frame = GenericFrame("frame1", 106.2)
        qubit = Qubit(1)
        mixed_frame = MixedFrame(qubit, frame)
        self.assertEqual(mixed_frame.logical_element, qubit)
        self.assertEqual(mixed_frame.frame, frame)
        self.assertEqual(mixed_frame.name, f"MixedFrame({qubit.name},{frame.name})")

    def test_cr_mixed_frame_initialization(self):
        """Test that CRMixedFrame type mixed frames are created correctly"""
        frame = QubitFrame(3)
        qubit = Qubit(1)
        mixed_frame = CRMixedFrame(qubit, frame)
        self.assertEqual(mixed_frame.logical_element, qubit)
        self.assertEqual(mixed_frame.logical_element, mixed_frame.qubit)
        self.assertEqual(mixed_frame.frame, frame)
        self.assertEqual(mixed_frame.qubit_frame, mixed_frame.frame)
        self.assertEqual(mixed_frame.name, f"MixedFrame({qubit.name},{frame.name})")

    def test_mixed_frames_comparison(self):
        """Test the comparison of various mixed frames"""
        self.assertEqual(
            MixedFrame(Qubit(1), GenericFrame("a", 10.2)),
            MixedFrame(Qubit(1), GenericFrame("a", 10.2)),
        )
        self.assertEqual(MixedFrame(Qubit(1), QubitFrame(3)), CRMixedFrame(Qubit(1), QubitFrame(3)))

        self.assertNotEqual(
            MixedFrame(Qubit(1), GenericFrame("a", 106.1)),
            MixedFrame(Qubit(2), GenericFrame("a", 106.1)),
        )
        self.assertNotEqual(
            MixedFrame(Qubit(1), GenericFrame("a", 106.1)),
            MixedFrame(Qubit(1), GenericFrame("b", 106.1)),
        )
