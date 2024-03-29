# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for pulse target for pulse compiler."""

from ddt import ddt, named_data, unpack

from qiskit.pulse import model
from qiskit.pulse.compiler.target import QiskitPulseTarget
from qiskit.pulse.exceptions import NotExistingComponent
from test import QiskitTestCase  # pylint: disable=wrong-import-order

# TODO update test


@ddt
class TestPulseTarget(QiskitTestCase):
    """Test cases of hardware information acquitision from QiskitPulseTarget with mock data."""

    def setUp(self):
        super().setUp()
        self.target = QiskitPulseTarget(
            qubit_frames={
                0: "Q0",
                1: "Q1",
            },
            meas_frames={
                0: "M0",
                1: "M1",
            },
            qubit_ports={
                0: "Port0",
                1: "Port1",
                2: "OnlyPort",
            },
            coupler_ports={
                (0, 1): "PortA",
            },
            mixed_frames={
                "Port0": ["Q0", "Q1", "M0", "EX0"],
                "Port1": ["Q1", "M1", "EX1", "EX2"],
            },
        )

    @named_data(
        ["qubit0", model.Qubit(0), "Port0"],
        ["qubit1", model.Qubit(1), "Port1"],
        ["coupler", model.Coupler(0, 1), "PortA"],
    )
    @unpack
    def test_get_port_uid(self, port, uid):
        """Test get Qiskit port UID from pulse target."""
        self.assertEqual(self.target.get_port_identifier(port), uid)

    def test_get_generic_port_fail(self):
        """Test get cannot get port UID with generic Port object."""
        with self.assertRaises(TypeError):
            self.target.get_port_identifier(model.Port("GenericPort"))

    def test_port_not_found(self):
        """Test port is not available in pulse target."""
        with self.assertRaises(NotExistingComponent):
            self.target.get_port_identifier(model.Qubit(100))

    @named_data(
        ["frame_q0", model.QubitFrame(0), "Q0"],
        ["frame_q1", model.QubitFrame(1), "Q1"],
        ["frame_m0", model.MeasurementFrame(0), "M0"],
        ["frame_m1", model.MeasurementFrame(1), "M1"],
    )
    def test_get_frame_uid(self, frame, uid):
        """Test get Qiskit frame UID from pulse target."""
        self.assertEqual(self.target.get_frame_identifier(frame), uid)

    def test_get_generic_frame_fail(self):
        """Test get cannot get frame UID with GenericFrame object."""
        with self.assertRaises(TypeError):
            self.target.get_frame_identifier(model.GenericFrame("GenericFrame"))

    def test_frame_not_found(self):
        """Test frame is not available in pulse target."""
        with self.assertRaises(NotExistingComponent):
            self.target.get_frame_identifier(model.QubitFrame(100))

    @named_data(
        [
            "qubit0",
            model.Qubit(0),
            [model.GenericFrame("EX0")],
        ],
        [
            "qubit1",
            model.Qubit(1),
            [model.GenericFrame("EX1"), model.GenericFrame("EX2")],
        ],
    )
    @unpack
    def test_get_extra_frames(self, port, frames):
        """Test get extra frames tied to a particular Qiskit port."""
        self.assertListEqual(self.target.extra_frames(port), frames)

    def test_get_extra_frames_not_available_for_port(self):
        """Test port exists but no mixed frame is defined."""
        with self.assertRaises(NotExistingComponent):
            self.target.extra_frames(model.Qubit(2))

    @named_data(
        ["q0_port0", model.MixedFrame(model.Qubit(0), model.QubitFrame(0))],
        ["q1_port0", model.MixedFrame(model.Qubit(0), model.QubitFrame(1))],
        ["m0_port0", model.MixedFrame(model.Qubit(0), model.MeasurementFrame(0))],
    )
    def test_check_mixed_frame_available(self, mixed_frame):
        """Test available mixed frames on this target."""
        self.assertTrue(self.target.is_mixed_frame_available(mixed_frame))

    @named_data(
        ["q0_port1", model.MixedFrame(model.Qubit(1), model.QubitFrame(0))],
        ["q0_port100", model.MixedFrame(model.Qubit(100), model.QubitFrame(0))],
        ["m1_port0", model.MixedFrame(model.Qubit(0), model.MeasurementFrame(1))],
    )
    def test_check_mixed_frame_not_available(self, mixed_frame):
        """Test unavailable mixed frames on this target."""
        self.assertFalse(self.target.is_mixed_frame_available(mixed_frame))

    def test_filter_mixed_frame_only_port(self):
        """Test get mixed frame list filtered by port."""
        self.assertListEqual(
            self.target.filter_mixed_frames(port=model.Qubit(0)),
            [
                model.MixedFrame(model.Qubit(0), model.QubitFrame(0)),
                model.MixedFrame(model.Qubit(0), model.QubitFrame(1)),
                model.MixedFrame(model.Qubit(0), model.MeasurementFrame(0)),
                model.MixedFrame(model.Qubit(0), model.GenericFrame("EX0")),
            ],
        )

    def test_filter_mixed_frame_only_frame(self):
        """Test get mixed frame list filtered by frame."""
        self.assertListEqual(
            self.target.filter_mixed_frames(frame=model.QubitFrame(1)),
            [
                model.MixedFrame(model.Qubit(0), model.QubitFrame(1)),
                model.MixedFrame(model.Qubit(1), model.QubitFrame(1)),
            ],
        )

    def test_filter_mixed_frame_both_port_frame(self):
        """Test get mixed frame list filtered by both port and frame."""
        self.assertListEqual(
            self.target.filter_mixed_frames(frame=model.QubitFrame(1), port=model.Qubit(0)),
            [
                model.MixedFrame(model.Qubit(0), model.QubitFrame(1)),
            ],
        )

    def test_filter_mixed_frame_non_existing(self):
        """Test no mixed frame mached with the condition."""
        self.assertListEqual(
            self.target.filter_mixed_frames(port=model.Qubit(100)),
            [],
        )
