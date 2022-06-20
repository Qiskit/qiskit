# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for the schedule block qpy loading and saving."""

import io
from ddt import ddt, data, unpack

import numpy as np

from qiskit.pulse import builder
from qiskit.pulse.library import (
    SymbolicPulse,
    Gaussian,
    GaussianSquare,
    Drag,
    Constant,
)
from qiskit.pulse.channels import (
    DriveChannel,
    ControlChannel,
    MeasureChannel,
    AcquireChannel,
    MemorySlot,
    RegisterSlot,
)
from qiskit.circuit import Parameter
from qiskit.test import QiskitTestCase
from qiskit.qpy import dump, load
from qiskit.utils import optionals as _optional


if _optional.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym


class QpyScheduleTestCase(QiskitTestCase):
    """QPY schedule testing platform."""

    def assert_roundtrip_equal(self, block):
        """QPY roundtrip equal test."""
        qpy_file = io.BytesIO()
        dump(block, qpy_file)
        qpy_file.seek(0)
        new_block = load(qpy_file)[0]

        self.assertEqual(block, new_block)


@ddt
class TestLoadFromQPY(QpyScheduleTestCase):
    """Test loading and saving schedule block to qpy file."""

    @data(
        (Gaussian, DriveChannel, 160, 0.1, 40),
        (GaussianSquare, DriveChannel, 800, 0.1, 64, 544),
        (Drag, DriveChannel, 160, 0.1, 40, 0.5),
        (Constant, DriveChannel, 800, 0.1),
        (Constant, ControlChannel, 800, 0.1),
        (Constant, MeasureChannel, 800, 0.1),
    )
    @unpack
    def test_library_pulse_play(self, envelope, channel, *params):
        """Test playing standard pulses."""
        with builder.build() as test_sched:
            builder.play(
                envelope(*params),
                channel(0),
            )
        self.assert_roundtrip_equal(test_sched)

    def test_playing_custom_symbolic_pulse(self):
        """Test playing a custom user pulse."""
        # pylint: disable=invalid-name
        t, amp, freq = sym.symbols("t, amp, freq")
        sym_envelope = 2 * amp * (freq * t - sym.floor(1 / 2 + freq * t))

        my_pulse = SymbolicPulse(
            pulse_type="Sawtooth",
            duration=100,
            parameters={"amp": 0.1, "freq": 0.05},
            envelope=sym_envelope,
            name="pulse1",
        )
        with builder.build() as test_sched:
            builder.play(my_pulse, DriveChannel(0))
        self.assert_roundtrip_equal(test_sched)

    def test_playing_waveform(self):
        """Test playing waveform."""
        # pylint: disable=invalid-name
        t = np.linspace(0, 1, 100)
        waveform = 0.1 * np.sin(2 * np.pi * t)
        with builder.build() as test_sched:
            builder.play(waveform, DriveChannel(0))
        self.assert_roundtrip_equal(test_sched)

    def test_phases(self):
        """Test phase."""
        with builder.build() as test_sched:
            builder.shift_phase(0.1, DriveChannel(0))
            builder.set_phase(0.4, DriveChannel(1))
        self.assert_roundtrip_equal(test_sched)

    def test_frequencies(self):
        """Test frequency."""
        with builder.build() as test_sched:
            builder.shift_frequency(10e6, DriveChannel(0))
            builder.set_frequency(5e9, DriveChannel(1))
        self.assert_roundtrip_equal(test_sched)

    def test_delay(self):
        """Test delay."""
        with builder.build() as test_sched:
            builder.delay(100, DriveChannel(0))
        self.assert_roundtrip_equal(test_sched)

    def test_barrier(self):
        """Test barrier."""
        with builder.build() as test_sched:
            builder.barrier(DriveChannel(0), DriveChannel(1), ControlChannel(2))
        self.assert_roundtrip_equal(test_sched)

    def test_measure(self):
        """Test measurement."""
        with builder.build() as test_sched:
            builder.acquire(100, AcquireChannel(0), MemorySlot(0))
            builder.acquire(100, AcquireChannel(1), RegisterSlot(1))
        self.assert_roundtrip_equal(test_sched)

    @data(
        (0, Parameter("dur"), 0.1, 40),
        (Parameter("ch1"), 160, 0.1, 40),
        (Parameter("ch1"), Parameter("dur"), Parameter("amp"), Parameter("sigma")),
        (0, 160, Parameter("amp") * np.exp(1j * Parameter("phase")), 40),
    )
    @unpack
    def test_parameterized(self, channel, *params):
        """Test playing parameterized pulse."""
        # pylint: disable=no-value-for-parameter
        with builder.build() as test_sched:
            builder.play(Gaussian(*params), DriveChannel(channel))
        self.assert_roundtrip_equal(test_sched)

    def test_nested_blocks(self):
        """Test nested blocks with different alignment contexts."""
        with builder.build() as test_sched:
            with builder.align_equispaced(duration=1200):
                with builder.align_left():
                    builder.delay(100, DriveChannel(0))
                    builder.delay(200, DriveChannel(1))
                with builder.align_right():
                    builder.delay(100, DriveChannel(0))
                    builder.delay(200, DriveChannel(1))
                with builder.align_sequential():
                    builder.delay(100, DriveChannel(0))
                    builder.delay(200, DriveChannel(1))
        self.assert_roundtrip_equal(test_sched)

    def test_bell_schedule(self):
        """Test complex schedule to create a Bell state."""
        with builder.build() as test_sched:
            with builder.align_sequential():
                # H
                builder.shift_phase(-1.57, DriveChannel(0))
                builder.play(Drag(160, 0.05, 40, 1.3), DriveChannel(0))
                builder.shift_phase(-1.57, DriveChannel(0))
                # ECR
                with builder.align_left():
                    builder.play(GaussianSquare(800, 0.05, 64, 544), DriveChannel(1))
                    builder.play(GaussianSquare(800, 0.1 - 0.2j, 64, 544), ControlChannel(0))
                builder.play(Drag(160, 0.1, 40, 1.5), DriveChannel(0))
                with builder.align_left():
                    builder.play(GaussianSquare(800, -0.05, 64, 544), DriveChannel(1))
                    builder.play(GaussianSquare(800, -0.1 + 0.2j, 64, 544), ControlChannel(0))
                builder.play(Drag(160, 0.1, 40, 1.5), DriveChannel(0))
                # Measure
                with builder.align_left():
                    builder.play(GaussianSquare(8000, 0.2, 64, 7744), MeasureChannel(0))
                    builder.acquire(8000, AcquireChannel(0), MemorySlot(0))

        self.assert_roundtrip_equal(test_sched)
