# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for Pulse Macro functions."""

from qiskit.providers.fake_provider import FakeOpenPulse2Q
from qiskit.pulse import (
    Schedule,
    AcquireChannel,
    Acquire,
    InstructionScheduleMap,
    MeasureChannel,
    MemorySlot,
    Constant,
    GaussianFallEdge,
    GaussianRiseEdge,
    GaussianSquare,
    Play,
    DriveChannel,
)
from qiskit.pulse import macros
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.transforms import block_to_schedule
from qiskit.test import QiskitTestCase


class TestMeasure(QiskitTestCase):
    """Pulse measure macro."""

    def setUp(self):
        super().setUp()
        self.backend = FakeOpenPulse2Q()
        self.inst_map = self.backend.defaults().instruction_schedule_map

    def test_measure(self):
        """Test macro - measure."""
        sched = macros.measure(qubits=[0], backend=self.backend)
        expected = Schedule(
            self.inst_map.get("measure", [0, 1]).filter(channels=[MeasureChannel(0)]),
            Acquire(10, AcquireChannel(0), MemorySlot(0)),
        )

        self.assertEqual(sched.instructions, expected.instructions)

    def test_measure_sched_with_qubit_mem_slots(self):
        """Test measure with custom qubit_mem_slots."""
        sched = macros.measure(qubits=[0], backend=self.backend, qubit_mem_slots={0: 1})
        expected = Schedule(
            self.inst_map.get("measure", [0, 1]).filter(channels=[MeasureChannel(0)]),
            Acquire(10, AcquireChannel(0), MemorySlot(1)),
        )
        self.assertEqual(sched.instructions, expected.instructions)

    def test_measure_sched_with_meas_map(self):
        """Test measure with custom meas_map as list and dict."""
        sched_with_meas_map_list = macros.measure(
            qubits=[0], backend=self.backend, meas_map=[[0, 1]]
        )
        sched_with_meas_map_dict = macros.measure(
            qubits=[0], backend=self.backend, meas_map={0: [0, 1], 1: [0, 1]}
        )
        expected = Schedule(
            self.inst_map.get("measure", [0, 1]).filter(channels=[MeasureChannel(0)]),
            Acquire(10, AcquireChannel(0), MemorySlot(0)),
        )
        self.assertEqual(sched_with_meas_map_list.instructions, expected.instructions)
        self.assertEqual(sched_with_meas_map_dict.instructions, expected.instructions)

    def test_measure_with_custom_inst_map(self):
        """Test measure with custom inst_map, meas_map with measure_name."""
        q0_sched = Play(GaussianSquare(1200, 1, 0.4, 1150), MeasureChannel(0))
        q0_sched += Acquire(1200, AcquireChannel(0), MemorySlot(0))
        inst_map = InstructionScheduleMap()
        inst_map.add("my_sched", 0, q0_sched)
        sched = macros.measure(
            qubits=[0], measure_name="my_sched", inst_map=inst_map, meas_map=[[0]]
        )
        self.assertEqual(sched.instructions, q0_sched.instructions)

        with self.assertRaises(PulseError):
            macros.measure(qubits=[0], measure_name="name", inst_map=inst_map, meas_map=[[0]])

    def test_fail_measure(self):
        """Test failing measure."""
        with self.assertRaises(PulseError):
            macros.measure(qubits=[0], meas_map=self.backend.configuration().meas_map)
        with self.assertRaises(PulseError):
            macros.measure(qubits=[0], inst_map=self.inst_map)


class TestMeasureAll(QiskitTestCase):
    """Pulse measure all macro."""

    def setUp(self):
        super().setUp()
        self.backend = FakeOpenPulse2Q()
        self.inst_map = self.backend.defaults().instruction_schedule_map

    def test_measure_all(self):
        """Test measure_all function."""
        sched = macros.measure_all(self.backend)
        expected = Schedule(self.inst_map.get("measure", [0, 1]))
        self.assertEqual(sched.instructions, expected.instructions)


class TestPlayChunkedPulse(QiskitTestCase):
    """Pulse macro test cases that plays chunked pulse within the builder context."""

    def setUp(self):
        super().setUp()
        self.backend = FakeOpenPulse2Q()

    def test_chunked_gaussian(self):
        """Test playing chunk divided Gaussian square pulse.

        Get pulse granularity from the fake backend.
        """
        channel = DriveChannel(0)
        duration = 1200
        amp = 0.1
        angle = 0.0
        sigma = 64
        risefall_sigma_ratio = 2

        gs_block = macros.chunking_gaussian_square(
            duration=duration,
            amp=amp,
            angle=angle,
            sigma=sigma,
            risefall_sigma_ratio=risefall_sigma_ratio,
            chunk_size=256,
            min_chunk_number=3,
            channel=channel,
            granularity=16,
        )
        schedule_to_test = block_to_schedule(gs_block)

        rise = GaussianRiseEdge(
            duration=224,
            amp=amp,
            angle=angle,
            sigma=sigma,
            risefall_sigma_ratio=risefall_sigma_ratio,
        )
        flat = Constant(
            duration=256,
            amp=amp,
            angle=angle,
        )
        fall = GaussianFallEdge(
            duration=224,
            amp=amp,
            angle=angle,
            sigma=sigma,
            risefall_sigma_ratio=risefall_sigma_ratio,
        )

        ref_schedule = Schedule()
        ref_schedule.insert(0, Play(rise, channel), inplace=True)
        ref_schedule.insert(224, Play(flat, channel), inplace=True)
        ref_schedule.insert(480, Play(flat, channel), inplace=True)
        ref_schedule.insert(736, Play(flat, channel), inplace=True)
        ref_schedule.insert(992, Play(fall, channel), inplace=True)

        self.assertEqual(schedule_to_test, ref_schedule)

    def test_chunked_gaussian_short(self):
        """Test playing a single Gaussian square pulse.

        When duration is shorter than minimum chunk size, it plays a normal GaussianSquare pulse.
        """
        channel = DriveChannel(0)
        duration = 800
        amp = 0.1
        angle = 0.0
        sigma = 64
        risefall_sigma_ratio = 2

        gs_block = macros.chunking_gaussian_square(
            duration=duration,
            amp=amp,
            angle=angle,
            sigma=sigma,
            risefall_sigma_ratio=risefall_sigma_ratio,
            chunk_size=256,
            min_chunk_number=3,
            channel=channel,
            granularity=16,
        )
        schedule_to_test = block_to_schedule(gs_block)

        gs_pulse = GaussianSquare(
            duration=duration,
            amp=amp,
            angle=angle,
            sigma=sigma,
            risefall_sigma_ratio=risefall_sigma_ratio,
        )
        ref_schedule = Schedule()
        ref_schedule.insert(0, Play(gs_pulse, channel), inplace=True)

        self.assertEqual(schedule_to_test, ref_schedule)

    def test_chunked_gaussian_invalid_chunk_size(self):
        """Test chunked Gaussian with invalid chunk size.

        Chunk size must be rounded to the nearest valid value with warning.
        """
        channel = DriveChannel(0)
        duration = 1200
        amp = 0.1
        angle = 0.0
        sigma = 64
        risefall_sigma_ratio = 2

        with self.assertWarns(UserWarning):
            gs_block = macros.chunking_gaussian_square(
                duration=duration,
                amp=amp,
                angle=angle,
                sigma=sigma,
                risefall_sigma_ratio=risefall_sigma_ratio,
                chunk_size=256,
                min_chunk_number=3,
                channel=channel,
                granularity=10,
            )
        schedule_to_test = block_to_schedule(gs_block)

        rise = GaussianRiseEdge(
            duration=230,
            amp=amp,
            angle=angle,
            sigma=sigma,
            risefall_sigma_ratio=risefall_sigma_ratio,
        )
        flat = Constant(
            duration=250,
            amp=amp,
            angle=angle,
        )
        fall = GaussianFallEdge(
            duration=230,
            amp=amp,
            angle=angle,
            sigma=sigma,
            risefall_sigma_ratio=risefall_sigma_ratio,
        )

        ref_schedule = Schedule()
        ref_schedule.insert(0, Play(rise, channel), inplace=True)
        ref_schedule.insert(230, Play(flat, channel), inplace=True)
        ref_schedule.insert(480, Play(flat, channel), inplace=True)
        ref_schedule.insert(730, Play(flat, channel), inplace=True)
        ref_schedule.insert(980, Play(fall, channel), inplace=True)

        self.assertEqual(schedule_to_test, ref_schedule)
