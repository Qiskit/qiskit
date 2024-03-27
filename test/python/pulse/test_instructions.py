# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit tests for pulse instructions."""

import numpy as np
from ddt import ddt, named_data, unpack

from qiskit import circuit
from qiskit.pulse import channels, configuration, instructions, library, exceptions
from qiskit.pulse.model import Qubit, QubitFrame, MixedFrame
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestAcquire(QiskitTestCase):
    """Acquisition tests."""

    def test_can_construct_valid_acquire_command_legacy(self):
        """Test if valid acquire command can be constructed."""
        kernel_opts = {"start_window": 0, "stop_window": 10}
        kernel = configuration.Kernel(name="boxcar", **kernel_opts)

        discriminator_opts = {
            "neighborhoods": [{"qubits": 1, "channels": 1}],
            "cal": "coloring",
            "resample": False,
        }
        discriminator = configuration.Discriminator(
            name="linear_discriminator", **discriminator_opts
        )

        acq = instructions.Acquire(
            10,
            channel=channels.AcquireChannel(0),
            mem_slot=channels.MemorySlot(0),
            kernel=kernel,
            discriminator=discriminator,
            name="acquire",
        )

        self.assertEqual(acq.duration, 10)
        self.assertEqual(acq.channel, channels.AcquireChannel(0))
        self.assertEqual(acq.qubit, channels.AcquireChannel(0))
        self.assertEqual(acq.discriminator.name, "linear_discriminator")
        self.assertEqual(acq.discriminator.params, discriminator_opts)
        self.assertEqual(acq.kernel.name, "boxcar")
        self.assertEqual(acq.kernel.params, kernel_opts)
        self.assertIsInstance(acq.id, int)
        self.assertEqual(acq.name, "acquire")
        self.assertEqual(
            acq.operands,
            (
                10,
                channels.AcquireChannel(0),
                channels.MemorySlot(0),
                None,
                kernel,
                discriminator,
            ),
        )

    def test_acquire_construction_new_model(self):
        """Test if valid acquire command can be constructed with the new model."""
        kernel_opts = {"start_window": 0, "stop_window": 10}
        kernel = configuration.Kernel(name="boxcar", **kernel_opts)

        discriminator_opts = {
            "neighborhoods": [{"qubits": 1, "channels": 1}],
            "cal": "coloring",
            "resample": False,
        }
        discriminator = configuration.Discriminator(
            name="linear_discriminator", **discriminator_opts
        )

        acq = instructions.Acquire(
            10,
            qubit=Qubit(1),
            mem_slot=channels.MemorySlot(0),
            kernel=kernel,
            discriminator=discriminator,
            name="acquire",
        )

        self.assertEqual(acq.duration, 10)
        self.assertEqual(acq.channel, None)
        self.assertEqual(acq.qubit, Qubit(1))
        self.assertEqual(acq.inst_targets, (Qubit(1), channels.MemorySlot(0)))
        self.assertEqual(acq.discriminator.name, "linear_discriminator")
        self.assertEqual(acq.discriminator.params, discriminator_opts)
        self.assertEqual(acq.kernel.name, "boxcar")
        self.assertEqual(acq.kernel.params, kernel_opts)
        self.assertIsInstance(acq.id, int)
        self.assertEqual(acq.name, "acquire")

    def test_instructions_hash(self):
        """Test hashing for acquire instruction."""
        acq_1 = instructions.Acquire(
            10,
            channel=channels.AcquireChannel(0),
            mem_slot=channels.MemorySlot(0),
            name="acquire",
        )
        acq_2 = instructions.Acquire(
            10,
            channel=channels.AcquireChannel(0),
            mem_slot=channels.MemorySlot(0),
            name="acquire",
        )

        hash_1 = hash(acq_1)
        hash_2 = hash(acq_2)

        self.assertEqual(hash_1, hash_2)


@ddt
class TestDelay(QiskitTestCase):
    """Delay tests."""

    def test_delay_legacy(self):
        """Test delay."""
        delay = instructions.Delay(10, channel=channels.DriveChannel(0), name="test_name")

        self.assertIsInstance(delay.id, int)
        self.assertEqual(delay.name, "test_name")
        self.assertEqual(delay.duration, 10)
        self.assertIsInstance(delay.duration, int)
        self.assertEqual(delay.channel, channels.DriveChannel(0))
        self.assertEqual(delay.inst_target, channels.DriveChannel(0))
        self.assertEqual(delay.operands, (10, channels.DriveChannel(0)))
        self.assertEqual(delay, instructions.Delay(10, channel=channels.DriveChannel(0)))
        self.assertNotEqual(delay, instructions.Delay(11, channel=channels.DriveChannel(1)))
        self.assertEqual(repr(delay), "Delay(10, DriveChannel(0), name='test_name')")

    def test_delay_np_int(self):
        """Test delay with numpy int duration"""
        delay = instructions.Delay(
            np.int32(10), channel=channels.DriveChannel(0), name="test_name2"
        )
        self.assertEqual(delay.duration, 10)
        self.assertIsInstance(delay.duration, np.integer)

    @named_data(
        ["target", Qubit(0), None, None],
        ["target_frame", Qubit(0), QubitFrame(0), None],
        ["mixed_frame", None, None, MixedFrame(Qubit(0), QubitFrame(0))],
    )
    @unpack
    def test_delay_new_model(self, target, frame, mixed_frame):
        """Test delay new model."""
        delay = instructions.Delay(
            10, target=target, frame=frame, mixed_frame=mixed_frame, name="test_name"
        )

        self.assertIsInstance(delay.id, int)
        self.assertEqual(delay.name, "test_name")
        self.assertEqual(delay.duration, 10)
        self.assertIsInstance(delay.duration, int)
        self.assertEqual(delay.channel, None)

        if frame is not None:
            ref = MixedFrame(target, frame)
        else:
            ref = target or mixed_frame

        self.assertEqual(delay.inst_target, ref)
        self.assertEqual(delay.inst_targets, (ref,))
        self.assertEqual(delay.operands, (10, ref))

    @named_data(
        ["none", False, False, False, False],
        ["channel_target", True, True, False, False],
        ["channel_frame", True, False, True, False],
        ["channel_mixed_frame", True, False, False, True],
        ["channel_target_frame", True, True, True, False],
        ["channel_target_mixed_frame", True, True, False, True],
        ["channel_frame_mixed_frame", True, False, True, True],
        ["all", True, True, True, True],
        ["target_mixed_frame", False, True, False, True],
        ["target_frame_mixed_frame", False, True, True, True],
        ["only_frame", False, False, True, False],
        ["frame_mixed_frame", False, False, True, True],
    )
    @unpack
    def test_delay_new_model_input_combination_validation(
        self, channel, target, frame, mixed_frame
    ):
        """Test delay instruction input combination validation.

        The combination is set according to the truth value of the arguments passed to the function.
        Only combinations which should raise an error should ba passed.
        """
        channel = channels.DriveChannel(0) if channel else None
        target = Qubit(0) if target else None
        frame = QubitFrame(0) if frame else None
        mixed_frame = MixedFrame(Qubit(0), QubitFrame(0)) if mixed_frame else None

        with self.assertRaises(exceptions.PulseError):
            instructions.Delay(
                10, channel=channel, target=target, frame=frame, mixed_frame=mixed_frame
            )

    def test_delay_new_model_input_type_validation(self):
        """Test delay instruction input type validation"""
        target = Qubit(0)
        frame = QubitFrame(0)
        with self.assertRaises(exceptions.PulseError):
            instructions.Delay(10, target=frame)
        with self.assertRaises(exceptions.PulseError):
            instructions.Delay(10, target=target, frame=target)
        with self.assertRaises(exceptions.PulseError):
            instructions.Delay(10, mixed_frame=frame)

    def test_operator_delay(self):
        """Test Operator(delay)."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import Operator

        circ = QuantumCircuit(1)
        circ.delay(10)
        op_delay = Operator(circ)

        expected = QuantumCircuit(1)
        expected.id(0)
        op_identity = Operator(expected)
        self.assertEqual(op_delay, op_identity)


_invalid_frame_input_combinations = [
    ["none", False, False, False, False],
    ["channel_target", True, True, False, False],
    ["channel_frame", True, False, True, False],
    ["channel_mixed_frame", True, False, False, True],
    ["channel_target_frame", True, True, True, False],
    ["channel_target_mixed_frame", True, True, False, True],
    ["channel_frame_mixed_frame", True, False, True, True],
    ["target", False, True, False, False],
    ["all", True, True, True, True],
    ["target_mixed_frame", False, True, False, True],
    ["target_frame_mixed_frame", False, True, True, True],
    ["frame_mixed_frame", False, False, True, True],
]

_valid_frame_input_combinations = [
    ["frame", QubitFrame(1), None, None],
    ["frame_target", QubitFrame(1), Qubit(1), None],
    ["mixed_frame", None, None, MixedFrame(Qubit(1), QubitFrame(2))],
]


@ddt
class TestSetFrequency(QiskitTestCase):
    """Set frequency tests."""

    def test_freq_legacy(self):
        """Test set frequency basic functionality."""
        set_freq = instructions.SetFrequency(4.5e9, channel=channels.DriveChannel(1), name="test")

        self.assertIsInstance(set_freq.id, int)
        self.assertEqual(set_freq.duration, 0)
        self.assertEqual(set_freq.frequency, 4.5e9)
        self.assertEqual(set_freq.channel, channels.DriveChannel(1))
        self.assertEqual(set_freq.inst_target, channels.DriveChannel(1))
        self.assertEqual(set_freq.inst_targets, (channels.DriveChannel(1),))
        self.assertEqual(set_freq.operands, (4.5e9, channels.DriveChannel(1)))
        self.assertEqual(
            set_freq,
            instructions.SetFrequency(4.5e9, channel=channels.DriveChannel(1), name="test"),
        )
        self.assertNotEqual(
            set_freq,
            instructions.SetFrequency(4.5e8, channel=channels.DriveChannel(1), name="test"),
        )
        self.assertEqual(repr(set_freq), "SetFrequency(4500000000.0, DriveChannel(1), name='test')")

    @named_data(*_valid_frame_input_combinations)
    @unpack
    def test_set_freq_new_model(self, frame, target, mixed_frame):
        """Test set frequency basic functionality with valid inputs."""
        set_freq = instructions.SetFrequency(
            4.5e9, frame=frame, target=target, mixed_frame=mixed_frame, name="test"
        )

        self.assertIsInstance(set_freq.id, int)
        self.assertEqual(set_freq.duration, 0)
        self.assertEqual(set_freq.frequency, 4.5e9)
        self.assertEqual(set_freq.channel, None)
        if target is not None:
            ref = MixedFrame(target, frame)
        else:
            ref = frame or mixed_frame
        self.assertEqual(set_freq.inst_target, ref)
        self.assertEqual(set_freq.inst_targets, (ref,))
        self.assertEqual(set_freq.operands, (4.5e9, ref))

    @named_data(*_invalid_frame_input_combinations)
    @unpack
    def test_set_freq_new_model_input_combination_validation(
        self, channel, target, frame, mixed_frame
    ):
        """Test set freq instruction input combination validation.

        The combination is set according to the truth value of the arguments passed to the function.
        Only combinations which should raise an error should ba passed.
        """
        channel = channels.DriveChannel(0) if channel else None
        target = Qubit(0) if target else None
        frame = QubitFrame(0) if frame else None
        mixed_frame = MixedFrame(Qubit(0), QubitFrame(0)) if mixed_frame else None

        with self.assertRaises(exceptions.PulseError):
            instructions.SetFrequency(
                10, channel=channel, target=target, frame=frame, mixed_frame=mixed_frame
            )

    def test_set_freq_new_model_input_type_validation(self):
        """Test set freq instruction input type validation."""
        target = Qubit(0)
        frame = QubitFrame(0)

        with self.assertRaises(exceptions.PulseError):
            instructions.SetFrequency(10, frame=target)
        with self.assertRaises(exceptions.PulseError):
            instructions.SetFrequency(10, mixed_frame=target)
        with self.assertRaises(exceptions.PulseError):
            instructions.SetFrequency(10, target=frame, frame=frame)

    def test_freq_non_pulse_channel(self):
        """Test set frequency constructor with illegal channel"""
        with self.assertRaises(exceptions.PulseError):
            instructions.SetFrequency(4.5e9, channel=channels.RegisterSlot(1), name="test")

    def test_parameter_expression(self):
        """Test getting all parameters assigned by expression."""
        p1 = circuit.Parameter("P1")
        p2 = circuit.Parameter("P2")
        expr = p1 + p2

        instr = instructions.SetFrequency(expr, channel=channels.DriveChannel(0))
        self.assertSetEqual(instr.parameters, {p1, p2})


@ddt
class TestShiftFrequency(QiskitTestCase):
    """Shift frequency tests."""

    def test_shift_freq_legacy(self):
        """Test shift frequency basic functionality."""
        shift_freq = instructions.ShiftFrequency(
            4.5e9, channel=channels.DriveChannel(1), name="test"
        )

        self.assertIsInstance(shift_freq.id, int)
        self.assertEqual(shift_freq.duration, 0)
        self.assertEqual(shift_freq.frequency, 4.5e9)
        self.assertEqual(shift_freq.channel, channels.DriveChannel(1))
        self.assertEqual(shift_freq.inst_target, channels.DriveChannel(1))
        self.assertEqual(shift_freq.inst_targets, (channels.DriveChannel(1),))
        self.assertEqual(shift_freq.operands, (4.5e9, channels.DriveChannel(1)))
        self.assertEqual(
            shift_freq,
            instructions.ShiftFrequency(4.5e9, channel=channels.DriveChannel(1), name="test"),
        )
        self.assertNotEqual(
            shift_freq,
            instructions.ShiftFrequency(4.5e8, channel=channels.DriveChannel(1), name="test"),
        )
        self.assertEqual(
            repr(shift_freq), "ShiftFrequency(4500000000.0, DriveChannel(1), name='test')"
        )

    @named_data(*_valid_frame_input_combinations)
    @unpack
    def test_shift_freq_new_model(self, frame, target, mixed_frame):
        """Test shift frequency basic functionality with valid inputs."""
        set_freq = instructions.ShiftFrequency(
            4.5e9, frame=frame, target=target, mixed_frame=mixed_frame, name="test"
        )

        self.assertIsInstance(set_freq.id, int)
        self.assertEqual(set_freq.duration, 0)
        self.assertEqual(set_freq.frequency, 4.5e9)
        self.assertEqual(set_freq.channel, None)
        if target is not None:
            ref = MixedFrame(target, frame)
        else:
            ref = frame or mixed_frame
        self.assertEqual(set_freq.inst_target, ref)
        self.assertEqual(set_freq.inst_targets, (ref,))
        self.assertEqual(set_freq.operands, (4.5e9, ref))

    @named_data(*_invalid_frame_input_combinations)
    @unpack
    def test_shift_freq_new_model_input_combination_validation(
        self, channel, target, frame, mixed_frame
    ):
        """Test shift freq instruction input combination validation.

        The combination is set according to the truth value of the arguments passed to the function.
        Only combinations which should raise an error should ba passed.
        """
        channel = channels.DriveChannel(0) if channel else None
        target = Qubit(0) if target else None
        frame = QubitFrame(0) if frame else None
        mixed_frame = MixedFrame(Qubit(0), QubitFrame(0)) if mixed_frame else None

        with self.assertRaises(exceptions.PulseError):
            instructions.ShiftFrequency(
                10, channel=channel, target=target, frame=frame, mixed_frame=mixed_frame
            )

    def test_shift_freq_new_model_input_type_validation(self):
        """Test shift freq instruction input type validation."""
        target = Qubit(0)
        frame = QubitFrame(0)

        with self.assertRaises(exceptions.PulseError):
            instructions.ShiftFrequency(10, frame=target)
        with self.assertRaises(exceptions.PulseError):
            instructions.ShiftFrequency(10, mixed_frame=target)
        with self.assertRaises(exceptions.PulseError):
            instructions.ShiftFrequency(10, target=frame, frame=frame)

    def test_freq_non_pulse_channel(self):
        """Test shift frequency constructor with illegal channel"""
        with self.assertRaises(exceptions.PulseError):
            instructions.ShiftFrequency(4.5e9, channel=channels.RegisterSlot(1), name="test")

    def test_parameter_expression(self):
        """Test getting all parameters assigned by expression."""
        p1 = circuit.Parameter("P1")
        p2 = circuit.Parameter("P2")
        expr = p1 + p2

        instr = instructions.ShiftFrequency(expr, channel=channels.DriveChannel(0))
        self.assertSetEqual(instr.parameters, {p1, p2})


@ddt
class TestSetPhase(QiskitTestCase):
    """Test the instruction construction."""

    def test_default(self):
        """Test basic SetPhase."""
        set_phase = instructions.SetPhase(1.57, channel=channels.DriveChannel(0))

        self.assertIsInstance(set_phase.id, int)
        self.assertEqual(set_phase.name, None)
        self.assertEqual(set_phase.duration, 0)
        self.assertEqual(set_phase.phase, 1.57)
        self.assertEqual(set_phase.channel, channels.DriveChannel(0))
        self.assertEqual(set_phase.inst_target, channels.DriveChannel(0))
        self.assertEqual(set_phase.inst_targets, (channels.DriveChannel(0),))
        self.assertEqual(set_phase.operands, (1.57, channels.DriveChannel(0)))
        self.assertEqual(
            set_phase, instructions.SetPhase(1.57, channel=channels.DriveChannel(0), name="test")
        )
        self.assertNotEqual(
            set_phase, instructions.SetPhase(1.57j, channel=channels.DriveChannel(0), name="test")
        )
        self.assertEqual(repr(set_phase), "SetPhase(1.57, DriveChannel(0))")

    @named_data(*_valid_frame_input_combinations)
    @unpack
    def test_set_phase_new_model(self, frame, target, mixed_frame):
        """Test set phase basic functionality with valid inputs."""
        set_freq = instructions.SetPhase(
            1.5, frame=frame, target=target, mixed_frame=mixed_frame, name="test"
        )

        self.assertIsInstance(set_freq.id, int)
        self.assertEqual(set_freq.duration, 0)
        self.assertEqual(set_freq.phase, 1.5)
        self.assertEqual(set_freq.channel, None)
        if target is not None:
            ref = MixedFrame(target, frame)
        else:
            ref = frame or mixed_frame
        self.assertEqual(set_freq.inst_target, ref)
        self.assertEqual(set_freq.inst_targets, (ref,))
        self.assertEqual(set_freq.operands, (1.5, ref))

    @named_data(*_invalid_frame_input_combinations)
    @unpack
    def test_set_phase_new_model_input_combination_validation(
        self, channel, target, frame, mixed_frame
    ):
        """Test set phase instruction input combination validation.

        The combination is set according to the truth value of the arguments passed to the function.
        Only combinations which should raise an error should ba passed.
        """
        channel = channels.DriveChannel(0) if channel else None
        target = Qubit(0) if target else None
        frame = QubitFrame(0) if frame else None
        mixed_frame = MixedFrame(Qubit(0), QubitFrame(0)) if mixed_frame else None

        with self.assertRaises(exceptions.PulseError):
            instructions.SetPhase(
                10, channel=channel, target=target, frame=frame, mixed_frame=mixed_frame
            )

    def test_set_phase_new_model_input_type_validation(self):
        """Test set phase instruction input type validation."""
        target = Qubit(0)
        frame = QubitFrame(0)

        with self.assertRaises(exceptions.PulseError):
            instructions.SetPhase(10, frame=target)
        with self.assertRaises(exceptions.PulseError):
            instructions.SetPhase(10, mixed_frame=target)
        with self.assertRaises(exceptions.PulseError):
            instructions.SetPhase(10, target=frame, frame=frame)

    def test_set_phase_non_pulse_channel(self):
        """Test shift phase constructor with illegal channel"""
        with self.assertRaises(exceptions.PulseError):
            instructions.SetPhase(1.57, channel=channels.RegisterSlot(1), name="test")

    def test_parameter_expression(self):
        """Test getting all parameters assigned by expression."""
        p1 = circuit.Parameter("P1")
        p2 = circuit.Parameter("P2")
        expr = p1 + p2

        instr = instructions.SetPhase(expr, channel=channels.DriveChannel(0))
        self.assertSetEqual(instr.parameters, {p1, p2})


@ddt
class TestShiftPhase(QiskitTestCase):
    """Test the instruction construction."""

    def test_default(self):
        """Test basic ShiftPhase."""
        shift_phase = instructions.ShiftPhase(1.57, channel=channels.DriveChannel(0))

        self.assertIsInstance(shift_phase.id, int)
        self.assertEqual(shift_phase.name, None)
        self.assertEqual(shift_phase.duration, 0)
        self.assertEqual(shift_phase.phase, 1.57)
        self.assertEqual(shift_phase.channel, channels.DriveChannel(0))
        self.assertEqual(shift_phase.inst_target, channels.DriveChannel(0))
        self.assertEqual(shift_phase.inst_targets, (channels.DriveChannel(0),))
        self.assertEqual(shift_phase.operands, (1.57, channels.DriveChannel(0)))
        self.assertEqual(
            shift_phase,
            instructions.ShiftPhase(1.57, channel=channels.DriveChannel(0), name="test"),
        )
        self.assertNotEqual(
            shift_phase,
            instructions.ShiftPhase(1.57j, channel=channels.DriveChannel(0), name="test"),
        )
        self.assertEqual(repr(shift_phase), "ShiftPhase(1.57, DriveChannel(0))")

    @named_data(*_valid_frame_input_combinations)
    @unpack
    def test_shift_phase_new_model(self, frame, target, mixed_frame):
        """Test shift phase basic functionality with valid inputs."""
        set_freq = instructions.ShiftPhase(
            1.5, frame=frame, target=target, mixed_frame=mixed_frame, name="test"
        )

        self.assertIsInstance(set_freq.id, int)
        self.assertEqual(set_freq.duration, 0)
        self.assertEqual(set_freq.phase, 1.5)
        self.assertEqual(set_freq.channel, None)
        if target is not None:
            ref = MixedFrame(target, frame)
        else:
            ref = frame or mixed_frame
        self.assertEqual(set_freq.inst_target, ref)
        self.assertEqual(set_freq.inst_targets, (ref,))
        self.assertEqual(set_freq.operands, (1.5, ref))

    @named_data(*_invalid_frame_input_combinations)
    @unpack
    def test_shift_phase_new_model_input_combination_validation(
        self, channel, target, frame, mixed_frame
    ):
        """Test shift phase instruction input combination validation.

        The combination is set according to the truth value of the arguments passed to the function.
        Only combinations which should raise an error should ba passed.
        """
        channel = channels.DriveChannel(0) if channel else None
        target = Qubit(0) if target else None
        frame = QubitFrame(0) if frame else None
        mixed_frame = MixedFrame(Qubit(0), QubitFrame(0)) if mixed_frame else None

        with self.assertRaises(exceptions.PulseError):
            instructions.ShiftPhase(
                10, channel=channel, target=target, frame=frame, mixed_frame=mixed_frame
            )

    def test_shift_phase_new_model_input_type_validation(self):
        """Test shift phase instruction input type validation."""
        target = Qubit(0)
        frame = QubitFrame(0)

        with self.assertRaises(exceptions.PulseError):
            instructions.ShiftPhase(10, frame=target)
        with self.assertRaises(exceptions.PulseError):
            instructions.ShiftPhase(10, mixed_frame=target)
        with self.assertRaises(exceptions.PulseError):
            instructions.ShiftPhase(10, target=frame, frame=frame)

    def test_shift_phase_non_pulse_channel(self):
        """Test shift phase constructor with illegal channel"""
        with self.assertRaises(exceptions.PulseError):
            instructions.ShiftPhase(1.57, channel=channels.RegisterSlot(1), name="test")

    def test_parameter_expression(self):
        """Test getting all parameters assigned by expression."""
        p1 = circuit.Parameter("P1")
        p2 = circuit.Parameter("P2")
        expr = p1 + p2

        instr = instructions.ShiftPhase(expr, channel=channels.DriveChannel(0))
        self.assertSetEqual(instr.parameters, {p1, p2})


class TestSnapshot(QiskitTestCase):
    """Snapshot tests."""

    def test_default(self):
        """Test default snapshot."""
        snapshot = instructions.Snapshot(label="test_name", snapshot_type="state")

        self.assertIsInstance(snapshot.id, int)
        self.assertEqual(snapshot.name, "test_name")
        self.assertEqual(snapshot.type, "state")
        self.assertEqual(snapshot.duration, 0)
        self.assertNotEqual(snapshot, instructions.Delay(10, channel=channels.DriveChannel(0)))
        self.assertEqual(repr(snapshot), "Snapshot(test_name, state, name='test_name')")


@ddt
class TestPlay(QiskitTestCase):
    """Play tests."""

    def setUp(self):
        """Setup play tests."""
        super().setUp()
        self.duration = 4
        self.pulse_op = library.Waveform([1.0] * self.duration, name="test")

    def test_play_legacy(self):
        """Test basic play instruction."""
        play = instructions.Play(self.pulse_op, channel=channels.DriveChannel(1))

        self.assertIsInstance(play.id, int)
        self.assertEqual(play.name, self.pulse_op.name)
        self.assertEqual(play.duration, self.duration)
        self.assertEqual(play.channel, channels.DriveChannel(1))
        self.assertEqual(play.inst_target, channels.DriveChannel(1))
        self.assertEqual(play.inst_targets, (channels.DriveChannel(1),))
        self.assertEqual(
            repr(play),
            "Play(Waveform(array([1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]), name='test'),"
            " DriveChannel(1), name='test')",
        )

    @named_data(
        ["target_frame", Qubit(0), QubitFrame(1), None],
        ["mixed_frame", None, None, MixedFrame(Qubit(0), QubitFrame(1))],
    )
    @unpack
    def test_play_new_model_target_frame(self, target, frame, mixed_frame):
        """Test basic play instruction."""
        play = instructions.Play(self.pulse_op, target=target, frame=frame, mixed_frame=mixed_frame)

        self.assertIsInstance(play.id, int)
        self.assertEqual(play.name, self.pulse_op.name)
        self.assertEqual(play.duration, self.duration)
        self.assertEqual(play.pulse, self.pulse_op)
        self.assertEqual(play.inst_target, MixedFrame(Qubit(0), QubitFrame(1)))
        self.assertEqual(play.inst_targets, (MixedFrame(Qubit(0), QubitFrame(1)),))
        self.assertEqual(play.channel, None)

    def test_play_non_pulse_ch_raises(self):
        """Test that play instruction on non-pulse channel raises a pulse error."""
        with self.assertRaises(exceptions.PulseError):
            instructions.Play(self.pulse_op, channel=channels.AcquireChannel(0))

    @named_data(
        ["none", False, False, False, False],
        ["channel_target", True, True, False, False],
        ["channel_frame", True, False, True, False],
        ["channel_mixed_frame", True, False, False, True],
        ["channel_target_frame", True, True, True, False],
        ["channel_target_mixed_frame", True, True, False, True],
        ["channel_frame_mixed_frame", True, False, True, True],
        ["all", True, True, True, True],
        ["target_mixed_frame", False, True, False, True],
        ["target_frame_mixed_frame", False, True, True, True],
        ["only_frame", False, False, True, False],
        ["only_target", False, False, True, False],
        ["frame_mixed_frame", False, False, True, True],
    )
    @unpack
    def test_play_arguments_combination_validation(self, channel, target, frame, mixed_frame):
        """Test that play instruction raises an error if the arguments don't specify a unique mixed
        frame"""
        channel = channels.DriveChannel(0) if channel else None
        target = Qubit(0) if target else None
        frame = QubitFrame(0) if frame else None
        mixed_frame = MixedFrame(Qubit(0), QubitFrame(0)) if mixed_frame else None

        with self.assertRaises(exceptions.PulseError):
            instructions.Play(
                self.pulse_op, frame=frame, target=target, mixed_frame=mixed_frame, channel=channel
            )

    def test_play_arguments_type_validation(self):
        """Test that play instruction raises an error if arguments of wrong type"""
        target = Qubit(0)
        frame = QubitFrame(0)

        with self.assertRaises(exceptions.PulseError):
            instructions.Play(self.pulse_op, target=frame, frame=frame)
        with self.assertRaises(exceptions.PulseError):
            instructions.Play(self.pulse_op, target=target, frame=target)
        with self.assertRaises(exceptions.PulseError):
            instructions.Play(self.pulse_op, mixed_frame=frame)


class TestDirectives(QiskitTestCase):
    """Test pulse directives."""

    def test_relative_barrier(self):
        """Test the relative barrier directive."""
        a0 = channels.AcquireChannel(0)
        d0 = channels.DriveChannel(0)
        m0 = channels.MeasureChannel(0)
        u0 = channels.ControlChannel(0)
        mem0 = channels.MemorySlot(0)
        reg0 = channels.RegisterSlot(0)
        chans = (a0, d0, m0, u0, mem0, reg0)
        name = "barrier"
        barrier = instructions.RelativeBarrier(*chans, name=name)

        self.assertEqual(barrier.name, name)
        self.assertEqual(barrier.duration, 0)
        self.assertEqual(barrier.channels, chans)
        self.assertEqual(barrier.operands, chans)
