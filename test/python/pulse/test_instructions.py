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

from qiskit import pulse, circuit
from qiskit.pulse import channels, configuration, instructions, library, exceptions
from qiskit.pulse.transforms import inline_subroutines, target_qobj_transform
from qiskit.test import QiskitTestCase


class TestAcquire(QiskitTestCase):
    """Acquisition tests."""

    def test_can_construct_valid_acquire_command(self):
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
            channels.AcquireChannel(0),
            channels.MemorySlot(0),
            kernel=kernel,
            discriminator=discriminator,
            name="acquire",
        )

        self.assertEqual(acq.duration, 10)
        self.assertEqual(acq.discriminator.name, "linear_discriminator")
        self.assertEqual(acq.discriminator.params, discriminator_opts)
        self.assertEqual(acq.kernel.name, "boxcar")
        self.assertEqual(acq.kernel.params, kernel_opts)
        self.assertIsInstance(acq.id, int)
        self.assertEqual(acq.name, "acquire")
        self.assertEqual(
            acq.operands, (10, channels.AcquireChannel(0), channels.MemorySlot(0), None)
        )

    def test_instructions_hash(self):
        """Test hashing for acquire instruction."""
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
        acq_1 = instructions.Acquire(
            10,
            channels.AcquireChannel(0),
            channels.MemorySlot(0),
            kernel=kernel,
            discriminator=discriminator,
            name="acquire",
        )
        acq_2 = instructions.Acquire(
            10,
            channels.AcquireChannel(0),
            channels.MemorySlot(0),
            kernel=kernel,
            discriminator=discriminator,
            name="acquire",
        )

        hash_1 = hash(acq_1)
        hash_2 = hash(acq_2)

        self.assertEqual(hash_1, hash_2)


class TestDelay(QiskitTestCase):
    """Delay tests."""

    def test_delay(self):
        """Test delay."""
        delay = instructions.Delay(10, channels.DriveChannel(0), name="test_name")

        self.assertIsInstance(delay.id, int)
        self.assertEqual(delay.name, "test_name")
        self.assertEqual(delay.duration, 10)
        self.assertIsInstance(delay.duration, int)
        self.assertEqual(delay.operands, (10, channels.DriveChannel(0)))
        self.assertEqual(delay, instructions.Delay(10, channels.DriveChannel(0)))
        self.assertNotEqual(delay, instructions.Delay(11, channels.DriveChannel(1)))
        self.assertEqual(repr(delay), "Delay(10, DriveChannel(0), name='test_name')")

        # Test numpy int for duration
        delay = instructions.Delay(np.int32(10), channels.DriveChannel(0), name="test_name2")
        self.assertEqual(delay.duration, 10)
        self.assertIsInstance(delay.duration, np.integer)

    def test_operator_delay(self):
        """Test Operator(delay)."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import Operator

        circ = QuantumCircuit(1)
        circ.delay(10)
        op_delay = Operator(circ)

        expected = QuantumCircuit(1)
        expected.i(0)
        op_identity = Operator(expected)
        self.assertEqual(op_delay, op_identity)


class TestSetFrequency(QiskitTestCase):
    """Set frequency tests."""

    def test_freq(self):
        """Test set frequency basic functionality."""
        set_freq = instructions.SetFrequency(4.5e9, channels.DriveChannel(1), name="test")

        self.assertIsInstance(set_freq.id, int)
        self.assertEqual(set_freq.duration, 0)
        self.assertEqual(set_freq.frequency, 4.5e9)
        self.assertEqual(set_freq.operands, (4.5e9, channels.DriveChannel(1)))
        self.assertEqual(
            set_freq, instructions.SetFrequency(4.5e9, channels.DriveChannel(1), name="test")
        )
        self.assertNotEqual(
            set_freq, instructions.SetFrequency(4.5e8, channels.DriveChannel(1), name="test")
        )
        self.assertEqual(repr(set_freq), "SetFrequency(4500000000.0, DriveChannel(1), name='test')")


class TestShiftPhase(QiskitTestCase):
    """Test the instruction construction."""

    def test_default(self):
        """Test basic ShiftPhase."""
        shift_phase = instructions.ShiftPhase(1.57, channels.DriveChannel(0))

        self.assertIsInstance(shift_phase.id, int)
        self.assertEqual(shift_phase.name, None)
        self.assertEqual(shift_phase.duration, 0)
        self.assertEqual(shift_phase.phase, 1.57)
        self.assertEqual(shift_phase.operands, (1.57, channels.DriveChannel(0)))
        self.assertEqual(
            shift_phase, instructions.ShiftPhase(1.57, channels.DriveChannel(0), name="test")
        )
        self.assertNotEqual(
            shift_phase, instructions.ShiftPhase(1.57j, channels.DriveChannel(0), name="test")
        )
        self.assertEqual(repr(shift_phase), "ShiftPhase(1.57, DriveChannel(0))")


class TestSnapshot(QiskitTestCase):
    """Snapshot tests."""

    def test_default(self):
        """Test default snapshot."""
        snapshot = instructions.Snapshot(label="test_name", snapshot_type="state")

        self.assertIsInstance(snapshot.id, int)
        self.assertEqual(snapshot.name, "test_name")
        self.assertEqual(snapshot.type, "state")
        self.assertEqual(snapshot.duration, 0)
        self.assertNotEqual(snapshot, instructions.Delay(10, channels.DriveChannel(0)))
        self.assertEqual(repr(snapshot), "Snapshot(test_name, state, name='test_name')")


class TestPlay(QiskitTestCase):
    """Play tests."""

    def setUp(self):
        """Setup play tests."""
        super().setUp()
        self.duration = 4
        self.pulse_op = library.Waveform([1.0] * self.duration, name="test")

    def test_play(self):
        """Test basic play instruction."""
        play = instructions.Play(self.pulse_op, channels.DriveChannel(1))

        self.assertIsInstance(play.id, int)
        self.assertEqual(play.name, self.pulse_op.name)
        self.assertEqual(play.duration, self.duration)
        self.assertEqual(
            repr(play),
            "Play(Waveform(array([1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]), name='test'),"
            " DriveChannel(1), name='test')",
        )

    def test_play_non_pulse_ch_raises(self):
        """Test that play instruction on non-pulse channel raises a pulse error."""
        with self.assertRaises(exceptions.PulseError):
            instructions.Play(self.pulse_op, channels.AcquireChannel(0))


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


class TestCall(QiskitTestCase):
    """Test call instruction."""

    def setUp(self):
        super().setUp()

        with pulse.build() as _subroutine:
            pulse.delay(10, pulse.DriveChannel(0))
        self.subroutine = _subroutine

        self.param1 = circuit.Parameter("amp1")
        self.param2 = circuit.Parameter("amp2")
        with pulse.build() as _function:
            pulse.play(pulse.Gaussian(160, self.param1, 40), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(160, self.param2, 40), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(160, self.param1, 40), pulse.DriveChannel(0))
        self.function = _function

    def test_call(self):
        """Test basic call instruction."""
        call = instructions.Call(subroutine=self.subroutine)

        self.assertEqual(call.duration, 10)
        self.assertEqual(call.subroutine, self.subroutine)

    def test_parameterized_call(self):
        """Test call instruction with parameterized subroutine."""
        call = instructions.Call(subroutine=self.function)

        self.assertTrue(call.is_parameterized())
        self.assertEqual(len(call.parameters), 2)

    def test_assign_parameters_to_call(self):
        """Test create schedule by calling subroutine and assign parameters to it."""
        init_dict = {self.param1: 0.1, self.param2: 0.5}

        with pulse.build() as test_sched:
            pulse.call(self.function)

        test_sched = test_sched.assign_parameters(value_dict=init_dict)
        test_sched = inline_subroutines(test_sched)

        with pulse.build() as ref_sched:
            pulse.play(pulse.Gaussian(160, 0.1, 40), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(160, 0.1, 40), pulse.DriveChannel(0))

        self.assertEqual(target_qobj_transform(test_sched), target_qobj_transform(ref_sched))

    def test_call_initialize_with_parameter(self):
        """Test call instruction with parameterized subroutine with initial dict."""
        init_dict = {self.param1: 0.1, self.param2: 0.5}
        call = instructions.Call(subroutine=self.function, value_dict=init_dict)

        with pulse.build() as ref_sched:
            pulse.play(pulse.Gaussian(160, 0.1, 40), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(160, 0.1, 40), pulse.DriveChannel(0))

        self.assertEqual(
            target_qobj_transform(call.assigned_subroutine()), target_qobj_transform(ref_sched)
        )

    def test_call_subroutine_with_different_parameters(self):
        """Test call subroutines with different parameters in the same schedule."""
        init_dict1 = {self.param1: 0.1, self.param2: 0.5}
        init_dict2 = {self.param1: 0.3, self.param2: 0.7}

        with pulse.build() as test_sched:
            pulse.call(self.function, value_dict=init_dict1)
            pulse.call(self.function, value_dict=init_dict2)

        with pulse.build() as ref_sched:
            pulse.play(pulse.Gaussian(160, 0.1, 40), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(160, 0.5, 40), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(160, 0.1, 40), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(160, 0.3, 40), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(160, 0.7, 40), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(160, 0.3, 40), pulse.DriveChannel(0))

        self.assertEqual(target_qobj_transform(test_sched), target_qobj_transform(ref_sched))
