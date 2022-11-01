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

"""Testing instruction alignment pass."""

from qiskit import QuantumCircuit, pulse
from qiskit.test import QiskitTestCase
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import (
    AlignMeasures,
    InstructionDurationCheck,
    ConstrainedReschedule,
    ValidatePulseGates,
    ALAPScheduleAnalysis,
    ASAPScheduleAnalysis,
    ALAPSchedule,
    PadDelay,
    SetIOLatency,
)


class TestAlignMeasures(QiskitTestCase):
    """A test for measurement alignment pass."""

    def setUp(self):
        super().setUp()

        self.instruction_durations = InstructionDurations(
            [
                ("rz", (0,), 0),
                ("rz", (1,), 0),
                ("x", (0,), 160),
                ("x", (1,), 160),
                ("sx", (0,), 160),
                ("sx", (1,), 160),
                ("cx", (0, 1), 800),
                ("cx", (1, 0), 800),
                ("measure", None, 1600),
            ]
        )

    def test_t1_experiment_type(self):
        """Test T1 experiment type circuit.

        (input)

             ┌───┐┌────────────────┐┌─┐
        q_0: ┤ X ├┤ Delay(100[dt]) ├┤M├
             └───┘└────────────────┘└╥┘
        c: 1/════════════════════════╩═
                                     0

        (aligned)

             ┌───┐┌────────────────┐┌─┐
        q_0: ┤ X ├┤ Delay(112[dt]) ├┤M├
             └───┘└────────────────┘└╥┘
        c: 1/════════════════════════╩═
                                     0

        This type of experiment slightly changes delay duration of interest.
        However the quantization error should be less than alignment * dt.
        """
        circuit = QuantumCircuit(1, 1)
        circuit.x(0)
        circuit.delay(100, 0, unit="dt")
        circuit.measure(0, 0)

        pm = PassManager(
            [
                # reproduce old behavior of 0.20.0 before #7655
                # currently default write latency is 0
                SetIOLatency(clbit_write_latency=1600, conditional_latency=0),
                ALAPScheduleAnalysis(durations=self.instruction_durations),
                ConstrainedReschedule(acquire_alignment=16),
                PadDelay(),
            ]
        )

        aligned_circuit = pm.run(circuit)

        ref_circuit = QuantumCircuit(1, 1)
        ref_circuit.x(0)
        ref_circuit.delay(112, 0, unit="dt")
        ref_circuit.measure(0, 0)

        self.assertEqual(aligned_circuit, ref_circuit)

    def test_hanh_echo_experiment_type(self):
        """Test Hahn echo experiment type circuit.

        (input)

             ┌────┐┌────────────────┐┌───┐┌────────────────┐┌────┐┌─┐
        q_0: ┤ √X ├┤ Delay(100[dt]) ├┤ X ├┤ Delay(100[dt]) ├┤ √X ├┤M├
             └────┘└────────────────┘└───┘└────────────────┘└────┘└╥┘
        c: 1/══════════════════════════════════════════════════════╩═
                                                                   0

        (output)

             ┌────┐┌────────────────┐┌───┐┌────────────────┐┌────┐┌──────────────┐┌─┐
        q_0: ┤ √X ├┤ Delay(100[dt]) ├┤ X ├┤ Delay(100[dt]) ├┤ √X ├┤ Delay(8[dt]) ├┤M├
             └────┘└────────────────┘└───┘└────────────────┘└────┘└──────────────┘└╥┘
        c: 1/══════════════════════════════════════════════════════════════════════╩═
                                                                                   0

        This type of experiment doesn't change duration of interest (two in the middle).
        However induces slight delay less than alignment * dt before measurement.
        This might induce extra amplitude damping error.
        """
        circuit = QuantumCircuit(1, 1)
        circuit.sx(0)
        circuit.delay(100, 0, unit="dt")
        circuit.x(0)
        circuit.delay(100, 0, unit="dt")
        circuit.sx(0)
        circuit.measure(0, 0)

        pm = PassManager(
            [
                # reproduce old behavior of 0.20.0 before #7655
                # currently default write latency is 0
                SetIOLatency(clbit_write_latency=1600, conditional_latency=0),
                ALAPScheduleAnalysis(durations=self.instruction_durations),
                ConstrainedReschedule(acquire_alignment=16),
                PadDelay(),
            ]
        )

        aligned_circuit = pm.run(circuit)

        ref_circuit = QuantumCircuit(1, 1)
        ref_circuit.sx(0)
        ref_circuit.delay(100, 0, unit="dt")
        ref_circuit.x(0)
        ref_circuit.delay(100, 0, unit="dt")
        ref_circuit.sx(0)
        ref_circuit.delay(8, 0, unit="dt")
        ref_circuit.measure(0, 0)

        self.assertEqual(aligned_circuit, ref_circuit)

    def test_mid_circuit_measure(self):
        """Test circuit with mid circuit measurement.

        (input)

             ┌───┐┌────────────────┐┌─┐┌───────────────┐┌───┐┌────────────────┐┌─┐
        q_0: ┤ X ├┤ Delay(100[dt]) ├┤M├┤ Delay(10[dt]) ├┤ X ├┤ Delay(120[dt]) ├┤M├
             └───┘└────────────────┘└╥┘└───────────────┘└───┘└────────────────┘└╥┘
        c: 2/════════════════════════╩══════════════════════════════════════════╩═
                                     0                                          1

        (output)

             ┌───┐┌────────────────┐┌─┐┌───────────────┐┌───┐┌────────────────┐┌─┐
        q_0: ┤ X ├┤ Delay(112[dt]) ├┤M├┤ Delay(10[dt]) ├┤ X ├┤ Delay(134[dt]) ├┤M├
             └───┘└────────────────┘└╥┘└───────────────┘└───┘└────────────────┘└╥┘
        c: 2/════════════════════════╩══════════════════════════════════════════╩═
                                     0                                          1

        Extra delay is always added to the existing delay right before the measurement.
        Delay after measurement is unchanged.
        """
        circuit = QuantumCircuit(1, 2)
        circuit.x(0)
        circuit.delay(100, 0, unit="dt")
        circuit.measure(0, 0)
        circuit.delay(10, 0, unit="dt")
        circuit.x(0)
        circuit.delay(120, 0, unit="dt")
        circuit.measure(0, 1)

        pm = PassManager(
            [
                # reproduce old behavior of 0.20.0 before #7655
                # currently default write latency is 0
                SetIOLatency(clbit_write_latency=1600, conditional_latency=0),
                ALAPScheduleAnalysis(durations=self.instruction_durations),
                ConstrainedReschedule(acquire_alignment=16),
                PadDelay(),
            ]
        )

        aligned_circuit = pm.run(circuit)

        ref_circuit = QuantumCircuit(1, 2)
        ref_circuit.x(0)
        ref_circuit.delay(112, 0, unit="dt")
        ref_circuit.measure(0, 0)
        ref_circuit.delay(10, 0, unit="dt")
        ref_circuit.x(0)
        ref_circuit.delay(134, 0, unit="dt")
        ref_circuit.measure(0, 1)

        self.assertEqual(aligned_circuit, ref_circuit)

    def test_mid_circuit_multiq_gates(self):
        """Test circuit with mid circuit measurement and multi qubit gates.

        (input)

             ┌───┐┌────────────────┐┌─┐             ┌─┐
        q_0: ┤ X ├┤ Delay(100[dt]) ├┤M├──■───────■──┤M├
             └───┘└────────────────┘└╥┘┌─┴─┐┌─┐┌─┴─┐└╥┘
        q_1: ────────────────────────╫─┤ X ├┤M├┤ X ├─╫─
                                     ║ └───┘└╥┘└───┘ ║
        c: 2/════════════════════════╩═══════╩═══════╩═
                                     0       1       0

        (output)

                    ┌───┐       ┌────────────────┐┌─┐     ┌─────────────────┐     ┌─┐»
        q_0: ───────┤ X ├───────┤ Delay(112[dt]) ├┤M├──■──┤ Delay(1600[dt]) ├──■──┤M├»
             ┌──────┴───┴──────┐└────────────────┘└╥┘┌─┴─┐└───────┬─┬───────┘┌─┴─┐└╥┘»
        q_1: ┤ Delay(1872[dt]) ├───────────────────╫─┤ X ├────────┤M├────────┤ X ├─╫─»
             └─────────────────┘                   ║ └───┘        └╥┘        └───┘ ║ »
        c: 2/══════════════════════════════════════╩═══════════════╩═══════════════╩═»
                                                   0               1               0 »
        «
        «q_0: ───────────────────
        «     ┌─────────────────┐
        «q_1: ┤ Delay(1600[dt]) ├
        «     └─────────────────┘
        «c: 2/═══════════════════
        «

        Delay for the other channel paired by multi-qubit instruction is also scheduled.
        Delay (1872dt) = X (160dt) + Delay (100dt + extra 12dt) + Measure (1600dt).
        """
        circuit = QuantumCircuit(2, 2)
        circuit.x(0)
        circuit.delay(100, 0, unit="dt")
        circuit.measure(0, 0)
        circuit.cx(0, 1)
        circuit.measure(1, 1)
        circuit.cx(0, 1)
        circuit.measure(0, 0)

        pm = PassManager(
            [
                # reproduce old behavior of 0.20.0 before #7655
                # currently default write latency is 0
                SetIOLatency(clbit_write_latency=1600, conditional_latency=0),
                ALAPScheduleAnalysis(durations=self.instruction_durations),
                ConstrainedReschedule(acquire_alignment=16),
                PadDelay(),
            ]
        )

        aligned_circuit = pm.run(circuit)

        ref_circuit = QuantumCircuit(2, 2)
        ref_circuit.x(0)
        ref_circuit.delay(112, 0, unit="dt")
        ref_circuit.measure(0, 0)
        ref_circuit.delay(160 + 112 + 1600, 1, unit="dt")
        ref_circuit.cx(0, 1)
        ref_circuit.delay(1600, 0, unit="dt")
        ref_circuit.measure(1, 1)
        ref_circuit.cx(0, 1)
        ref_circuit.delay(1600, 1, unit="dt")
        ref_circuit.measure(0, 0)

        self.assertEqual(aligned_circuit, ref_circuit)

    def test_alignment_is_not_processed(self):
        """Test avoid pass processing if delay is aligned."""
        circuit = QuantumCircuit(2, 2)
        circuit.x(0)
        circuit.delay(160, 0, unit="dt")
        circuit.measure(0, 0)
        circuit.cx(0, 1)
        circuit.measure(1, 1)
        circuit.cx(0, 1)
        circuit.measure(0, 0)

        # pre scheduling is not necessary because alignment is skipped
        # this is to minimize breaking changes to existing code.
        pm = PassManager()

        pm.append(InstructionDurationCheck(acquire_alignment=16))
        pm.run(circuit)

        self.assertFalse(pm.property_set["reschedule_required"])

    def test_circuit_using_clbit(self):
        """Test a circuit with instructions using a common clbit.

        (input)
             ┌───┐┌────────────────┐┌─┐
        q_0: ┤ X ├┤ Delay(100[dt]) ├┤M├──────────────
             └───┘└────────────────┘└╥┘   ┌───┐
        q_1: ────────────────────────╫────┤ X ├──────
                                     ║    └─╥─┘   ┌─┐
        q_2: ────────────────────────╫──────╫─────┤M├
                                     ║ ┌────╨────┐└╥┘
        c: 1/════════════════════════╩═╡ c_0 = T ╞═╩═
                                     0 └─────────┘ 0

        (aligned)
                    ┌───┐       ┌────────────────┐┌─┐┌────────────────┐
        q_0: ───────┤ X ├───────┤ Delay(112[dt]) ├┤M├┤ Delay(160[dt]) ├───
             ┌──────┴───┴──────┐└────────────────┘└╥┘└─────┬───┬──────┘
        q_1: ┤ Delay(1872[dt]) ├───────────────────╫───────┤ X ├──────────
             └┬────────────────┤                   ║       └─╥─┘       ┌─┐
        q_2: ─┤ Delay(432[dt]) ├───────────────────╫─────────╫─────────┤M├
              └────────────────┘                   ║    ┌────╨────┐    └╥┘
        c: 1/══════════════════════════════════════╩════╡ c_0 = T ╞═════╩═
                                                   0    └─────────┘     0

        Looking at the q_0, the total schedule length T becomes
        160 (x) + 112 (aligned delay) + 1600 (measure) + 160 (delay) = 2032.
        The last delay comes from ALAP scheduling called before the AlignMeasure pass,
        which aligns stop times as late as possible, so the start time of x(1).c_if(0)
        and the stop time of measure(0, 0) become T - 160.
        """
        circuit = QuantumCircuit(3, 1)
        circuit.x(0)
        circuit.delay(100, 0, unit="dt")
        circuit.measure(0, 0)
        circuit.x(1).c_if(0, 1)
        circuit.measure(2, 0)

        pm = PassManager(
            [
                # reproduce old behavior of 0.20.0 before #7655
                # currently default write latency is 0
                SetIOLatency(clbit_write_latency=1600, conditional_latency=0),
                ALAPScheduleAnalysis(durations=self.instruction_durations),
                ConstrainedReschedule(acquire_alignment=16),
                PadDelay(fill_very_end=False),
            ]
        )

        aligned_circuit = pm.run(circuit)

        self.assertEqual(aligned_circuit.duration, 2032)

        ref_circuit = QuantumCircuit(3, 1)
        ref_circuit.x(0)
        ref_circuit.delay(112, 0, unit="dt")
        ref_circuit.delay(1872, 1, unit="dt")  # 2032 - 160
        ref_circuit.delay(432, 2, unit="dt")  # 2032 - 1600
        ref_circuit.measure(0, 0)
        ref_circuit.x(1).c_if(0, 1)
        ref_circuit.measure(2, 0)

        self.assertEqual(aligned_circuit, ref_circuit)

    def test_programmed_delay_preserved(self):
        """Intentionally programmed delay will be kept after reschedule.

        No delay
        ++++++++

        (input)
             ┌────────────────┐┌───┐ ░ ┌───┐
        q_0: ┤ Delay(100[dt]) ├┤ X ├─░─┤ X ├
             ├────────────────┤└───┘ ░ └───┘
        q_1: ┤ Delay(272[dt]) ├──────░──────
             └────────────────┘      ░

        (aligned)
             ┌────────────────┐┌───┐ ░ ┌───┐
        q_0: ┤ Delay(112[dt]) ├┤ X ├─░─┤ X ├
             ├────────────────┤└───┘ ░ └───┘
        q_1: ┤ Delay(272[dt]) ├──────░──────
             └────────────────┘      ░

        With delay (intentional post buffer)
        ++++++++++++++++++++++++++++++++++++

        (input) ... this is identical to no delay pattern without reschedule
             ┌────────────────┐┌───┐┌───────────────┐ ░ ┌───┐
        q_0: ┤ Delay(100[dt]) ├┤ X ├┤ Delay(10[dt]) ├─░─┤ X ├
             ├────────────────┤└───┘└───────────────┘ ░ └───┘
        q_1: ┤ Delay(272[dt]) ├───────────────────────░──────
             └────────────────┘                       ░

        (aligned)
             ┌────────────────┐┌───┐┌───────────────┐ ░ ┌──────────────┐┌───┐
        q_0: ┤ Delay(112[dt]) ├┤ X ├┤ Delay(10[dt]) ├─░─┤ Delay(6[dt]) ├┤ X ├
             ├────────────────┤└───┘└───────────────┘ ░ └──────────────┘└───┘
        q_1: ┤ Delay(282[dt]) ├───────────────────────░──────────────────────
             └────────────────┘                       ░

        """

        pm = PassManager(
            [
                ASAPScheduleAnalysis(durations=self.instruction_durations),
                ConstrainedReschedule(pulse_alignment=16),
                PadDelay(fill_very_end=False),
            ]
        )

        pm_only_schedule = PassManager(
            [
                ASAPScheduleAnalysis(durations=self.instruction_durations),
                PadDelay(fill_very_end=False),
            ]
        )

        circuit_no_delay = QuantumCircuit(2)
        circuit_no_delay.delay(100, 0, unit="dt")
        circuit_no_delay.x(0)  # q0 ends here at t = 260, t = 260 - 272 is free
        circuit_no_delay.delay(160 + 112, 1, unit="dt")
        circuit_no_delay.barrier()  # q0 and q1 is aligned here at t = 272 dt
        circuit_no_delay.x(0)

        ref_no_delay = QuantumCircuit(2)
        ref_no_delay.delay(112, 0, unit="dt")
        ref_no_delay.x(0)
        ref_no_delay.delay(160 + 100 + 12, 1, unit="dt")  # this t0 doesn't change
        ref_no_delay.barrier()
        ref_no_delay.x(0)  # no buffer

        self.assertEqual(pm.run(circuit_no_delay), ref_no_delay)

        circuit_with_delay = QuantumCircuit(2)
        circuit_with_delay.delay(100, 0, unit="dt")
        circuit_with_delay.x(0)  # q0 ends here at t = 260
        circuit_with_delay.delay(10, 0, unit="dt")  # intentional post buffer of 10 dt to next X(0)
        circuit_with_delay.delay(160 + 112, 1, unit="dt")  # q0 and q1 is aligned here at t = 272 dt
        circuit_with_delay.barrier()
        circuit_with_delay.x(0)

        ref_with_delay = QuantumCircuit(2)
        ref_with_delay.delay(112, 0, unit="dt")
        ref_with_delay.x(0)
        ref_with_delay.delay(10, 0, unit="dt")  # this delay survive
        ref_with_delay.delay(160 + 100 + 12 + 10, 1, unit="dt")
        ref_with_delay.barrier()
        ref_with_delay.delay(6, 0, unit="dt")  # extra delay for next X0
        ref_with_delay.x(0)  # at least 10dt buffer is preserved

        self.assertEqual(pm.run(circuit_with_delay), ref_with_delay)

        # check if circuit is identical without reschedule
        self.assertEqual(
            pm_only_schedule.run(circuit_no_delay),
            pm_only_schedule.run(circuit_with_delay),
        )

    def test_both_pulse_and_acquire_alignment(self):
        """Test when both acquire and pulse alignment are specified.

        (input)
             ┌────────────────┐┌───┐┌───────────────┐┌─┐
          q: ┤ Delay(100[dt]) ├┤ X ├┤ Delay(10[dt]) ├┤M├
             └────────────────┘└───┘└───────────────┘└╥┘
        c: 1/═════════════════════════════════════════╩═
                                                      0

        (aligned)
             ┌────────────────┐┌───┐┌───────────────┐┌─┐
          q: ┤ Delay(112[dt]) ├┤ X ├┤ Delay(16[dt]) ├┤M├
             └────────────────┘└───┘└───────────────┘└╥┘
        c: 1/═════════════════════════════════════════╩═
                                                      0
        """
        pm = PassManager(
            [
                ALAPScheduleAnalysis(durations=self.instruction_durations),
                ConstrainedReschedule(pulse_alignment=16, acquire_alignment=16),
                PadDelay(fill_very_end=False),
            ]
        )

        circuit = QuantumCircuit(1, 1)
        circuit.delay(100, 0, unit="dt")
        circuit.x(0)
        circuit.delay(10, 0, unit="dt")
        circuit.measure(0, 0)

        ref_circ = QuantumCircuit(1, 1)
        ref_circ.delay(112, 0, unit="dt")
        ref_circ.x(0)
        ref_circ.delay(16, 0, unit="dt")
        ref_circ.measure(0, 0)

        self.assertEqual(pm.run(circuit), ref_circ)

    def test_deprecated_align_measure(self):
        """Test if old AlignMeasures can be still used and warning is raised."""
        circuit = QuantumCircuit(1, 1)
        circuit.x(0)
        circuit.delay(100)
        circuit.measure(0, 0)

        with self.assertWarns(PendingDeprecationWarning):
            pm_old = PassManager(
                [
                    ALAPSchedule(durations=self.instruction_durations),
                    AlignMeasures(alignment=16),
                ]
            )

        pm_new = PassManager(
            [
                ALAPSchedule(durations=self.instruction_durations),
                AlignMeasures(alignment=16),
            ]
        )

        self.assertEqual(pm_old.run(circuit), pm_new.run(circuit))


class TestPulseGateValidation(QiskitTestCase):
    """A test for pulse gate validation pass."""

    def test_invalid_pulse_duration(self):
        """Kill pass manager if invalid pulse gate is found."""

        # this is invalid duration pulse
        # this will cause backend error since this doesn't fit with waveform memory chunk.
        custom_gate = pulse.Schedule(name="custom_x_gate")
        custom_gate.insert(
            0, pulse.Play(pulse.Constant(100, 0.1), pulse.DriveChannel(0)), inplace=True
        )

        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.add_calibration("x", qubits=(0,), schedule=custom_gate)

        pm = PassManager(ValidatePulseGates(granularity=16, min_length=64))
        with self.assertRaises(TranspilerError):
            pm.run(circuit)

    def test_short_pulse_duration(self):
        """Kill pass manager if invalid pulse gate is found."""

        # this is invalid duration pulse
        # this will cause backend error since this doesn't fit with waveform memory chunk.
        custom_gate = pulse.Schedule(name="custom_x_gate")
        custom_gate.insert(
            0, pulse.Play(pulse.Constant(32, 0.1), pulse.DriveChannel(0)), inplace=True
        )

        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.add_calibration("x", qubits=(0,), schedule=custom_gate)

        pm = PassManager(ValidatePulseGates(granularity=16, min_length=64))
        with self.assertRaises(TranspilerError):
            pm.run(circuit)

    def test_short_pulse_duration_multiple_pulse(self):
        """Kill pass manager if invalid pulse gate is found."""

        # this is invalid duration pulse
        # however total gate schedule length is 64, which accidentally satisfies the constraints
        # this should fail in the validation
        custom_gate = pulse.Schedule(name="custom_x_gate")
        custom_gate.insert(
            0, pulse.Play(pulse.Constant(32, 0.1), pulse.DriveChannel(0)), inplace=True
        )
        custom_gate.insert(
            32, pulse.Play(pulse.Constant(32, 0.1), pulse.DriveChannel(0)), inplace=True
        )

        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.add_calibration("x", qubits=(0,), schedule=custom_gate)

        pm = PassManager(ValidatePulseGates(granularity=16, min_length=64))
        with self.assertRaises(TranspilerError):
            pm.run(circuit)

    def test_valid_pulse_duration(self):
        """No error raises if valid calibration is provided."""

        # this is valid duration pulse
        custom_gate = pulse.Schedule(name="custom_x_gate")
        custom_gate.insert(
            0, pulse.Play(pulse.Constant(160, 0.1), pulse.DriveChannel(0)), inplace=True
        )

        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.add_calibration("x", qubits=(0,), schedule=custom_gate)

        # just not raise an error
        pm = PassManager(ValidatePulseGates(granularity=16, min_length=64))
        pm.run(circuit)

    def test_no_calibration(self):
        """No error raises if no calibration is addedd."""

        circuit = QuantumCircuit(1)
        circuit.x(0)

        # just not raise an error
        pm = PassManager(ValidatePulseGates(granularity=16, min_length=64))
        pm.run(circuit)
