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

"""Testing legacy instruction alignment pass."""

from qiskit import QuantumCircuit, pulse
from qiskit.test import QiskitTestCase
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import ValidatePulseGates


class TestPulseGateValidation(QiskitTestCase):
    """A test for pulse gate validation pass."""

    def setUp(self):
        super().setUp()
        self.pulse_gate_validation_pass = ValidatePulseGates(granularity=16, min_length=64)

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

        with self.assertRaises(TranspilerError):
            self.pulse_gate_validation_pass(circuit)

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

        with self.assertRaises(TranspilerError):
            self.pulse_gate_validation_pass(circuit)

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

        with self.assertRaises(TranspilerError):
            self.pulse_gate_validation_pass(circuit)

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
        self.pulse_gate_validation_pass(circuit)

    def test_no_calibration(self):
        """No error raises if no calibration is addedd."""

        circuit = QuantumCircuit(1)
        circuit.x(0)

        # just not raise an error
        self.pulse_gate_validation_pass(circuit)
