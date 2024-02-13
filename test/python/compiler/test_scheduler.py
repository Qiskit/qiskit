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

"""Scheduler Test."""
from qiskit import transpile
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.duration import convert_durations_to_dt
from qiskit.exceptions import QiskitError
from qiskit.pulse import InstructionScheduleMap, Schedule
from qiskit.providers.fake_provider import FakeOpenPulse3Q, GenericBackendV2
from qiskit.compiler.scheduler import schedule
from qiskit.scheduler import ScheduleConfig
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestCircuitScheduler(QiskitTestCase):
    """Tests for scheduling."""

    def setUp(self):
        super().setUp()
        qr = QuantumRegister(2, name="q")
        cr = ClassicalRegister(2, name="c")
        self.circ = QuantumCircuit(qr, cr, name="circ")
        self.circ.cx(qr[0], qr[1])
        self.circ.measure(qr, cr)

        qr2 = QuantumRegister(2, name="q")
        cr2 = ClassicalRegister(2, name="c")
        self.circ2 = QuantumCircuit(qr2, cr2, name="circ2")
        self.circ2.cx(qr2[0], qr2[1])
        self.circ2.measure(qr2, cr2)

        self.backend = FakeOpenPulse3Q()
        self.backend_config = self.backend.configuration()
        self.num_qubits = self.backend_config.n_qubits

    def test_instruction_map_and_backend_not_supplied(self):
        """Test instruction map and backend not supplied."""
        with self.assertRaisesRegex(
            QiskitError,
            r"Must supply either a backend or InstructionScheduleMap for scheduling passes.",
        ):
            schedule(self.circ)

    def test_instruction_map_and_backend_defaults_unavailable(self):
        """Test backend defaults unavailable when backend is provided, but instruction map is not."""
        self.backend._defaults = None
        with self.assertRaisesRegex(
            QiskitError, r"The backend defaults are unavailable. The backend may not support pulse."
        ):
            schedule(self.circ, self.backend)

    def test_measurement_map_and_backend_not_supplied(self):
        """Test measurement map and backend not supplied."""
        with self.assertRaisesRegex(
            QiskitError,
            r"Must supply either a backend or a meas_map for scheduling passes.",
        ):
            schedule(self.circ, inst_map=InstructionScheduleMap())

    def test_schedules_single_circuit(self):
        """Test scheduling of a single circuit."""
        circuit_schedule = schedule(self.circ, self.backend)

        self.assertIsInstance(circuit_schedule, Schedule)
        self.assertEqual(circuit_schedule.name, "circ")

    def test_schedules_multiple_circuits(self):
        """Test scheduling of multiple circuits."""
        self.enable_parallel_processing()

        circuits = [self.circ, self.circ2]
        circuit_schedules = schedule(circuits, self.backend, method="asap")
        self.assertEqual(len(circuit_schedules), len(circuits))

        circuit_one_schedule = circuit_schedules[0]
        circuit_two_schedule = circuit_schedules[1]

        self.assertEqual(
            circuit_one_schedule,
            schedule(self.circ, self.backend, method="asap"),
        )

        self.assertEqual(
            circuit_two_schedule,
            schedule(self.circ2, self.backend, method="asap"),
        )

    def test_convert_duration_to_dt(self):
        """Test that circuit duration unit conversion is applied only when necessary.
        Tests fix for bug reported in PR #11782."""

        backend = GenericBackendV2(num_qubits=3, calibrate_instructions=True, seed=10)
        schedule_config = ScheduleConfig(
            inst_map=backend.target.instruction_schedule_map(),
            meas_map=backend.meas_map,
            dt=backend.dt,
        )

        for circuit in [self.circ, self.circ2]:
            with self.subTest(circuit=circuit):
                transpiled_circ = transpile(circuit, backend, scheduling_method="asap")
                converted_circ = convert_durations_to_dt(
                    transpiled_circ, dt_in_sec=schedule_config.dt, inplace=False
                )
                self.assertEqual(
                    converted_circ.duration,
                    transpiled_circ.duration,
                )
                self.assertEqual(
                    converted_circ.unit,
                    transpiled_circ.unit,
                )
