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

# pylint: disable=missing-function-docstring

"""Test scheduled circuit (quantum circuit with duration)."""

from qiskit import QuantumCircuit, QiskitError
from qiskit import transpile, execute, assemble
from qiskit.circuit.duration import duration_in_dt
from qiskit.test.mock.backends import FakeParis, FakeVigo
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations

from qiskit.test.base import QiskitTestCase


class TestScheduledCircuit(QiskitTestCase):
    """Test scheduled circuit (quantum circuit with duration)."""
    def setUp(self):
        super().setUp()
        self.backend = FakeParis()
        self.dt = self.backend.configuration().dt
        self.backend_with_dt = FakeParis()
        self.backend_without_dt = FakeVigo()

    def test_scinario_1_with_backend(self):
        """[Scenario 1: dt is known to transpiler]"""
        qc = QuantumCircuit(2)
        qc.delay(0.1, 0, unit='ms')
        qc.delay(100, 0, unit='ns')
        qc.h(0)
        qc.h(1)
        sc = transpile(qc, self.backend_with_dt, scheduling_method='alap')
        qobj = assemble(sc, self.backend_with_dt)

    def test_scinario_1_without_backend(self):
        qc = QuantumCircuit(2)
        qc.delay(0.1, 0, unit='ms')
        qc.delay(100, 0, unit='ns')
        qc.h(0)
        qc.h(1)
        sc = transpile(qc, scheduling_method='alap', instruction_durations=[('h', None, 50)],
                       dt=0.123)

    def test_scinario_2_with_backend(self):
        """[Scenario 2: dt is unknown and all delays and gate times are in SI]"""
        qc = QuantumCircuit(2)
        qc.delay(0.1, 0, unit='ms')
        qc.delay(100, 0, unit='ns')
        qc.h(0)
        qc.h(1)
        sc = transpile(qc, self.backend_without_dt, scheduling_method='alap')
        with self.assertRaises(QiskitError):
            assemble(sc, self.backend_without_dt)

    def test_scinario_3_with_backend(self):
        """[Scenario 3: dt is unknown but delays and gate times have a mix of SI and dt]"""
        qc = QuantumCircuit(2)
        qc.delay(100, 0, unit='ns')
        qc.delay(30, 0, unit='dt')
        qc.h(0)
        qc.h(1)
        with self.assertRaises(QiskitError):
            transpile(qc, self.backend_without_dt, scheduling_method='alap')

    def test_cannot_execute_delay_circuit_when_schedule_circuit_off(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        with self.assertRaises(QiskitError):
            execute(qc, backend=self.backend, schedule_circuit=False)

    def test_transpile_t1_circuit(self):
        qc = QuantumCircuit(1)
        qc.x(0)  # 320 [dt]
        qc.delay(1000, 0, unit='ns')  # 4500 [dt]
        qc.measure_all()  # 19200 [dt]
        scheduled = transpile(qc, backend=self.backend, scheduling_method='alap')
        self.assertEqual(scheduled.duration, 24020)

    def test_transpile_delay_circuit_with_backend(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(100, 1, unit='ns')  # 450 [dt]
        qc.cx(0, 1)  # 1408 [dt]
        scheduled = transpile(qc, backend=self.backend, scheduling_method='alap')
        self.assertEqual(scheduled.duration, 1858)

    def test_transpile_delay_circuit_without_backend(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        scheduled = transpile(qc,
                              scheduling_method='alap',
                              instruction_durations=[('h', 0, 200), ('cx', [0, 1], 700)])
        self.assertEqual(scheduled.duration, 1200)

    def test_transpile_delay_circuit_without_scheduling_method_as_normal_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        transpiled = transpile(qc)
        self.assertEqual(transpiled.duration, None)

    def test_raise_error_if_transpile_with_scheduling_method_but_without_durations(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        with self.assertRaises(TranspilerError):
            transpile(qc, scheduling_method="alap")

    def test_invalidate_schedule_circuit_if_new_instruction_is_appended(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500*self.dt, 1, 's')
        qc.cx(0, 1)
        scheduled = transpile(qc,
                              backend=self.backend,
                              scheduling_method='alap')
        # append a gate to a scheduled circuit
        scheduled.h(0)
        self.assertEqual(scheduled.duration, None)

    def test_default_units_for_my_own_duration_users(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        # accept None for qubits
        scheduled = transpile(qc,
                              basis_gates=['h', 'cx', 'delay'],
                              scheduling_method='alap',
                              instruction_durations=[('h', 0, 200),
                                                     ('cx', None, 900)]
                              )
        self.assertEqual(scheduled.duration, 1400)
        # prioritize specified qubits over None
        scheduled = transpile(qc,
                              basis_gates=['h', 'cx', 'delay'],
                              scheduling_method='alap',
                              instruction_durations=[('h', 0, 200),
                                                     ('cx', None, 900),
                                                     ('cx', [0, 1], 800)]
                              )
        self.assertEqual(scheduled.duration, 1300)

    def test_unit_seconds_for_users_who_uses_durations_given_by_backend(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500*self.dt, 1, 's')
        qc.cx(0, 1)
        # usual case
        scheduled = transpile(qc,
                              backend=self.backend,
                              scheduling_method='alap'
                              )
        self.assertEqual(scheduled.duration, 1908)

        # update durations
        scheduled = transpile(qc,
                              backend=self.backend,
                              scheduling_method='alap',
                              instruction_durations=[('cx', [0, 1], 1000*self.dt, 's')]
                              )
        self.assertEqual(scheduled.duration, 1500)

        my_own_durations = InstructionDurations([('cx', [0, 1], 1000*self.dt, 's')])
        scheduled = transpile(qc,
                              backend=self.backend,  # unit='s'
                              scheduling_method='alap',
                              instruction_durations=my_own_durations
                              )
        self.assertEqual(scheduled.duration, 1500)
