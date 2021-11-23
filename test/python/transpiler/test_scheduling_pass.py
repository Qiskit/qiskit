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

"""Test the Scheduling passes"""

import unittest

from ddt import ddt, data
from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes import ASAPSchedule, ALAPSchedule
from qiskit.transpiler.passmanager import PassManager


@ddt
class TestSchedulingPass(QiskitTestCase):
    """Tests the Scheduling passes"""

    def test_alap_agree_with_reverse_asap_reverse(self):
        """Test if ALAP schedule agrees with doubly-reversed ASAP schedule."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        qc.measure_all()

        durations = InstructionDurations(
            [("h", 0, 200), ("cx", [0, 1], 700), ("measure", None, 1000)]
        )

        pm = PassManager(ALAPSchedule(durations))
        alap_qc = pm.run(qc)

        pm = PassManager(ASAPSchedule(durations))
        new_qc = pm.run(qc.reverse_ops())
        new_qc = new_qc.reverse_ops()
        new_qc.name = new_qc.name

        self.assertEqual(alap_qc, new_qc)

    @data(ALAPSchedule, ASAPSchedule)
    def test_classically_controlled_gate_after_measure(self, schedule_pass):
        """Test if ALAP/ASAP schedules circuits with c_if after measure with a common clbit.
        See: https://github.com/Qiskit/qiskit-terra/issues/7006

        (input)
             ┌─┐
        q_0: ┤M├───────────
             └╥┘   ┌───┐
        q_1: ─╫────┤ X ├───
              ║    └─╥─┘
              ║ ┌────╨────┐
        c: 1/═╩═╡ c_0 = T ╞
              0 └─────────┘

        (scheduled)
                                ┌─┐┌────────────────┐
        q_0: ───────────────────┤M├┤ Delay(200[dt]) ├
             ┌─────────────────┐└╥┘└─────┬───┬──────┘
        q_1: ┤ Delay(1000[dt]) ├─╫───────┤ X ├───────
             └─────────────────┘ ║       └─╥─┘
                                 ║    ┌────╨────┐
        c: 1/════════════════════╩════╡ c_0 = T ╞════
                                 0    └─────────┘
        """
        qc = QuantumCircuit(2, 1)
        qc.measure(0, 0)
        qc.x(1).c_if(0, True)

        durations = InstructionDurations([("x", None, 200), ("measure", None, 1000)])
        pm = PassManager(schedule_pass(durations))
        scheduled = pm.run(qc)

        expected = QuantumCircuit(2, 1)
        expected.measure(0, 0)
        expected.delay(200, 0)
        expected.delay(1000, 1)  # x.c_if starts after measure
        expected.x(1).c_if(0, True)

        self.assertEqual(expected, scheduled)

    @data(ALAPSchedule, ASAPSchedule)
    def test_measure_after_measure(self, schedule_pass):
        """Test if ALAP/ASAP schedules circuits with measure after measure with a common clbit.
        See: https://github.com/Qiskit/qiskit-terra/issues/7006

        (input)
             ┌───┐┌─┐
        q_0: ┤ X ├┤M├───
             └───┘└╥┘┌─┐
        q_1: ──────╫─┤M├
                   ║ └╥┘
        c: 1/══════╩══╩═
                   0  0

        (scheduled)
                   ┌───┐       ┌─┐
        q_0: ──────┤ X ├───────┤M├───
             ┌─────┴───┴──────┐└╥┘┌─┐
        q_1: ┤ Delay(200[dt]) ├─╫─┤M├
             └────────────────┘ ║ └╥┘
        c: 1/═══════════════════╩══╩═
                                0  0
        """
        qc = QuantumCircuit(2, 1)
        qc.x(0)
        qc.measure(0, 0)
        qc.measure(1, 0)

        durations = InstructionDurations([("x", None, 200), ("measure", None, 1000)])
        pm = PassManager(schedule_pass(durations))
        scheduled = pm.run(qc)

        expected = QuantumCircuit(2, 1)
        expected.x(0)
        expected.measure(0, 0)
        expected.delay(200, 1)  # 2nd measure starts at the same time as 1st measure starts
        expected.measure(1, 0)

        self.assertEqual(expected, scheduled)

    @data(ALAPSchedule, ASAPSchedule)
    def test_c_if_on_different_qubits(self, schedule_pass):
        """Test if ALAP/ASAP schedules circuits with `c_if`s on different qubits.

        (input)
             ┌─┐
        q_0: ┤M├──────────────────────
             └╥┘   ┌───┐
        q_1: ─╫────┤ X ├──────────────
              ║    └─╥─┘      ┌───┐
        q_2: ─╫──────╫────────┤ X ├───
              ║      ║        └─╥─┘
              ║ ┌────╨────┐┌────╨────┐
        c: 1/═╩═╡ c_0 = T ╞╡ c_0 = T ╞
              0 └─────────┘└─────────┘

        (scheduled)
                                ┌─┐┌────────────────┐
        q_0: ───────────────────┤M├┤ Delay(200[dt]) ├───────────
             ┌─────────────────┐└╥┘└─────┬───┬──────┘
        q_1: ┤ Delay(1000[dt]) ├─╫───────┤ X ├──────────────────
             ├─────────────────┤ ║       └─╥─┘          ┌───┐
        q_2: ┤ Delay(1000[dt]) ├─╫─────────╫────────────┤ X ├───
             └─────────────────┘ ║         ║            └─╥─┘
                                 ║    ┌────╨────┐    ┌────╨────┐
        c: 1/════════════════════╩════╡ c_0 = T ╞════╡ c_0 = T ╞
                                 0    └─────────┘    └─────────┘
        """
        qc = QuantumCircuit(3, 1)
        qc.measure(0, 0)
        qc.x(1).c_if(0, True)
        qc.x(2).c_if(0, True)

        durations = InstructionDurations([("x", None, 200), ("measure", None, 1000)])
        pm = PassManager(schedule_pass(durations))
        scheduled = pm.run(qc)

        expected = QuantumCircuit(3, 1)
        expected.measure(0, 0)
        expected.delay(200, 0)
        expected.delay(1000, 1)
        expected.delay(1000, 2)
        expected.x(1).c_if(0, True)
        expected.x(2).c_if(0, True)

        self.assertEqual(expected, scheduled)

    @data(ALAPSchedule, ASAPSchedule)
    def test_shorter_measure_after_measure(self, schedule_pass):
        """Test if ALAP/ASAP schedules circuits with shorter measure after measure with a common clbit.

        (input)
             ┌─┐
        q_0: ┤M├───
             └╥┘┌─┐
        q_1: ─╫─┤M├
              ║ └╥┘
        c: 1/═╩══╩═
              0  0

        (scheduled)
                               ┌─┐
        q_0: ──────────────────┤M├───
             ┌────────────────┐└╥┘┌─┐
        q_1: ┤ Delay(300[dt]) ├─╫─┤M├
             └────────────────┘ ║ └╥┘
        c: 1/═══════════════════╩══╩═
                                0  0
        """
        qc = QuantumCircuit(2, 1)
        qc.measure(0, 0)
        qc.measure(1, 0)

        durations = InstructionDurations([("measure", 0, 1000), ("measure", 1, 700)])
        pm = PassManager(schedule_pass(durations))
        scheduled = pm.run(qc)

        expected = QuantumCircuit(2, 1)
        expected.measure(0, 0)
        expected.delay(300, 1)
        expected.measure(1, 0)

        self.assertEqual(expected, scheduled)

    def test_measure_after_c_if(self):
        """Test if ALAP/ASAP schedules circuits with c_if after measure with a common clbit.

        (input)
             ┌─┐
        q_0: ┤M├──────────────
             └╥┘   ┌───┐
        q_1: ─╫────┤ X ├──────
              ║    └─╥─┘   ┌─┐
        q_2: ─╫──────╫─────┤M├
              ║ ┌────╨────┐└╥┘
        c: 1/═╩═╡ c_0 = T ╞═╩═
              0 └─────────┘ 0

        (scheduled - ASAP)
                                ┌─┐┌────────────────┐
        q_0: ───────────────────┤M├┤ Delay(200[dt]) ├─────────────────────
             ┌─────────────────┐└╥┘└─────┬───┬──────┘
        q_1: ┤ Delay(1000[dt]) ├─╫───────┤ X ├────────────────────────────
             └─────────────────┘ ║       └─╥─┘       ┌─┐┌────────────────┐
        q_2: ────────────────────╫─────────╫─────────┤M├┤ Delay(200[dt]) ├
                                 ║    ┌────╨────┐    └╥┘└────────────────┘
        c: 1/════════════════════╩════╡ c_0 = T ╞═════╩═══════════════════
                                 0    └─────────┘     0

        (scheduled - ALAP)
                                ┌─┐┌────────────────┐
        q_0: ───────────────────┤M├┤ Delay(200[dt]) ├───
             ┌─────────────────┐└╥┘└─────┬───┬──────┘
        q_1: ┤ Delay(1000[dt]) ├─╫───────┤ X ├──────────
             └┬────────────────┤ ║       └─╥─┘       ┌─┐
        q_2: ─┤ Delay(200[dt]) ├─╫─────────╫─────────┤M├
              └────────────────┘ ║    ┌────╨────┐    └╥┘
        c: 1/════════════════════╩════╡ c_0 = T ╞═════╩═
                                 0    └─────────┘     0
        """
        qc = QuantumCircuit(3, 1)
        qc.measure(0, 0)
        qc.x(1).c_if(0, 1)
        qc.measure(2, 0)

        durations = InstructionDurations([("x", None, 200), ("measure", None, 1000)])
        actual_asap = PassManager(ASAPSchedule(durations)).run(qc)
        actual_alap = PassManager(ALAPSchedule(durations)).run(qc)

        # start times of 2nd measure depends on ASAP/ALAP
        expected_asap = QuantumCircuit(3, 1)
        expected_asap.measure(0, 0)
        expected_asap.delay(200, 0)
        expected_asap.delay(1000, 1)
        expected_asap.x(1).c_if(0, 1)
        expected_asap.measure(2, 0)
        expected_asap.delay(200, 2)  # delay after measure on q_2
        self.assertEqual(expected_asap, actual_asap)

        expected_aslp = QuantumCircuit(3, 1)
        expected_aslp.measure(0, 0)
        expected_aslp.delay(200, 0)
        expected_aslp.delay(1000, 1)
        expected_aslp.x(1).c_if(0, 1)
        expected_aslp.delay(200, 2)
        expected_aslp.measure(2, 0)  # delay before measure on q_2
        self.assertEqual(expected_aslp, actual_alap)


if __name__ == "__main__":
    unittest.main()
