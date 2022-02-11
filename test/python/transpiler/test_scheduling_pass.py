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
        qc.barrier()
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
                                ┌─┐
        q_0: ───────────────────┤M├───────────
             ┌─────────────────┐└╥┘   ┌───┐
        q_1: ┤ Delay(1000[dt]) ├─╫────┤ X ├───
             └─────────────────┘ ║    └─╥─┘
                                 ║ ┌────╨────┐
        c: 1/════════════════════╩═╡ c_0=0x1 ╞
                                 0 └─────────┘
        """
        qc = QuantumCircuit(2, 1)
        qc.measure(0, 0)
        qc.x(1).c_if(0, True)

        durations = InstructionDurations([("x", None, 200), ("measure", None, 1000)])
        pm = PassManager(schedule_pass(durations))
        scheduled = pm.run(qc)

        expected = QuantumCircuit(2, 1)
        expected.measure(0, 0)
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
        q_0: ───────┤ X ├───────┤M├───
             ┌──────┴───┴──────┐└╥┘┌─┐
        q_1: ┤ Delay(1200[dt]) ├─╫─┤M├
             └─────────────────┘ ║ └╥┘
        c: 1/════════════════════╩══╩═
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
        expected.delay(1200, 1)
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

                                ┌─┐
        q_0: ───────────────────┤M├──────────────────────
             ┌─────────────────┐└╥┘   ┌───┐
        q_1: ┤ Delay(1000[dt]) ├─╫────┤ X ├──────────────
             ├─────────────────┤ ║    └─╥─┘      ┌───┐
        q_2: ┤ Delay(1000[dt]) ├─╫──────╫────────┤ X ├───
             └─────────────────┘ ║      ║        └─╥─┘
                                 ║ ┌────╨────┐┌────╨────┐
        c: 1/════════════════════╩═╡ c_0=0x1 ╞╡ c_0=0x1 ╞
                                 0 └─────────┘└─────────┘
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
        q_0: ───────────────────┤M├───
             ┌─────────────────┐└╥┘┌─┐
        q_1: ┤ Delay(1000[dt]) ├─╫─┤M├
             └─────────────────┘ ║ └╥┘
        c: 1/════════════════════╩══╩═
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
        expected.delay(1000, 1)
        expected.measure(1, 0)

        self.assertEqual(expected, scheduled)

    @data(ALAPSchedule, ASAPSchedule)
    def test_measure_after_c_if(self, schedule_pass):
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

        (scheduled)
                                ┌─┐
        q_0: ───────────────────┤M├──────────────
             ┌─────────────────┐└╥┘   ┌───┐
        q_1: ┤ Delay(1000[dt]) ├─╫────┤ X ├──────
             ├─────────────────┤ ║    └─╥─┘   ┌─┐
        q_2: ┤ Delay(1200[dt]) ├─╫──────╫─────┤M├
             └─────────────────┘ ║ ┌────╨────┐└╥┘
        c: 1/════════════════════╩═╡ c_0=0x1 ╞═╩═
                                 0 └─────────┘ 0
        """
        qc = QuantumCircuit(3, 1)
        qc.measure(0, 0)
        qc.x(1).c_if(0, 1)
        qc.measure(2, 0)

        durations = InstructionDurations([("x", None, 200), ("measure", None, 1000)])
        pm = PassManager(schedule_pass(durations))
        scheduled = pm.run(qc)

        expected = QuantumCircuit(3, 1)
        expected.delay(1000, 1)
        expected.delay(1200, 2)
        expected.measure(0, 0)
        expected.x(1).c_if(0, 1)
        expected.measure(2, 0)
        expected.draw()

        self.assertEqual(expected, scheduled)

    def test_parallel_gate_different_length(self):
        """Test circuit having two parallel instruction with different length.

        (input)
             ┌───┐┌─┐
        q_0: ┤ X ├┤M├───
             ├───┤└╥┘┌─┐
        q_1: ┤ X ├─╫─┤M├
             └───┘ ║ └╥┘
        c: 2/══════╩══╩═
                   0  1

        (expected, ALAP)
             ┌────────────────┐┌───┐┌─┐
        q_0: ┤ Delay(200[dt]) ├┤ X ├┤M├
             └─────┬───┬──────┘└┬─┬┘└╥┘
        q_1: ──────┤ X ├────────┤M├──╫─
                   └───┘        └╥┘  ║
        c: 2/════════════════════╩═══╩═
                                 1   0

        (expected, ASAP)
             ┌───┐┌─┐
        q_0: ┤ X ├┤M├───
             ├───┤└╥┘┌─┐
        q_1: ┤ X ├─╫─┤M├
             └───┘ ║ └╥┘
        c: 2/══════╩══╩═
                   0  1
        """
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.x(1)
        qc.measure(0, 0)
        qc.measure(1, 1)

        durations = InstructionDurations(
            [("x", [0], 200), ("x", [1], 400), ("measure", None, 1000)]
        )
        pm = PassManager(ALAPSchedule(durations))
        qc_alap = pm.run(qc)

        alap_expected = QuantumCircuit(2, 2)
        alap_expected.delay(200, 0)
        alap_expected.x(0)
        alap_expected.x(1)
        alap_expected.measure(0, 0)
        alap_expected.measure(1, 1)

        self.assertEqual(qc_alap, alap_expected)

        pm = PassManager(ASAPSchedule(durations))
        qc_asap = pm.run(qc)

        asap_expected = QuantumCircuit(2, 2)
        asap_expected.x(0)
        asap_expected.x(1)
        asap_expected.measure(0, 0)  # immediately start after X gate
        asap_expected.measure(1, 1)

        self.assertEqual(qc_asap, asap_expected)

    def test_parallel_gate_different_length_with_barrier(self):
        """Test circuit having two parallel instruction with different length with barrier.

        (input)
             ┌───┐┌─┐
        q_0: ┤ X ├┤M├───
             ├───┤└╥┘┌─┐
        q_1: ┤ X ├─╫─┤M├
             └───┘ ║ └╥┘
        c: 2/══════╩══╩═
                   0  1

        (expected, ALAP)
             ┌────────────────┐┌───┐ ░ ┌─┐
        q_0: ┤ Delay(200[dt]) ├┤ X ├─░─┤M├───
             └─────┬───┬──────┘└───┘ ░ └╥┘┌─┐
        q_1: ──────┤ X ├─────────────░──╫─┤M├
                   └───┘             ░  ║ └╥┘
        c: 2/═══════════════════════════╩══╩═
                                        0  1

        (expected, ASAP)
             ┌───┐┌────────────────┐ ░ ┌─┐
        q_0: ┤ X ├┤ Delay(200[dt]) ├─░─┤M├───
             ├───┤└────────────────┘ ░ └╥┘┌─┐
        q_1: ┤ X ├───────────────────░──╫─┤M├
             └───┘                   ░  ║ └╥┘
        c: 2/═══════════════════════════╩══╩═
                                        0  1
        """
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.x(1)
        qc.barrier()
        qc.measure(0, 0)
        qc.measure(1, 1)

        durations = InstructionDurations(
            [("x", [0], 200), ("x", [1], 400), ("measure", None, 1000)]
        )
        pm = PassManager(ALAPSchedule(durations))
        qc_alap = pm.run(qc)

        alap_expected = QuantumCircuit(2, 2)
        alap_expected.delay(200, 0)
        alap_expected.x(0)
        alap_expected.x(1)
        alap_expected.barrier()
        alap_expected.measure(0, 0)
        alap_expected.measure(1, 1)

        self.assertEqual(qc_alap, alap_expected)

        pm = PassManager(ASAPSchedule(durations))
        qc_asap = pm.run(qc)

        asap_expected = QuantumCircuit(2, 2)
        asap_expected.x(0)
        asap_expected.delay(200, 0)
        asap_expected.x(1)
        asap_expected.barrier()
        asap_expected.measure(0, 0)
        asap_expected.measure(1, 1)

        self.assertEqual(qc_asap, asap_expected)


if __name__ == "__main__":
    unittest.main()
