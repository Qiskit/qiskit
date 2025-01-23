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

"""Test the legacy Scheduling passes"""

import unittest

from ddt import ddt, data, unpack

from qiskit import QuantumCircuit
from qiskit.circuit import Delay, Parameter
from qiskit.circuit.library.standard_gates import XGate, YGate, CXGate
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes import ASAPSchedule, ALAPSchedule, DynamicalDecoupling
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.target import Target, InstructionProperties
from test import QiskitTestCase  # pylint: disable=wrong-import-order


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

        with self.assertWarns(DeprecationWarning):
            pm = PassManager(ALAPSchedule(durations))
        alap_qc = pm.run(qc)

        with self.assertWarns(DeprecationWarning):
            pm = PassManager(ASAPSchedule(durations))
        new_qc = pm.run(qc.reverse_ops())
        new_qc = new_qc.reverse_ops()
        new_qc.name = new_qc.name

        self.assertEqual(alap_qc, new_qc)

    @data(ALAPSchedule, ASAPSchedule)
    def test_measure_after_measure(self, schedule_pass):
        """Test if ALAP/ASAP schedules circuits with measure after measure with a common clbit.
        See: https://github.com/Qiskit/qiskit-terra/issues/7654

        (input)
             ┌───┐┌─┐
        q_0: ┤ X ├┤M├───
             └───┘└╥┘┌─┐
        q_1: ──────╫─┤M├
                   ║ └╥┘
        c: 1/══════╩══╩═
                   0  0

        (scheduled)
                    ┌───┐       ┌─┐┌─────────────────┐
        q_0: ───────┤ X ├───────┤M├┤ Delay(1000[dt]) ├
             ┌──────┴───┴──────┐└╥┘└───────┬─┬───────┘
        q_1: ┤ Delay(1200[dt]) ├─╫─────────┤M├────────
             └─────────────────┘ ║         └╥┘
        c: 1/════════════════════╩══════════╩═════════
                                 0          0
        """
        qc = QuantumCircuit(2, 1)
        qc.x(0)
        qc.measure(0, 0)
        qc.measure(1, 0)

        durations = InstructionDurations([("x", None, 200), ("measure", None, 1000)])
        with self.assertWarns(DeprecationWarning):
            pm = PassManager(schedule_pass(durations))
        scheduled = pm.run(qc)

        expected = QuantumCircuit(2, 1)
        expected.x(0)
        expected.measure(0, 0)
        expected.delay(1200, 1)
        expected.measure(1, 0)
        expected.delay(1000, 0)

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
                                ┌─┐┌────────────────┐
        q_0: ───────────────────┤M├┤ Delay(700[dt]) ├
             ┌─────────────────┐└╥┘└──────┬─┬───────┘
        q_1: ┤ Delay(1000[dt]) ├─╫────────┤M├────────
             └─────────────────┘ ║        └╥┘
        c: 1/════════════════════╩═════════╩═════════
                                 0         0
        """
        qc = QuantumCircuit(2, 1)
        qc.measure(0, 0)
        qc.measure(1, 0)

        durations = InstructionDurations([("measure", [0], 1000), ("measure", [1], 700)])
        with self.assertWarns(DeprecationWarning):
            pm = PassManager(schedule_pass(durations))
        scheduled = pm.run(qc)

        expected = QuantumCircuit(2, 1)
        expected.measure(0, 0)
        expected.delay(1000, 1)
        expected.measure(1, 0)
        expected.delay(700, 0)

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
             ┌───┐┌─┐┌────────────────┐
        q_0: ┤ X ├┤M├┤ Delay(200[dt]) ├
             ├───┤└╥┘└──────┬─┬───────┘
        q_1: ┤ X ├─╫────────┤M├────────
             └───┘ ║        └╥┘
        c: 2/══════╩═════════╩═════════
                   0         1

        """
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.x(1)
        qc.measure(0, 0)
        qc.measure(1, 1)

        durations = InstructionDurations(
            [("x", [0], 200), ("x", [1], 400), ("measure", None, 1000)]
        )
        with self.assertWarns(DeprecationWarning):
            pm = PassManager(ALAPSchedule(durations))
        qc_alap = pm.run(qc)

        alap_expected = QuantumCircuit(2, 2)
        alap_expected.delay(200, 0)
        alap_expected.x(0)
        alap_expected.x(1)
        alap_expected.measure(0, 0)
        alap_expected.measure(1, 1)

        self.assertEqual(qc_alap, alap_expected)

        with self.assertWarns(DeprecationWarning):
            pm = PassManager(ASAPSchedule(durations))
        qc_asap = pm.run(qc)

        asap_expected = QuantumCircuit(2, 2)
        asap_expected.x(0)
        asap_expected.x(1)
        asap_expected.measure(0, 0)  # immediately start after X gate
        asap_expected.measure(1, 1)
        asap_expected.delay(200, 0)

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
        with self.assertWarns(DeprecationWarning):
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

        with self.assertWarns(DeprecationWarning):
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

    @data(ALAPSchedule, ASAPSchedule)
    def test_respect_target_instruction_constraints(self, schedule_pass):
        """Test if ALAP/ASAP does not pad delays for qubits that do not support delay instructions.
        See: https://github.com/Qiskit/qiskit-terra/issues/9993
        """
        target = Target(dt=1)
        target.add_instruction(XGate(), {(1,): InstructionProperties(duration=200)})
        # delays are not supported

        qc = QuantumCircuit(2)
        qc.x(1)

        with self.assertWarns(DeprecationWarning):
            pm = PassManager(schedule_pass(target=target))
        scheduled = pm.run(qc)

        expected = QuantumCircuit(2)
        expected.x(1)
        # no delay on qubit 0

        self.assertEqual(expected, scheduled)

    def test_dd_respect_target_instruction_constraints(self):
        """Test if DD pass does not pad delays for qubits that do not support delay instructions
        and does not insert DD gates for qubits that do not support necessary gates.
        See: https://github.com/Qiskit/qiskit-terra/issues/9993
        """
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)

        target = Target(dt=1)
        # Y is partially supported (not supported on qubit 2)
        target.add_instruction(
            XGate(), {(q,): InstructionProperties(duration=100) for q in range(2)}
        )
        target.add_instruction(
            CXGate(),
            {
                (0, 1): InstructionProperties(duration=1000),
                (1, 2): InstructionProperties(duration=1000),
            },
        )
        # delays are not supported

        # No DD instructions nor delays are padded due to no delay support in the target
        with self.assertWarns(DeprecationWarning):
            pm_scheduler = PassManager(
                [
                    ALAPSchedule(target=target),
                    DynamicalDecoupling(
                        durations=None, dd_sequence=[XGate(), XGate()], target=target
                    ),
                ]
            )
        scheduled = pm_scheduler.run(qc)
        self.assertEqual(qc, scheduled)

        # Fails since Y is not supported in the target
        with self.assertWarns(DeprecationWarning):
            with self.assertRaises(TranspilerError):
                PassManager(
                    [
                        ALAPSchedule(target=target),
                        DynamicalDecoupling(
                            durations=None,
                            dd_sequence=[XGate(), YGate(), XGate(), YGate()],
                            target=target,
                        ),
                    ]
                )

        # Add delay support to the target
        target.add_instruction(Delay(Parameter("t")), {(q,): None for q in range(3)})
        # No error but no DD on qubit 2 (just delay is padded) since X is not supported on it
        scheduled = pm_scheduler.run(qc)

        expected = QuantumCircuit(3)
        expected.delay(1000, [2])
        expected.cx(0, 1)
        expected.cx(1, 2)
        expected.delay(200, [0])
        expected.x([0])
        expected.delay(400, [0])
        expected.x([0])
        expected.delay(200, [0])
        self.assertEqual(expected, scheduled)


if __name__ == "__main__":
    unittest.main()
