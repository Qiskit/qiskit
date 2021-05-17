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

from qiskit import QuantumCircuit
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes import ASAPSchedule, ALAPSchedule
from qiskit.transpiler.passmanager import PassManager

from qiskit.test import QiskitTestCase


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
        new_qc = new_qc.reverse_ops()  # pylint: disable=no-member
        new_qc.name = new_qc.name

        self.assertEqual(alap_qc, new_qc)


if __name__ == "__main__":
    unittest.main()
