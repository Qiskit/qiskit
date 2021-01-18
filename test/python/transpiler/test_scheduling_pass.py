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
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes import ASAPSchedule, ALAPSchedule

from qiskit.test import QiskitTestCase


class TestSchedulingPass(QiskitTestCase):
    """Tests the Scheduling passes"""

    def test_alap_agree_with_reverse_asap_reverse(self):
        """Test if ALAP schedule agrees with doubly-reversed ASAP schedule.
        """
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        qc.measure_all()

        dag = circuit_to_dag(qc)
        durations = InstructionDurations([('h', 0, 200), ('cx', [0, 1], 700),
                                          ('measure', None, 1000)])

        alap_dag = ALAPSchedule(durations).run(dag, time_unit="dt")

        new_dag = dag.reverse_ops()
        new_dag = ASAPSchedule(durations).run(new_dag, time_unit="dt")
        new_dag = new_dag.reverse_ops()
        new_dag.name = dag.name

        self.assertEqual(alap_dag, new_dag)


if __name__ == '__main__':
    unittest.main()
