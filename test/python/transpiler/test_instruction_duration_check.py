# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test the InstructionDurationCheck pass"""

from qiskit import QuantumCircuit
from qiskit.transpiler.passes import InstructionDurationCheck
from test import QiskitTestCase
from ddt import ddt, data, unpack

@ddt
class TestInstructionDurationCheck(QiskitTestCase):
    """Test InstructionDurationCheck class"""

    @data(
        (16, 1, 100, True),
        (16, 1, 112, False),
    )
    @unpack
    def test_instruction_duration_check_alignment(self, acquire_alignment, pulse_alignment, delay, expected):
        """Test alignment check in InstructionDurationCheck"""
        qc = QuantumCircuit(1, 1)
        qc.delay(delay, 0)
        qc.measure(0, 0)

        check = InstructionDurationCheck(acquire_alignment, pulse_alignment)
        check(qc)
        self.assertEqual(check.property_set["reschedule_required"], expected)