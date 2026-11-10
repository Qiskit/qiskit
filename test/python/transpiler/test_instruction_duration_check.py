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

"""Test delay alignment checks before rescheduling."""

import unittest

from ddt import ddt

from qiskit import QuantumCircuit
from qiskit.transpiler import Target
from qiskit.transpiler.passes import InstructionDurationCheck
from test import QiskitTestCase, combine


@ddt
class TestInstructionDurationCheck(QiskitTestCase):
    """Both acquisition and pulse alignment must hold for delay durations."""

    @combine(
        alignments=[(16, 1), (1, 16), (16, 8), (8, 16), (16, 16), (1, 1)],
        duration=[0, 8, 16, 100],
        use_target=[False, True],
    )
    def test_delay_alignment(self, alignments, duration, use_target):
        """Check asymmetric constraints through explicit arguments and a target."""
        acquire, pulse = alignments
        circuit = QuantumCircuit(1, 1)
        circuit.delay(duration, 0)
        circuit.measure(0, 0)
        if use_target:
            check = InstructionDurationCheck(
                target=Target(num_qubits=1, acquire_alignment=acquire, pulse_alignment=pulse)
            )
        else:
            check = InstructionDurationCheck(acquire_alignment=acquire, pulse_alignment=pulse)
        check(circuit)
        # 16 is aligned to every constraint here; 8 is only aligned to 1 and 8.
        aligned = duration in (0, 16) or max(alignments) == 1
        self.assertEqual(check.property_set["reschedule_required"], not aligned)


if __name__ == "__main__":
    unittest.main()
