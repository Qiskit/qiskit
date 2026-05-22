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

"""Test the ConstrainedReschedule pass."""

import unittest

from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.transpiler.passes import ALAPScheduleAnalysis, ConstrainedReschedule
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.target import InstructionProperties, Target

from test import QiskitTestCase


class TestConstrainedReschedule(QiskitTestCase):
    """Tests for the :class:`.ConstrainedReschedule` pass."""

    def _make_target(self):
        """Build a tiny 2-qubit target with one CX gate and alignment constraints."""
        dt = 2.22e-10
        target = Target(dt=dt, num_qubits=2, acquire_alignment=16, pulse_alignment=1)
        target.add_instruction(CXGate(), {(0, 1): InstructionProperties(duration=100 * dt)})
        return target

    def test_reschedule_with_barrier(self):
        """Regression test for #16135: a directive such as ``barrier`` must not
        raise ``"Unknown operation type"`` in :class:`.ConstrainedReschedule`.

        The original Python implementation treated compiler directives the same
        as :class:`.Delay`: no alignment is enforced (they have no hardware
        duration), but they are still visited so that successor overlap is
        computed.  The 2.4 Rust port inverted that branch.
        """
        target = self._make_target()

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.barrier(0)

        pm = PassManager(
            [ALAPScheduleAnalysis(target=target), ConstrainedReschedule(target=target)]
        )
        pm.run(qc)  # would raise TranspilerError on the buggy code path


if __name__ == "__main__":
    unittest.main()
