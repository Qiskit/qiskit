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
from qiskit.transpiler import InstructionDurations
from qiskit.transpiler.passes import ALAPScheduleAnalysis, ConstrainedReschedule
from qiskit.transpiler.passmanager import PassManager

from test import QiskitTestCase


class TestConstrainedReschedule(QiskitTestCase):
    """Tests for the :class:`.ConstrainedReschedule` pass."""

    def test_alignment_only_construction(self):
        """Regression test for #16245: constructing :class:`.ConstrainedReschedule`
        without a ``target=`` (the alignment-only path, the way the docstring
        still advertises) must not raise ``AttributeError`` when the pass is
        run.  Previously ``self.target`` was only assigned inside the
        ``if target is not None`` branch of ``__init__``.
        """
        durations = InstructionDurations([("cx", [0, 1], 100, "dt")], dt=2.22e-10)

        qc = QuantumCircuit(2)
        qc.cx(0, 1)

        pm = PassManager(
            [
                ALAPScheduleAnalysis(durations),
                ConstrainedReschedule(acquire_alignment=16, pulse_alignment=1),
            ]
        )
        pm.run(qc)  # would raise AttributeError on the buggy code path


if __name__ == "__main__":
    unittest.main()
