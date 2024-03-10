# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A scheduling pass for Qiskit PulseIR compilation."""

from __future__ import annotations

from qiskit.pulse.compiler.basepasses import TransformationPass
from qiskit.pulse.ir import SequenceIR


class SchedulePass(TransformationPass):
    """Concretely schedule ``SequenceIR`` object.

    The pass traverses the ``SequenceIR``, and recursively sets initial time for every
    node in the sequence (and sub-sequences). The scheduling is done according to the
    alignment strategy, and requires that the ``sequence`` property is already sequenced,
    typically with the pass :class:`~qiskit.pulse.compiler.passes.SetSequence`."""

    def __init__(self):
        """Create new Schedule pass"""
        super().__init__(target=None)

    def run(
        self,
        passmanager_ir: SequenceIR,
    ) -> SequenceIR:

        self._schedule_recursion(passmanager_ir)
        return passmanager_ir

    def _schedule_recursion(self, prog: SequenceIR) -> None:
        """A helper function to recurse through the sequence"""
        # Parent sequences depend on the scheduling of child sequences, so we recurse first
        for elm in prog.elements():
            if isinstance(elm, SequenceIR):
                self._schedule_recursion(elm)

        prog.alignment.schedule(prog.sequence, prog.time_table)

    def __hash__(self):
        return hash((self.__class__.__name__,))
