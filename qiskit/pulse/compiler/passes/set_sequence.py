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

"""Sequencing pass for Qiskit PulseIR compilation."""

from __future__ import annotations

from qiskit.pulse.compiler.basepasses import TransformationPass
from qiskit.pulse.ir import SequenceIR


class SetSequence(TransformationPass):
    """Sets the sequence of a ``SequenceIR`` object.

    The pass traverses the ``SequenceIR``, recursively sets the sequence, by adding edges to
    the ``sequence`` property. Sequencing is done according to the alignment strategy.

    For parallel alignment types, the pass depends on the results of the analysis pass
    :class:`~qiskit.pulse.compiler.passes.MapMixedFrame`.
    """

    def __init__(self):
        """Create new SetSequence pass"""
        super().__init__(target=None)

    def run(
        self,
        passmanager_ir: SequenceIR,
    ) -> SequenceIR:

        self._set_sequence_recursion(passmanager_ir)
        return passmanager_ir

    def _set_sequence_recursion(self, prog: SequenceIR) -> None:
        """A helper function to recurse through the sequence"""
        prog.alignment.set_sequence(
            prog.sequence, mixed_frames_mapping=self.property_set["mixed_frames_mapping"]
        )
        for elm in prog.elements():
            if isinstance(elm, SequenceIR):
                self._set_sequence_recursion(elm)

    def __hash__(self):
        return hash((self.__class__.__name__,))
