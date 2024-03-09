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

"""A base pass for Qiskit PulseIR compilation."""

from __future__ import annotations

from qiskit.pulse.compiler.basepasses import TransformationPass
from qiskit.pulse.ir import SequenceIR


class SetSequence(TransformationPass):
    """Map the dependencies of all ``MixedFrame``s on ``PulseTaraget`` and ``Frame``.

    The pass recursively scans the ``SequenceIR``, identifies all ``MixedFrame``s and
    tracks the dependencies of them on ``PulseTarget`` and ``Frame``. The analysis result
    is added as a dictionary to the property set under key "mixed_frames_mapping". The
    added dictionary is keyed on every ``PulseTarget`` and ``Frame`` in ``SequenceIR``
    with the value being a set of all ``MixedFrame``s associated with the key.
    """

    def __init__(self):
        """Create new MapMixedFrames pass"""
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
