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

"""MixedFrames mapping analysis pass"""

from __future__ import annotations
from collections import defaultdict

from qiskit.pulse.compiler.basepasses import AnalysisPass
from qiskit.pulse.ir import SequenceIR
from qiskit.pulse.model import MixedFrame


class MapMixedFrame(AnalysisPass):
    r"""Map the dependencies of all class:`.MixedFrame`\s
    on class:`~qiskit.pulse.PulseTaraget` and :class:`~qiskit.pulse.Frame`.

    The pass recursively scans the :class:`.SequenceIR`, identifies all :class:`.MixedFrame`\s and
    tracks the dependencies of them on class:`~qiskit.pulse.PulseTaraget` and
    :class:`~qiskit.pulse.Frame`. The analysis result
    is added as a dictionary to the property set under key "mixed_frames_mapping". The
    added dictionary is keyed on every class:`~qiskit.pulse.PulseTaraget` and
    :class:`~qiskit.pulse.Frame` in :class:`.SequenceIR`
    with the value being a set of all class:`.MixedFrame`\s associated with the key.

    .. notes::
        The pass will override results of previous ``MapMixedFrame`` runs.
    """

    def __init__(self):
        """Create new ``MapMixedFrames`` pass"""
        super().__init__(target=None)

    def run(
        self,
        passmanager_ir: SequenceIR,
    ) -> None:
        """Run ``MapMixedFrame`` pass"""
        mixed_frames_mapping = defaultdict(set)

        for inst_target in passmanager_ir.inst_targets:
            if isinstance(inst_target, MixedFrame):
                mixed_frames_mapping[inst_target.frame].add(inst_target)
                mixed_frames_mapping[inst_target.pulse_target].add(inst_target)
        self.property_set["mixed_frames_mapping"] = mixed_frames_mapping

    def __hash__(self):
        return hash((self.__class__.__name__,))

    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__
