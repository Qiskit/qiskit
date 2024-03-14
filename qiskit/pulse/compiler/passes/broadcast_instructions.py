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

"""Broadcasting pass for Qiskit PulseIR compilation."""

from __future__ import annotations

from qiskit.pulse.compiler.basepasses import TransformationPass
from qiskit.pulse.ir import SequenceIR
from qiskit.pulse import Frame, PulseTarget
from qiskit.pulse.exceptions import PulseCompilerError


class BroadcastInstructions(TransformationPass):
    r"""Broadcast :class:`~qiskit.pulse.Frame` or :class:`~qiskit.pulse.PulseTarget` only instructions.

    Some :class:`~qiskit.pulse.Instruction`\s could be defined on :class:`~qiskit.pulse.Frame`
    or :class:`~qiskit.pulse.PulseTarget` instead of the typical :class:`~qiskit.pulse.MixedFrame`.
    While the :class:`~qiskit.pulse.compiler.passes.SetSequence` pass will sequence these instructions
    correctly, in some cases it might be needed to convert them into instructions defined on
    :class:`~qiskit.pulse.MixedFrame` - broadcasting the instruction to all relevant
    :class:`~qiskit.pulse.MixedFrame`\s. It should be noted that in other contexts the term
    "broadcasting" is also used to describe the sequencing of :class:`~qiskit.pulse.Frame`
    or :class:`~qiskit.pulse.PulseTarget` only instructions.

    The pass recursively traverses through the IR, and replaces every :class:`~qiskit.pulse.Frame`
    or :class:`~qiskit.pulse.PulseTarget` only instruction with the set of instructions acting on the
    relevant :class:`~qiskit.pulse.MixedFrame`\s. The new instructions will have the same timing,
    as well as the same sequencing as the original instructions.

    .. notes::
        The pass depends on the results of the analysis pass
        :class:`~qiskit.pulse.compiler.passes.MapMixedFrame`.

    .. notes::
        Running this pass before
        :class:`~qiskit.pulse.compiler.passes.SetSequence` will not raise an error,
        but might change the resulting sequence, as :class:`~qiskit.pulse.Frame` or
        :class:`~qiskit.pulse.PulseTarget` only instructions are sequenced differently
        than :class:`~qiskit.pulse.MixedFrame`\s instructions.
    """

    def __init__(self):
        """Create new BroadcastInstruction pass"""
        super().__init__(target=None)

    def run(
        self,
        passmanager_ir: SequenceIR,
    ) -> SequenceIR:
        """Run broadcasting pass.

        Arguments:
            passmanager_ir: The IR object to undergo broadcasting.

        Raises:
            PulseCompilerError: if ``property_set`` does not include a mixed_frames_mapping dictionary.
        """
        if self.property_set["mixed_frames_mapping"] is None:
            raise PulseCompilerError(
                "broadcasting requires mixed frames mapping."
                " Run MapMixedFrame before broadcasting."
            )

        self._broadcast_recursion(passmanager_ir)
        return passmanager_ir

    def _broadcast_recursion(self, prog: SequenceIR) -> None:
        """Recursively broadcast the IR.

        Arguments:
            prog: The IR object to undergo broadcasting.
        """
        mixed_frames_mapping = self.property_set["mixed_frames_mapping"]

        for ind in prog.sequence.node_indices():
            if ind in (0, 1):
                continue
            elem = prog.sequence.get_node_data(ind)
            if isinstance(elem, SequenceIR):
                self._broadcast_recursion(elem)
            elif isinstance(inst_target := elem.inst_target, (Frame, PulseTarget)):
                in_edges = [x[0] for x in prog.sequence.in_edges(ind)]
                out_edges = [x[1] for x in prog.sequence.out_edges(ind)]
                initial_time = prog.time_table[ind]
                if mixed_frames := mixed_frames_mapping[inst_target]:
                    for mixed_frame in mixed_frames:
                        # The relevant instructions are delay and set\shift phase\frequency, and they all
                        # have the same signature.
                        new_ind = prog.sequence.add_node(
                            elem.__class__(
                                elem.operands[0], mixed_frame=mixed_frame, name=elem.name
                            )
                        )
                        prog.sequence.add_edges_from_no_data(
                            [(in_edge, new_ind) for in_edge in in_edges]
                        )
                        prog.sequence.add_edges_from_no_data(
                            [(new_ind, out_edge) for out_edge in out_edges]
                        )
                        prog.time_table[new_ind] = initial_time
                    prog.sequence.remove_node(ind)
                    del prog.time_table[ind]

    def __hash__(self):
        return hash((self.__class__.__name__,))

    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__
