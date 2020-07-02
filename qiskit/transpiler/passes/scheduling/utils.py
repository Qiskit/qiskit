# -*- coding: utf-8 -*-

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

"""Utilities for scheduling passes."""

from qiskit.circuit import Barrier, Delay
from qiskit.dagcircuit import DAGNode
from qiskit.transpiler.instruction_durations import InstructionDurations


class DurationMapper:
    """Providing duration for each DAGNode."""

    def __init__(self, instruction_durations: InstructionDurations):
        self.instruction_durations = instruction_durations

    def get(self, node: DAGNode) -> int:
        """Get the duration [dt] of the node.

        Args:
            node: A node whose duration to be returned.

        Returns:
            Duration of the node in dt.
        """

        if isinstance(node.op, Barrier):
            return 0
        elif isinstance(node.op, Delay):
            return node.op.duration
        # consult instruction_durations
        qubits = [q.index for q in node.qargs]
        return self.instruction_durations.get(node.op.name, qubits)
