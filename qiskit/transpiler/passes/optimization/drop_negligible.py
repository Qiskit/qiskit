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

"""Transpiler pass to drop gates with negligible effects."""

from __future__ import annotations

import numpy as np
from qiskit.circuit.library import (
    CPhaseGate,
    PhaseGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    XXMinusYYGate,
    XXPlusYYGate,
)
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

# List of Gate classes with the property that if the gate's parameters are all
# (close to) zero then the gate has (close to) no effect.
DROP_NEGLIGIBLE_GATE_CLASSES = (
    CPhaseGate,
    PhaseGate,
    RXGate,
    RYGate,
    RZGate,
    RXXGate,
    RYYGate,
    RZZGate,
    XXPlusYYGate,
    XXMinusYYGate,
)


class DropNegligible(TransformationPass):
    """Drop gates with negligible effects."""

    def __init__(self, atol: float = 1e-8) -> None:
        """Initialize the transpiler pass.

        Args:
            atol: Absolute numerical tolerance for determining whether a gate's effect
                is negligible.
        """
        self.atol = atol
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.op_nodes():
            if not isinstance(node.op, DROP_NEGLIGIBLE_GATE_CLASSES):
                continue
            if not all(isinstance(param, (int, float, complex)) for param in node.op.params):
                continue
            if np.allclose(node.op.params, 0, atol=self.atol):
                dag.remove_op_node(node)
        return dag
