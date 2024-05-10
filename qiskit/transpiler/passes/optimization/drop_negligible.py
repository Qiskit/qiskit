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

import math
from collections.abc import Iterable

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
    """Drop gates with negligible effects.

    Removes certain gates whose parameters are all close to zero up to the specified
    tolerance. By default, the gates subject to removal are those present in a
    hard-coded list, specified below. Additional gate types to consider can be passed
    as an argument to the constructor of this class.

    By default, the following gate classes are considered for removal:

    - :class:`CPhaseGate`
    - :class:`PhaseGate`
    - :class:`RXGate`
    - :class:`RYGate`
    - :class:`RZGate`
    - :class:`RXXGate`
    - :class:`RYYGate`
    - :class:`RZZGate`
    - :class:`XXPlusYYGate`
    - :class:`XXMinusYYGate`
    """

    def __init__(
        self, *, atol: float = 1e-8, additional_gate_types: Iterable[type] | None = None
    ) -> None:
        """Initialize the transpiler pass.

        Args:
            atol: Absolute numerical tolerance for determining whether a gate's effect
                is negligible.
            additional_gate_types: List of :class:`Gate` subclasses that should be
                considered for dropping in addition to the built-in gates.
        """
        self.atol = atol
        self.gate_types = DROP_NEGLIGIBLE_GATE_CLASSES
        if additional_gate_types is not None:
            self.gate_types += tuple(additional_gate_types)
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.op_nodes():
            if not isinstance(node.op, self.gate_types):
                continue
            if not all(isinstance(param, (int, float, complex)) for param in node.op.params):
                continue
            if all(
                math.isclose(param, 0, rel_tol=0, abs_tol=self.atol) for param in node.op.params
            ):
                dag.remove_op_node(node)
        return dag
