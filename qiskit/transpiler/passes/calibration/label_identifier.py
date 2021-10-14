# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Instruction replacement for labeled gate."""

from qiskit.dagcircuit import DAGCircuit
from qiskit.pulse import InstructionScheduleMap
from qiskit.transpiler.basepasses import TransformationPass


class LabelIdentifier(TransformationPass):
    """Replace instruction with Gate with unique name if label is defined.

    Note:
        Once this pass is called, the labeled gate will be no longer recognized by the
        circuit equivalence library. Thus further optimization on the
        instruction will be just ignored.
    """

    def __init__(
        self,
        inst_map: InstructionScheduleMap,
    ):
        """Create new pass.

        Args:
            inst_map: Instruction schedule map that user may override.
        """
        super().__init__()
        self.inst_map = inst_map

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run label identifier pass on dag.

        Args:
            dag: DAG to identify instruction label.

        Returns:
            DAG circuit with replaced instruction.
        """
        for node in dag.topological_op_nodes():
            label = getattr(node.op, "label", None)
            if label:
                qubits = list(dag.qubits.index(q) for q in node.qargs)
                if self.inst_map.has(instruction=node.op, qubits=qubits):
                    node.name = f"{node.name}_{label}"

        return dag
