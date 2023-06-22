# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Remove all babeled ops from a circuit"""

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow


class RemoveLabeledOps(TransformationPass):
    """Remove all operations with a specific label..

    This transformation pass is used to remove

    Example:

        .. plot::
           :include-source:

            from qiskit import QuantumCircuit
            from qiskit.transpiler.passes import RemoveBarriers

            circuit = QuantumCircuit(1)
            circuit.x(0, label='foo')
            circuit.barrier()
            circuit.h(0)

            circuit = RemoveLabeledOps('foo')(circuit)
            circuit.draw('mpl')

    """

    def __init__(self, label: str):
        super().__init__()
        self.label = label

    @control_flow.trivial_recurse
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the RemoveBarriers pass on `dag`."""
        for node in dag.op_nodes():
            if node.op.label == self.label:
                dag.remove_op_node(node)
        return dag
