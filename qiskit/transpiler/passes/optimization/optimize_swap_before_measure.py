# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Remove the swaps followed by measurement (and adapt the measurement)."""

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import Layout
from qiskit.dagcircuit import DAGCircuit


class OptimizeSwapBeforeMeasure(TransformationPass):
    """Remove the swaps followed by measurement (and adapt the measurement).

    Transpiler pass to remove swaps in front of measurements by re-targeting
    the classical bit of the measure instruction.
    """

    def run(self, dag):
        """Run the OptimizeSwapBeforeMeasure pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """

        new_dag = DAGCircuit()
        new_dag.metadata = dag.metadata
        new_dag._global_phase = dag._global_phase
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)

        _layout = Layout.generate_trivial_layout(*dag.qregs.values())
        _trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())

        for node in dag.topological_op_nodes():
            if node.type == 'op':
                qargs = [_trivial_layout[_layout[qarg]] for qarg in node.qargs]
                if isinstance(node.op, SwapGate):
                    swap = node
                    final_successor = []
                    for successor in dag.successors(swap):
                        if successor.type == 'op' and successor.op.name == 'measure':
                            is_final_measure = all([s.type == 'out'
                                                    for s in dag.successors(successor)])
                        else:
                            is_final_measure = False
                        final_successor.append(successor.type == 'out' or is_final_measure)
                    if all(final_successor):
                        _layout.swap(*qargs)
                        continue
                new_dag.apply_operation_back(node.op, qargs, node.cargs)
        return new_dag
