# -*- coding: utf-8 -*-

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

"""Remove final measurements and barriers at the end of a circuit."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit


class RemoveFinalMeasurements(TransformationPass):
    """Remove final measurements and barriers at the end of a circuit.

    This pass removes final barriers and final measurements, as well as the
    ClassicalRegisters they are connected to if the ClassicalRegister
    is unused. Measurements and barriers are considered final if they are
    followed by no other operations (aside from other measurements or barriers.)
    """

    def run(self, dag):
        """Run the RemoveFinalMeasurements pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        final_op_types = ['measure', 'barrier']
        final_ops = []
        cregs_to_remove = dict()
        clbits_with_final_measures = set()

        for candidate_node in dag.named_nodes(*final_op_types):
            is_final_op = True

            for _, child_successors in dag.bfs_successors(candidate_node):
                if any(suc.type == 'op' and suc.name not in final_op_types
                       for suc in child_successors):
                    is_final_op = False
                    break

            if is_final_op:
                final_ops.append(candidate_node)

        if not final_ops:
            return dag

        new_dag = DAGCircuit()

        for node in final_ops:
            for carg in node.cargs:
                # Add the clbit that was attached to the measure we are removing
                clbits_with_final_measures.add(carg)
            dag.remove_op_node(node)

        # If the clbit is idle, add its register to list of registers we may remove
        for clbit in clbits_with_final_measures:
            if clbit in dag.idle_wires():
                if clbit.register in cregs_to_remove:
                    cregs_to_remove[clbit.register] += 1
                else:
                    cregs_to_remove[clbit.register] = 1

        # Remove creg if all of its clbits were added above
        for key, val in zip(list(dag.cregs.keys()), list(dag.cregs.values())):
            if val in cregs_to_remove and cregs_to_remove[val] == val.size:
                del dag.cregs[key]

        # Fill new DAGCircuit
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        for node in dag.topological_op_nodes():
            # copy the condition over too
            if node.condition:
                new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs,
                                             condition=node.condition)
            else:
                new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)

        return new_dag
