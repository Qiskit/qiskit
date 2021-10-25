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

from collections import defaultdict
from qiskit.pulse.builder import barrier
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGOpNode


class RemoveFinalMeasurements(TransformationPass):
    """Remove final measurements and barriers at the end of a circuit.

    This pass removes final barriers and final measurements, as well as the
    ClassicalRegisters they are connected to if the ClassicalRegister
    is unused. Measurements and barriers are considered final if they are
    followed by no other operations (aside from other measurements or barriers.)
    """

    def __init__(self, recurse=True):
        self._recurse = recurse

    def _calc_final_ops(self, dag):
        final_op_types = {"measure", "barrier"}
        final_ops = []

        final_qubit_inputs = (next(dag.predecessors(dag.output_map[qubit])) for qubit in dag.qubits)
        to_visit = list(final_qubit_inputs)
        barrier_encounters_remaining = dict()

        while to_visit:
            node = to_visit.pop()
            if not isinstance(node, DAGOpNode):
                continue
            if node.op.name == "barrier":
                # Barrier is final if all children are final, so we track
                # how many times we still need to encounter each barrier
                # via a child node.
                if node not in barrier_encounters_remaining:
                    barrier_encounters_remaining[node] = sum(1 for _ in dag.quantum_successors(node))
                if barrier_encounters_remaining[node] - 1 > 0:
                    # We've encountered the barrier, but not (yet) via all children.
                    # Record the encounter, and bail!
                    barrier_encounters_remaining[node] -= 1
                    continue
            if node.name in final_op_types:
                # Current node is either a measure, or a barrier with all final op children.
                final_ops.append(node)
                to_visit.extend(dag.quantum_predecessors(node))

        return final_ops

    def run(self, dag):
        """Run the RemoveFinalMeasurements pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        final_ops = self._calc_final_ops(dag)
        if not final_ops:
            return dag

        cregs_to_remove = dict()
        clbits_with_final_measures = set()
        clbit_registers = {clbit: creg for creg in dag.cregs.values() for clbit in creg}

        for node in final_ops:
            for carg in node.cargs:
                # Add the clbit that was attached to the measure we are removing
                clbits_with_final_measures.add(carg)
            dag.remove_op_node(node)

        # If the clbit is idle, add its register to list of registers we may remove
        for clbit in clbits_with_final_measures:
            if clbit in dag.idle_wires():
                creg = clbit_registers[clbit]
                if creg in cregs_to_remove:
                    cregs_to_remove[creg] += 1
                else:
                    cregs_to_remove[creg] = 1

        # Remove creg if all of its clbits were added above
        for key, val in zip(list(dag.cregs.keys()), list(dag.cregs.values())):
            if val in cregs_to_remove and cregs_to_remove[val] == val.size:
                del dag.cregs[key]

        return dag
