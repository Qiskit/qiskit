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

        clbits_with_final_measures = set()
        for node in final_ops:
            for carg in node.cargs:
                # Add the clbit that was attached to the measure we are removing
                clbits_with_final_measures.add(carg)
            dag.remove_op_node(node)

        # A creg is removable if all:
        #   - it contains a final measure bit
        #   - all of its bits are idle, including the final measure bit
        #
        # A bit is removable if all:
        #   - it appears in a creg that is removable
        #   - it does not appear in a creg that is non-removable
        idle_wires = set(dag.idle_wires())
        cregs_to_remove = set()
        clbits_to_remove = set()
        for creg in dag.cregs.values():
            clbits = set(creg)
            if not clbits.isdisjoint(clbits_with_final_measures) and clbits.issubset(idle_wires):
                cregs_to_remove.add(creg)
                clbits_to_remove.update(clbits)

        registerless_clbits_to_remove = clbits_with_final_measures.intersection(idle_wires) - clbits_to_remove

        # Remove cregs from DAG
        dag.remove_cregs(*cregs_to_remove)
        dag.remove_idle_clbits(*registerless_clbits_to_remove)
        return dag
