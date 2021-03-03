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

from qiskit.circuit import Measure
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import Layout
from qiskit.dagcircuit import DAGCircuit


class OptimizeSwapBeforeMeasure(TransformationPass):
    """Remove or move the swaps followed by measurement (and adapt the measurement)."""

    def __init__(self, all_measurement=False, move_swap=False):
        """Remove/move the swaps followed by measurement (and adapt the measurement).

        Transpiler pass to remove swaps in front of measurements by re-targeting
        the classical bit of the measure instruction.

        Args:
            all_measurement (bool): If `True` (default is `False`)`, the SWAP to be removed
                 has to be measured on both wires. Otherwise, it stays.
            move_swap (bool): If `True`, it moves the swap gate behind the measures instead of
                 removing it.
        """
        self.all_measurement = all_measurement
        self.move_swap = move_swap
        super().__init__()

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

        nodes = dag.topological_op_nodes()
        nodes_to_skip = []
        for node in nodes:
            if node in nodes_to_skip:
                nodes_to_skip.remove(node)
                continue
            if node.type == 'op':
                qargs = [_trivial_layout[_layout[qarg]] for qarg in node.qargs]
                swap_successors = list(dag.successors(node))
                if isinstance(node.op, SwapGate) and self.should_remove_swap(swap_successors, dag):
                    if self.move_swap:
                        [q1, q2] = [s.qargs[0] for s in swap_successors]
                        [c1, c2] = [s.cargs[0] for s in swap_successors]
                        new_dag.apply_operation_back(Measure(), [q1], [c2])
                        new_dag.apply_operation_back(Measure(), [q2], [c1])
                        new_dag.apply_operation_back(SwapGate(), qargs, node.cargs)
                        nodes_to_skip += swap_successors  # skip the successors (they are measure)
                    else:
                        _layout.swap(*qargs)
                    continue
                new_dag.apply_operation_back(node.op, qargs, node.cargs)
        return new_dag

    def should_remove_swap(self, swap_successors, dag):
        """Based on the swap successor characteristics, should that swap be removed/moved? """
        final_successor = []
        followed_by_measures = []
        for successor in swap_successors:
            is_final_measure = False
            if successor.type == 'op' and successor.op.name == 'measure':
                followed_by_measures.append(True)
                if self.move_swap:
                    is_final_measure = True
                else:
                    is_final_measure = all(s.type == 'out' for s in dag.successors(successor))
            else:
                followed_by_measures.append(False)
            final_successor.append(successor.type == 'out' or is_final_measure)
        if self.all_measurement:
            return all(followed_by_measures) and all(final_successor)
        if self.move_swap:
            return all(followed_by_measures)
        return all(final_successor)
