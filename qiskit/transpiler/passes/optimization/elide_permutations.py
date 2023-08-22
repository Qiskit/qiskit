# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Remove any swap gates in the circuit by pushing it through into a qubit permutation."""

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.library.generalized_gates import PermutationGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout


class ElidePermutations(TransformationPass):
    r"""Remove permutation operations from a pre-layout circuit

    This pass is intended to be run before a layout (mapping virtual qubits
    to physical qubits) is set during the transpilation pipeline. This
    pass iterates over the :class:`~.DAGCircuit` and when a :class:`~.SwapGate`
    is encountered it permutes the virtual qubits in the circuit and removes
    the swap gate. This will effectively remove any :class:`~SwapGate`\s in
    the circuit prior to running layout.
    """

    def run(self, dag):
        """Run the ElideSwaps pass on ``dag``.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        op_count = dag.count_ops()
        if op_count.get("swap", 0) == 0 and op_count.get("permutation", 0) == 0:
            return dag

        new_dag = dag.copy_empty_like()
        qubit_mapping = list(range(len(dag.qubits)))

        def _apply_mapping(qargs):
            return tuple(dag.qubits[qubit_mapping[dag.find_bit(qubit).index]] for qubit in qargs)

        for node in dag.topological_op_nodes():
            if not isinstance(node.op, (SwapGate, PermutationGate)):
                new_dag.apply_operation_back(node.op, _apply_mapping(node.qargs), node.cargs)
            elif getattr(node.op, "condition", None) is not None:
                new_dag.apply_operation_back(node.op, _apply_mapping(node.qargs), node.cargs)
            elif isinstance(node.op, SwapGate):
                index_0 = dag.find_bit(node.qargs[0]).index
                index_1 = dag.find_bit(node.qargs[1]).index
                qubit_mapping[index_1], qubit_mapping[index_0] = (
                    qubit_mapping[index_0],
                    qubit_mapping[index_1],
                )
            elif isinstance(node.op, PermutationGate):
                starting_indices = [qubit_mapping[dag.find_bit(qarg).index] for qarg in node.qargs]
                pattern = node.op.params[0]
                pattern_indices = [qubit_mapping[idx] for idx in pattern]
                for i, j in zip(starting_indices, pattern_indices):
                    qubit_mapping[i] = j
        input_qubit_mapping = {qubit: index for index, qubit in enumerate(dag.qubits)}
        self.property_set["original_layout"] = Layout(input_qubit_mapping)
        if self.property_set["original_qubit_indices"] is None:
            self.property_set["original_qubit_indices"] = input_qubit_mapping
        self.property_set["elision_final_layout"] = Layout(
            {dag.qubits[out]: idx for idx, out in enumerate(qubit_mapping)}
        )
        return new_dag
