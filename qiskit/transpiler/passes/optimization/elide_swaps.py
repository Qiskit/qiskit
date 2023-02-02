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
from qiskit.transpiler.layout import Layout


class ElideSwaps(TransformationPass):
    r"""Remove :class:`~SwapGate`\s from a pre-layout circuit

    This pass is intended to be run before a layout (mapping logical qubits
    to physical qubits) is set during the transpilation pipeline. This
    pass iterates over the :class:`~.DAGCircuit` and when a :class:`~.SwapGate`
    is encountered it permutes the virtual qubits in the circuit and removes
    the swap gate. This will effectively remove any :class:`~SwapGate`\s in
    the circuit prior to running layout.
    """

    def run(self, dag):
        """Run the OptimizeSwapBeforeMeasure pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        if dag.count_ops().get("swap", 0) == 0:
            return dag
        new_dag = dag.copy_empty_like()
        qubit_mapping = list(range(len(dag.qubits)))
        input_qubit_mapping = {qubit: index for index, qubit in enumerate(dag.qubits)}

        def _apply_mapping(qargs):
            return tuple(dag.qubits[qubit_mapping[input_qubit_mapping[qubit]]] for qubit in qargs)

        for node in dag.topological_op_nodes():
            if not isinstance(node.op, SwapGate):
                new_dag.apply_operation_back(node.op, _apply_mapping(node.qargs), node.cargs)
            elif getattr(node.op, "condition", None) is not None:
                new_dag.apply_operation_back(node.op, _apply_mapping(node.qargs), node.cargs)
            else:
                index_0 = input_qubit_mapping[node.qargs[0]]
                index_1 = input_qubit_mapping[node.qargs[1]]
                qubit_mapping[index_1], qubit_mapping[index_0] = (
                    qubit_mapping[index_0],
                    qubit_mapping[index_1],
                )
        self.property_set["original_layout"] = Layout(input_qubit_mapping)
        self.property_set["original_qubit_indices"] = input_qubit_mapping
        self.property_set["final_layout"] = Layout(dict(zip(dag.qubits, qubit_mapping)))
        return new_dag
