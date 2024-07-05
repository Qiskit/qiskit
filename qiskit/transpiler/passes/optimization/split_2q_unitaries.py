# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from qiskit.circuit.library import UnitaryGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.synthesis.two_qubit.local_invariance import two_qubit_local_invariants
from qiskit.synthesis.two_qubit.two_qubit_decompose import decompose_two_qubit_product_gate
from qiskit.transpiler import TransformationPass


class Split2QUnitaries(TransformationPass):
    """Splits each two-qubit gate in the `dag` into two single-qubit gates, if possible without error."""

    def run(self, dag: DAGCircuit):
        """Run the Split2QUnitaries pass on `dag`."""
        for node in dag.topological_op_nodes():
            # skip operations without two-qubits and for which we can not determine a potential 1q split
            if (
                len(node.cargs) > 0
                or len(node.qargs) != 2
                or not (hasattr(node.op, "to_matrix") and hasattr(node.op, "__array__"))
                or (hasattr(node.op, "is_parameterized") and node.op.is_parameterized())
            ):
                # getattr(node.op, "_directive", False) or (hasattr(node.op, 'is_parameterized') and node.op.is_parameterized())):
                continue

            # check if the node can be represented by single-qubit gates
            nmat = node.op.to_matrix()
            if np.all(two_qubit_local_invariants(nmat) == [1, 0, 3]):
                ul, ur, phase = decompose_two_qubit_product_gate(nmat)
                dag_node = DAGCircuit()
                dag_node.add_qubits(node.qargs)

                dag_node.apply_operation_back(UnitaryGate(ur), qargs=(node.qargs[0],))
                dag_node.apply_operation_back(UnitaryGate(ul), qargs=(node.qargs[1],))
                dag_node.global_phase += phase
                dag.substitute_node_with_dag(node, dag_node)

        return dag
