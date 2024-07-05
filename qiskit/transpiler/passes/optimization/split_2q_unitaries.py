import numpy as np
from qiskit.circuit.library import UnitaryGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.synthesis.two_qubit.local_invariance import two_qubit_local_invariants
from qiskit.synthesis.two_qubit.two_qubit_decompose import decompose_two_qubit_product_gate
from qiskit.transpiler import TransformationPass


class Split2QUnitaries(TransformationPass):
    """Splits each two-qubit gate in the `dag` into two single-qubit gates, if possible without error.

    """

    def run(self, dag: DAGCircuit):
        """Run the Split2QUnitaries pass on `dag`."""
        for node in dag.topological_op_nodes():
            # skip operations without two-qubits and for which we can not determine a potential 1q split
            if (len(node.cargs) > 0 or len(node.qargs) != 2 or hasattr(node.op, '_directive')
                    or (hasattr(node.op, 'is_parameterized') and node.op.is_parameterized())):
                continue

            # check if the node can be represented by single-qubit gates
            if np.all(two_qubit_local_invariants(node.op) == [1, 0, 3]):
                ul, ur, phase = decompose_two_qubit_product_gate(node.op)
                dag_node = DAGCircuit()
                dag_node.add_qubits(node.qargs)
                dag_node.apply_operation_back(UnitaryGate(ul), qargs=(node.qargs[0],))
                dag_node.apply_operation_back(UnitaryGate(ur), qargs=(node.qargs[1],))
                dag.substitute_node_with_dag(node, dag_node)
