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

"""Remove reset gate when the qubit is in zero state."""

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister, Reset
from qiskit.circuit.library import CXGate, HGate, Measure, SwapGate
from qiskit.circuit.controlflow.if_else import IfElseOp
from qiskit.dagcircuit import DAGCircuit, DAGInNode
from qiskit.transpiler.basepasses import TransformationPass


class ReplaceSwapWithZeroState(TransformationPass):
    """Remove reset gate when the qubit is in zero state."""

    def run(self, dag):
        """Run the RemoveResetInZeroState pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        swaps = dag.op_nodes(SwapGate)
        for swap in swaps:
            predecessor = next(dag.predecessors(swap))
            if isinstance(predecessor, DAGInNode) or isinstance(predecessor, Reset):
                zero_qubit = predecessor.wire.index
                for qarg in swap.qargs:
                    if qarg.index != zero_qubit:
                        data_qubit = qarg.index

                mini_dag = DAGCircuit()
                qreg = QuantumRegister(2)
                creg = ClassicalRegister(1)
                mini_dag.add_qreg(qreg)
                mini_dag.add_creg(creg)

                mini_dag.apply_operation_back(CXGate(), [qreg[1], qreg[0]])
                mini_dag.apply_operation_back(HGate(), [qreg[1]])
                mini_dag.apply_operation_back(Measure(), [qreg[1]], [creg[0]])

                true_body = QuantumCircuit(qreg)
                true_body.z(0)
                true_body.x(1)

                mini_dag.apply_operation_back(
                    IfElseOp((creg[0], 1), true_body), [qreg[0], qreg[1]], [creg[0]]
                )
                dag.substitute_node_with_dag(swap, mini_dag, wires=[qreg[0], qreg[1], creg[0]])

        return dag
