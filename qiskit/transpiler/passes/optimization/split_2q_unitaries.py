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

"""Splits each two-qubit gate in the `dag` into two single-qubit gates, if possible without error."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.quantumcircuitdata import CircuitInstruction
from qiskit.dagcircuit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.library.generalized_gates import UnitaryGate
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition


class Split2QUnitaries(TransformationPass):
    """Attempt to splits two-qubit unitaries in a :class:`.DAGCircuit` into two single-qubit gates.

    This pass will analyze all :class:`.UnitaryGate` instances and determine whether the
    matrix is actually a product of 2 single qubit gates. In these cases the 2q gate can be
    simplified into two single qubit gates and this pass will perform this optimization and will
    replace the two qubit gate with two single qubit :class:`.UnitaryGate`.
    """

    def __init__(self, fidelity: float = 1.0 - 1e-16):
        """
        Args:
            fidelity: Allowed tolerance for splitting two-qubit unitaries and gate decompositions.
        """
        super().__init__()
        self.requested_fidelity = fidelity

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the Split2QUnitaries pass on `dag`."""

        for node in dag.topological_op_nodes():
            # We only attempt to split UnitaryGate objects, but this could be extended in future
            # -- however we need to ensure that we can compile the resulting single-qubit unitaries
            # to the supported basis gate set.
            if not (len(node.qargs) == 2 and node.op.name == "unitary"):
                continue

            decomp = TwoQubitWeylDecomposition(node.matrix, fidelity=self.requested_fidelity)
            if (
                decomp._inner_decomposition.specialization
                == TwoQubitWeylDecomposition._specializations.IdEquiv
            ):
                new_dag = DAGCircuit()
                new_dag.add_qubits(node.qargs)

                ur = decomp.K1r
                ur_node = DAGOpNode.from_instruction(
                    CircuitInstruction(UnitaryGate(ur), qubits=(node.qargs[0],)), dag=new_dag
                )

                ul = decomp.K1l
                ul_node = DAGOpNode.from_instruction(
                    CircuitInstruction(UnitaryGate(ul), qubits=(node.qargs[1],)), dag=new_dag
                )
                new_dag._apply_op_node_back(ur_node)
                new_dag._apply_op_node_back(ul_node)
                new_dag.global_phase = decomp.global_phase
                dag.substitute_node_with_dag(node, new_dag)
            elif (
                decomp._inner_decomposition.specialization
                == TwoQubitWeylDecomposition._specializations.SWAPEquiv
            ):
                # TODO maybe also look into swap-gate-like gates? Things to consider:
                #   * As the qubit mapping may change, we'll always need to build a new dag in this pass
                #   * There may not be many swap-gate-like gates in an arbitrary input circuit
                #   * Removing swap gates from a user-routed input circuit here is unexpected
                pass
        return dag
