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

"""Cancel the redundant (self-adjoint) gates through commutation relations."""
from __future__ import annotations

from qiskit.circuit.commutation_checker import CommutationChecker
from qiskit.circuit.commutation_library import standard_gates_commutations
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library.generalized_gates.pauli import PauliGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import Pauli
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils.remove_final_measurements import calc_final_ops

commutator = CommutationChecker(standard_gates_commutations)


class LightCone(TransformationPass):
    """Remove the gates that do not affect the outcome of a measurement on a circuit.

    Pass for computing the light-cone of an observable or measurement. The Pass can
    handle either a Pauli operator one would like to measure or a measurement on a set
    of qubits.
    """

    def __init__(self, observable: Pauli | None = None):
        """
        Args:
            observable: If None the lightcone will be computed for the set
                of measurements in the circuit. If a Pauli operator is specified,
                the lightcone will correspond to the reduced circuit with the
                same expectation value for the observable.
        """
        super().__init__()
        self.observable = observable

    @staticmethod
    def _find_measurement_qubits(dag: DAGCircuit):
        final_nodes = calc_final_ops(dag, {"measure"})
        qubits_measured = set()
        for node in final_nodes:
            qubits_measured |= set(node.qargs)
        return qubits_measured

    def _get_initial_lightcone(
        self, dag: DAGCircuit
    ) -> tuple[set[int], list[tuple[PauliGate, list[int]]]]:
        """Returns the initial lightcone.
        If obervable is `None`, the lightcone is the set of measured qubits.
        If a Pauli observable is provided, the qubit corresponding to
        the non-trivial Paulis define the lightcone.
        """
        if self.observable is None:
            light_cone = self._find_measurement_qubits(dag)
            light_cone_ops = [(PauliGate("Z"), [qubit_index]) for qubit_index in light_cone]
        else:
            # Check if the size of the observable matches the number of qubits in the circuit
            if self.observable.num_qubits != dag.num_qubits():
                raise ValueError(
                    "Observable size does not match the number of qubits in the circuit."
                )
            non_trivial_indices = [i for i, p in enumerate(self.observable) if p != Pauli("I")]
            light_cone = [dag.qubits[i] for i in non_trivial_indices]
            stripped_pauli_label = "".join(
                [self.observable[i].to_label() for i in reversed(non_trivial_indices)]
            )
            light_cone_ops = [(PauliGate(stripped_pauli_label), light_cone)]

        return set(light_cone), light_cone_ops

    def run(self, dag: DAGCircuit):
        """Run the LightCone pass on `dag`.

        Args:
            dag: The DAG to reduce.

        Returns:
            The DAG reduced to the light cone of the observable.
        """
        # Get the initial light cone and operations
        light_cone, light_cone_ops = self._get_initial_lightcone(dag)

        #  Initialize a new, empty DAG
        new_dag = dag.copy_empty_like()

        # Iterate over the nodes in reverse topological order
        for node in reversed(list(dag.topological_op_nodes())):
            # Check if the node belongs to the light cone
            if light_cone.intersection(node.qargs):
                # Check commutation with all previous operations
                commutes_bool = True
                for op in light_cone_ops:
                    commute_bool = commutator.commute(op[0], op[1], [], node.op, node.qargs, [])
                    if not commute_bool:
                        # If the current node does not commute, update the light cone
                        light_cone.update(node.qargs)
                        light_cone_ops.append((node.op, node.qargs))
                        commutes_bool = False
                        break

                # If the node is in the light cone and commutes with previous ops,
                # add it to the new DAG at the front
                if not commutes_bool:
                    new_dag.apply_operation_front(node.op, node.qargs, node.cargs)
        return new_dag
