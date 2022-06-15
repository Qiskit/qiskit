# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A generic InverseCancellation pass for any set of gate-inverse pairs.
"""
from typing import List, Tuple, Union

from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class CommutativeInverseCancellation(TransformationPass):
    """Cancel specific Gates which are inverses of each other when they occur back-to-
    back."""

    def __init__(self, gates_to_cancel: List[Union[Gate, Tuple[Gate, Gate]]]):
        """Initialize InverseCancellation pass.

        Args:
            gates_to_cancel: list of gates to cancel

        Raises:
            TranspilerError:
                Initalization raises an error when the input is not a self-inverse gate
                or a two-tuple of inverse gates.
        """

        for gates in gates_to_cancel:
            if isinstance(gates, Gate):
                if gates != gates.inverse():
                    raise TranspilerError(f"Gate {gates.name} is not self-inverse")
            elif isinstance(gates, tuple):
                if len(gates) != 2:
                    raise TranspilerError(
                        f"Too many or too few inputs: {gates}. Only two are allowed."
                    )
                if gates[0] != gates[1].inverse():
                    raise TranspilerError(
                        f"Gate {gates[0].name} and {gates[1].name} are not inverse."
                    )
            else:
                raise TranspilerError(
                    "InverseCancellation pass does not take input type {}. Input must be"
                    " a Gate.".format(type(gates))
                )

        self.self_inverse_gates = []
        self.inverse_gate_pairs = []

        for gates in gates_to_cancel:
            if isinstance(gates, Gate):
                self.self_inverse_gates.append(gates)
            else:
                self.inverse_gate_pairs.append(gates)

        super().__init__()

    def run(self, dag: DAGCircuit):
        """Run the InverseCancellation pass on `dag`.

        Args:
            dag: the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """

        dag = self._run_on_self_inverse(dag, self.self_inverse_gates)
        dag2 = self._run_on_inverse_pairs(dag, self.inverse_gate_pairs)
        return dag2

    def _run_on_self_inverse(self, dag: DAGCircuit, self_inverse_gates: List[Gate]):
        """
        Run self-inverse gates on `dag`.

        Args:
            dag: the directed acyclic graph to run on.
            self_inverse_gates: list of gates who cancel themeselves in pairs

        Returns:
            DAGCircuit: Transformed DAG.
        """
        topo_sorted_nodes = []
        for node in dag.topological_op_nodes():
            # print(node)
            topo_sorted_nodes.append(node)
        # print(topo_sorted_nodes)

        circ_size = len(topo_sorted_nodes)

        removed = [False for _ in range(circ_size)]

        from .commutation_checker import CommutationChecker

        cc = CommutationChecker()
        # cc.print()

        gate_names = [gate.name for gate in self.self_inverse_gates]

        for idx1 in range(0, circ_size):
            if topo_sorted_nodes[idx1].name not in gate_names:
                continue

            matched_idx2 = -1

            for idx2 in range(idx1 - 1, -1, -1):
                if removed[idx2]:
                    continue

                removed2 = False

                if (
                    topo_sorted_nodes[idx2].name == topo_sorted_nodes[idx1].name
                    and topo_sorted_nodes[idx2].op.params == topo_sorted_nodes[idx1].op.params
                    and topo_sorted_nodes[idx2].qargs == topo_sorted_nodes[idx1].qargs
                ):
                    matched_idx2 = idx2
                    break

                if removed2:
                    print("Should not happen")
                    print(f"{topo_sorted_nodes[idx2]}")
                    print(f"{topo_sorted_nodes[idx1]}")
                    print(f"{topo_sorted_nodes[idx2].op}")
                    print(f"{topo_sorted_nodes[idx1].op}")

                    assert(False)

                if not cc.commute(topo_sorted_nodes[idx1], topo_sorted_nodes[idx2]):
                    break

            if matched_idx2 != -1:
                removed[idx1] = True
                removed[matched_idx2] = True

        # print(f"At the end:")
        num_removed = 0
        for i in range(len(removed)):
            if removed[i]:
                num_removed += 1
        # print(f"{num_removed = }")
        # cc.print()

        for idx in range(circ_size):
            if removed[idx]:
                dag.remove_op_node(topo_sorted_nodes[idx])

        return dag

    def _run_on_inverse_pairs(self, dag: DAGCircuit, inverse_gate_pairs: List[Tuple[Gate, Gate]]):
        """
        Run inverse gate pairs on `dag`.

        Args:
            dag: the directed acyclic graph to run on.
            inverse_gate_pairs: list of gates with inverse angles that cancel each other.

        Returns:
            DAGCircuit: Transformed DAG.
        """

        # NOT IMPLEMENTED FOR NOW
        return dag
