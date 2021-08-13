# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A generic gate-inverse cancellation pass for a broad set of gate-inverse pairs.
"""
from typing import List, Tuple, Union

from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import TranspilerError
from qiskit.transpiler.basepasses import TransformationPass


class Cancellation(TransformationPass):
    """Cancel back-to-back `gates` in dag."""

    def __init__(self, gates_to_cancel: List[Union[Gate, Tuple[Gate, Gate]]]):
        """Initialize gates_to_cancel.

        Args:
            gates_to_cancel: TODO
        """
        # TODO: iterate through each item in the input
        #       if it's a single item: check if self inverse
        #       else: check if they multiply to identity
        self.gates_to_cancel = gates_to_cancel
        super().__init__()

    def run(self, dag: DAGCircuit):
        """Run the Cancellation pass on `dag`.

        Args:
            dag: the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        self_inverse_gates = []
        inverse_gate_pairs = []

        for gates in self.gates_to_cancel:
            if isinstance(gates, Gate):
                self_inverse_gates.append(gates)
            else:
                inverse_gate_pairs.append(gates)

        dag = self._run_on_self_inverse(dag, self_inverse_gates)
        return self._run_on_inverse_pairs(dag, inverse_gate_pairs)

    def _run_on_self_inverse(self, dag: DAGCircuit, self_inverse_gates: List[Gate]):
        """
        Run self-inverse gates on `dag`.

        Args:
            dag: the directed acyclic graph to run on.
            self_inverse_gates: TODO

        Returns:
            DAGCircuit: Transformed DAG.
        """
        gate_cancel_runs = dag.collect_runs([gate.name for gate in self_inverse_gates])
        for gate_cancel_run in gate_cancel_runs:
            if len(gate_cancel_run) % 2 == 0:
                dag.remove_op_node(gate_cancel_run[0])
            for node in gate_cancel_run[1:]:
                dag.remove_op_node(node)
        return dag

    def _run_on_inverse_pairs(self, dag: DAGCircuit, inverse_gate_pairs: List[Tuple[Gate, Gate]]):
        """
        Run inverse gate pairs on `dag`.
        
        Args:
            dag: the directed acyclic graph to run on.
            inverse_gate_pairs: TODO

        Returns:
            DAGCircuit: Transformed DAG.
        """
        for pair in inverse_gate_pairs:
            gate_cancel_runs = dag.collect_runs([pair[0].name]) 
            for dag_node in gate_cancel_runs:
                for i in range(len(dag_node) - 1):
                    if dag_node[i].op == pair[0] and dag_node[i + 1].op == pair[1]:
                        dag.remove_op_node(dag_node[i])
                        dag.remove_op_node(dag_node[i + 1])
                    elif dag_node[i].op == pair[1] and dag_node[i + 1].op == pair[0]:
                        dag.remove_op_node(dag_node[i])
                        dag.remove_op_node(dag_node[i + 1])

        return dag
