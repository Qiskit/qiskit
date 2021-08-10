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

from qiskit.transpiler.basepasses import TransformationPass


class Cancellation(TransformationPass):
    """Cancel back-to-back `gates` in dag."""

    def __init__(self, gates_to_cancel):
        """Initialize gates_to_cancel"""
        self.gates_to_cancel = gates_to_cancel
        super().__init__()

    def run(self, dag):
        """Run the Cancellation pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        self_inverse_gates = []
        inverse_gate_pairs = []
  
        for gates in self.gates_to_cancel:
            if isinstance(gates, str):
                self_inverse_gates.append(gates)
            else:
                inverse_gate_pairs.append(gates)

        dag = self._run_on_self_inverse(dag, self_inverse_gates)
        return self._run_on_inverse_pairs(dag, inverse_gate_pairs)

    def _run_on_self_inverse(self, dag, self_inverse_gates):
        """
        Run self-inverse gates on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        gate_cancel_runs = dag.collect_runs(self.gates_to_cancel)
        for gate_cancel_run in gate_cancel_runs:
            if len(gate_cancel_run) % 2 == 0:
                dag.remove_op_node(gate_cancel_run[0])
            for node in gate_cancel_run[1:]:
                dag.remove_op_node(node)
        return dag

    def _run_on_inverse_pairs(self, dag, inverse_gate_pairs):
        """
        Run inverse gate pairs on `dag`.
        
        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        for pair in inverse_gate_pairs:
            gate_cancel_runs = dag.collect_runs([pair[0].name]) 
            for gate_cancel_run in gate_cancel_runs:
                for i in range(len(gate_cancel_run) - 1):
                    #TODO Change i & i+1 from a dag node to a gate node
                    if gate_cancel_run[i] == pair[0] and gate_cancel_run[i+1] == pair[1]:
                        dag.remove_op_node(gate_cancel_run[i])
                        dag.remove_op_node(gate_cancel_run[i+1])
                    elif gate_cancel_run[i] == pair[1] and gate_cancel_run[i+1] == pair[0]:
                        dag.remove_op_node(gate_cancel_run[i])
                        dag.remove_op_node(gate_cancel_run[i+1])

        return dag
