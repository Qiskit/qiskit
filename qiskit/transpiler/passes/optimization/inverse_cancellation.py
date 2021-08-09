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
        # EXAMPLE gates_to_cancel = ["h", (RXGate(theta), RXGate(-theta)), "cx"]
        self_inverse_gates = []
        inverse_gate_pairs = []
        # TODO your code here...
        # loop through each item in self.gates_to_cancel:
        #     if it's a single item (not a tuple):  <== check the type of the input "type checking"
        #         add to self_inverse
        #     otherwise add to inverse pairs. Check that there are exactly two items
        # EXAMPLE self_inverse_gates = ["h", "cx"]
        # EXAMPLE inverse_gate_pairs = [(RXGate(theta), RXGate(-theta))]
        return self._run_on_self_inverse(dag, self.gates_to_cancel)
        # return self._run_on_inverse_pairs(dag, inverse_gate_pairs)

    def _run_on_self_inverse(self, dag, self_inverse_gates):
        """
        TODO
        """
        import ipdb; ipdb.set_trace()
        # EXAMPLE ["h", "jfdlsj", "fjdsjd", "fjdsf", "jkdjsk"]
        gate_cancel_runs = dag.collect_runs(self.gates_to_cancel)
        # Generalize input using self.gates_to_cancel
        for gate_cancel_run in gate_cancel_runs:
            # Partition the gate_cancel_run into chunks with equal gate arguments
            partition = []
            chunk = []
            for i in range(len(gate_cancel_run) - 1):
                chunk.append(gate_cancel_run[i])

                qargs0 = gate_cancel_run[i].qargs
                qargs1 = gate_cancel_run[i + 1].qargs

                if qargs0 != qargs1:
                    partition.append(chunk)
                    chunk = []
            chunk.append(gate_cancel_run[-1])
            partition.append(chunk)
            # Simplify each chunk in the partition
            for chunk in partition:
                if len(chunk) % 2 == 0:
                    for n in chunk:
                        dag.remove_op_node(n)
                else:
                    for n in chunk[1:]:
                        dag.remove_op_node(n)
        return dag

    def _run_on_inverse_pairs(self, dag, inverse_gate_pairs):
        """
        TODO
        """
        # EXAMPLE DAG RX(pi/4) 0, RX(pi/4) 0, RX(-pi/4) 0, ...
        # EXAMPLE DAG RX(pi/4) 0, RX(pi/4) 1, RX(-pi/4) 0, RX(-pi/4) 1, ... 
        # EXAMPLE [(RXGate(pi/4), RXGate(-pi/4)), (RZ(theta), RZ(-theta))]
        for pair in inverse_gate_pairs:
            # EXAMPLE pair = (RXGate(pi/4), RXGate(-pi/4))
            # EXAMPLE pair[0] = RXGate(pi/4)
            gate_cancel_runs = dag.collect_runs([pair[0].name])  # "rx"
            for gate_cancel_run in gate_cancel_runs:
                partition = []
                chunk = []
                for i in range(len(gate_cancel_run) - 1):
                    chunk.append(gate_cancel_run[i])

                    qargs0 = gate_cancel_run[i].qargs
                    qargs1 = gate_cancel_run[i + 1].qargs

                    if qargs0 != qargs1:
                        partition.append(chunk)
                        chunk = []
                chunk.append(gate_cancel_run[-1])
                partition.append(chunk)
                # Simplify each chunk in the partition
                for chunk in partition:
                    # EXAMPLE
                    # RX(pi/4) 0, RX(pi/4) 0, RX(-pi/4) 0
                    # TODO: this only checks the first two items, we want to check every pair
                    if chunk[0] == pair[0] and chunk[1] == pair[1]:
                        dag.remove_op_node(chunk[0])
                        dag.remove_op_node(chunk[1])
                    # TODO: check also the inverse order

        return dag
