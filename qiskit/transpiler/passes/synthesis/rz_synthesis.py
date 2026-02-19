# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""A temporary RZSynthesis pass. This will be replaced by Janani's Rust implementation."""


from qiskit.transpiler.basepasses import TransformationPass
from qiskit.synthesis import gridsynth_rz


class RZSynthesis(TransformationPass):
    """Convert Clifford+T+RZ circuit to Clifford+T circuit."""

    def __init__(self, approximation_degree: float = 1.0):
        self.approximation_degree = approximation_degree
        super().__init__()

    def run(self, dag):
        """Run RZSynthesis on `dag`.

        This will be replaced by a much better rust version of the pass.
        """
        ops = dag.count_ops()

        if ops.get("rz", 0) == 0:
            return dag

        tol = max(1 - self.approximation_degree, 1e-10)

        new_dag = dag.copy_empty_like()

        for node in dag.topological_op_nodes():
            if node.name == "rz":
                qubit = node.qargs[0]
                angle = node.op.params[0]
                inner_t_circuit = gridsynth_rz(angle, tol)
                for inner_t_node in inner_t_circuit:
                    new_dag.apply_operation_back(inner_t_node.operation, [qubit], [])
                new_dag.global_phase += inner_t_circuit.global_phase
            else:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        return new_dag
