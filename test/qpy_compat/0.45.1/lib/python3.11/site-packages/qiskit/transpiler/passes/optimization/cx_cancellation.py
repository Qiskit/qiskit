# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Cancel back-to-back ``cx`` gates in dag."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow


class CXCancellation(TransformationPass):
    """Cancel back-to-back ``cx`` gates in dag."""

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the CXCancellation pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        cx_runs = dag.collect_runs(["cx"])
        for cx_run in cx_runs:
            # Although one `cx_run` will always have the same set of qubits,
            # the order of these qubits matters (CXi_j != CXj_i).
            # Therefore, we have to partition each run into chunks where the qargs
            # are in the same order, and only cancel out CXs within each chunk.
            partitions = []
            chunk = []
            for i in range(len(cx_run) - 1):
                chunk.append(cx_run[i])
                if cx_run[i].qargs != cx_run[i + 1].qargs:
                    partitions.append(chunk)
                    chunk = []
            chunk.append(cx_run[-1])
            partitions.append(chunk)

            # Remove an even number of CX gates from each chunk
            for chunk in partitions:
                if len(chunk) % 2 == 0:
                    dag.remove_op_node(chunk[0])
                for node in chunk[1:]:
                    dag.remove_op_node(node)

        return dag
