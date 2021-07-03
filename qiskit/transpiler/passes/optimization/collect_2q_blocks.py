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

"""Collect sequences of uninterrupted gates acting on 2 qubits."""

from collections import defaultdict

import retworkx

from qiskit.circuit import Gate
from qiskit.circuit.quantumregister import Qubit
from qiskit.transpiler.basepasses import AnalysisPass


class Collect2qBlocks(AnalysisPass):
    """Collect two-qubit subcircuits."""

    def run(self, dag):
        """Run the Collect2qBlocks pass on `dag`.

        The blocks contain "op" nodes in topological order such that all gates
        in a block act on the same qubits and are adjacent in the circuit.

        After the execution, ``property_set['block_list']`` is set to a list of
        tuples of "op" node.
        """
        self.property_set["commutation_set"] = defaultdict(list)

        def filter_fn(node):
            if node.type == "op":
                return (
                    isinstance(node._op, Gate)
                    and len(node._qargs) <= 2
                    and not node._op.condition
                    and not node._op.is_parameterized()
                )
            else:
                return None

        to_qid = dict()
        for i, qubit in enumerate(dag.qubits):
            to_qid[qubit] = i

        def color_fn(edge):
            if isinstance(edge, Qubit):
                return to_qid[edge]
            else:
                return -1

        self.property_set["block_list"] = retworkx.collect_bicolor_runs(
            dag._multi_graph, filter_fn, color_fn
        )
        return dag
