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

"""Analysis pass to find commutation relations between DAG nodes."""

from collections import defaultdict
import numpy as np
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.quantum_info.operators import Operator
from qiskit.dagcircuit import DAGOpNode

_CUTOFF_PRECISION = 1e-10


class CommutationAnalysis(AnalysisPass):
    """Analysis pass to find commutation relations between DAG nodes.

    Property_set['commutation_set'] is a dictionary that describes
    the commutation relations on a given wire, all the gates on a wire
    are grouped into a set of gates that commute.

    TODO: the current pass determines commutativity through matrix multiplication.
    A rule-based analysis would be potentially faster, but more limited.
    """

    def __init__(self):
        super().__init__()
        self.cache = {}

    def run(self, dag):
        """Run the CommutationAnalysis pass on `dag`.

        Run the pass on the DAG, and write the discovered commutation relations
        into the property_set.
        """
        # Initiate the commutation set
        self.property_set["commutation_set"] = defaultdict(list)

        # Build a dictionary to keep track of the gates on each qubit
        # The key with format (wire) will store the lists of commutation sets
        # The key with format (node, wire) will store the index of the commutation set
        # on the specified wire, thus, for example:
        # self.property_set['commutation_set'][wire][(node, wire)] will give the
        # commutation set that contains node.

        for wire in dag.wires:
            self.property_set["commutation_set"][wire] = []

        # Add edges to the dictionary for each qubit
        for node in dag.topological_op_nodes():
            for (_, _, edge_wire) in dag.edges(node):
                self.property_set["commutation_set"][(node, edge_wire)] = -1

        # Construct the commutation set
        for wire in dag.wires:

            for current_gate in dag.nodes_on_wire(wire):

                current_comm_set = self.property_set["commutation_set"][wire]
                if not current_comm_set:
                    current_comm_set.append([current_gate])

                if current_gate not in current_comm_set[-1]:
                    prev_gate = current_comm_set[-1][-1]
                    does_commute = False
                    try:
                        does_commute = _commute(current_gate, prev_gate, self.cache)
                    except TranspilerError:
                        pass
                    if does_commute:
                        current_comm_set[-1].append(current_gate)

                    else:
                        current_comm_set.append([current_gate])

                temp_len = len(current_comm_set)
                self.property_set["commutation_set"][(current_gate, wire)] = temp_len - 1


_COMMUTE_ID_OP = {}


def _hashable_parameters(params):
    """Convert the parameters of a gate into a hashable format for lookup in a dictionary.

    This aims to be fast in common cases, and is not intended to work outside of the lifetime of a
    single commutation pass; it does not handle mutable state correctly if the state is actually
    changed."""
    try:
        hash(params)
        return params
    except TypeError:
        pass
    if isinstance(params, (list, tuple)):
        return tuple(_hashable_parameters(x) for x in params)
    if isinstance(params, np.ndarray):
        # We trust that the arrays will not be mutated during the commutation pass, since nothing
        # would work if they were anyway. Using the id can potentially cause some additional cache
        # misses if two UnitaryGate instances are being compared that have been separately
        # constructed to have the same underlying matrix, but in practice the cost of string-ifying
        # the matrix to get a cache key is far more expensive than just doing a small matmul.
        return (np.ndarray, id(params))
    # Catch anything else with a slow conversion.
    return ("fallback", str(params))


def _commute(node1, node2, cache):
    if not isinstance(node1, DAGOpNode) or not isinstance(node2, DAGOpNode):
        return False
    for nd in [node1, node2]:
        if nd.op._directive or nd.name in {"measure", "reset", "delay"}:
            return False
    if node1.op.condition or node2.op.condition:
        return False
    if node1.op.is_parameterized() or node2.op.is_parameterized():
        return False

    # Assign indices to each of the qubits such that all `node1`'s qubits come first, followed by
    # any _additional_ qubits `node2` addresses.  This helps later when we need to compose one
    # operator with the other, since we can easily expand `node1` with a suitable identity.
    qarg = {q: i for i, q in enumerate(node1.qargs)}
    num_qubits = len(qarg)
    for q in node2.qargs:
        if q not in qarg:
            qarg[q] = num_qubits
            num_qubits += 1
    qarg1 = tuple(qarg[q] for q in node1.qargs)
    qarg2 = tuple(qarg[q] for q in node2.qargs)

    node1_key = (node1.op.name, _hashable_parameters(node1.op.params), qarg1)
    node2_key = (node2.op.name, _hashable_parameters(node2.op.params), qarg2)
    try:
        # We only need to try one orientation of the keys, since if we've seen the compound key
        # before, we've set it in both orientations.
        return cache[node1_key, node2_key]
    except KeyError:
        pass

    operator_1 = Operator(node1.op, input_dims=(2,) * len(qarg1), output_dims=(2,) * len(qarg1))
    operator_2 = Operator(node2.op, input_dims=(2,) * len(qarg2), output_dims=(2,) * len(qarg2))

    if qarg1 == qarg2:
        # Use full composition if possible to get the fastest matmul paths.
        op12 = operator_1.compose(operator_2)
        op21 = operator_2.compose(operator_1)
    else:
        # Expand operator_1 to be large enough to contain operator_2 as well; this relies on qargs1
        # being the lowest possible indices so the identity can be tensored before it.
        extra_qarg2 = num_qubits - len(qarg1)
        if extra_qarg2:
            try:
                id_op = _COMMUTE_ID_OP[extra_qarg2]
            except KeyError:
                id_op = _COMMUTE_ID_OP[extra_qarg2] = Operator(
                    np.eye(2**extra_qarg2),
                    input_dims=(2,) * extra_qarg2,
                    output_dims=(2,) * extra_qarg2,
                )
            operator_1 = id_op.tensor(operator_1)
        op12 = operator_1.compose(operator_2, qargs=qarg2, front=False)
        op21 = operator_1.compose(operator_2, qargs=qarg2, front=True)
    cache[node1_key, node2_key] = cache[node2_key, node1_key] = ret = op12 == op21
    return ret
