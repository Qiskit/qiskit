# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Code from commutative_analysis pass that checks commutation relations between DAG nodes."""

from functools import lru_cache

import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit.dagcircuit import DAGOpNode, DAGDepNode


@lru_cache(maxsize=None)
def _identity_op(num_qubits):
    """Cached identity matrix"""
    return Operator(
        np.eye(2**num_qubits), input_dims=(2,) * num_qubits, output_dims=(2,) * num_qubits
    )


class CommutationChecker:
    """This code is essentially copy-pasted from commutative_analysis.py.
    This code cleverly hashes commutativity and non-commutativity results between DAG nodes and seems
    quite efficient for large Clifford circuits.
    They may be other possible efficiency improvements: using rule-based commutativity analysis,
    evicting from the cache less useful entries, etc.
    """

    def __init__(self):
        super().__init__()
        self.cache = {}

    def _hashable_parameters(self, params):
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
            return tuple(self._hashable_parameters(x) for x in params)
        if isinstance(params, np.ndarray):
            # We trust that the arrays will not be mutated during the commutation pass, since nothing
            # would work if they were anyway. Using the id can potentially cause some additional cache
            # misses if two UnitaryGate instances are being compared that have been separately
            # constructed to have the same underlying matrix, but in practice the cost of string-ifying
            # the matrix to get a cache key is far more expensive than just doing a small matmul.
            return (np.ndarray, id(params))
        # Catch anything else with a slow conversion.
        return ("fallback", str(params))

    def commute(self, node1, node2):
        """Checks if two DAG nodes commute."""

        # The function works both with nodes from DagCircuit and from DagDependency
        if not isinstance(node1, (DAGOpNode, DAGDepNode)) or not isinstance(
            node2, (DAGOpNode, DAGDepNode)
        ):
            return False

        # These lines are adapted from dag_dependency and say that two gates over
        # different quantum and classical bits necessarily commute. This is more
        # permissive that the check from commutation_analysis, as for example it
        # allows to commute X(1) and Measure(0, 0).
        # Presumably this check was not present in commutation_analysis as
        # it was only called on pairs of connected nodes from DagCircuit.
        intersection_q = set(node1.qargs).intersection(set(node2.qargs))
        intersection_c = set(node1.cargs).intersection(set(node2.cargs))
        if not (intersection_q or intersection_c):
            return True

        # This lines are adapted from commutation_analysis, which is more restrictive
        # than the check from dag_dependency when considering nodes with "_directive"
        # or "condition". It would be nice to think which optimizations
        # from dag_dependency can indeed be used.
        for nd in [node1, node2]:
            if (
                getattr(nd.op, "_directive", False)
                or nd.name in {"measure", "reset", "delay"}
                or getattr(nd.op, "condition", None)
                or nd.op.is_parameterized()
            ):
                return False

        # The main code is adapted from commutative analysis.
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

        node1_key = (node1.op.name, self._hashable_parameters(node1.op.params), qarg1)
        node2_key = (node2.op.name, self._hashable_parameters(node2.op.params), qarg2)
        try:
            # We only need to try one orientation of the keys, since if we've seen the compound key
            # before, we've set it in both orientations.
            return self.cache[node1_key, node2_key]
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
                id_op = _identity_op(2**extra_qarg2)
                operator_1 = id_op.tensor(operator_1)
            op12 = operator_1.compose(operator_2, qargs=qarg2, front=False)
            op21 = operator_1.compose(operator_2, qargs=qarg2, front=True)
        self.cache[node1_key, node2_key] = self.cache[node2_key, node1_key] = ret = op12 == op21
        return ret
