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
from typing import List, Union
import numpy as np

from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info.operators import Operator
from qiskit.circuit._standard_gates_commutations import standard_gates_commutations

StandardGateCommutations = standard_gates_commutations


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

    def __init__(self, cache_max_entries: int = 10**6):
        super().__init__()
        self._standard_commutations = StandardGateCommutations
        self._cached_commutations = {}
        self._cache_max_entries = cache_max_entries
        self._current_cache_entries = 0

    def commute(
        self,
        op1: Operation,
        qargs1: List,
        cargs1: List,
        op2: Operation,
        qargs2: List,
        cargs2: List,
        max_num_qubits: int = 4,
    ) -> bool:
        """
        Checks if two Operations commute. The return value of `True` means that the operations
        truly commute, and the return value of `False` means that either the operations do not
        commute or that the commutation check was skipped (for example, when the operations
        have conditions or have too many qubits).

        Args:
            op1: first operation.
            qargs1: first operation's qubits.
            cargs1: first operation's clbits.
            op2: second operation.
            qargs2: second operation's qubits.
            cargs2: second operation's clbits.
            max_num_qubits: the maximum number of qubits to consider, the check may be skipped if
                the number of qubits for either operation exceeds this amount.

        Returns:
            bool: whether two operations commute.
        """
        structural_commutation = _commutation_precheck(
            op1, qargs1, cargs1, op2, qargs2, cargs2, max_num_qubits
        )

        if structural_commutation is not None:
            return structural_commutation

        commutation_lookup = self.check_commutation_entries(
            op1, qargs1, cargs1, op2, qargs2, cargs2
        )

        if commutation_lookup is not None:
            return commutation_lookup

        # Compute commutation via matrix multiplication
        is_commuting = _commute_matmul(op1, qargs1, op2, qargs2)

        # Store result in this session's commutation_library
        # TODO implement LRU cache or similar
        # Rebuild cache if current cache exceeded max size
        if self._current_cache_entries >= self._cache_max_entries:
            self._cached_commutations = {}

        first_op_tuple, second_op_tuple = _order_operations(
            op1, qargs1, cargs1, op2, qargs2, cargs2
        )
        first_op, first_qargs, _ = first_op_tuple
        second_op, second_qargs, _ = second_op_tuple
        first_params = first_op.params
        second_params = second_op.params

        self._cached_commutations.setdefault((first_op.name, second_op.name), {}).setdefault(
            _get_relative_placement(first_qargs, second_qargs), {}
        )[(_hashable_parameters(first_params), _hashable_parameters(second_params))] = is_commuting
        self._current_cache_entries += 1

        return is_commuting

    def check_commutation_entries(
        self, op1: Operation, qargs1: List, cargs1: List, op2: Operation, qargs2: List, cargs2: List
    ) -> Union[bool, None]:
        """Returns stored commutation relation if any

        Args:
            op1: first operation.
            qargs1: first operation's qubits.
            cargs1: first operation's clbits.
            op2: second operation.
            qargs2: second operation's qubits.
            cargs2: second operation's clbits.

        Return:
            bool: True if the gates commute and false if it is not the case.
        """

        # We don't precompute commutations for parameterized gates, yet
        commutation = _query_commutation(
            op1, qargs1, cargs1, op2, qargs2, cargs2, self._standard_commutations
        )
        if commutation is not None:
            return commutation

        commutation = _query_commutation(
            op1, qargs1, cargs1, op2, qargs2, cargs2, self._cached_commutations
        )

        return commutation


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
        # Using the bytes of the matrix as key is runtime efficient but requires more space: 128 bits
        # times the number of parameters instead of a single 64 bit id. However, by using the bytes as
        # an id, we can reuse the cached commutations between different passes.
        return (np.ndarray, params.tobytes())
    # Catch anything else with a slow conversion.
    return ("fallback", str(params))


_skipped_op_names = {"measure", "reset", "delay"}


def _commutation_precheck(
    op1: Operation,
    qargs1: List,
    cargs1: List,
    op2: Operation,
    qargs2: List,
    cargs2: List,
    max_num_qubits,
):
    # pylint: disable=too-many-return-statements

    # We don't support commutation of conditional gates for now due to bugs in
    # CommutativeCancellation.  See gh-8553.
    if getattr(op1, "condition", None) is not None or getattr(op2, "condition", None) is not None:
        return False

    # Commutation of ControlFlow gates also not supported yet. This may be
    # pending a control flow graph.
    if isinstance(op1, ControlFlowOp) or isinstance(op2, ControlFlowOp):
        return False

    # These lines are adapted from dag_dependency and say that two gates over
    # different quantum and classical bits necessarily commute. This is more
    # permissive that the check from commutation_analysis, as for example it
    # allows to commute X(1) and Measure(0, 0).
    # Presumably this check was not present in commutation_analysis as
    # it was only called on pairs of connected nodes from DagCircuit.
    intersection_q = set(qargs1).intersection(set(qargs2))
    intersection_c = set(cargs1).intersection(set(cargs2))
    if not (intersection_q or intersection_c):
        return True

    # Skip the check if the number of qubits for either operation is too large
    if len(qargs1) > max_num_qubits or len(qargs2) > max_num_qubits:
        return False

    # These lines are adapted from commutation_analysis, which is more restrictive than the
    # check from dag_dependency when considering nodes with "_directive".  It would be nice to
    # think which optimizations from dag_dependency can indeed be used.
    if op1.name in _skipped_op_names or op2.name in _skipped_op_names:
        return False

    if getattr(op1, "_directive", False) or getattr(op2, "_directive", False):
        return False
    if (getattr(op1, "is_parameterized", False) and op1.is_parameterized()) or (
        getattr(op2, "is_parameterized", False) and op2.is_parameterized()
    ):
        return False

    return None


def _get_relative_placement(first_qargs: List[Qubit], second_qargs: List[Qubit]) -> tuple:
    """Determines the relative qubit placement of two gates. Note: this is NOT symmetric.

    Args:
        first_qargs (DAGOpNode): first gate
        second_qargs (DAGOpNode): second gate

    Return:
        A tuple that describes the relative qubit placement: E.g.
        _get_relative_placement(CX(0, 1), CX(1, 2)) would return (None, 0) as there is no overlap on
        the first qubit of the first gate but there is an overlap on the second qubit of the first gate,
        i.e. qubit 0 of the second gate. Likewise,
        _get_relative_placement(CX(1, 2), CX(0, 1)) would return (1, None)
    """
    qubits_g2 = {q_g1: i_g1 for i_g1, q_g1 in enumerate(second_qargs)}
    return tuple(qubits_g2.get(q_g0, None) for q_g0 in first_qargs)


@lru_cache(maxsize=10**3)
def _persistent_id(op_name: str) -> int:
    """Returns an integer id of a string that is persistent over different python executions (note that
        hash() can not be used, i.e. its value can change over two python executions)
    Args:
        op_name (str): The string whose integer id should be determined.
    Return:
        The integer id of the input string.
    """
    return int.from_bytes(bytes(op_name, encoding="ascii"), byteorder="big", signed=True)


def _order_operations(
    op1: Operation, qargs1: List, cargs1: List, op2: Operation, qargs2: List, cargs2: List
):
    """Orders two operations in a canonical way that is persistent over
    @different python versions and executions
    Args:
        op1: first operation.
        qargs1: first operation's qubits.
        cargs1: first operation's clbits.
        op2: second operation.
        qargs2: second operation's qubits.
        cargs2: second operation's clbits.
    Return:
        The input operations in a persistent, canonical order.
    """
    op1_tuple = (op1, qargs1, cargs1)
    op2_tuple = (op2, qargs2, cargs2)
    least_qubits_op, most_qubits_op = (
        (op1_tuple, op2_tuple) if op1.num_qubits < op2.num_qubits else (op2_tuple, op1_tuple)
    )
    # prefer operation with the least number of qubits as first key as this results in shorter keys
    if op1.num_qubits != op2.num_qubits:
        return least_qubits_op, most_qubits_op
    else:
        return (
            (op1_tuple, op2_tuple)
            if _persistent_id(op1.name) < _persistent_id(op2.name)
            else (op2_tuple, op1_tuple)
        )


def _query_commutation(
    op1: Operation,
    qargs1: List,
    cargs1: List,
    op2: Operation,
    qargs2: List,
    cargs2: List,
    _commutation_lib: dict,
) -> Union[bool, None]:
    """Queries and returns the commutation of a pair of operations from a provided commutation library
    Args:
    Args:
        op1: first operation.
        qargs1: first operation's qubits.
        cargs1: first operation's clbits.
        op2: second operation.
        qargs2: second operation's qubits.
        cargs2: second operation's clbits.
        _commutation_lib (dict): dictionary of commutation relations
    Return:
        True if op1 and op2 commute, False if they do not commute and
        None if the commutation is not in the library
    """
    first_op_tuple, second_op_tuple = _order_operations(op1, qargs1, cargs1, op2, qargs2, cargs2)
    first_op, first_qargs, _ = first_op_tuple
    second_op, second_qargs, _ = second_op_tuple
    first_params = first_op.params
    second_params = second_op.params

    commutation = _commutation_lib.get((first_op.name, second_op.name), None)

    # Return here if the commutation is constant over all relative placements of the operations
    if commutation is None or isinstance(commutation, bool):
        return commutation

    # If we arrive here, there is an entry in the commutation library but it depends on the
    # placement of the operations and also possibly on operation parameters
    if isinstance(commutation, dict):
        placement_commutation = commutation.get(
            _get_relative_placement(first_qargs, second_qargs), None
        )
        if (len(op1.params) > 0 or len(op2.params) > 0) and placement_commutation is not None:
            # Param commutation entry exists and must be a dict
            return placement_commutation.get(
                (_hashable_parameters(first_params), _hashable_parameters(second_params)), None
            )
        else:
            # queried commutation is True, False or None
            return placement_commutation
    else:
        raise ValueError("Expected commutation to be None, bool or a dict")


def _commute_matmul(op1: Operation, qargs1: List, op2: Operation, qargs2: List):
    qarg = {q: i for i, q in enumerate(qargs1)}
    num_qubits = len(qarg)
    for q in qargs2:
        if q not in qarg:
            qarg[q] = num_qubits
            num_qubits += 1
    qarg1 = tuple(qarg[q] for q in qargs1)
    qarg2 = tuple(qarg[q] for q in qargs2)

    operator_1 = Operator(op1, input_dims=(2,) * len(qarg1), output_dims=(2,) * len(qarg1))
    operator_2 = Operator(op2, input_dims=(2,) * len(qarg2), output_dims=(2,) * len(qarg2))

    if qarg1 == qarg2:
        # Use full composition if possible to get the fastest matmul paths.
        op12 = operator_1.compose(operator_2)
        op21 = operator_2.compose(operator_1)
    else:
        # Expand operator_1 to be large enough to contain operator_2 as well; this relies on qargs1
        # being the lowest possible indices so the identity can be tensored before it.
        extra_qarg2 = num_qubits - len(qarg1)
        if extra_qarg2:
            id_op = _identity_op(extra_qarg2)
            operator_1 = id_op.tensor(operator_1)
        op12 = operator_1.compose(operator_2, qargs=qarg2, front=False)
        op21 = operator_1.compose(operator_2, qargs=qarg2, front=True)
    ret = op12 == op21
    return ret
