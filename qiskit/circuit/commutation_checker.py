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

from qiskit import QiskitError
from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.quantum_info.operators import Operator

_skipped_op_names = {"measure", "reset", "delay", "initialize"}
_no_cache_op_names = {"annotated"}


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

    def __init__(self, standard_gate_commutations: dict = None, cache_max_entries: int = 10**6):
        super().__init__()
        if standard_gate_commutations is None:
            self._standard_commutations = {}
        else:
            self._standard_commutations = standard_gate_commutations
        self._cache_max_entries = cache_max_entries

        # self._cached_commutation has the same structure as standard_gate_commutations, i.e. a
        # dict[pair of gate names][relative placement][tuple of gate parameters] := True/False
        self._cached_commutations = {}
        self._current_cache_entries = 0
        self._cache_miss = 0
        self._cache_hit = 0

    def commute_nodes(
        self,
        op1,
        op2,
        max_num_qubits: int = 3,
    ) -> bool:
        """Checks if two DAGOpNodes commute."""
        qargs1 = op1.qargs
        cargs1 = op2.cargs
        if not op1.is_standard_gate:
            op1 = op1.op
        qargs2 = op2.qargs
        cargs2 = op2.cargs
        if not op2.is_standard_gate:
            op2 = op2.op
        return self.commute(op1, qargs1, cargs1, op2, qargs2, cargs2, max_num_qubits)

    def commute(
        self,
        op1: Operation,
        qargs1: List,
        cargs1: List,
        op2: Operation,
        qargs2: List,
        cargs2: List,
        max_num_qubits: int = 3,
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

        first_op_tuple, second_op_tuple = _order_operations(
            op1, qargs1, cargs1, op2, qargs2, cargs2
        )
        first_op, first_qargs, _ = first_op_tuple
        second_op, second_qargs, _ = second_op_tuple

        skip_cache = first_op.name in _no_cache_op_names or second_op.name in _no_cache_op_names

        if skip_cache:
            return _commute_matmul(first_op, first_qargs, second_op, second_qargs)

        commutation_lookup = self.check_commutation_entries(
            first_op, first_qargs, second_op, second_qargs
        )

        if commutation_lookup is not None:
            return commutation_lookup

        # Compute commutation via matrix multiplication
        is_commuting = _commute_matmul(first_op, first_qargs, second_op, second_qargs)

        # Store result in this session's commutation_library
        # TODO implement LRU cache or similar
        # Rebuild cache if current cache exceeded max size
        if self._current_cache_entries >= self._cache_max_entries:
            self.clear_cached_commutations()

        first_params = getattr(first_op, "params", [])
        second_params = getattr(second_op, "params", [])
        if len(first_params) > 0 or len(second_params) > 0:
            self._cached_commutations.setdefault((first_op.name, second_op.name), {}).setdefault(
                _get_relative_placement(first_qargs, second_qargs), {}
            )[
                (_hashable_parameters(first_params), _hashable_parameters(second_params))
            ] = is_commuting
        else:
            self._cached_commutations.setdefault((first_op.name, second_op.name), {})[
                _get_relative_placement(first_qargs, second_qargs)
            ] = is_commuting
        self._current_cache_entries += 1

        return is_commuting

    def num_cached_entries(self):
        """Returns number of cached entries"""
        return self._current_cache_entries

    def clear_cached_commutations(self):
        """Clears the dictionary holding cached commutations"""
        self._current_cache_entries = 0
        self._cache_miss = 0
        self._cache_hit = 0
        self._cached_commutations = {}

    def check_commutation_entries(
        self,
        first_op: Operation,
        first_qargs: List,
        second_op: Operation,
        second_qargs: List,
    ) -> Union[bool, None]:
        """Returns stored commutation relation if any

        Args:
            first_op: first operation.
            first_qargs: first operation's qubits.
            second_op: second operation.
            second_qargs: second operation's qubits.

        Return:
            bool: True if the gates commute and false if it is not the case.
        """

        # We don't precompute commutations for parameterized gates, yet
        commutation = _query_commutation(
            first_op,
            first_qargs,
            second_op,
            second_qargs,
            self._standard_commutations,
        )

        if commutation is not None:
            return commutation

        commutation = _query_commutation(
            first_op,
            first_qargs,
            second_op,
            second_qargs,
            self._cached_commutations,
        )
        if commutation is None:
            self._cache_miss += 1
        else:
            self._cache_hit += 1
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


def is_commutation_supported(op):
    """
    Filter operations whose commutation is not supported due to bugs in transpiler passes invoking
    commutation analysis.
    Args:
        op (Operation): operation to be checked for commutation relation
    Return:
        True if determining the commutation of op is currently supported
    """
    # Bug in CommutativeCancellation, e.g. see gh-8553
    if getattr(op, "condition", False):
        return False

    # Commutation of ControlFlow gates also not supported yet. This may be pending a control flow graph.
    if op.name in CONTROL_FLOW_OP_NAMES:
        return False

    return True


def is_commutation_skipped(op, qargs, max_num_qubits):
    """
    Filter operations whose commutation will not be determined.
    Args:
        op (Operation): operation to be checked for commutation relation
        qargs (List): operation qubits
        max_num_qubits (int): the maximum number of qubits to consider, the check may be skipped if
                the number of qubits for either operation exceeds this amount.
    Return:
        True if determining the commutation of op is currently not supported
    """
    if (
        len(qargs) > max_num_qubits
        or getattr(op, "_directive", False)
        or op.name in _skipped_op_names
    ):
        return True

    if getattr(op, "is_parameterized", False) and op.is_parameterized():
        return True

    from qiskit.dagcircuit.dagnode import DAGOpNode

    # we can proceed if op has defined: to_operator, to_matrix and __array__, or if its definition can be
    # recursively resolved by operations that have a matrix. We check this by constructing an Operator.
    if (
        isinstance(op, DAGOpNode)
        or (hasattr(op, "to_matrix") and hasattr(op, "__array__"))
        or hasattr(op, "to_operator")
    ):
        return False

    return False


def _commutation_precheck(
    op1: Operation,
    qargs1: List,
    cargs1: List,
    op2: Operation,
    qargs2: List,
    cargs2: List,
    max_num_qubits,
):
    if not is_commutation_supported(op1) or not is_commutation_supported(op2):
        return False

    if set(qargs1).isdisjoint(qargs2) and set(cargs1).isdisjoint(cargs2):
        return True

    if is_commutation_skipped(op1, qargs1, max_num_qubits) or is_commutation_skipped(
        op2, qargs2, max_num_qubits
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
    return int.from_bytes(bytes(op_name, encoding="utf-8"), byteorder="big", signed=True)


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
    first_op: Operation,
    first_qargs: List,
    second_op: Operation,
    second_qargs: List,
    _commutation_lib: dict,
) -> Union[bool, None]:
    """Queries and returns the commutation of a pair of operations from a provided commutation library
    Args:
        first_op: first operation.
        first_qargs: first operation's qubits.
        first_cargs: first operation's clbits.
        second_op: second operation.
        second_qargs: second operation's qubits.
        second_cargs: second operation's clbits.
        _commutation_lib (dict): dictionary of commutation relations
    Return:
        True if first_op and second_op commute, False if they do not commute and
        None if the commutation is not in the library
    """

    commutation = _commutation_lib.get((first_op.name, second_op.name), None)

    # Return here if the commutation is constant over all relative placements of the operations
    if commutation is None or isinstance(commutation, bool):
        return commutation

    # If we arrive here, there is an entry in the commutation library but it depends on the
    # placement of the operations and also possibly on operation parameters
    if isinstance(commutation, dict):
        commutation_after_placement = commutation.get(
            _get_relative_placement(first_qargs, second_qargs), None
        )
        # if we have another dict in commutation_after_placement, commutation depends on params
        if isinstance(commutation_after_placement, dict):
            # Param commutation entry exists and must be a dict
            first_params = getattr(first_op, "params", [])
            second_params = getattr(second_op, "params", [])
            return commutation_after_placement.get(
                (_hashable_parameters(first_params), _hashable_parameters(second_params)),
                None,
            )
        else:
            # queried commutation is True, False or None
            return commutation_after_placement
    else:
        raise ValueError("Expected commutation to be None, bool or a dict")


def _commute_matmul(
    first_ops: Operation, first_qargs: List, second_op: Operation, second_qargs: List
):
    qarg = {q: i for i, q in enumerate(first_qargs)}
    num_qubits = len(qarg)
    for q in second_qargs:
        if q not in qarg:
            qarg[q] = num_qubits
            num_qubits += 1

    first_qarg = tuple(qarg[q] for q in first_qargs)
    second_qarg = tuple(qarg[q] for q in second_qargs)

    from qiskit.dagcircuit.dagnode import DAGOpNode

    # If we have a DAGOpNode here we've received a StandardGate definition from
    # rust and we can manually pull the matrix to use for the Operators
    if isinstance(first_ops, DAGOpNode):
        first_ops = first_ops.matrix
    if isinstance(second_op, DAGOpNode):
        second_op = second_op.matrix

    # try to generate an Operator out of op, if this succeeds we can determine commutativity, otherwise
    # return false
    try:
        operator_1 = Operator(
            first_ops, input_dims=(2,) * len(first_qarg), output_dims=(2,) * len(first_qarg)
        )
        operator_2 = Operator(
            second_op, input_dims=(2,) * len(second_qarg), output_dims=(2,) * len(second_qarg)
        )
    except QiskitError:
        return False

    if first_qarg == second_qarg:
        # Use full composition if possible to get the fastest matmul paths.
        op12 = operator_1.compose(operator_2)
        op21 = operator_2.compose(operator_1)
    else:
        # Expand operator_1 to be large enough to contain operator_2 as well; this relies on qargs1
        # being the lowest possible indices so the identity can be tensored before it.
        extra_qarg2 = num_qubits - len(first_qarg)
        if extra_qarg2:
            id_op = _identity_op(extra_qarg2)
            operator_1 = id_op.tensor(operator_1)
        op12 = operator_1.compose(operator_2, qargs=second_qarg, front=False)
        op21 = operator_1.compose(operator_2, qargs=second_qarg, front=True)
    ret = op12 == op21
    return ret
