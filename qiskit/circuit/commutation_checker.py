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

from typing import List, Union, Set, Optional
import numpy as np

from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.utils import deprecate_func
from qiskit._accelerate.commutation_checker import CommutationChecker as RustChecker


class CommutationChecker:
    """This code is essentially copy-pasted from commutative_analysis.py.
    This code cleverly hashes commutativity and non-commutativity results between DAG nodes and seems
    quite efficient for large Clifford circuits.
    They may be other possible efficiency improvements: using rule-based commutativity analysis,
    evicting from the cache less useful entries, etc.
    """

    @deprecate_func(
        additional_msg=(
            "This Python implementation will stop to be maintained in the future. Instead, use the Rust"
            " implementation at qiskit.circuit.commutation_library.SessionCommutationChecker."
        ),
        since="1.3.0",
        pending=True,
    )
    def __init__(
        self,
        standard_gate_commutations: dict = None,
        cache_max_entries: int = 10**6,
        *,
        gates: Optional[Set[str]] = None,
    ):
        self.cc = RustChecker(standard_gate_commutations, cache_max_entries, gates)

    def commute_nodes(
        self,
        op1,
        op2,
        max_num_qubits: int = 3,
    ) -> bool:
        """Checks if two DAGOpNodes commute."""
        return self.cc.commute_nodes(op1, op2, max_num_qubits)

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
        return self.cc.commute(op1, qargs1, cargs1, op2, qargs2, cargs2, max_num_qubits)

    def num_cached_entries(self):
        """Returns number of cached entries"""
        return self.cc.num_cached_entries()

    def clear_cached_commutations(self):
        """Clears the dictionary holding cached commutations"""
        self.cc.clear_cached_commutations()

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
        # Note to the interested reader: This function has a Rust equivalent in the
        # Python-exposed class ``qiskit._accelerate.commutation_checker.CommutationLibrary``,
        # but is no longer part of the Rust version of ``CommutationChecker``.

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
                (
                    _hashable_parameters(first_params),
                    _hashable_parameters(second_params),
                ),
                None,
            )
        else:
            # queried commutation is True, False or None
            return commutation_after_placement
    else:
        raise ValueError("Expected commutation to be None, bool or a dict")
