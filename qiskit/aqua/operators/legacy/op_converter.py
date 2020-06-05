# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" op converter """

# pylint: disable=cyclic-import

from typing import Union, Callable, cast
import itertools
import logging
import sys

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.tools.parallel import parallel_map
from qiskit.tools.events import TextProgressBar

from qiskit.aqua import AquaError, aqua_globals
from .weighted_pauli_operator import WeightedPauliOperator
from .matrix_operator import MatrixOperator
from .tpb_grouped_weighted_pauli_operator import TPBGroupedWeightedPauliOperator

logger = logging.getLogger(__name__)


def _conversion(basis, matrix):
    pauli = Pauli.from_label(''.join(basis))
    trace_value = np.sum(matrix.dot(pauli.to_spmatrix()).diagonal())
    return trace_value, pauli


def to_weighted_pauli_operator(
        operator: Union[WeightedPauliOperator, TPBGroupedWeightedPauliOperator, MatrixOperator]) \
        -> WeightedPauliOperator:
    """
    Converting a given operator to `WeightedPauliOperator`

    Args:
        operator: one of supported operator type
    Returns:
        The converted weighted pauli operator
    Raises:
        AquaError: Unsupported type to convert

    Warnings:
        Converting time from a MatrixOperator to a Pauli-type Operator grows exponentially.
        If you are converting a system with large number of qubits, it will take time.
        You can turn on DEBUG logging to check the progress.
    """
    if operator.__class__ == WeightedPauliOperator:
        return cast(WeightedPauliOperator, operator)
    elif operator.__class__ == TPBGroupedWeightedPauliOperator:
        # destroy the grouping but keep z2 symmetries info
        op_tpb = cast(TPBGroupedWeightedPauliOperator, operator)
        return WeightedPauliOperator(paulis=op_tpb.paulis, z2_symmetries=op_tpb.z2_symmetries,
                                     name=op_tpb.name)
    elif operator.__class__ == MatrixOperator:
        op_m = cast(MatrixOperator, operator)
        if op_m.is_empty():
            return WeightedPauliOperator(paulis=[])
        if op_m.num_qubits > 10:
            logger.warning("Converting time from a MatrixOperator to a Pauli-type Operator grows "
                           "exponentially. If you are converting a system with large number of "
                           "qubits, it will take time. And now you are converting a %s-qubit "
                           "Hamiltonian. You can turn on DEBUG logging to check the progress."
                           "", op_m.num_qubits)
        num_qubits = op_m.num_qubits
        coeff = 2 ** (-num_qubits)

        paulis = []
        possible_basis = 'IXYZ'
        if op_m.dia_matrix is not None:
            possible_basis = 'IZ'

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Converting a MatrixOperator to a Pauli-type Operator:")
            TextProgressBar(sys.stderr)
        results = parallel_map(_conversion,
                               list(itertools.product(possible_basis, repeat=num_qubits)),
                               task_kwargs={"matrix": op_m._matrix},
                               num_processes=aqua_globals.num_processes)
        for trace_value, pauli in results:
            weight = trace_value * coeff
            if weight != 0.0 and np.abs(weight) > op_m.atol:
                paulis.append([weight, pauli])

        return WeightedPauliOperator(paulis, z2_symmetries=operator.z2_symmetries,
                                     name=operator.name)
    else:
        raise AquaError("Unsupported type to convert to WeightedPauliOperator: "
                        "{}".format(operator.__class__))


def to_matrix_operator(
        operator: Union[WeightedPauliOperator, TPBGroupedWeightedPauliOperator, MatrixOperator])\
        -> MatrixOperator:
    """
    Converting a given operator to `MatrixOperator`

    Args:
        operator: one of supported operator type
    Returns:
        the converted matrix operator
    Raises:
        AquaError: Unsupported type to convert
    """
    if operator.__class__ == WeightedPauliOperator:
        op_w = cast(WeightedPauliOperator, operator)
        if op_w.is_empty():
            return MatrixOperator(None)
        hamiltonian = 0
        for weight, pauli in op_w.paulis:
            hamiltonian += weight * pauli.to_spmatrix()
        return MatrixOperator(matrix=hamiltonian, z2_symmetries=op_w.z2_symmetries,
                              name=op_w.name)
    elif operator.__class__ == TPBGroupedWeightedPauliOperator:
        op_tpb = cast(TPBGroupedWeightedPauliOperator, operator)
        op = WeightedPauliOperator(paulis=op_tpb.paulis, z2_symmetries=op_tpb.z2_symmetries,
                                   name=op_tpb.name)
        return to_matrix_operator(op)
    elif operator.__class__ == MatrixOperator:
        return cast(MatrixOperator, operator)
    else:
        raise AquaError("Unsupported type to convert to MatrixOperator: "
                        "{}".format(operator.__class__))


# pylint: disable=invalid-name
def to_tpb_grouped_weighted_pauli_operator(
        operator: Union[WeightedPauliOperator, TPBGroupedWeightedPauliOperator, MatrixOperator],
        grouping_func: Callable, **kwargs: int) -> TPBGroupedWeightedPauliOperator:
    """

    Args:
        operator: one of supported operator type
        grouping_func: a callable function that grouped the paulis in the operator.
        kwargs: other setting for `grouping_func` function

    Returns:
        the converted tensor-product-basis grouped weighted pauli operator

    Raises:
        AquaError: Unsupported type to convert
    """
    if operator.__class__ == WeightedPauliOperator:
        return grouping_func(operator, **kwargs)
    elif operator.__class__ == TPBGroupedWeightedPauliOperator:
        # different tpb grouping approach is asked
        op_tpb = cast(TPBGroupedWeightedPauliOperator, operator)
        if grouping_func != op_tpb.grouping_func and kwargs != op_tpb.kwargs:
            return grouping_func(op_tpb, **kwargs)
        else:
            return op_tpb
    elif operator.__class__ == MatrixOperator:
        op = to_weighted_pauli_operator(operator)
        return grouping_func(op, **kwargs)
    else:
        raise AquaError("Unsupported type to convert to TPBGroupedWeightedPauliOperator: "
                        "{}".format(operator.__class__))
